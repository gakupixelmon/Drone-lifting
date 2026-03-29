import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np

class LiftingController(Node):
    def __init__(self):
        super().__init__('lifting_controller')

        self.set_parameters([rclpy.parameter.Parameter(
            'use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.publisher = self.create_publisher(Twist, '/m5fly/cmd_vel', 10)
        self.debug_publisher = self.create_publisher(Image, '/camera/debug_image', 10)
        self.bridge = CvBridge()

        # PID ゲイン
        self.kp_z = 0.008
        self.ki_z = 0.0003
        self.kd_z = 0.001
        self.kp_y = 0.01

        self.default_height = 400
        self.MAX_VZ = 0.5

        self.state = 'TRACKING'
        self.state_timer = 0.0
        self.state_smash_vz = 1.0  # SMASHステートで使う速度(発火時に確定)

        self.integral_z = 0.0
        self.prev_error_z = None
        self.prev_time = None
        self.derivative_z_filtered = 0.0

        self.lost_start_time = None
        self.prev_ball_pos = None
        self.ball_vy = 0.0

        # ---------------------------------------------------------------
        # 適応型SMASH速度制御
        #
        # 「できました」時のパラメータ(dist/duration)はそのまま維持。
        # 変えるのは固定値 1.5 → ball_vy から計算した適応VZのみ。
        #
        # 弾性衝突式 (上方向を正):
        #   v_ball_after = ratio_fall * v_fall + ratio_vel * VZ
        #   ratio_vel  = 2M/(M+m) = 1.857
        #   ratio_fall = (M-m)/(M+m) = 0.873
        #
        #   速く落ちるボール(v_fall大) → ratio_fall*v_fall が大きい
        #   → 必要なVZは小さくなる → 毎回同じ高さに到達できる
        #
        # TARGET_BALL_RISE_PX: スマッシュ後の目標到達高さ [px]
        #   ★ここを変えると高さが変わる
        #   40px ≈ 23cm: 控えめ
        #   50px ≈ 29cm: 中程度  ← デフォルト
        #   60px ≈ 35cm: 高め
        #
        # VZ_MAX = 1.5: 「できました」と同じ上限。これ以上は出ない。
        # VZ_MIN = 0.3: ボールが速く落ちてもある程度は打ち上げる。
        # ---------------------------------------------------------------
        self.TARGET_BALL_RISE_PX = 50   # ★高さ調整はここだけ
        self.SMASH_VZ_MIN  = 0.3        # m/s
        self.SMASH_VZ_MAX  = 1.5        # m/s (「できました」と同じ上限)
        self.SMASH_DURATION = 0.08      # s   (「できました」と同じ)

        # 「できました」と同じSMASH発火条件
        self.SMASH_DIST_MIN      = 5    # px
        self.SMASH_DIST_MAX      = 120  # px
        self.SMASH_DX_MAX        = 80   # px
        self.SMASH_BALL_VY_MIN   = 10.0 # px/s (落下中のみ)
        self.SMASH_STABLE_FRAMES = 1
        self.COOLDOWN_SEC        = 0.3

        # 物理定数
        self.DRONE_MASS = 0.035   # kg
        self.BALL_MASS  = 0.0027  # kg
        self.GRAVITY    = 9.8     # m/s²
        self.M_PER_PX   = 0.0058  # 1px ≈ 0.58cm (距離4m, FOV1.5rad)

        self.smash_candidate_frames = 0
        self._log_count = 0
        self._smash_log_count = 0

        self.get_logger().info(
            f"Controller started: TARGET_BALL_RISE={self.TARGET_BALL_RISE_PX}px"
            f"({self.TARGET_BALL_RISE_PX*self.M_PER_PX*100:.0f}cm), "
            f"VZ=[{self.SMASH_VZ_MIN},{self.SMASH_VZ_MAX}]m/s")

    # -------------------------------------------------------------------
    def calc_smash_vz(self, ball_vy_px: float) -> float:
        """
        ボール落下速度から一定高さに到達するSMASH速度を計算。

        弾性衝突: v_ball_after = ratio_fall*v_fall + ratio_vel*VZ
        → VZ = (v_ball_needed - ratio_fall*v_fall) / ratio_vel
        """
        M, m, g = self.DRONE_MASS, self.BALL_MASS, self.GRAVITY
        h = self.TARGET_BALL_RISE_PX * self.M_PER_PX      # 目標高さ [m]
        v_need = np.sqrt(2 * g * h)                        # 必要ボール速度 [m/s]
        v_fall = max(0.0, ball_vy_px * self.M_PER_PX)     # 落下速度 [m/s]
        ratio_vel  = 2 * M / (M + m)                       # 1.857
        ratio_fall = (M - m) / (M + m)                     # 0.873
        vz = (v_need - ratio_fall * v_fall) / ratio_vel
        return float(np.clip(vz, self.SMASH_VZ_MIN, self.SMASH_VZ_MAX))

    # -------------------------------------------------------------------
    def get_best_blob(self, hsv_image, lower_color, upper_color,
                      debug_image, draw_color, label, min_area=50):
        mask = cv2.inRange(hsv_image, lower_color, upper_color)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_pos = None
        if contours:
            valid = [c for c in contours if cv2.contourArea(c) > min_area]
            if valid:
                cnt = max(valid, key=cv2.contourArea)
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.circle(debug_image, (cX, cY), 5, draw_color, -1)
                    cv2.putText(debug_image, f"{label} Y:{cY}",
                                (cX + 10, cY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 2)
                    best_pos = (cX, cY)
        return best_pos

    # -------------------------------------------------------------------
    def image_callback(self, msg):
        current_time = self.get_clock().now().nanoseconds / 1e9

        if self.prev_time is None:
            self.prev_time = current_time
            return
        dt = current_time - self.prev_time
        if dt <= 0:
            return
        self.prev_time = current_time

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception:
            return

        debug_img = cv_image.copy()
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        drone_pos = self.get_best_blob(
            hsv, np.array([100, 120, 80]), np.array([130, 255, 255]),
            debug_img, (255, 0, 0), "Drone", min_area=50)

        ball_pos = self.get_best_blob(
            hsv, np.array([10, 100, 100]), np.array([25, 255, 255]),
            debug_img, (0, 0, 255), "Ball", min_area=20)

        if ball_pos is not None and self.prev_ball_pos is not None:
            raw_vy = (ball_pos[1] - self.prev_ball_pos[1]) / dt
            self.ball_vy = 0.7 * self.ball_vy + 0.3 * raw_vy
        elif ball_pos is None:
            self.ball_vy = 0.0
        self.prev_ball_pos = ball_pos

        twist = Twist()
        twist.linear.x = 0.0
        cmd_z_val = 0.0
        error_z_display = 0.0

        if drone_pos is not None:
            if self.lost_start_time is not None:
                self.prev_error_z = None
                self.derivative_z_filtered = 0.0
            self.lost_start_time = None

            dx, dy = drone_pos
            target_x = ball_pos[0] if ball_pos else 640
            twist.linear.y = float(
                np.clip(self.kp_y * (dx - target_x), -1.0, 1.0))

            target_h = self.default_height
            error_z = dy - target_h
            error_z_display = error_z

            if self.prev_error_z is None:
                self.prev_error_z = error_z
                self.derivative_z_filtered = 0.0

            if abs(error_z) < 80:
                self.integral_z += error_z * dt
                self.integral_z = np.clip(self.integral_z, -100.0, 100.0)
            else:
                self.integral_z *= 0.95

            raw_derivative = (error_z - self.prev_error_z) / dt
            self.derivative_z_filtered = (0.6 * self.derivative_z_filtered
                                          + 0.4 * raw_derivative)
            self.prev_error_z = error_z

            pid_cmd_z = float(np.clip(
                self.kp_z * error_z
                + self.ki_z * self.integral_z
                + self.kd_z * self.derivative_z_filtered,
                -self.MAX_VZ, self.MAX_VZ))

            if self.state == 'TRACKING':
                twist.linear.z = pid_cmd_z

                if ball_pos:
                    dist_to_ball = dy - ball_pos[1]
                    dx_ball = abs(dx - ball_pos[0])
                    adaptive_vz = self.calc_smash_vz(self.ball_vy)

                    is_stable      = abs(dy - target_h) < 40
                    is_ball_above  = (self.SMASH_DIST_MIN <= dist_to_ball <= self.SMASH_DIST_MAX) \
                                     and (dx_ball < self.SMASH_DX_MAX)
                    is_ball_falling = self.ball_vy > self.SMASH_BALL_VY_MIN

                    self._smash_log_count += 1
                    if self._smash_log_count >= 5:
                        self._smash_log_count = 0
                        self.get_logger().info(
                            f"[SMASH?] dist={dist_to_ball:.0f}px "
                            f"ball_vy={self.ball_vy:.0f}px/s "
                            f"adaptVZ={adaptive_vz:.3f}m/s "
                            f"stable={is_stable} above={is_ball_above} "
                            f"falling={is_ball_falling}")

                    if is_stable and is_ball_above and is_ball_falling:
                        self.smash_candidate_frames += 1
                    else:
                        self.smash_candidate_frames = 0

                    if self.smash_candidate_frames >= self.SMASH_STABLE_FRAMES:
                        self.smash_candidate_frames = 0
                        # 発火時のball_vyで適応VZを確定
                        self.state_smash_vz = self.calc_smash_vz(self.ball_vy)
                        self.state = 'SMASH'
                        self.state_timer = current_time
                        self.integral_z = 0.0
                        self.get_logger().info(
                            f">>> SMASH! dist={dist_to_ball:.0f}px "
                            f"ball_vy={self.ball_vy:.0f}px/s "
                            f"VZ={self.state_smash_vz:.3f}m/s "
                            f"(target={self.TARGET_BALL_RISE_PX}px)")
                else:
                    self.smash_candidate_frames = 0

            elif self.state == 'SMASH':
                twist.linear.z = self.state_smash_vz  # 適応VZ
                if current_time - self.state_timer > self.SMASH_DURATION:
                    self.state = 'COOLDOWN'
                    self.state_timer = current_time

            elif self.state == 'COOLDOWN':
                twist.linear.z = pid_cmd_z
                if current_time - self.state_timer > self.COOLDOWN_SEC:
                    self.state = 'TRACKING'

            cmd_z_val = twist.linear.z

            # デバッグ描画
            cv2.line(debug_img, (0, target_h), (1280, target_h), (0, 255, 0), 2)
            cv2.rectangle(debug_img,
                          (0, target_h - 40), (1280, target_h + 40), (0, 255, 0), 1)
            if ball_pos:
                dist_to_ball = dy - ball_pos[1]
                adaptive_vz = self.calc_smash_vz(self.ball_vy)
                vy_color = (0, 255, 255) if self.ball_vy > 10 else (128, 128, 128)
                cv2.putText(debug_img,
                            f"BallVy:{self.ball_vy:.0f}  dist:{dist_to_ball:.0f}px",
                            (10, 510), cv2.FONT_HERSHEY_SIMPLEX, 0.7, vy_color, 2)
                cv2.putText(debug_img,
                            f"AdaptVZ:{adaptive_vz:.3f}m/s",
                            (10, 540), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 128, 0), 2)
            cv2.putText(debug_img,
                        f"I:{self.integral_z:.1f}  Err:{error_z:.0f}px",
                        (10, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)

        else:
            if self.lost_start_time is None:
                self.lost_start_time = current_time
                self.prev_error_z = None
                self.derivative_z_filtered = 0.0
            lost_duration = current_time - self.lost_start_time

            cmd_z_val = -0.2 if lost_duration < 0.5 else -0.3

            if lost_duration >= 1.5:
                self.state = 'TRACKING'
                self.prev_error_z = None
                self.integral_z = 0.0
                self.derivative_z_filtered = 0.0

            self.smash_candidate_frames = 0
            twist.linear.z = cmd_z_val
            cv2.putText(debug_img,
                        f"LOST {lost_duration:.1f}s",
                        (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        self._log_count += 1
        if self._log_count >= 30:
            self._log_count = 0
            if drone_pos is not None:
                self.get_logger().info(
                    f"[TRACK] state={self.state} "
                    f"drone=({drone_pos[0]},{drone_pos[1]}) "
                    f"err_z={error_z_display:.0f}px cmdZ={cmd_z_val:.3f}")
            else:
                lost_dur = (current_time - self.lost_start_time
                            if self.lost_start_time else 0.0)
                self.get_logger().warn(
                    f"[LOST] {lost_dur:.1f}s cmdZ={cmd_z_val:.3f}")

        cv2.putText(debug_img, f"State: {self.state}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(debug_img, f"CmdZ: {cmd_z_val:.2f}",
                    (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        self.publisher.publish(twist)
        try:
            self.debug_publisher.publish(
                self.bridge.cv2_to_imgmsg(debug_img, "bgr8"))
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = LiftingController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
