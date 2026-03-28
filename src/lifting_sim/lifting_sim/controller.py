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

        # ---------------------------------------------------------------
        # PID ゲイン
        #
        # Bug4修正: カスケード比率の改善
        #   m5model.sdf の velocityGain を 5.0→15.0 に変更することで
        #   τ_inner = 1/15.0 ≈ 0.067s
        #   τ_outer = 1/(kp_z × 171.8px/m) = 1/(0.008×171.8) ≈ 0.73s
        #   比率 0.067/0.73 ≈ 0.092 → 安定カスケード条件(<0.1)を満たす
        #
        # Bug2修正: kd_z を 0.008→0.001 に大幅削減
        #   30Hz検出ノイズ±2px → raw derivative ≈ ±60px/s
        #   旧: 0.008×60 = ±0.48 m/s (MAX_VZの96%がノイズ)
        #   新: 0.001×60 = ±0.06 m/s (12%に抑制)
        #   加えてEMAフィルターでさらにノイズを低減
        # ---------------------------------------------------------------
        self.kp_z = 0.008
        self.ki_z = 0.0003
        self.kd_z = 0.001
        self.kp_y = 0.01

        # ---------------------------------------------------------------
        # 目標高さ (画像 Y 座標)
        #
        # 計算根拠:
        #   カメラ z=0.55m(モデルz=0.5m + センサーオフセット0.05m)
        #   水平FOV=1.5rad, 距離4m → 垂直FOV≈55.8度
        #   ドローン目標 z=0.30m → カメラより0.25m下 → 仰角3.58度(下向き)
        #   → Y = 360 + 3.58/27.9 × 360 ≈ 406px ≈ 400px
        #
        # 目安: z=0.1m → Y≈454,  z=0.3m → Y≈406,  z=0.5m → Y≈360
        #        z=0.7m → Y≈314,  z=0.9m → Y≈268
        # ---------------------------------------------------------------
        self.default_height = 400

        # ---------------------------------------------------------------
        # Z軸最大速度 / スマッシュ制御パラメータ
        # ---------------------------------------------------------------
        self.MAX_VZ = 0.5
        self.smash_base_vz = 1.5
        self.smash_vy_target = 80.0
        self.smash_vy_gain = 0.005
        self.smash_ball_vy = 0.0

        self.state = 'TRACKING'
        self.state_timer = 0.0

        self.integral_z = 0.0
        self.prev_error_z = None
        self.prev_time = None

        # Bug2修正: derivative EMAフィルター用
        # raw derivative を毎フレーム 0.6:0.4 でブレンドしノイズを除去
        self.derivative_z_filtered = 0.0

        # LOST管理
        self.lost_start_time = None

        # ボール速度追跡（スマッシュ条件: 落下中のみスマッシュ）
        self.prev_ball_pos = None
        self.ball_vy = 0.0

        self.get_logger().info(
            f"Controller started: default_height={self.default_height}, "
            f"MAX_VZ={self.MAX_VZ}")

    # -------------------------------------------------------------------
    def get_best_blob(self, hsv_image, lower_color, upper_color,
                      debug_image, draw_color, label):
        """HSVマスクで最大面積の輪郭を検出する"""
        mask = cv2.inRange(hsv_image, lower_color, upper_color)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_pos = None
        if contours:
            valid = [c for c in contours if cv2.contourArea(c) > 10]
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

        # ---------------------------------------------------------------
        # HSV検出 (OGRE2対応の広範囲)
        # H[85-155] S[40-255] V[20-255] に拡張
        # ---------------------------------------------------------------
        drone_pos = self.get_best_blob(
            hsv,
            np.array([85,  40,  20]),
            np.array([155, 255, 255]),
            debug_img, (255, 0, 0), "Drone")

        ball_pos = self.get_best_blob(
            hsv,
            np.array([10, 100, 100]),
            np.array([25, 255, 255]),
            debug_img, (0, 0, 255), "Ball")

        # ---------------------------------------------------------------
        # ボール速度計算（スマッシュ判定に使用）
        # 指数移動平均でノイズを除去
        # ---------------------------------------------------------------
        if ball_pos is not None and self.prev_ball_pos is not None:
            raw_vy = (ball_pos[1] - self.prev_ball_pos[1]) / dt
            self.ball_vy = 0.7 * self.ball_vy + 0.3 * raw_vy
        elif ball_pos is None:
            self.ball_vy = 0.0
        self.prev_ball_pos = ball_pos

        # ---------------------------------------------------------------
        twist = Twist()
        twist.linear.x = 0.0
        cmd_z_val = 0.0

        if drone_pos is not None:
            # -----------------------------------------------------------
            # Bug3修正: LOSTから復帰した瞬間にprev_error_zをリセット
            #
            # 旧: LOST<1.5sで再検出してもprev_error_zが古い値のまま残る
            #     → derivative = (現在値 - 古い値) / dt が巨大スパイクになる
            #     → 制御出力が瞬間的に飽和し上方向にバイアスが乗る
            # 修正: lost_start_timeがセットされていた(=LOSTからの復帰)場合に
            #       prevをNoneにリセットし、このフレームのderivativeを0にする
            # -----------------------------------------------------------
            if self.lost_start_time is not None:
                self.prev_error_z = None
                self.derivative_z_filtered = 0.0
            self.lost_start_time = None

            dx, dy = drone_pos

            # -----------------------------------------------------------
            # Bug1修正: 横制御の符号を反転
            #
            # カメラ pose yaw=π のため座標変換:
            #   image +u (右)  = world -Y
            #   image +v (下)  = world -Z
            #
            # ドローンが world +Y にいる → image上では左(dx < target_x)
            # 旧: kp_y * (target_x - dx) → dx<target_xで正 → linear.y>0
            #     → world +Y へ移動 → さらに左に映る → 誤差増大 = 正帰還(発散)
            # 修正: kp_y * (dx - target_x) → dx<target_xで負 → linear.y<0
            #     → world -Y へ移動 → 画像右に戻る = 負帰還(安定) ✓
            # -----------------------------------------------------------
            target_x = ball_pos[0] if ball_pos else 640
            twist.linear.y = float(
                np.clip(self.kp_y * (dx - target_x), -1.0, 1.0))

            # -----------------------------------------------------------
            # 縦制御 (PID)
            # error_z > 0: ドローンが目標より下(Y大) → 上昇コマンド
            # error_z < 0: ドローンが目標より上(Y小) → 降下コマンド
            # -----------------------------------------------------------
            target_h = self.default_height
            error_z = dy - target_h

            if self.prev_error_z is None:
                # 初回または復帰時: derivativeは0として扱う
                self.prev_error_z = error_z
                self.derivative_z_filtered = 0.0

            # 積分ワインドアップ対策
            # 目標付近(±80px)のみ積分を有効化し蓄積を防ぐ
            if abs(error_z) < 80:
                self.integral_z += error_z * dt
                self.integral_z = np.clip(self.integral_z, -100.0, 100.0)
            else:
                self.integral_z *= 0.95

            # Bug2修正: raw derivativeをEMAフィルタリング
            # ノイズ±2px/0.033s=±60px/s のうち40%のみ採用し残り60%は前回値
            # → 実効ノイズ影響を 1フレームで約40%、2フレームで約16% に低減
            raw_derivative = (error_z - self.prev_error_z) / dt
            self.derivative_z_filtered = (0.6 * self.derivative_z_filtered
                                          + 0.4 * raw_derivative)
            self.prev_error_z = error_z

            pid_cmd_z = float(np.clip(
                self.kp_z * error_z
                + self.ki_z * self.integral_z
                + self.kd_z * self.derivative_z_filtered,
                -self.MAX_VZ, self.MAX_VZ))

            # -----------------------------------------------------------
            # ステートマシン
            # -----------------------------------------------------------
            if self.state == 'TRACKING':
                twist.linear.z = pid_cmd_z

                if ball_pos:
                    # dist_to_ball > 0: ドローンが画像で下=ボールがドローンの上
                    dist_to_ball = dy - ball_pos[1]
                    dx_ball = abs(dx - ball_pos[0])

                    is_stable      = abs(dy - target_h) < 40
                    is_ball_above  = (15 < dist_to_ball < 80) and (dx_ball < 60)
                    # ボール落下中のみスマッシュ (上昇中・静止中は無視)
                    is_ball_falling = self.ball_vy > 20.0

                    if is_stable and is_ball_above and is_ball_falling:
                        self.state = 'SMASH'
                        self.state_timer = current_time
                        self.integral_z = 0.0
                        self.smash_ball_vy = self.ball_vy
                        self.get_logger().info(
                            f"SMASH: drone_y={dy}, ball_y={ball_pos[1]}, "
                            f"dist={dist_to_ball:.0f}px, "
                            f"ball_vy={self.ball_vy:.0f}px/s")

            elif self.state == 'SMASH':
                calculated_vz = self.smash_base_vz - self.smash_vy_gain * (self.smash_ball_vy - self.smash_vy_target)
                twist.linear.z = float(np.clip(calculated_vz, 0.5, 2.0))
                if current_time - self.state_timer > 0.08:
                    self.state = 'COOLDOWN'
                    self.state_timer = current_time

            elif self.state == 'COOLDOWN':
                twist.linear.z = pid_cmd_z
                if current_time - self.state_timer > 0.3:
                    self.state = 'TRACKING'

            cmd_z_val = twist.linear.z

            # デバッグ描画
            cv2.line(debug_img, (0, target_h), (1280, target_h),
                     (0, 255, 0), 2)
            cv2.rectangle(debug_img,
                          (0, target_h - 40), (1280, target_h + 40),
                          (0, 255, 0), 1)
            if ball_pos:
                vy_color = (0, 255, 255) if self.ball_vy > 20 else (128, 128, 128)
                cv2.putText(debug_img,
                            f"BallVy:{self.ball_vy:.0f}px/s",
                            (10, 510), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, vy_color, 2)
            cv2.putText(debug_img,
                        f"I:{self.integral_z:.1f}",
                        (10, 480), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (200, 200, 0), 2)

        else:
            # -----------------------------------------------------------
            # LOST処理
            #
            # Bug3修正: LOST開始時にprev_error_zをリセット
            #   LOSTが短時間(例0.3s)で終わり再検出された場合でも
            #   「再検出フレームでのリセット処理」が走るため二重になるが
            #   ここでも先にリセットしておくことで確実性を高める
            #
            # 降下方向のみコマンドを出し上昇バイアスを防ぐ:
            #   0〜0.5s : -0.2 m/s
            #   0.5s〜  : -0.3 m/s
            #   1.5s〜  : PID完全リセット
            # -----------------------------------------------------------
            if self.lost_start_time is None:
                self.lost_start_time = current_time
                self.prev_error_z = None          # 追加: スパイク防止
                self.derivative_z_filtered = 0.0  # 追加: フィルター値もリセット
            lost_duration = current_time - self.lost_start_time

            if lost_duration < 0.5:
                cmd_z_val = -0.2
            else:
                cmd_z_val = -0.3

            if lost_duration >= 1.5:
                self.state = 'TRACKING'
                self.prev_error_z = None
                self.integral_z = 0.0
                self.derivative_z_filtered = 0.0

            twist.linear.z = cmd_z_val
            cv2.putText(debug_img,
                        f"LOST {lost_duration:.1f}s  cmdZ:{cmd_z_val:.2f}",
                        (10, 200), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)

        cv2.putText(debug_img, f"State: {self.state}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(debug_img, f"Cmd Z: {cmd_z_val:.2f}",
                    (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        self.publisher.publish(twist)
        try:
            self.debug_publisher.publish(
                self.bridge.cv2_to_imgmsg(debug_img, "bgr8"))
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    controller = LiftingController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()