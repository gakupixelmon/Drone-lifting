FROM osrf/ros:humble-desktop

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-opencv \
    ros-humble-cv-bridge \
    ros-humble-actuator-msgs \
    ignition-fortress \
    ros-humble-ros-gz-sim \
    ros-humble-ros-gz-bridge \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# Build the ROS 2 workspace
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && colcon build"

# Source both setups at runtime
CMD ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && source /app/install/setup.bash && ros2 launch lifting_sim simulation.launch.py"]
