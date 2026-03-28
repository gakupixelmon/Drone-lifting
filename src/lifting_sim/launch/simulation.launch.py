import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('lifting_sim')

    model_path = os.path.join(pkg_share, 'models')
    for var in ['IGN_GAZEBO_RESOURCE_PATH', 'GZ_SIM_RESOURCE_PATH']:
        if var in os.environ:
            os.environ[var] += ':' + model_path
        else:
            os.environ[var] = model_path

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ros_gz_sim'),
                'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            'gz_args': f'-r {os.path.join(pkg_share, "worlds", "lifting.sdf")}'
        }.items(),
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
            '/m5fly/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist',
            '/camera/image_raw@sensor_msgs/msg/Image[ignition.msgs.Image',
        ],
        output='screen'
    )

    controller = Node(
        package='lifting_sim',
        executable='controller',
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    return LaunchDescription([gz_sim, bridge, controller])
