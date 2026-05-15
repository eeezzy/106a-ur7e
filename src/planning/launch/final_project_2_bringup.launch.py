from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    ur_type    = LaunchConfiguration('ur_type',    default='ur7e')
    launch_rviz = LaunchConfiguration('launch_rviz', default='true')

    # RealSense camera
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('realsense2_camera'),
                'launch', 'rs_launch.py'
            )
        ),
        launch_arguments={
            'pointcloud.enable':          'true',
            'rgb_camera.color_profile':   '1920x1080x30',
            'publish_tf':                 'false',
        }.items(),
    )

    # Colored-block perception node
    perception_node = Node(
        package='perception',
        executable='process_pointcloud',
        name='colored_block_perception',
        output='screen',
    )

    # Camera TF broadcaster (wrist_3_link -> camera frames)
    tf_node = Node(
        package='planning',
        executable='tf',
        name='tf_node',
        output='screen',
    )

    # MoveIt
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ur_moveit_config'),
                'launch', 'ur_moveit.launch.py'
            )
        ),
        launch_arguments={
            'ur_type':    ur_type,
            'launch_rviz': launch_rviz,
        }.items(),
    )

    # ArUco detection — tag #10, 5x5_50 dictionary, 50 mm marker size
    aruco_node = Node(
        package='ros2_aruco',
        executable='aruco_node',
        name='aruco_node',
        parameters=[{
            'marker_size':         0.05,
            'aruco_dictionary_id': 'DICT_5X5_50',
            'image_topic':         '/camera/camera/color/image_raw',
            'camera_info_topic':   '/camera/camera/color/camera_info',
        }],
        output='screen',
    )

    # Main pick-and-drop planning node
    planning_node = Node(
        package='planning',
        executable='main',
        name='final_project_2_node',
        output='screen',
    )

    # Delay perception, TF, ArUco, and planning nodes until MoveIt is up
    # and base_link is being published in the TF tree.
    delayed_nodes = TimerAction(
        period=10.0,
        actions=[
            tf_node,
            perception_node,
            aruco_node,
            planning_node,
        ]
    )

    return LaunchDescription([
        realsense_launch,
        moveit_launch,
        delayed_nodes,
    ])
