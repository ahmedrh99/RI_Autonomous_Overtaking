"""
Launch file used to ruin the multi object tracker 

Carla_ros_bridge: 
used to communicate between ros2 and carla

spawn_lidar.py:
used to spawn lidar and camera produce camera detections

lidar_centerpoint:
takes lidar data and produces detections 

fusion.py:
used for sensor fusion between camera detections and lidar center points

multi_object_tracker:
used from autoware.universe to track detections


"""
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource, AnyLaunchDescriptionSource
from launch.substitutions import ThisLaunchFileDir
import os

def generate_launch_description():
    return LaunchDescription([

        # Launch CARLA ROS bridge
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                os.path.join(
                    get_package_share_directory('carla_ros_bridge'),
                    'launch',
                    'carla_ros_bridge.launch.py'
                )
            ]),
            launch_arguments={
                'host': 'host.docker.internal',
            }.items()
        ),

        # Spawn LiDAR
        ExecuteProcess(
            cmd=['ros2', 'run', 'carla_ros_bridge', 'spawn_lidar', '--ros-args', '-p', 'use_sim_time:=true'],
            output='screen'
        ),

        # LiDAR Enricher
        ExecuteProcess(
            cmd=['ros2', 'run', 'carla_ros_bridge', 'carla_lidar_enricher', '--ros-args', '-p', 'use_sim_time:=true'],
            output='screen'
        ),

        # Custom Fusion Node
        ExecuteProcess(
            cmd=['ros2', 'run', 'carla_ros_bridge', 'fusion', '--ros-args', '-p', 'use_sim_time:=true'],
            output='screen'
        ),

        # Autoware Pointcloud Preprocessor
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource([
                os.path.join(
                    get_package_share_directory('autoware_pointcloud_preprocessor'),
                    'launch',
                    'preprocessor.launch.xml'
                )
            ])
        ),

        # Autoware CenterPoint Detector
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource([
                os.path.join(
                    get_package_share_directory('autoware_lidar_centerpoint'),
                    'launch',
                    'lidar_centerpoint.launch.xml'
                )
            ]),
            launch_arguments={
                'input/pointcloud': '/points_raw/cropbox/filtered',
                'output/objects': '/perception/objects',
                'data_path': '/root/autoware_data',
                'model_name': 'centerpoint'
            }.items()
        ),

        # Multi Object Tracker
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource([
                os.path.join(
                    get_package_share_directory('autoware_multi_object_tracker'),
                    'launch',
                    'multi_object_tracker.launch.xml'
                )
            ])
        )
    ])
