"""Launch FAST-LIPEDE together with its RViz visualization."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = Path(get_package_share_directory("fast_lipede"))
    rviz_config = str(share / "rviz" / "fast_lipede.rviz")

    input_topic = LaunchConfiguration("input_topic")
    output_topic = LaunchConfiguration("output_topic")
    people_topic = LaunchConfiguration("people_topic")
    device = LaunchConfiguration("device")

    return LaunchDescription([
        DeclareLaunchArgument("input_topic", default_value="/ouster/points"),
        DeclareLaunchArgument("output_topic", default_value="/ouster/points/processed"),
        DeclareLaunchArgument("people_topic", default_value="/ouster/points/people"),
        DeclareLaunchArgument("device", default_value="cuda"),
        Node(
            package="fast_lipede",
            executable="fast_lipede_node",
            name="fast_lipede",
            output="screen",
            parameters=[{
                "input_topic": input_topic,
                "output_topic": output_topic,
                "people_topic": people_topic,
                "device": device,
            }],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="fast_lipede_rviz",
            output="screen",
            arguments=["-d", rviz_config],
            remappings=[
                ("/ouster/points", input_topic),
                ("/ouster/points/processed", output_topic),
                ("/ouster/points/people", people_topic),
            ],
        ),
    ])
