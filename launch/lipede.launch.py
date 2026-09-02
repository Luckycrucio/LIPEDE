"""Launch FAST-LIPEDE together with its RViz visualization."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    share = Path(get_package_share_directory("lipede"))
    rviz_config = str(share / "rviz" / "lipede.rviz")

    input_topic = LaunchConfiguration("input_topic")
    output_topic = LaunchConfiguration("output_topic")
    people_topic = LaunchConfiguration("people_topic")
    device = LaunchConfiguration("device")
    mode = LaunchConfiguration("mode")
    is_realtime = IfCondition(PythonExpression(["'", mode, "' == 'real_time'"]))
    is_offline = IfCondition(PythonExpression(["'", mode, "' == 'offline'"]))

    return LaunchDescription([
        DeclareLaunchArgument("input_topic", default_value="/ouster/points"),
        DeclareLaunchArgument("output_topic", default_value="/ouster/points/processed"),
        DeclareLaunchArgument("people_topic", default_value="/ouster/points/people"),
        DeclareLaunchArgument("device", default_value="cuda"),
        DeclareLaunchArgument(
            "mode", default_value="real_time",
            description="Processing mode: 'real_time' or 'offline'",
        ),
        DeclareLaunchArgument(
            "bag_path", default_value="",
            description="Input rosbag URI (required in offline mode)",
        ),
        DeclareLaunchArgument(
            "output_bag_path", default_value="",
            description="Offline output URI; defaults to <bag_path>_lipede",
        ),
        DeclareLaunchArgument("overwrite_output", default_value="false"),
        Node(
            package="lipede",
            executable="lipede_node",
            name="lipede",
            output="screen",
            condition=is_realtime,
            parameters=[{
                "input_topic": input_topic,
                "output_topic": output_topic,
                "people_topic": people_topic,
                "device": device,
            }],
        ),
        Node(
            package="lipede",
            executable="lipede_offline",
            name="lipede_offline",
            output="screen",
            condition=is_offline,
            parameters=[{
                "bag_path": LaunchConfiguration("bag_path"),
                "output_bag_path": LaunchConfiguration("output_bag_path"),
                "overwrite_output": LaunchConfiguration("overwrite_output"),
                "input_topic": input_topic,
                "output_topic": output_topic,
                "people_topic": people_topic,
                "device": device,
            }],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="lipede_rviz",
            output="screen",
            arguments=["-d", rviz_config],
            remappings=[
                ("/ouster/points", input_topic),
                ("/ouster/points/processed", output_topic),
                ("/ouster/points/people", people_topic),
            ],
        ),
    ])
