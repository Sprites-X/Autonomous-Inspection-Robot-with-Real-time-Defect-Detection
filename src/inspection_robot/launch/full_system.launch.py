from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # map_yaml and use_ble_fallback are launch args (not hardcoded) because the
    # same launch file is reused across the simulated map and the real site —
    # only the map file and whether GPS-denied fallback is needed should change
    # between them, not the launch structure itself.
    map_yaml_arg = DeclareLaunchArgument(
        'map', default_value='maps/sim_map.yaml', description='Path to map YAML'
    )
    use_ble_fallback_arg = DeclareLaunchArgument(
        'use_ble_fallback', default_value='false',
        description='Enable BLE trilateration fallback localization'
    )

    pkg_share = get_package_share_directory('inspection_robot')
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')

    # Nav2's own launch file is included rather than re-declaring its dozen
    # internal nodes here — this file only owns composing the inspection-
    # specific pieces (perception, patrol logic, optional fallback) around it.
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_dir, 'launch', 'bringup_launch.py')
        ),
        launch_arguments={'map': LaunchConfiguration('map')}.items(),
    )

    perception_node = Node(
        package='inspection_robot',
        executable='defect_detector_server',
        name='defect_detector_server',
        output='screen',
        parameters=[os.path.join(pkg_share, 'config', 'perception_params.yaml')],
    )

    # patrol_navigator is started after perception/nav2 are declared, but ROS2
    # doesn't guarantee startup ordering from launch file position alone — the
    # node itself is written to wait_for_server()/service_is_ready() rather
    # than assuming its dependencies are up, so this ordering is for
    # readability, not correctness.
    navigator_node = Node(
        package='inspection_robot',
        executable='patrol_navigator',
        name='patrol_navigator',
        output='screen',
        parameters=[os.path.join(pkg_share, 'config', 'patrol_waypoints.yaml')],
    )

    # BLE fallback is conditional rather than always-on: it publishes an
    # independent pose estimate (see ble_localizer.py), and running it in
    # areas with good GPS/AMCL coverage would just add noise for no benefit.
    ble_node = Node(
        package='inspection_robot',
        executable='ble_localizer',
        name='ble_localizer',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_ble_fallback')),
    )

    return LaunchDescription([
        map_yaml_arg,
        use_ble_fallback_arg,
        nav2_launch,
        perception_node,
        navigator_node,
        ble_node,
    ])
