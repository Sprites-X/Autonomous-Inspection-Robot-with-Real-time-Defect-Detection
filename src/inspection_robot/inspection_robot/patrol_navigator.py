import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup

from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
import tf2_ros

from inspection_interfaces.srv import DetectDefect


class PatrolNavigator(Node):
    """
    Drives the patrol loop and owns the link between "where the robot is" and
    "what the robot saw", since defect_detector_server intentionally has no
    access to /tf (kept a pure perception node — see its own comments).

    Waypoints are consumed one at a time via the NavigateToPose action rather
    than a single multi-point route, so a defect check can be inserted between
    every leg without fighting Nav2's own path execution.
    """

    def __init__(self):
        super().__init__('patrol_navigator')

        self.declare_parameter('waypoints_frame', 'map')
        self.frame = self.get_parameter('waypoints_frame').value

        # Loaded from config/patrol_waypoints.yaml so the route can change
        # without a rebuild. The defaults below are the fallback route used
        # when no params file is passed.
        self.declare_parameter(
            'waypoints', [1.0, 2.0, 0.0, 3.5, 2.0, 0.0, 3.5, -1.0, 0.0]
        )
        self.waypoints = self._load_waypoints()
        self.current_index = 0

        cb_group = ReentrantCallbackGroup()

        self.nav_client = ActionClient(
            self, NavigateToPose, 'navigate_to_pose', callback_group=cb_group
        )
        self.detect_client = self.create_client(
            DetectDefect, 'detect_defect', callback_group=cb_group
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Perception runs on the latest cached frame rather than a fresh
        # subscription per check, since the camera topic is high-rate and we
        # only need "whatever the robot currently sees" at each waypoint.
        self.latest_image = None
        self.create_subscription(Image, '/camera/image_raw', self._cache_image, 1)

        self.get_logger().info(f'patrol_navigator ready, {len(self.waypoints)} waypoints loaded')
        self._send_next_goal()

    def _load_waypoints(self):
        """Chunk the flat [x, y, yaw, ...] parameter into (x, y, yaw) tuples.

        The parameter is flat because ROS2 has no list-of-lists parameter type;
        a trailing partial triplet means the config is malformed, so it is
        dropped with a warning rather than silently navigating to a half-read
        coordinate.
        """
        flat = list(self.get_parameter('waypoints').value)

        remainder = len(flat) % 3
        if remainder:
            self.get_logger().warn(
                f'waypoints has {len(flat)} values, not a multiple of 3; '
                f'ignoring the trailing {remainder}'
            )
            flat = flat[:len(flat) - remainder]

        return [tuple(flat[i:i + 3]) for i in range(0, len(flat), 3)]

    def _cache_image(self, msg):
        self.latest_image = msg

    def _send_next_goal(self):
        # An empty route is a config error, not a finished patrol — returning
        # here keeps the node alive and diagnosable instead of raising out of
        # __init__ with an IndexError.
        if not self.waypoints:
            self.get_logger().error('no waypoints configured, patrol not started')
            return

        if self.current_index >= len(self.waypoints):
            self.get_logger().info('patrol loop complete, restarting')
            self.current_index = 0

        x, y, yaw = self.waypoints[self.current_index]

        goal = PoseStamped()
        goal.header.frame_id = self.frame
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = x
        goal.pose.position.y = y
        # yaw -> quaternion omitted here for brevity; production code should
        # use tf_transformations rather than hand-rolling the conversion.

        self.nav_client.wait_for_server()
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal

        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            # Nav2 rejects goals it can't plan for (e.g. inside an obstacle).
            # Skipping to the next waypoint rather than retrying avoids the
            # robot getting stuck on a single unreachable point.
            self.get_logger().warn('goal rejected, skipping waypoint')
            self.current_index += 1
            self._send_next_goal()
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_nav_result)

    def _on_nav_result(self, future):
        # Reaching the waypoint is the trigger for a defect check, not a timer —
        # inspection only makes sense once the robot has actually stopped and
        # the camera view has settled.
        self._check_for_defect()
        self.current_index += 1
        self._send_next_goal()

    def _check_for_defect(self):
        if self.latest_image is None or not self.detect_client.service_is_ready():
            self.get_logger().warn('skipping defect check: no image or service unavailable')
            return

        req = DetectDefect.Request()
        req.image = self.latest_image
        req.request_id = f'wp_{self.current_index}'

        future = self.detect_client.call_async(req)
        future.add_done_callback(self._on_defect_result)

    def _on_defect_result(self, future):
        response = future.result()
        if not response.defect_found:
            return

        # world_position is stamped here, at the navigator, using the robot's
        # pose in the map frame at the moment of detection — this is the one
        # piece of state defect_detector_server deliberately doesn't own.
        try:
            transform = self.tf_buffer.lookup_transform(
                self.frame, 'base_link', rclpy.time.Time()
            )
            pos = transform.transform.translation
            self.get_logger().info(
                f'defect "{response.defect_type}" '
                f'(confidence={response.confidence:.2f}) at ({pos.x:.2f}, {pos.y:.2f})'
            )
        except tf2_ros.TransformException as exc:
            self.get_logger().warn(f'defect found but position lookup failed: {exc}')


def main(args=None):
    rclpy.init(args=args)
    node = PatrolNavigator()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
