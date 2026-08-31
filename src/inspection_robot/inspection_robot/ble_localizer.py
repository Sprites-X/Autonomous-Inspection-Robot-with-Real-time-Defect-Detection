import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped

from inspection_robot.utils.trilateration import estimate_position, rssi_to_distance


class BleLocalizer(Node):
    """
    Fallback localization for zones where GPS (and often even good AMCL
    convergence) is unavailable — enclosed or underground areas. This node
    does not replace Nav2's localization; it publishes an independent pose
    estimate that a supervisor/fusion layer can fall back to when the primary
    source is unreliable, rather than fighting AMCL for the same topic.

    Design assumption: beacon positions are static and known in advance
    (surveyed once during site setup), so only the robot's position is solved
    for at runtime — this keeps the problem to plain trilateration instead of
    full SLAM.
    """

    def __init__(self):
        super().__init__('ble_localizer')

        self.declare_parameter('min_beacons_required', 3)
        self.min_beacons = self.get_parameter('min_beacons_required').value

        # beacon_id -> (x, y) in the map frame. Loaded from config in practice;
        # hardcoded here to keep the node's logic visible without a config dep.
        self.beacon_positions = {
            'beacon_1': (0.0, 0.0),
            'beacon_2': (5.0, 0.0),
            'beacon_3': (2.5, 4.0),
        }

        # RSSI is noisy per-reading; a short rolling window trades a bit of
        # latency for a distance estimate stable enough that trilateration
        # doesn't jitter the pose every scan.
        self.rssi_window = {}
        self.window_size = 5

        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/ble_localizer/pose', 10
        )

        # Subscription type is left generic here since the actual BLE scan
        # message depends on the driver in use (e.g. a custom msg from the
        # BLE gateway node) — swap BeaconScan for whatever that driver emits.
        from inspection_interfaces.msg import BeaconScan
        self.create_subscription(BeaconScan, '/ble/scan', self._on_scan, 10)

        self.get_logger().info('ble_localizer ready (fallback mode)')

    def _on_scan(self, msg):
        beacon_id = msg.beacon_id
        self.rssi_window.setdefault(beacon_id, []).append(msg.rssi)
        if len(self.rssi_window[beacon_id]) > self.window_size:
            self.rssi_window[beacon_id].pop(0)

        self._try_localize()

    def _try_localize(self):
        # Only beacons with a full window contribute — a beacon seen once
        # is more likely noise than signal, and including it would let a
        # single bad reading pull the position estimate off.
        ready_beacons = {
            bid: readings for bid, readings in self.rssi_window.items()
            if len(readings) == self.window_size
        }

        if len(ready_beacons) < self.min_beacons:
            # Trilateration is under-determined below 3 beacons; publishing
            # an estimate anyway would be worse than publishing nothing, since
            # a fusion layer has no way to tell a bad fix from a good one.
            return

        distances = {}
        for bid, readings in ready_beacons.items():
            avg_rssi = sum(readings) / len(readings)
            distances[bid] = rssi_to_distance(avg_rssi)

        known_points = {
            bid: self.beacon_positions[bid]
            for bid in distances if bid in self.beacon_positions
        }

        if len(known_points) < self.min_beacons:
            self.get_logger().warn('beacon(s) detected with unknown surveyed position, skipping')
            return

        x, y = estimate_position(known_points, distances)
        self._publish_pose(x, y)

    def _publish_pose(self, x, y):
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y

        # Covariance is set noticeably higher than a typical AMCL estimate —
        # RSSI-based trilateration is meters-level accurate at best, and a
        # fusion layer needs that reflected honestly rather than reported as
        # confident as the primary localization source.
        msg.pose.covariance[0] = 1.0   # x variance
        msg.pose.covariance[7] = 1.0   # y variance

        self.pose_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = BleLocalizer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
