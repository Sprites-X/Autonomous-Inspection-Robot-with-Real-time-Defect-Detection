import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

from inspection_interfaces.srv import DetectDefect

# Placeholder for the actual model wrapper (Ultralytics YOLO or ONNX/TensorRT runtime).
# Kept as a separate module so the inference backend can be swapped (PyTorch -> TensorRT)
# without touching the service logic below.
from inspection_robot.utils.yolo_inference import YoloDetector


class DefectDetectorServer(Node):
    """
    Exposes /detect_defect as a synchronous ROS2 service.

    Design choice: service (request/response) rather than a topic, because the
    caller (patrol_navigator) needs a direct answer tied to a specific frame,
    not a continuous stream it has to buffer and correlate itself.
    """

    def __init__(self):
        super().__init__('defect_detector_server')

        self.bridge = CvBridge()

        # Model is loaded once at startup, not per-request, since weight loading
        # dominates latency and the node is expected to stay alive for the whole patrol.
        self.declare_parameter('model_path', 'models/best.onnx')
        self.declare_parameter('confidence_threshold', 0.5)

        model_path = self.get_parameter('model_path').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.detector = YoloDetector(model_path)

        self.srv = self.create_service(
            DetectDefect, 'detect_defect', self.handle_detect_defect
        )

        self.get_logger().info(f'defect_detector_server ready (model={model_path})')

    def handle_detect_defect(self, request, response):
        # cv_bridge converts the ROS Image message into a numpy array the model expects.
        # Any conversion failure is treated as a service-level error rather than a crash,
        # since a malformed frame shouldn't take the whole node down mid-patrol.
        try:
            frame = self.bridge.imgmsg_to_cv2(request.image, desired_encoding='bgr8')
        except Exception as exc:
            response.defect_found = False
            response.message = f'image conversion failed: {exc}'
            return response

        detections = self.detector.infer(frame)

        # Only the highest-confidence detection above threshold is reported.
        # The service contract is "one defect per call" by design — if multiple
        # defects need reporting, the caller re-queries after moving the region of
        # interest, keeping the response schema simple.
        best = max(detections, key=lambda d: d.confidence, default=None)

        if best is None or best.confidence < self.confidence_threshold:
            response.defect_found = False
            response.message = 'no defect above confidence threshold'
            return response

        response.defect_found = True
        response.defect_type = best.label
        response.confidence = best.confidence
        response.bbox_x = best.x
        response.bbox_y = best.y
        response.bbox_width = best.width
        response.bbox_height = best.height

        # world_position is left at its zero default here; it gets filled in by
        # patrol_navigator using the robot's pose at capture time, since this node
        # has no access to /tf and shouldn't need to for a pure perception task.
        response.message = 'ok'
        return response


def main(args=None):
    rclpy.init(args=args)
    node = DefectDetectorServer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
