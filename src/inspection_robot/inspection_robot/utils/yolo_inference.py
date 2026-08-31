from dataclasses import dataclass


@dataclass
class Detection:
    label: str
    confidence: float
    x: int
    y: int
    width: int
    height: int


class YoloDetector:
    """
    Thin wrapper around the actual inference backend. The rest of the codebase
    (defect_detector_server) only depends on this class's interface, not on
    Ultralytics/ONNXRuntime/TensorRT directly — which is what lets the model
    move from a PyTorch .pt file in early development to a TensorRT .engine
    file for edge deployment (Phase 5) without touching the service node.
    """

    def __init__(self, model_path):
        self.model_path = model_path
        self.backend = self._load_backend(model_path)

    def _load_backend(self, model_path):
        # Dispatch on file extension rather than a config flag — the model
        # file itself is the source of truth for which runtime it needs,
        # so there's no way for the declared backend and the actual weights
        # to drift out of sync.
        if model_path.endswith('.pt'):
            from ultralytics import YOLO
            return ('ultralytics', YOLO(model_path))

        elif model_path.endswith('.onnx'):
            import onnxruntime as ort
            session = ort.InferenceSession(
                model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            return ('onnxruntime', session)

        elif model_path.endswith('.engine'):
            # TensorRT engines are hardware- and version-locked (built for a
            # specific GPU + TensorRT version), which is why this path stays
            # separate from ONNX rather than trying to unify them — a .engine
            # file built on one Jetson won't load on another without rebuilding.
            from inspection_robot.utils.tensorrt_runner import TensorRTRunner
            return ('tensorrt', TensorRTRunner(model_path))

        else:
            raise ValueError(f'unsupported model format: {model_path}')

    def infer(self, frame):
        backend_type, model = self.backend

        if backend_type == 'ultralytics':
            results = model(frame, verbose=False)[0]
            return self._parse_ultralytics(results)

        elif backend_type == 'onnxruntime':
            return self._infer_onnx(model, frame)

        elif backend_type == 'tensorrt':
            return model.infer(frame)

    def _parse_ultralytics(self, results):
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(Detection(
                label=results.names[int(box.cls[0])],
                confidence=float(box.conf[0]),
                x=int(x1),
                y=int(y1),
                width=int(x2 - x1),
                height=int(y2 - y1),
            ))
        return detections

    def _infer_onnx(self, session, frame):
        # Preprocessing (resize/normalize/transpose to NCHW) and postprocessing
        # (NMS, box decoding) are model-export-specific and intentionally left
        # as a stub — they must match whatever export script produced the
        # .onnx file, so hardcoding them here would silently break on re-export.
        raise NotImplementedError(
            'ONNX pre/post-processing must match the export pipeline; fill in '
            'once the export script (scripts/export_to_onnx.sh) is finalized'
        )
