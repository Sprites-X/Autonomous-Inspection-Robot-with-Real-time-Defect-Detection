import numpy as np

# pycuda/tensorrt are imported lazily inside __init__ rather than at module
# level, since this module gets imported by yolo_inference.py even when the
# active backend is ONNX or Ultralytics — a machine without a GPU/TensorRT
# install shouldn't fail to import the rest of the package over an unused path.


class TensorRTRunner:
    """
    Loads a pre-built TensorRT engine (.engine) and runs inference on it.

    Unlike the ONNX/Ultralytics paths, there is no "build the engine here"
    step: engines are produced offline by scripts/convert_tensorrt.sh on the
    target hardware, because an engine built on a dev machine's GPU is not
    guaranteed to load on the Jetson/edge device it will actually run on
    (see the backend-selection comment in yolo_inference.py).
    """

    def __init__(self, engine_path):
        import pycuda.autoinit  # noqa: F401 - registers the CUDA context
        import pycuda.driver as cuda
        import tensorrt as trt

        self.cuda = cuda
        logger = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, 'rb') as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self._allocate_buffers()

    def _allocate_buffers(self):
        # Buffers are allocated once at load time and reused every inference
        # call, rather than per-call — repeated cudaMalloc/cudaFree on every
        # frame is the kind of overhead that defeats the point of using
        # TensorRT for real-time edge inference in the first place.
        self.bindings = []
        self.host_inputs, self.host_outputs = [], []
        self.device_inputs, self.device_outputs = [], []

        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            size = int(np.prod(shape))
            dtype = np.float32

            host_mem = self.cuda.pagelocked_empty(size, dtype)
            device_mem = self.cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                self.host_inputs.append(host_mem)
                self.device_inputs.append(device_mem)
            else:
                self.host_outputs.append(host_mem)
                self.device_outputs.append(device_mem)

    def infer(self, frame):
        # Preprocessing must exactly match what the ONNX export used before
        # conversion to TensorRT (same resize/normalize/layout) — any mismatch
        # here produces a model that runs without error but predicts garbage,
        # which is why this stays a hard NotImplementedError until the export
        # pipeline (scripts/export_to_onnx.sh) is finalized and both sides
        # of the conversion can be written to agree.
        raise NotImplementedError(
            'preprocessing/postprocessing must mirror the ONNX export pipeline; '
            'implement once scripts/export_to_onnx.sh is finalized'
        )
