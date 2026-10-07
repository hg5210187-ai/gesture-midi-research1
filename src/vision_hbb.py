"""Single-model vision backend: one gesture detector for both hands.

Runs one self-trained YOLO26 **detect** model (plain horizontal boxes, e.g.
``hbb-s-fold1-320.pt``) with three gesture classes (``thumbout`` / ``openhand``
/ ``closedhand``). One inference per frame carries every hand as a box plus a
gesture class, so there is no separate left-hand pose model:

  * the **right hand**'s box drives pitch (Y-center) and volume (area), and
  * the **left hand**'s ``closedhand`` class toggles the looper

(see ``GestureLogic.process`` in ``logic_obb.py``).
"""
import os
import time
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from src.vision_obb import LatestFrameCamera, _resolve_model


class HbbVision(LatestFrameCamera):
    """``get_frame`` blocks for the camera's newest frame and runs the model.

    The default model is the Core ML export (``hbb-s-fold1-320.mlpackage``,
    ~2 ms per frame on an M4 versus ~16-22 ms for the ``.pt`` in PyTorch) when
    it exists, else the ``.pt``. Create the export with:

        uv run yolo export model=hbb-s-fold1-320.pt format=coreml imgsz=320

    Environment overrides: ``HBB_MODEL`` (model file), ``HBB_IMGSZ`` (default
    320, the size the model was trained and exported at) and ``HBB_DEVICE``
    (``mps``/``cpu``; PyTorch models only), plus the camera ones on
    ``LatestFrameCamera``.
    """

    def __init__(self, model=None, device=None, imgsz=None,
                 cam_width=640, cam_height=480,
                 camera_index=None, camera_fps=None):
        model = model or os.environ.get("HBB_MODEL")
        if model is None:
            model = "hbb-s-fold1-320.mlpackage"
            if not Path(_resolve_model(model)).exists():
                model = "hbb-s-fold1-320.pt"
        model_path = _resolve_model(model)
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Gesture model not found: {model_path}")
        self.model = YOLO(model_path, task="detect")

        if device is None:
            device = os.environ.get("HBB_DEVICE") or (
                "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        if imgsz is None:
            imgsz = int(os.environ.get("HBB_IMGSZ", 320))
        self.imgsz = imgsz

        # Warm up once so the first real frame isn't a stall.
        warm = self._predict(np.zeros((cam_height, cam_width, 3), dtype=np.uint8))
        self.names = warm.names
        print(f"Gesture model: {model_path}  classes={self.names}")
        print(f"Vision: imgsz {self.imgsz}"
              + ("" if model_path.endswith(".mlpackage") else f" on {self.device}"))

        self._open_camera(cam_width, cam_height, camera_index, camera_fps)
        self._start_thread(self._capture_loop)

    def _predict(self, frame):
        """Run the model, permanently dropping MPS->CPU on a runtime error."""
        try:
            return self.model.predict(source=frame, device=self.device,
                                      imgsz=self.imgsz, stream=False, verbose=False)[0]
        except Exception as e:
            if self.device == "cpu":
                raise
            print(f"Inference on {self.device} failed ({e}); switching to CPU.")
            self.device = "cpu"
            return self.model.predict(source=frame, device=self.device,
                                      imgsz=self.imgsz, stream=False, verbose=False)[0]

    def get_frame(self):
        frame, ts, self._last_id = self._wait_new_frame(self._last_id)
        if frame is None:
            return False, None, None

        t0 = time.perf_counter()
        result = self._predict(frame)
        self.last_infer_ms = (time.perf_counter() - t0) * 1000.0
        self.last_capture_ts = ts
        return True, frame.copy(), result
