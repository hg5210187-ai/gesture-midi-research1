"""Single-model vision backend for the rotation instrument (``main_rotate.py``).

Unlike ``vision_obb.py`` (which runs a right-hand OBB model *and* a left-hand
pose model), this backend runs **one** self-trained model — ``YOLO26n-MIDI.pt``,
a YOLO26n **OBB** (oriented-bounding-box) model with three gesture classes
(``thumbout`` / ``openhand`` / ``closedhand``). A single inference per frame
carries every hand as an oriented box, so:

  * the **right hand**'s box *angle* drives pitch (see ``logic_rotate.py``),
  * the box *area* drives volume, and
  * the **left hand**'s gesture *class* (``closedhand``) toggles the looper —

with no separate pose model. The OBB angle is the whole point: it is what lets
the player create sound by *rotating* the hand.

Tuned to stay light on Apple Silicon: inference runs on the GPU via MPS when
available and at a reduced ``imgsz`` (the model trains at 1024 but detects hands
— which fill much of the frame — comfortably at 640, which runs several times
faster than the 20 FPS the loop needs).
"""
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Models + the version file live in the project root, next to this package.
# Resolve relative to the repo root so the app finds them regardless of the
# current working directory.
_ROOT = Path(__file__).resolve().parent.parent


def _resolve(name):
    p = Path(name)
    return str(p if p.is_absolute() else _ROOT / p)


def _version_file_for(model):
    """Map a model filename to its version file (``YOLO26<size>-MIDI.pt`` ->
    ``YOLO26<size>-ver``); fall back to ``<stem>-ver`` for custom paths."""
    name = Path(model).name
    ver = re.sub(r"-MIDI\.pt$", "-ver", name)
    return ver if ver != name else Path(name).stem + "-ver"


def load_version_info(path="YOLO26n-ver"):
    """Read a ``YOLO26<size>-ver`` model version file (JSON), or ``None``.

    Missing or malformed version files never break startup — the model still
    loads; we simply skip the banner.
    """
    try:
        with open(_resolve(path), "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"({path} not read: {e})")
        return None


class Yolo26Vision:
    """One-model hand inference for the rotation instrument.

    ``get_frame`` returns ``(success, frame, obb_result)`` where ``obb_result``
    is a single Ultralytics result whose ``.obb`` holds every detected hand
    (both hands, each with a class + oriented box). ``GestureLogic.process``
    (``src/logic_rotate.py``) splits them into left/right by screen position.
    """

    def __init__(self, model="YOLO26n-MIDI.pt", device=None, imgsz=640,
                 cam_width=640, cam_height=480):
        # Announce which model version is running. Each model has its own
        # YOLO26<size>-ver provenance file, derived from the model filename.
        info = load_version_info(_version_file_for(model))
        if info:
            fps = info.get('approx_fps_640_mps')
            print(f"Model: {info.get('model_file', model)} "
                  f"v{info.get('version', '?')} "
                  f"(task={info.get('task', '?')}, "
                  f"classes={info.get('classes', '?')}, "
                  f"trained={info.get('trained', '?')}, "
                  f"mAP50-95={info.get('metrics_mAP50_95', '?')}"
                  + (f", ~{fps}fps@640)" if fps else ")"))

        model_path = _resolve(model)
        if not Path(model_path).is_file():
            raise FileNotFoundError(
                f"Rotation model not found: {model_path}\n"
                f"    main_rotate.py needs the trained YOLO26n-MIDI.pt in the "
                f"project root.")
        self.model = YOLO(model_path)
        self.names = self.model.names
        print(f"Rotation OBB model: {model_path}  classes={self.names}")

        # Prefer Apple's GPU (MPS); fall back to CPU when it isn't available.
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.imgsz = imgsz
        print(f"Vision device: {self.device}, imgsz: {self.imgsz}")

        self.cap = cv2.VideoCapture(0)
        # Capture at a modest resolution so we don't decode/copy large frames
        # (e.g. from a Continuity Camera) only to downscale them for inference.
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
        aw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera capture resolution: {aw}x{ah}")

        # Warm up the model/device once (the first MPS call compiles kernels and
        # is slow) so the first real frame isn't a stall. ``_predict`` falls back
        # from MPS to CPU on its own if the chosen device can't run.
        warm = np.zeros((cam_height, cam_width, 3), dtype=np.uint8)
        self._predict(warm)

    def _predict(self, frame):
        """Run the model, permanently dropping MPS->CPU on a runtime error.

        If the GPU path hiccups we switch ``self.device`` to CPU for good so the
        vision thread keeps working rather than crashing.
        """
        try:
            return self.model.predict(source=frame, device=self.device,
                                      imgsz=self.imgsz, stream=False,
                                      verbose=False)[0]
        except Exception as e:
            if self.device != "cpu":
                print(f"Inference on {self.device} failed ({e}); "
                      f"switching to CPU.")
                self.device = "cpu"
                return self.model.predict(source=frame, device=self.device,
                                          imgsz=self.imgsz, stream=False,
                                          verbose=False)[0]
            raise

    def get_frame(self):
        success, frame = self.cap.read()
        if not success:
            return False, None, None

        # Mirror view. Keep the RAW frame (no result.plot()) so the lightweight
        # rotation overlay can be drawn downstream.
        frame = cv2.flip(frame, 1)

        obb_result = self._predict(frame)
        return True, frame, obb_result

    def release(self):
        self.cap.release()


if __name__ == "__main__":
    vision = Yolo26Vision()
    try:
        while True:
            success, frame, result = vision.get_frame()
            if not success:
                break
            cv2.imshow("YOLO26n Rotation Vision", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        vision.release()
        cv2.destroyAllWindows()
