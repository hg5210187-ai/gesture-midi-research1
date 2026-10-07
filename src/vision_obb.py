from ultralytics import YOLO
import cv2
import numpy as np
import os
import threading
import time
import torch
from pathlib import Path

# Models are referenced by bare filename and live in the project root next to
# the existing pose model. Resolve relative to the repo root so the app finds
# them regardless of the current working directory.
_ROOT = Path(__file__).resolve().parent.parent


def _resolve_model(name):
    p = Path(name)
    return str(p if p.is_absolute() else _ROOT / p)


class LatestFrameCamera:
    """Camera whose capture thread always holds the NEWEST (mirrored) frame.

    ``_wait_new_frame`` blocks until a frame newer than the caller's last one
    exists, so a reader never gets a stale buffered frame and never handles a
    frame twice. Environment overrides: ``CAMERA_INDEX`` and ``CAMERA_FPS``
    (requested; the camera may grant less).
    """

    def _open_camera(self, cam_width=640, cam_height=480,
                     camera_index=None, camera_fps=None):
        if camera_index is None:
            camera_index = int(os.environ.get("CAMERA_INDEX", 0))
        if camera_fps is None:
            camera_fps = float(os.environ.get("CAMERA_FPS", 0))
        self.cap = cv2.VideoCapture(camera_index)
        # Capture at a modest resolution so we don't decode/copy large frames
        # (e.g. from a Continuity Camera) only to downscale them for inference.
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
        if camera_fps > 0:
            self.cap.set(cv2.CAP_PROP_FPS, camera_fps)
        aw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera {camera_index} capture: {aw}x{ah} @ "
              f"{self.cap.get(cv2.CAP_PROP_FPS):.0f} fps (reported)")

        # Newest mirrored frame. It is never drawn on (get_frame hands out a
        # copy), so every reader always sees the clean camera image.
        self._cond = threading.Condition()
        self._frame = None
        self._frame_ts = 0.0
        self._frame_id = 0
        self._last_id = 0
        self._running = True
        self._threads = []

        # Timing of the last frame returned by get_frame(), for latency readouts:
        # when it left the camera (perf_counter) and how long inference took.
        self.last_capture_ts = 0.0
        self.last_infer_ms = 0.0

    def _start_thread(self, target):
        t = threading.Thread(target=target, daemon=True)
        self._threads.append(t)
        t.start()

    def _capture_loop(self):
        """Keep ``self._frame`` at the camera's newest (mirrored) frame."""
        while self._running:
            success, frame = self.cap.read()
            ts = time.perf_counter()
            if not success:
                time.sleep(0.01)
                continue
            # Mirror view. Keep the RAW frame (no result.plot()) so a
            # lightweight custom overlay can be drawn downstream.
            frame = cv2.flip(frame, 1)
            with self._cond:
                self._frame = frame
                self._frame_ts = ts
                self._frame_id += 1
                self._cond.notify_all()

    def _wait_new_frame(self, last_id, timeout=1.0):
        """Block until a frame newer than ``last_id`` exists -> (frame, ts, id).

        Returns ``(None, 0.0, last_id)`` on timeout or shutdown.
        """
        with self._cond:
            self._cond.wait_for(
                lambda: self._frame_id != last_id or not self._running, timeout)
            if not self._running or self._frame_id == last_id:
                return None, 0.0, last_id
            return self._frame, self._frame_ts, self._frame_id

    def release(self):
        with self._cond:
            if not self._running:
                return
            self._running = False
            self._cond.notify_all()
        for t in self._threads:
            if t is not threading.current_thread():
                t.join(timeout=2.0)
        self.cap.release()


class UltralyticsVision(LatestFrameCamera):
    """Two-model hand inference, tuned for low gesture->MIDI latency.

    The **right hand** is driven by a self-trained YOLO11n-OBB model (oriented
    bounding box): pitch from the box Y-center, volume from the box area, and a
    rotation angle reserved for a future tilt parameter. The **left hand** stays
    on the original YOLOv8 hand-pose model, whose keypoints drive fist/looper
    control.

    Three things keep the note path short:
      * a capture thread always holds the NEWEST camera frame, so ``get_frame``
        never returns a stale buffered one and never handles a frame twice;
      * only the right-hand OBB model runs inside ``get_frame``; the left-hand
        pose model (coarse, gated by a 1s cooldown) runs in its own worker
        thread and its latest result is simply picked up;
      * the two models run on different devices (OBB on the CPU, pose on the
        GPU via MPS). Measured on an M4, sharing one device between the two
        threads doubles the OBB time (~14ms -> ~28ms); splitting them doesn't,
        and keeping the OBB off the GPU avoids stalls when the UI redraws.

    ``get_frame`` blocks until the camera delivers a new frame, so the caller's
    loop runs at camera rate with no pacing of its own.

    Environment override: ``OBB_IMGSZ`` (e.g. 256 for faster, coarser
    right-hand inference), plus the camera ones on ``LatestFrameCamera``.
    """

    def __init__(self, pose_model="yolov8n-hand-pose.pt",
                 obb_model="right_hand_obb.pt",
                 device="cpu", pose_device=None, imgsz=384, obb_imgsz=None,
                 cam_width=640, cam_height=480,
                 camera_index=None, camera_fps=None):
        # Left-hand pose model (always required).
        self.pose_model = YOLO(_resolve_model(pose_model))
        self._last_pose_result = None

        # Right-hand OBB model. If the file isn't present yet (e.g. not yet
        # downloaded from Drive), degrade gracefully: the app still runs with
        # the left hand / looper, and the right hand stays silent until the
        # model is dropped in. The warning is loud so it isn't missed.
        self.obb_model = None
        obb_path = _resolve_model(obb_model)
        if Path(obb_path).is_file():
            self.obb_model = YOLO(obb_path)
            print(f"Right-hand OBB model: {obb_path}")
        else:
            print("=" * 72)
            print(f"WARNING: right-hand OBB model not found:\n    {obb_path}")
            print("    The RIGHT hand (pitch/volume) is DISABLED until it exists.")
            print("    Copy your trained model (Drive: MyDrive/midi_hand_model/")
            print("    right_hand_obb.pt) to the path above, then restart.")
            print("=" * 72)

        # Right-hand OBB on the CPU, left-hand pose on Apple's GPU (MPS) when
        # available, so the two inference threads don't compete for one device.
        if pose_device is None:
            pose_device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.pose_device = pose_device
        self.imgsz = imgsz
        if obb_imgsz is None:
            obb_imgsz = int(os.environ.get("OBB_IMGSZ", imgsz))
        self.obb_imgsz = obb_imgsz
        print(f"Vision: obb on {self.device} @ {self.obb_imgsz}, "
              f"pose on {self.pose_device} @ {self.imgsz}")

        # Warm up both models once so the first real frame isn't a stall.
        warm = np.zeros((cam_height, cam_width, 3), dtype=np.uint8)
        if self.obb_model:
            self._predict(self.obb_model, warm, self.obb_imgsz, self.device)
        self._predict_pose(warm)

        self._open_camera(cam_width, cam_height, camera_index, camera_fps)
        self._start_thread(self._capture_loop)
        self._start_thread(self._pose_loop)

    def _predict(self, model, frame, imgsz, device):
        return model.predict(source=frame, device=device,
                             imgsz=imgsz, stream=False, verbose=False)[0]

    def _predict_pose(self, frame):
        """Run the pose model, permanently dropping MPS->CPU on a runtime error.

        If the GPU path hiccups we switch ``self.pose_device`` to CPU for good
        so the left hand keeps working rather than going dead.
        """
        try:
            return self._predict(self.pose_model, frame, self.imgsz, self.pose_device)
        except Exception as e:
            if self.pose_device == "cpu":
                raise
            print(f"Inference on {self.pose_device} failed ({e}); switching to CPU.")
            self.pose_device = "cpu"
            return self._predict(self.pose_model, frame, self.imgsz, self.pose_device)

    def _pose_loop(self):
        """Left-hand pose inference on the newest frame, off the note path."""
        last_id = 0
        while self._running:
            frame, _, last_id = self._wait_new_frame(last_id)
            if frame is None:
                continue
            try:
                self._last_pose_result = self._predict_pose(frame)
            except Exception as e:
                print(f"Pose worker: dropping frame after error: {e}")
                time.sleep(0.05)

    def get_frame(self):
        frame, ts, self._last_id = self._wait_new_frame(self._last_id)
        if frame is None:
            return False, None, None

        # Right hand <- OBB model (run here, every frame); left hand <- the pose
        # worker's latest result. Returned as a (pose_result, obb_result) pair
        # consumed by GestureLogic.process().
        t0 = time.perf_counter()
        obb_result = (self._predict(self.obb_model, frame, self.obb_imgsz, self.device)
                      if self.obb_model else None)
        self.last_infer_ms = (time.perf_counter() - t0) * 1000.0
        self.last_capture_ts = ts
        return True, frame.copy(), (self._last_pose_result, obb_result)


if __name__ == "__main__":
    vision = UltralyticsVision()
    try:
        while True:
            success, frame, result = vision.get_frame()
            if not success:
                break
            cv2.imshow("Ultralytics MIDI Vision", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        vision.release()
        cv2.destroyAllWindows()
