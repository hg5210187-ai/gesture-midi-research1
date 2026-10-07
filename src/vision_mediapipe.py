import time
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# The MediaPipe Tasks HandLandmarker needs a model bundle. Download it once:
#   curl -L -o hand_landmarker.task \
#     https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
_ROOT = Path(__file__).resolve().parent.parent
_MODEL = _ROOT / "hand_landmarker.task"


# --- Minimal stand-ins so a MediaPipe frame duck-types an Ultralytics Results --
# GestureLogic.process() reads result.boxes.xyxyn / .conf and result.keypoints.xyn
# and calls .cpu().numpy() on each. These tiny wrappers expose that same surface
# from MediaPipe output, so the EXISTING (tested) GestureLogic + overlay work
# unchanged — only the detector is different.

class _T:
    """A tensor stand-in: supports ``.cpu().numpy()`` like a torch tensor."""
    def __init__(self, a):
        self._a = np.asarray(a)
    def cpu(self):
        return self
    def numpy(self):
        return self._a


class _Boxes:
    def __init__(self, xyxyn, conf):
        self.xyxyn = _T(xyxyn)   # [N,4] normalized x1,y1,x2,y2
        self.conf = _T(conf)     # [N]
    def __len__(self):
        return len(self.xyxyn.numpy())


class _Keypoints:
    def __init__(self, xyn):
        self.xyn = _T(xyn)       # [N,21,2] normalized
    def __len__(self):
        return len(self.xyn.numpy())


class _Result:
    """Duck-typed Ultralytics Results: exposes .boxes (.xyxyn,.conf),
    .keypoints (.xyn) and .orig_shape so GestureLogic.process() runs as-is."""
    def __init__(self, boxes, keypoints, orig_shape):
        self.boxes = boxes
        self.keypoints = keypoints
        self.orig_shape = orig_shape


class MediaPipeVision:
    """Hand detection via the MediaPipe Tasks HandLandmarker, exposed with the
    SAME interface as ``UltralyticsVision`` so it drops straight into the app.

    Why this exists: the YOLOv8 hand-pose model hallucinates a "hand" on
    faces/curtains/shadows (the phantom-detection problem). MediaPipe runs a
    dedicated palm detector and is far more robust against those false
    positives, and returns 21 normalized landmarks per hand — exactly what the
    right-hand (bounding box -> pitch/volume) and left-hand (fist keypoints ->
    looper) logic needs. It runs on CPU but is light.

    Each detected hand becomes a bounding box (from the landmark extents) plus
    its 21 normalized landmarks, with the per-hand detection score as the
    confidence. The frame is mirrored (selfie view) so the user's right hand
    sits on the right half, matching GestureLogic's center-x convention.

    Note: this build of MediaPipe ships only the Tasks API (not the legacy
    ``mediapipe.solutions.hands``), which is why the HandLandmarker + a
    ``hand_landmarker.task`` model bundle are used.
    """

    def __init__(self, num_hands=2, min_detection_confidence=0.7,
                 min_tracking_confidence=0.5, cam_width=640, cam_height=480,
                 model_path=None):
        model_path = str(model_path or _MODEL)
        if not Path(model_path).is_file():
            raise FileNotFoundError(
                f"MediaPipe hand model not found: {model_path}\n"
                "Download it once with:\n  curl -L -o hand_landmarker.task "
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                "hand_landmarker/float16/1/hand_landmarker.task")

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)

        # VIDEO mode wants a strictly increasing timestamp (ms) per frame.
        self._t0 = time.time()
        self._last_ts = -1

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
        aw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"MediaPipe vision: camera {aw}x{ah}, "
              f"min_detection_confidence={min_detection_confidence}")

    def get_frame(self):
        success, frame = self.cap.read()
        if not success:
            return False, None, None

        # Mirror (selfie) view to match the rest of the app, then detect.
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ts = int((time.time() - self._t0) * 1000)
        if ts <= self._last_ts:          # guarantee monotonic timestamps
            ts = self._last_ts + 1
        self._last_ts = ts
        result = self.landmarker.detect_for_video(mp_image, ts)

        bboxes, confs, kpts = [], [], []
        hand_landmarks = result.hand_landmarks or []
        handed = result.handedness or []
        for i, lms in enumerate(hand_landmarks):
            # 21 normalized (x, y) landmarks (MediaPipe normalizes to [0,1]).
            pts = np.array([[lm.x, lm.y] for lm in lms], dtype=float)
            pts[:, 0] = np.clip(pts[:, 0], 0.0, 1.0)
            pts[:, 1] = np.clip(pts[:, 1], 0.0, 1.0)

            x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
            x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
            bboxes.append([x1, y1, x2, y2])

            score = 1.0
            if i < len(handed) and handed[i]:
                score = float(handed[i][0].score)
            confs.append(score)
            kpts.append(pts)

        boxes = _Boxes(
            np.array(bboxes, dtype=float).reshape(-1, 4),
            np.array(confs, dtype=float).reshape(-1),
        )
        keypoints = _Keypoints(
            np.array(kpts, dtype=float).reshape(-1, 21, 2) if kpts
            else np.empty((0, 21, 2), dtype=float)
        )
        return True, frame, _Result(boxes, keypoints, (h, w))

    def release(self):
        self.cap.release()
        self.landmarker.close()


if __name__ == "__main__":
    vision = MediaPipeVision()
    try:
        while True:
            success, frame, result = vision.get_frame()
            if not success:
                break
            cv2.imshow("MediaPipe MIDI Vision", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        vision.release()
        cv2.destroyAllWindows()
