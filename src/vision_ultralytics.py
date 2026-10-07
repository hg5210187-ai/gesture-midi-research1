from ultralytics import YOLO
import cv2
import numpy as np
import torch


class UltralyticsVision:
    """Hand-pose inference wrapper, tuned to stay light on Apple Silicon.

    The app felt "heavy" because Ultralytics defaults to CPU on macOS and runs
    at imgsz=640, which pins several CPU cores every frame. We instead:
      * run inference on the GPU via MPS when available (frees the CPU), and
      * shrink the inference size (imgsz) and camera capture resolution,
    which together cut per-frame cost and power dramatically while keeping
    hands (which fill much of the frame) easy to detect.
    """

    def __init__(self, model_variant="yolov8n-hand-pose.pt",
                 device=None, imgsz=384, cam_width=640, cam_height=480):
        self.model = YOLO(model_variant)

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
        # is slow) so the first real frame isn't a stall. If the chosen device
        # can't run, fall back to CPU now rather than failing mid-stream.
        warm = np.zeros((cam_height, cam_width, 3), dtype=np.uint8)
        try:
            self.model.predict(warm, device=self.device, imgsz=self.imgsz, verbose=False)
        except Exception as e:
            print(f"Warmup on {self.device} failed ({e}); falling back to CPU.")
            self.device = "cpu"
            self.model.predict(warm, device=self.device, imgsz=self.imgsz, verbose=False)

    def get_frame(self):
        success, frame = self.cap.read()
        if not success:
            return False, None, None

        # Mirror view. Keep the RAW frame (no result.plot()) so a lightweight
        # custom overlay can be drawn downstream.
        frame = cv2.flip(frame, 1)

        # Single pose inference: result carries BOTH .boxes and .keypoints
        # (index-aligned). The right hand uses .boxes; the left hand uses
        # .keypoints for fist detection.
        try:
            results = self.model.predict(source=frame, device=self.device,
                                         imgsz=self.imgsz, stream=False, verbose=False)
        except Exception as e:
            # If the GPU path hiccups at runtime, drop to CPU permanently so the
            # app keeps working rather than crashing the vision thread.
            if self.device != "cpu":
                print(f"Inference on {self.device} failed ({e}); switching to CPU.")
                self.device = "cpu"
                results = self.model.predict(source=frame, device=self.device,
                                             imgsz=self.imgsz, stream=False, verbose=False)
            else:
                raise

        return True, frame, results[0]

    def release(self):
        self.cap.release()


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
