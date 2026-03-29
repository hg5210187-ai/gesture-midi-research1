from ultralytics import YOLO
import cv2

class UltralyticsVision:
    def __init__(self, model_variant="yolov8n-hand-pose.pt"):
        # Load the specialized hand pose model
        self.model = YOLO(model_variant)
        self.cap = cv2.VideoCapture(0)
        
    def get_frame(self):
        success, frame = self.cap.read()
        if not success:
            return False, None, None
            
        frame = cv2.flip(frame, 1)
        
        # Run inference
        results = self.model.predict(source=frame, stream=False, verbose=False)
        result = results[0]
        
        # Render detection on frame
        annotated_frame = result.plot()
        
        # result.keypoints.xyn contains normalized coordinates [N, 21, 2]
        return True, annotated_frame, result

    def release(self):
        self.cap.release()

if __name__ == "__main__":
    vision = UltralyticsVision()
    try:
        while True:
            success, frame, result = vision.get_frame()
            if not success: break
            cv2.imshow("Ultralytics MIDI Vision", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        vision.release()
        cv2.destroyAllWindows()
