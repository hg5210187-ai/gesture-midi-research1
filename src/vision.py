import cv2
import mediapipe as mp

class VisionModule:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise Exception("Cannot open camera")

        self.mp_hands = mp.solutions.hands
        # Limit to 2 hands max (one left, one right)
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def get_frame(self):
        success, image = self.cap.read()
        if not success:
            return False, None, None

        # Flip image horizontally for a selfie-view display
        image = cv2.flip(image, 1)

        # Convert the BGR image to RGB before processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # To improve performance, optionally mark the image as not writeable
        image_rgb.flags.writeable = False
        results = self.hands.process(image_rgb)
        image_rgb.flags.writeable = True

        # Convert back to BGR for rendering
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        hand_data = {
            'Left': None,
            'Right': None
        }

        # Draw the hand annotations on the image.
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # handedness.classification[0].label is "Left" or "Right"
                # Keep in mind that mediapipe's Left/Right might be flipped if we flip the image,
                # but since we flipped the image *before* processing, mediapipe will see it as a mirror.
                # Actually, Mediapipe assumes non-mirrored for accurate L/R detection.
                # When we flip, the label might be reversed. 
                # Let's extract the label:
                label = handedness.classification[0].label
                
                # Extract landmarks
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append({'x': lm.x, 'y': lm.y, 'z': lm.z})
                
                # Assign to our data structure based on the label
                # Note: Because we flipped the frame, Mediapipe will think our physical right hand is the "Left" hand in the image
                # Let's map it back to physical hands. 
                # If label == "Left" in a flipped image, it means it's physically the Right hand.
                physical_label = "Right" if label == "Left" else "Left"
                hand_data[physical_label] = landmarks

                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS)
                    
        return True, image, hand_data

    def release(self):
        self.cap.release()
        self.hands.close()

if __name__ == "__main__":
    vision = VisionModule()
    while True:
        success, frame, data = vision.get_frame()
        if not success:
            break
            
        cv2.imshow('MIDI Gesture Vision', frame)
        if cv2.waitKey(5) & 0xFF == 27: # ESC to quit
            break
            
    vision.release()
    cv2.destroyAllWindows()
