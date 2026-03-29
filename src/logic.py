import time
import numpy as np
import subprocess

class GestureLogic:
    def __init__(self, midi_engine):
        self.midi = midi_engine
        
        # State tracking
        self.is_recording = False
        self.active_track = 1 # Assume starting on Track 1
        self.current_pitch_note = -1
        self.gesture_cooldown = 0
        
        # Scale mapping (default Chromatic A2)
        self.scale = [45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57]
        self.scale_type = "chromatic"
        self.base_theremin_note = 60 # C4

    def _send_keystroke_to_gb(self, key_code_or_stroke):
        """Sends a keystroke directly to GarageBand using macOS AppleScript."""
        try:
            script = f'''
            tell application "GarageBand" to activate
            tell application "System Events"
                {key_code_or_stroke}
            end tell
            '''
            subprocess.run(["osascript", "-e", script], check=True)
        except Exception as e:
            print(f"Failed to send keystroke to GarageBand: {e}")

    def _select_track(self, target_track):
        """Navigates to the target track in GarageBand using Up/Down arrows."""
        if target_track == self.active_track:
            return
            
        diff = target_track - self.active_track
        direction = "key code 125" if diff > 0 else "key code 126" # 125=Down, 126=Up
        
        for _ in range(abs(diff)):
            self._send_keystroke_to_gb(direction)
            time.sleep(0.1)
            
        self.active_track = target_track
        print(f"Moved to Track {self.active_track}")

    def process(self, yolo_result):
        """Processes YOLOv8-pose results (21 keypoints)."""
        if yolo_result.keypoints is None or len(yolo_result.keypoints) == 0:
            # If no hands, stop active note
            if self.current_pitch_note != -1:
                self.midi.send_note_off(self.current_pitch_note)
                self.current_pitch_note = -1
            if self.scale_type == "theremin":
                self.midi.send_note_off(self.base_theremin_note)
            return

        # Extract normalized keypoints (x, y)
        # xyn shape: (num_hands, 21, 2)
        kpts = yolo_result.keypoints.xyn.cpu().numpy()
        
        # Sort hands by X-coordinate of wrist (landmark 0) to distinguish Left/Right
        # Hand on the left side of image (lower x) = Left hand
        # Hand on the right side of image (higher x) = Right hand
        hands = []
        for i in range(len(kpts)):
            wrist_x = kpts[i][0][0]
            hands.append({'wrist_x': wrist_x, 'landmarks': kpts[i]})
        
        hands = sorted(hands, key=lambda x: x['wrist_x'])
        
        left_hand = None
        right_hand = None
        
        if len(hands) == 1:
            # If only one hand, determine if it's left or right based on screen half
            if hands[0]['wrist_x'] < 0.5:
                left_hand = hands[0]['landmarks']
            else:
                right_hand = hands[0]['landmarks']
        else:
            left_hand = hands[0]['landmarks']
            right_hand = hands[1]['landmarks']

        current_time = time.time()
        
        if left_hand is not None:
            self._process_left_hand(left_hand, current_time)
            
        if right_hand is not None:
            self._process_right_hand(right_hand)
        else:
            if self.current_pitch_note != -1:
                self.midi.send_note_off(self.current_pitch_note)
                self.current_pitch_note = -1
            if self.scale_type == "theremin":
                self.midi.send_note_off(self.base_theremin_note)

    def _process_left_hand(self, landmarks, current_time):
        # Landmarks: 0:wrist, 4:thumb_tip, 8:index_tip, 12:middle_tip, 16:ring_tip, 20:pinky_tip
        index_tip_x = landmarks[8][0]
        
        # Screen zones (1-3)
        if index_tip_x < 0.33: zone = 1
        elif index_tip_x < 0.66: zone = 2
        else: zone = 3
            
        gesture = self._detect_gesture(landmarks)
        
        if current_time - self.gesture_cooldown > 1.5: # 1.5s cooldown to prevent double presses
            if gesture == "Open" and not self.is_recording:
                print(f"Start Recording (Zone {zone})")
                self._select_track(zone)
                self._send_keystroke_to_gb('keystroke "r"') # Record
                self.is_recording = True
                self.gesture_cooldown = current_time
            elif gesture == "Closed" and self.is_recording:
                print(f"Stop Recording")
                self._send_keystroke_to_gb('keystroke space') # Stop (Spacebar)
                self.is_recording = False
                self.gesture_cooldown = current_time
            elif gesture == "Pointing":
                print(f"Play (Zone {zone})")
                self._select_track(zone)
                self._send_keystroke_to_gb('keystroke space') # Play (Spacebar)
                self.gesture_cooldown = current_time
            elif gesture == "Victory":
                print(f"Delete Track/Clip (Zone {zone})")
                self._select_track(zone)
                self._send_keystroke_to_gb('key code 51') # Backspace/Delete key
                self.gesture_cooldown = current_time

    def _process_right_hand(self, landmarks):
        # Pitch: Map Y coordinate of middle finger knuckle (landmark 9)
        # Note: 1.0 - y because image origin is top-left
        y = 1.0 - landmarks[9][1]
        
        if self.scale_type == "theremin":
            # Continuous pitch using Pitch Bend
            # MIDI Pitch Bend is 14-bit: 0 to 16383, center is 8192
            # mido pitch is -8192 to 8191
            pitch_bend = int(y * 16383) - 8192
            self.midi.send_pitch_bend(pitch_bend)
            
            # Send note on if not already playing
            if self.current_pitch_note != self.base_theremin_note:
                self.midi.send_note_on(self.base_theremin_note)
                self.current_pitch_note = self.base_theremin_note
        else:
            # Scale mapping
            note_index = int(y * len(self.scale))
            note_index = max(0, min(len(self.scale) - 1, note_index))
            new_note = self.scale[note_index]
            
            if new_note != self.current_pitch_note:
                if self.current_pitch_note != -1:
                    self.midi.send_note_off(self.current_pitch_note)
                self.midi.send_note_on(new_note)
                self.current_pitch_note = new_note
            
        # Volume: Map thumb tip (4) distance from index knuckle (5)
        thumb_tip = landmarks[4]
        index_knuckle = landmarks[5]
        dist = np.linalg.norm(thumb_tip - index_knuckle)
        
        # Normalize distance (approx 0.05 to 0.25)
        vol_norm = np.clip((dist - 0.05) / 0.2, 0, 1)
        volume = int(vol_norm * 127)
        self.midi.send_cc(7, volume)

    def _detect_gesture(self, landmarks):
        # Simple heuristic: tip.y < pip.y means finger is extended
        # Fingers: 8:Index, 12:Middle, 16:Ring, 20:Pinky
        # PIP joints: 6:Index, 10:Middle, 14:Ring, 18:Pinky
        
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        
        extended = []
        for tip, pip in zip(tips, pips):
            # Remember lower y is higher in screen
            extended.append(landmarks[tip][1] < landmarks[pip][1])
            
        num_extended = sum(extended)
        
        if num_extended >= 3:
            return "Open"
        elif num_extended == 0:
            return "Closed"
        elif num_extended == 1 and extended[0]: # Only index
            return "Pointing"
        elif num_extended == 2 and extended[0] and extended[1]: # Index + Middle
            return "Victory"
            
        return "Unknown"
