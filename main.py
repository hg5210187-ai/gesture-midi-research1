"""Working version: one YOLOv8 hand-pose model drives BOTH hands.

Right hand <- bounding box (pitch from Y-center, volume from area); left hand
<- keypoints (fist -> looper). This is the reliable version (src/vision_ultralytics.py
+ src/logic.py) and needs no custom model.

    Run:  uv run python main.py

The experimental trained-OBB version is main_obb.py (uv run python main_obb.py).
"""
import cv2
import threading
import queue
from src.vision_ultralytics import UltralyticsVision
from src.midi_engine import MidiEngine
from src.logic import GestureLogic
from src.looper import MidiLooper
from src.ui import SettingsUI
from src.scales import ScaleGenerator
from src.overlay import render_overlay

class MIDIApp:
    def __init__(self):
        # Data & Logic
        self.midi_engine = MidiEngine()
        self.looper = MidiLooper(self.midi_engine, channel=1)
        self.logic = GestureLogic(self.midi_engine, self.looper)
        self.vision = UltralyticsVision()
        self.running = True

        # Vision/MIDI sample rate. Lower = lighter (less CPU/GPU and power),
        # higher = snappier gesture response. 20 FPS (50ms) is a good balance on
        # Apple Silicon; raise toward 30 for crisper fast playing.
        self.target_fps = 20

        # Last MIDI program (instrument) sent, so we only emit a Program Change
        # when the selection actually changes rather than on every settings tweak.
        self._current_program = None
        
        # Thread-safe queue for UI image updates
        self.frame_queue = queue.Queue(maxsize=2)
        
        # UI (Settings Window)
        self.ui = SettingsUI(self.update_settings, self.on_calibrate)
        
        # Processing thread
        self.process_thread = threading.Thread(target=self.vision_loop, daemon=True)

    def update_settings(self, settings):
        """Callback from UI to update engine logic."""
        print(f"Updating settings: {settings}")

        # Volume sensitivity + gate + overlay toggles (read by the vision thread)
        self.logic.volume_gain = settings.get("volume_gain", self.logic.volume_gain)
        self.logic.volume_gate = settings.get("volume_gate", self.logic.volume_gate)
        self.logic.show_borderlines = settings.get("show_borderlines", False)
        self.logic.show_pitch = settings.get("show_pitch", False)

        # Instrument (MIDI Program Change). Send only on an actual change, on
        # both the live channel (0) and the looper channel (1) so recorded loops
        # use the same instrument as live play.
        program = settings.get("instrument_program")
        if program is not None and program != self._current_program:
            self.midi_engine.send_program_change(program, channel=0)
            self.midi_engine.send_program_change(program, channel=self.looper.channel)
            self._current_program = program
            print(f"Instrument changed -> program {program}")

        # Update scale
        raw_scale_type = settings["scale_type"]
        scale_type = raw_scale_type.lower().replace(" ", "_")
            
        self.logic.scale_type = scale_type
        
        range_preset = settings.get("range_preset", "Full Range (5 Octaves)")
        if range_preset == "Melody (2 Octaves)":
            start_octave = 4
            num_octaves = 2
        elif range_preset == "Melody (1 Octave)":
            start_octave = 4
            num_octaves = 1
        elif "Bass" in range_preset:
            start_octave = 2
            num_octaves = 1
        else:
            start_octave = 2
            num_octaves = 5
        
        if scale_type != "theremin":
            new_scale = ScaleGenerator.get_scale(settings["root_note"], scale_type, start_octave=start_octave, num_octaves=num_octaves)
            self.logic.scale = new_scale
            print(f"Scale updated to {settings['root_note']} {scale_type} ({num_octaves} octaves)")
        else:
            print("Theremin mode active: Continuous pitch tracking.")

    def on_calibrate(self):
        """UI 'Calibrate Min' button: capture the current right-hand box area."""
        area = self.logic.calibrate_min()
        if area is not None:
            self.ui.set_calib_status(f"Calibrated (min area = {area:.4f})", ok=True)
        else:
            self.ui.set_calib_status("No right hand detected - try again", ok=False)

    def vision_loop(self):
        """Camera processing and MIDI logic."""
        print("Vision loop started.")
        import time
        # Cap the loop to self.target_fps. Running flat-out keeps a core pinned
        # for no benefit (the camera + inference cost ~33ms/frame); pacing the
        # loop lets the thread sleep, cutting CPU/GPU load and power. When a
        # frame takes longer than the budget we simply don't sleep.
        target_dt = 1.0 / self.target_fps
        # A single bad frame (a transient inference/overlay error) should be
        # dropped, not tear down the camera + MIDI. The per-frame body is
        # wrapped so we log and skip it; we only give up after errors persist
        # (a real fault), and cleanup() runs once on genuine loop exit.
        consecutive_errors = 0
        try:
            while self.running:
                loop_start = time.time()
                try:
                    success, frame, result = self.vision.get_frame()
                    if not success:
                        print("Vision loop: Failed to get frame.")
                        time.sleep(0.1)
                        continue

                    # Process gestures and send MIDI
                    self.logic.process(result)

                    # Draw the lightweight overlay (replaces the YOLO skeleton)
                    render_overlay(frame, self.logic)

                    # Put frame into the queue (drop oldest if full to avoid lag)
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame)

                    consecutive_errors = 0
                except Exception as e:
                    consecutive_errors += 1
                    print(f"Vision loop: dropping frame after error "
                          f"({consecutive_errors}): {e}")
                    if consecutive_errors >= 60:
                        print("Vision loop: too many consecutive errors; stopping.")
                        break
                    time.sleep(0.05)

                # Pace to the target FPS so we don't burn power on frames the UI
                # never displays.
                sleep_t = target_dt - (time.time() - loop_start)
                if sleep_t > 0:
                    time.sleep(sleep_t)
        finally:
            self.cleanup()

    def update_ui_video(self):
        """Runs in the main thread to safely update Tkinter."""
        if not self.frame_queue.empty():
            frame = self.frame_queue.get()
            self.ui.update_image(frame)

        # Update looper status in UI
        self.ui.update_looper_status(self.looper.state)

        if self.running:
            # Re-schedule itself every 40ms (~25 FPS). This is display only;
            # gesture->MIDI still samples at the vision loop's ~30 FPS. Rebuilding
            # the video image is main-thread work, so a slightly lower refresh
            # keeps the UI light without affecting playing responsiveness.
            self.ui.after(40, self.update_ui_video)

    def cleanup(self):
        self.running = False
        self.looper.cleanup()
        self.vision.release()
        self.midi_engine.close()

    def run(self):
        self.process_thread.start()
        # Start UI update polling
        self.ui.after(100, self.update_ui_video)
        # Main thread runs the settings UI
        self.ui.mainloop()
        self.cleanup()

if __name__ == "__main__":
    app = MIDIApp()
    app.run()
