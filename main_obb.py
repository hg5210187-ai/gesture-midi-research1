"""Experimental version: one self-trained gesture detector for both hands.

A single YOLO26 detect model (src/vision_hbb.py, default hbb-s-fold1-320.pt in
the project root) finds both hands as boxes with a gesture class: the right
hand's box plays pitch/volume, the left hand's ``closedhand`` toggles the
looper (src/logic_obb.py). The earlier two-model backend (right-hand OBB +
left-hand pose) is still in src/vision_obb.py.

    Run:  uv run python main_obb.py

The reliable single-model version is main.py (uv run python main.py).
"""
import cv2
import os
import threading
import time
import queue
from src.vision_hbb import HbbVision
from src.midi_engine import MidiEngine
from src.logic_obb import GestureLogic
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
        self.vision = HbbVision()
        self.running = True

        # MIDI_LATENCY_DEBUG=1 prints the vision loop's rate and gesture->MIDI
        # timings every few seconds (see vision_loop).
        self.latency_debug = os.environ.get("MIDI_LATENCY_DEBUG") == "1"

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
        # No pacing here: get_frame() blocks until the camera delivers a new
        # frame, so the loop runs at camera rate (lowest latency) without
        # spinning a core.
        # A single bad frame (a transient inference/overlay error) should be
        # dropped, not tear down the camera + MIDI. The per-frame body is
        # wrapped so we log and skip it; we only give up after errors persist
        # (a real fault), and cleanup() runs once on genuine loop exit.
        consecutive_errors = 0
        try:
            while self.running:
                try:
                    success, frame, result = self.vision.get_frame()
                    if not success:
                        print("Vision loop: Failed to get frame.")
                        time.sleep(0.1)
                        continue

                    # Process gestures and send MIDI
                    self.logic.process(result)
                    if self.latency_debug:
                        self._log_latency()

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
        finally:
            self.cleanup()

    def _log_latency(self, every=5.0):
        """Collect per-frame timings; print mean/max every ``every`` seconds.

        "frame age at MIDI" is camera frame delivered -> MIDI sent. It excludes
        the camera's own internal delay, which can't be observed from here.
        """
        now = time.perf_counter()
        if not hasattr(self, "_lat_ages"):
            self._lat_ages, self._lat_infer, self._lat_since = [], [], now
        self._lat_ages.append((now - self.vision.last_capture_ts) * 1000.0)
        self._lat_infer.append(self.vision.last_infer_ms)
        elapsed = now - self._lat_since
        if elapsed >= every:
            n = len(self._lat_ages)
            print(f"[latency] {n / elapsed:.1f} fps | "
                  f"inference mean {sum(self._lat_infer) / n:.1f} ms, "
                  f"max {max(self._lat_infer):.1f} ms | "
                  f"frame age at MIDI mean {sum(self._lat_ages) / n:.1f} ms, "
                  f"max {max(self._lat_ages):.1f} ms")
            self._lat_ages, self._lat_infer, self._lat_since = [], [], now

    def update_ui_video(self):
        """Runs in the main thread to safely update Tkinter."""
        if not self.frame_queue.empty():
            frame = self.frame_queue.get()
            self.ui.update_image(frame)

        # Update looper status in UI
        self.ui.update_looper_status(self.looper.state)

        if self.running:
            # Re-schedule itself every 40ms (~25 FPS). This is display only;
            # gesture->MIDI samples at the vision loop's camera rate. Rebuilding
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
        # Let the vision thread finish its current frame before tearing down;
        # exiting the interpreter while it is mid-inference crashes torch.
        self.running = False
        self.process_thread.join(timeout=3)
        self.cleanup()

if __name__ == "__main__":
    app = MIDIApp()
    app.run()
