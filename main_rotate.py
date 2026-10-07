"""Rotation instrument: play sound by ROTATING the hand.

Uses the self-trained ``YOLO26n-MIDI.pt`` OBB model (src/vision_yolo26.py +
src/logic_rotate.py). The oriented-bounding-box angle of the RIGHT hand drives
pitch; the box area drives volume; a LEFT-hand fist (the model's ``closedhand``
gesture) toggles the looper. One model, both hands, no pose model.

Two pitch modes, switchable live from the "Rotation Mode" control in the UI:
  * Scale      – rotation snaps through the current scale's notes.
  * Continuous – rotation glides the pitch smoothly (theremin-style).

The model comes in five sizes (all OBB, same 3 gesture classes) — pick one with
``--model`` (default nano). Larger = more accurate but slower on MPS @ imgsz 640:
    n ~90fps   s ~29fps   m ~25fps   l ~24fps   x ~11fps
    (by mAP50-95: l 0.80, s 0.77, m 0.67, n 0.49, x 0.37)

    Run:  uv run python main_rotate.py                 # nano (default)
          uv run python main_rotate.py --model l       # large
          uv run python main_rotate.py -m s --imgsz 640

The reliable single-model (height-based) version is main.py; the two-model OBB
experiment is main_obb.py.
"""
import math
import queue
import threading

import cv2

from src.vision_yolo26 import Yolo26Vision
from src.logic_rotate import GestureLogic
from src.midi_engine import MidiEngine
from src.looper import MidiLooper
from src.ui import SettingsUI
from src.scales import ScaleGenerator
from src.overlay import render_overlay


class MIDIApp:
    def __init__(self, model="YOLO26n-MIDI.pt", imgsz=640):
        # Data & Logic
        self.midi_engine = MidiEngine()
        self.looper = MidiLooper(self.midi_engine, channel=1)
        self.logic = GestureLogic(self.midi_engine, self.looper)
        self.vision = Yolo26Vision(model=model, imgsz=imgsz)
        self.running = True

        # Vision/MIDI sample rate. Lower = lighter; higher = snappier gesture
        # response. 20 FPS (50ms) is a good balance on Apple Silicon.
        self.target_fps = 20

        # Last MIDI program (instrument) sent, so we only emit a Program Change
        # when the selection actually changes.
        self._current_program = None

        # Thread-safe queue for UI image updates
        self.frame_queue = queue.Queue(maxsize=2)

        # UI (Settings Window). show_rotation_mode reveals the Scale/Continuous
        # toggle and the rotation-range calibration buttons that only this
        # version uses.
        self.ui = SettingsUI(self.update_settings, self.on_calibrate,
                             show_rotation_mode=True,
                             on_calibrate_rotation=self.on_calibrate_rotation)

        # Processing thread
        self.process_thread = threading.Thread(target=self.vision_loop,
                                               daemon=True)

    def update_settings(self, settings):
        """Callback from UI to update engine logic."""
        print(f"Updating settings: {settings}")

        # Volume sensitivity + gate + overlay toggles (read by the vision thread)
        self.logic.volume_gain = settings.get("volume_gain", self.logic.volume_gain)
        self.logic.volume_gate = settings.get("volume_gate", self.logic.volume_gate)
        self.logic.show_borderlines = settings.get("show_borderlines", False)
        self.logic.show_pitch = settings.get("show_pitch", False)

        # Rotation pitch mode (Scale / Continuous).
        mode = settings.get("rotation_mode")
        if mode:
            self.logic.rotation_mode = mode.strip().lower()

        # Instrument (MIDI Program Change), sent only on an actual change, on
        # both the live channel (0) and the looper channel (1).
        program = settings.get("instrument_program")
        if program is not None and program != self._current_program:
            self.midi_engine.send_program_change(program, channel=0)
            self.midi_engine.send_program_change(program, channel=self.looper.channel)
            self._current_program = program
            print(f"Instrument changed -> program {program}")

        # Scale. Rotation always needs an ordered note list: it is what scale
        # mode steps through and what defines continuous mode's pitch range, so
        # (unlike the height versions) we build one even for "Theremin", falling
        # back to chromatic there.
        raw_scale_type = settings["scale_type"]
        scale_type = raw_scale_type.lower().replace(" ", "_")
        if scale_type == "theremin":
            scale_type = "chromatic"
        self.logic.scale_type = scale_type

        range_preset = settings.get("range_preset", "Full Range (5 Octaves)")
        if range_preset == "Melody (2 Octaves)":
            start_octave, num_octaves = 4, 2
        elif range_preset == "Melody (1 Octave)":
            start_octave, num_octaves = 4, 1
        elif "Bass" in range_preset:
            start_octave, num_octaves = 2, 1
        else:
            start_octave, num_octaves = 2, 5

        new_scale = ScaleGenerator.get_scale(settings["root_note"], scale_type,
                                             start_octave=start_octave,
                                             num_octaves=num_octaves)
        if new_scale:
            self.logic.scale = new_scale
            print(f"Scale updated to {settings['root_note']} {scale_type} "
                  f"({num_octaves} octaves), mode={self.logic.rotation_mode}")

    def on_calibrate(self):
        """UI 'Calibrate Min' button: capture the current right-hand box area."""
        area = self.logic.calibrate_min()
        if area is not None:
            self.ui.set_calib_status(f"Calibrated (min area = {area:.4f})", ok=True)
        else:
            self.ui.set_calib_status("No right hand detected - try again", ok=False)

    def on_calibrate_rotation(self, which):
        """UI 'Rotation Range' buttons: capture the low/high angle (or reset).

        Hold the right hand at the tilt for the lowest note and press Set Low,
        then rotate to the highest-note tilt and press Set High. Reset restores
        the default full-span window.
        """
        if which == "reset":
            self.logic.reset_rotation_range()
            lo = math.degrees(self.logic.angle_min)
            hi = math.degrees(self.logic.angle_max)
            self.ui.set_rotation_calib_status(
                f"Range reset to default ({lo:+.0f}° to {hi:+.0f}°)", ok=True)
            return

        deg = self.logic.calibrate_rotation(which)
        if deg is None:
            self.ui.set_rotation_calib_status(
                "No right hand detected - hold hand up and retry", ok=False)
            return

        lo = math.degrees(self.logic.angle_min)
        hi = math.degrees(self.logic.angle_max)
        span = abs(hi - lo)
        label = "Low note" if which == "low" else "High note"
        if span < 10.0:
            # Too small a span makes pitch hyper-sensitive to jitter; nudge the
            # user to rotate further between the two captures.
            self.ui.set_rotation_calib_status(
                f"{label} set to {deg:+.0f}°. Span only {span:.0f}° — rotate "
                f"further between Low and High.", ok=False)
        else:
            self.ui.set_rotation_calib_status(
                f"{label} set to {deg:+.0f}° (range {lo:+.0f}°..{hi:+.0f}°, "
                f"span {span:.0f}°)", ok=True)

    def vision_loop(self):
        """Camera processing and MIDI logic."""
        print("Vision loop started.")
        import time
        # Cap the loop to self.target_fps so the thread can sleep, cutting
        # CPU/GPU load and power. When a frame takes longer than the budget we
        # simply don't sleep.
        target_dt = 1.0 / self.target_fps
        # A single bad frame (a transient inference/overlay error) should be
        # dropped, not tear down the camera + MIDI; we only give up after errors
        # persist, and cleanup() runs once on genuine loop exit.
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

                    # Draw the lightweight overlay (rotation gauge + boxes)
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

                # Pace to the target FPS.
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
    import argparse

    SIZES = {"n", "s", "m", "l", "x"}
    parser = argparse.ArgumentParser(
        description="Rotation instrument driven by a YOLO26-MIDI OBB model.")
    parser.add_argument(
        "--model", "-m", default="n",
        help="Model size (n/s/m/l/x) or a path to a .pt file. A size maps to "
             "YOLO26<size>-MIDI.pt. Speed@640 on MPS: n~90 s~29 m~25 l~24 "
             "x~11 fps; by mAP50-95: l>s>m>n>x. Default: n.")
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Inference image size (default 640; s/m/l/x were trained at 640).")
    args = parser.parse_args()

    model = args.model
    if model.lower() in SIZES:
        model = f"YOLO26{model.lower()}-MIDI.pt"

    app = MIDIApp(model=model, imgsz=args.imgsz)
    app.run()
