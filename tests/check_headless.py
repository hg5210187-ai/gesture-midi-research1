"""Headless checks of the instrument logic (no camera, no MIDI port, no UI).

    uv run python tests/check_headless.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


class FakeMidi:
    def __init__(self):
        self.log = []

    def send_note_on(self, note, velocity=64, channel=0):
        self.log.append(("on", note, channel))

    def send_note_off(self, note, velocity=0, channel=0):
        self.log.append(("off", note, channel))

    def send_cc(self, control, value, channel=0):
        self.log.append(("cc", control, value, channel))

    def send_pitch_bend(self, pitch, channel=0):
        self.log.append(("bend", pitch, channel))

    def send_program_change(self, program, channel=0):
        self.log.append(("pc", program, channel))

    def notes_on(self):
        return [e[1] for e in self.log if e[0] == "on"]

    def last_cc(self):
        return [e[2] for e in self.log if e[0] == "cc"][-1]


class T:
    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class Boxes:
    def __init__(self, xyxyn, conf, cls=None):
        self.xyxyn, self.conf = T(xyxyn), T(conf)
        if cls is not None:
            self.cls = T(cls)

    def __len__(self):
        return len(self.xyxyn.a)


class Kpts:
    def __init__(self, xyn):
        self.xyn = T(xyn)

    def __len__(self):
        return len(self.xyn.a)


class Obb:
    def __init__(self, xywhr, polys, conf, cls):
        self.xywhr, self.xyxyxyxyn = T(xywhr), T(polys)
        self.conf, self.cls = T(conf), T(cls)

    def __len__(self):
        return len(self.xywhr.a)


class Res:
    pass


NAMES = {0: "thumbout", 1: "openhand", 2: "closedhand"}
ok = True


def check(label, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label} {detail}")


def box(cx, cy, w=0.2, h=0.3):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def open_hand_kpts(cx, cy):
    k = np.zeros((21, 2))
    k[0] = [cx, cy + 0.10]           # wrist
    k[9] = [cx, cy]                  # middle knuckle
    for tip in (8, 12, 16, 20):
        k[tip] = [cx, cy - 0.20]     # fingertips far from the palm
    return k


def fist_kpts(cx, cy):
    k = open_hand_kpts(cx, cy)
    for tip in (8, 12, 16, 20):
        k[tip] = [cx, cy + 0.05]     # fingertips on the palm centre
    return k


# --------------------------------------------------------------------------
print("1. entry points import (no app constructed)")
for mod in ("main", "main_mediapipe", "main_obb", "main_rotate", "run_benchmark"):
    try:
        __import__(mod)
        check(mod, True)
    except Exception as e:                      # noqa: BLE001
        check(mod, False, repr(e))

from src.looper import MidiLooper
from src.overlay import render_overlay
from src.scales import ScaleGenerator, midi_to_freq, midi_to_name_octave

print("2. scales")
check("A minor pentatonic starts at A2=45",
      ScaleGenerator.get_scale("A", "minor_pentatonic")[:5] == [45, 48, 50, 52, 55])
check("default 5 octaves of chromatic = 60 notes",
      len(ScaleGenerator.get_scale("C", "chromatic")) == 60)
check("69 -> A4 / 440 Hz",
      midi_to_name_octave(69) == "A4" and abs(midi_to_freq(69) - 440) < 1e-9)

# --------------------------------------------------------------------------
print("3. src/logic.py  (pose path: main.py / main_mediapipe.py)")
from src.logic import GestureLogic as PoseLogic


def pose_result(hands):
    """hands: list of (bbox, conf, kpts)."""
    r = Res()
    r.boxes = Boxes([h[0] for h in hands], [h[1] for h in hands])
    r.keypoints = Kpts([h[2] for h in hands])
    r.orig_shape = (480, 640)
    return r


def right_only(cy, w=0.2, h=0.3, conf=0.95):
    return pose_result([(box(0.75, cy, w, h), conf, open_hand_kpts(0.75, cy))])


m = FakeMidi()
lg = PoseLogic(m, MidiLooper(m, channel=1))
lg.process(right_only(0.5, 0.10, 0.15))       # small hand seeds area_min
check("first frame seeds area_min, gate shut (no note)", m.notes_on() == [])
lg.process(right_only(0.15))                  # big hand at the TOP edge of reach
check("hand at top -> highest scale note (57)", m.notes_on()[-1:] == [57], str(m.notes_on()))
lg.process(right_only(0.85))                  # BOTTOM edge of reach
check("hand at bottom -> lowest scale note (45)", m.notes_on()[-1:] == [45], str(m.notes_on()))
check("volume sent as CC7 and > gate", m.last_cc() > lg.volume_gate, str(m.last_cc()))
lg.process(right_only(0.85, conf=0.30))       # below min_confidence
check("low-confidence box is dropped -> note released",
      lg.current_pitch_note == -1 and m.log[-1][0] != "on")

m = FakeMidi()
looper = MidiLooper(m, channel=1)
lg = PoseLogic(m, looper)
states = []
for i in range(4):
    jit = 0.02 * i                            # jitter so the box is not "stationary"
    lg.process(pose_result([(box(0.25 + jit, 0.5), 0.95, fist_kpts(0.25 + jit, 0.5))]))
    states.append(looper.state)
check("left fist toggles looper only on the 3rd consecutive frame",
      states == ["IDLE", "IDLE", "RECORDING", "RECORDING"], str(states))
check("fist confirm frames == 3", lg._fist_confirm_frames == 3)
check("min_confidence == 0.7, play_margin == 0.15",
      lg.min_confidence == 0.7 and lg.play_margin == 0.15)

# stationary phantom: a perfectly still, confident right-side box goes silent
m = FakeMidi()
lg = PoseLogic(m, None)
lg.process(right_only(0.5, 0.10, 0.15))
for _ in range(40):
    lg.process(right_only(0.3))
check("motionless box is suppressed after ~30 frames",
      lg.current_pitch_note == -1 and lg.last_right_bbox is None)

# theremin: held C4 + full-range bend
m = FakeMidi()
lg = PoseLogic(m, None)
lg.scale_type = "theremin"
lg.process(right_only(0.5, 0.10, 0.15))
lg.process(right_only(0.15))
bends = [e[1] for e in m.log if e[0] == "bend"]
check("theremin holds C4 (60) and bends to +8191 at the top",
      m.notes_on() == [60] and bends[-1] == 8191, f"{m.notes_on()} {bends[-1:]}")

frame = np.zeros((480, 640, 3), np.uint8)
lg.show_borderlines = True
render_overlay(frame, lg)
check("overlay renders (pose logic)", frame.any())

# --------------------------------------------------------------------------
print("4. src/logic_obb.py  (single gesture detector: main_obb.py)")
from src.logic_obb import GestureLogic as HbbLogic


def det_result(dets):
    """dets: list of (bbox, conf, cls)."""
    r = Res()
    r.boxes = Boxes([d[0] for d in dets], [d[1] for d in dets], [d[2] for d in dets])
    r.names = NAMES
    r.orig_shape = (480, 640)
    return r


m = FakeMidi()
looper = MidiLooper(m, channel=1)
lg = HbbLogic(m, looper)
lg.process(det_result([(box(0.75, 0.5, 0.10, 0.15), 0.9, 1)]))
lg.process(det_result([(box(0.75, 0.15), 0.9, 1)]))
check("right-hand box at top -> 57", m.notes_on()[-1:] == [57], str(m.notes_on()))
lg.process(det_result([(box(0.75, 0.5, 0.7, 0.7), 0.9, 1)]))   # area 0.49 > 0.30
check("background-sized box rejected (max_box_area 0.30)", lg.current_pitch_note == -1)
states = []
for _ in range(6):
    lg.process(det_result([(box(0.25, 0.5), 0.9, 2)]))          # left closedhand
    states.append(looper.state)
check("left closedhand toggles looper on the 5th frame",
      states == ["IDLE"] * 4 + ["RECORDING"] * 2, str(states))
check("defaults: min_confidence 0.6, confirm 5, release 5",
      lg.min_confidence == 0.6 and lg._fist_confirm_frames == 5
      and lg._fist_release_frames == 5)
lg2 = HbbLogic(FakeMidi(), None)
lg2.process(det_result([(box(0.25, 0.5), 0.9, 1)]))            # left open hand only
check("left half never plays a note", lg2.current_pitch_note == -1)

# --------------------------------------------------------------------------
print("5. src/logic_rotate.py  (rotation instrument: main_rotate.py)")
from src.logic_rotate import GestureLogic as RotLogic


def obb_result(dets):
    """dets: list of (cx, cy, w, h, r, conf, cls) in normalized units."""
    W, H = 640, 480
    r = Res()
    xywhr, polys = [], []
    for cx, cy, w, h, rot, _, _ in dets:
        xywhr.append([cx * W, cy * H, w * W, h * H, rot])
        polys.append([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2],
                      [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]])
    r.obb = Obb(xywhr, polys, [d[5] for d in dets], [d[6] for d in dets])
    r.orig_shape = (H, W)
    return r


m = FakeMidi()
lg = RotLogic(m, None)
lg.process(obb_result([(0.75, 0.5, 0.10, 0.15, 0.0, 0.9, 0)]))   # seed area_min
lg.process(obb_result([(0.75, 0.5, 0.20, 0.30, 0.0, 0.9, 0)]))   # vertical hand
mid = m.notes_on()[-1]
check("vertical hand (angle 0) -> middle of the scale (51)", mid == 51, str(mid))
check("angle window defaults to +/-90 deg",
      abs(lg.angle_min + math.pi / 2) < 1e-9 and abs(lg.angle_max - math.pi / 2) < 1e-9)
lg.process(obb_result([(0.75, 0.5, 0.20, 0.30, 0.6, 0.9, 0)]))   # rotate ~+34 deg
check("rotating changes the note", m.notes_on()[-1] != mid, str(m.notes_on()))
lg.rotation_mode = "continuous"
lg.process(obb_result([(0.75, 0.5, 0.20, 0.30, 0.65, 0.9, 0)]))
check("continuous mode sends pitch bend", any(e[0] == "bend" for e in m.log))
frame = np.zeros((480, 640, 3), np.uint8)
render_overlay(frame, lg)
check("overlay renders the rotation gauge", frame.any())

print()
print("ALL PASS" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
