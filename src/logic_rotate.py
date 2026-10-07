"""Rotation instrument logic for ``main_rotate.py``.

The player creates sound by **rotating the hand**. Every detection from the
``YOLO26n-MIDI.pt`` OBB model carries an oriented box, so the box **angle** is
turned into pitch. Two pitch modes (switchable live from the UI):

  * **scale**      – the angle is quantised to the current scale, so rotating
                     the hand steps cleanly through the notes.
  * **continuous** – the angle maps to smooth, unquantised pitch (a chromatic
                     anchor note plus a pitch bend for the sub-semitone part),
                     so rotating the hand glides the pitch like a theremin.

In both modes:
  * **volume** comes from the box AREA relative to a calibrated minimum
    (rotation-invariant ``w*h``), and the note is gated off below the volume
    gate — same feel as the other versions, and it means rotation changes pitch
    without a rogue note when the hand is small/absent.
  * the **left hand** toggles the looper by making a fist. With this model the
    fist is read directly from the ``closedhand`` gesture class — no keypoints
    and no separate pose model.

Left/right roles follow the box center-x convention (the right hand sits on the
right half of the mirrored frame).

Set ``ROT_DEBUG=1`` to print each right-hand detection's angle / normalized
position / note, so the angle window (``angle_min`` / ``angle_max`` /
``invert_rotation``) can be tuned to how you hold the hand:
    ROT_DEBUG=1 uv run python main_rotate.py
"""
import math
import os
import time

import numpy as np

_ROT_DEBUG = os.environ.get("ROT_DEBUG") == "1"

# Gesture class indices for YOLO26n-MIDI.pt (names={0:thumbout,1:openhand,
# 2:closedhand}). Read from the model at runtime where possible; these are the
# fallbacks / semantic anchors used by the logic.
CLS_THUMBOUT = 0
CLS_OPENHAND = 1
CLS_CLOSEDHAND = 2
_CLASS_NAMES = {CLS_THUMBOUT: "thumbout",
                CLS_OPENHAND: "openhand",
                CLS_CLOSEDHAND: "closedhand"}


class GestureLogic:
    def __init__(self, midi_engine, looper=None):
        self.midi = midi_engine
        self.looper = looper

        # Right hand state
        self.current_pitch_note = -1

        # Scale mapping (default Chromatic A2). ``scale`` is the ordered list of
        # MIDI notes rotation sweeps through in scale mode, and its endpoints
        # define the pitch range in continuous mode.
        self.scale = [45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57]
        self.scale_type = "chromatic"

        # Pitch mode: "scale" (snap to self.scale) or "continuous" (smooth
        # chromatic glide via pitch bend). Set live from the UI.
        self.rotation_mode = "scale"

        # --- Angle -> pitch mapping window ---
        # The angle fed to the mapping is first recentered (see
        # _center_and_unwrap) so a VERTICAL hand (thumb straight up) reads 0 and
        # lands in the MIDDLE of the pitch range; rotating toward horizontal
        # moves toward the ends. The default window is the full physical
        # half-turn (+/- 90 deg = 180 deg span), mapped to [0, 1] by
        # _angle_to_t. Narrow it (or use the Set Low/High calibration) to match
        # how far you actually rotate — use ROT_DEBUG to read your live angles.
        self.angle_min = -math.pi / 2.0     # -90 deg -> lowest note
        self.angle_max = math.pi / 2.0      # +90 deg -> highest note
        self.invert_rotation = False
        # Defaults kept so the "Reset Range" button can restore the full span.
        self._default_angle_min = self.angle_min
        self._default_angle_max = self.angle_max

        # Recenter/unwrap state. Ultralytics regularizes the raw OBB angle to
        # [0, pi/2), which puts a discontinuity right at the vertical hand pose
        # and spans only ~90 deg. Unwrapping across frames stitches successive
        # quadrants together so a continuous rotation reaches a full ~180 deg
        # instead of snapping back. Set unwrap_rotation=False for a rock-solid
        # (but +/-45 deg) single-frame range if it ever feels jumpy.
        self.unwrap_rotation = True
        self._prev_centered = None
        self._wrap_offset = 0.0

        # Left-hand looper control (fist == closedhand gesture)
        self._prev_fist = False        # confirmed (sustained) fist state
        self._last_toggle_time = 0
        self._toggle_cooldown = 1.0    # seconds between toggles
        # Require a sustained fist (held several consecutive frames) before it
        # counts, so a single-frame phantom can't toggle the looper.
        self._fist_frames = 0
        self._fist_confirm_frames = 3
        # Release debounce: the model can drop the left hand for a frame or two;
        # releasing the held-fist state on a single dropout would read the
        # re-detection as a fresh grip and fire a spurious toggle, so the fist is
        # only considered released after sustained absence.
        self._left_absent_frames = 0
        self._fist_release_frames = 3

        # Throttle recording of continuous events into the looper
        self._last_recorded_volume = -1
        self._last_recorded_pitch = -99999
        self._cur_bend = 0            # last pitch-bend value actually sent

        # Last computed values for logging / overlay
        self.last_pitch_cents = None
        self.last_angle_deg = None    # right-hand angle in degrees (overlay)
        self.last_angle_t = None      # normalized rotation position [0, 1]
        self.last_right_gesture = None
        self.last_left_gesture = None

        # Box-area driven volume state
        self.area_min = None          # calibrated min box area -> volume 0
        self.volume_gain = 2.0        # sensitivity gain G (set from UI slider)
        # Volume gate (0-127): below this the note is fully released (note_off),
        # not just made quiet, so shrinking/removing the hand cuts the sound.
        self.volume_gate = 8
        self.last_right_area = None   # latest right-hand box area (for calibrate)
        self.last_right_bbox = None   # latest right-hand axis-aligned bbox (overlay)
        self.last_right_obb = None    # latest right-hand OBB corners [4,2] norm (overlay)
        self.last_right_rotation = None  # latest OBB angle, radians (overlay)
        self.last_left_bbox = None    # latest left-hand bbox (for overlay)

        # --- Phantom-rejection gates ---
        # The model can fire on textured background; two cheap gates reject
        # those: a confidence floor and a max box area (a real hand never fills
        # this fraction of the frame at playing distance). Tune from ROT_DEBUG.
        self.min_confidence = 0.5
        self.max_box_area = 0.35

        # Tells the shared overlay to draw the rotation gauge (and skip the
        # height-based pitch guides, which are meaningless here).
        self.pitch_from_rotation = True

        # Overlay toggles (set from UI checkboxes; read by render_overlay).
        self.show_borderlines = False
        self.show_pitch = True

    # ---- frame processing -------------------------------------------------

    def process(self, obb_result):
        """Process one frame's OBB result (all hands in ``obb_result.obb``)."""
        right, left = self._extract_hands(obb_result)

        # ---- Right hand: rotation -> pitch, area -> volume ----
        if right is not None:
            # Recenter (thumb-up -> 0 -> middle note) and unwrap the raw OBB
            # angle before it drives pitch / is stored for calibration+overlay.
            angle = self._center_and_unwrap(right["rotation"])
            self.last_right_obb = right["poly"]
            self.last_right_bbox = right["bbox"]
            self.last_right_rotation = angle
            self.last_right_gesture = _CLASS_NAMES.get(right["cls"], "?")
            self._process_right_hand(angle, right["area"])
        else:
            self.last_right_obb = None
            self.last_right_bbox = None
            self.last_right_rotation = None
            self.last_right_area = None
            self.last_angle_deg = None
            self.last_angle_t = None
            self.last_right_gesture = None
            # Restart unwrapping cleanly when the hand reappears.
            self._prev_centered = None
            self._wrap_offset = 0.0
            self._silence_right_hand()

        # ---- Left hand: closedhand gesture -> looper toggle ----
        if left is not None:
            self._left_absent_frames = 0
            self.last_left_bbox = left["bbox"]
            self.last_left_gesture = _CLASS_NAMES.get(left["cls"], "?")
            if self.looper is not None:
                self._process_left_hand(left["cls"] == CLS_CLOSEDHAND)
        else:
            self.last_left_bbox = None
            self.last_left_gesture = None
            self._fist_frames = 0
            # Debounce: only release the held-fist state after sustained absence.
            self._left_absent_frames += 1
            if self._left_absent_frames >= self._fist_release_frames:
                self._prev_fist = False

    def _extract_hands(self, obb_result):
        """Split OBB detections into (right, left) hand dicts (either may be None).

        The rightmost detection on the right half of the frame is the right
        hand; the leftmost on the left half is the left hand. Each dict carries
        ``cx``/``cy`` (normalized center), ``area`` (normalized ``w*h``),
        ``rotation`` (radians), ``cls`` (gesture class id), ``bbox``
        (axis-aligned ``[x1,y1,x2,y2]`` normalized) and ``poly`` (``[4,2]``
        normalized corners).
        """
        if obb_result is None:
            return None, None
        obb = getattr(obb_result, "obb", None)
        if obb is None or len(obb) == 0:
            return None, None

        h, w = obb_result.orig_shape            # (height, width) in pixels
        xywhr = obb.xywhr.cpu().numpy()         # [N,5] px: cx, cy, w, h, r(rad)
        polys = obb.xyxyxyxyn.cpu().numpy()     # [N,4,2] normalized corners
        confs = (obb.conf.cpu().numpy()
                 if getattr(obb, "conf", None) is not None else None)
        clss = (obb.cls.cpu().numpy().astype(int)
                if getattr(obb, "cls", None) is not None else None)

        right = None
        left = None
        for i in range(len(xywhr)):
            conf = float(confs[i]) if confs is not None else 1.0
            cls = int(clss[i]) if clss is not None else -1
            cx_px, cy_px, bw_px, bh_px, r = xywhr[i]
            cx = float(cx_px / w)
            cy = float(cy_px / h)
            area = float((bw_px * bh_px) / (w * h))

            # Phantom rejection: weak confidence or a box too large to be a hand.
            if conf < self.min_confidence:
                continue
            if self.max_box_area is not None and area > self.max_box_area:
                continue

            poly = polys[i]                      # [4,2] normalized
            x1 = float(poly[:, 0].min()); x2 = float(poly[:, 0].max())
            y1 = float(poly[:, 1].min()); y2 = float(poly[:, 1].max())
            cand = {
                "cx": cx, "cy": cy, "area": area,
                "rotation": float(r), "cls": cls,
                "bbox": np.array([x1, y1, x2, y2], dtype=float),
                "poly": poly,
            }

            if cx >= 0.5:
                # Right half -> right-hand candidate (keep the rightmost).
                if right is None or cx > right["cx"]:
                    right = cand
            else:
                # Left half -> left-hand candidate (keep the leftmost).
                if left is None or cx < left["cx"]:
                    left = cand

        return right, left

    def _silence_right_hand(self):
        # current_pitch_note tracks whatever is sounding; this single guarded
        # release covers every mode. Also relax any active pitch bend.
        if self.current_pitch_note != -1:
            self.midi.send_note_off(self.current_pitch_note)
            if self.looper:
                self.looper.record_event('note_off', note=self.current_pitch_note)
            self.current_pitch_note = -1
        self._set_bend(0)

    # ---- angle -> pitch ---------------------------------------------------

    def _center_and_unwrap(self, r):
        """Recenter (and optionally unwrap) the raw OBB angle, in radians.

        Ultralytics regularizes the OBB angle to ``[0, pi/2)``, which places a
        discontinuity exactly at the vertical hand pose and spans only ~90 deg.
        This returns an angle where **0 == hand vertical (thumb up)** — so
        thumb-up lands in the middle of the pitch range — with the fold moved
        out to the +/-45 deg diagonals. When ``unwrap_rotation`` is on, jumps at
        that fold are stitched across frames so a continuous rotation keeps going
        toward a full +/-90 deg (180 deg) sweep instead of snapping back.
        """
        half = math.pi / 2.0
        quarter = math.pi / 4.0
        # Recenter: vertical (r ~ 0 or ~ pi/2) -> 0; fold now sits at +/-45 deg.
        a = r if r <= quarter else r - half
        if not self.unwrap_rotation:
            self._prev_centered = a
            return a
        # Unwrap: a per-frame jump near the +/-45 deg fold is a quadrant wrap;
        # accumulate +/-90 deg so the angle stays continuous as rotation
        # continues past the fold (human motion between 20 FPS frames is far
        # smaller than the 45 deg threshold, so real motion is never mistaken
        # for a wrap).
        if self._prev_centered is not None:
            d = a - self._prev_centered
            if d > quarter:
                self._wrap_offset -= half
            elif d < -quarter:
                self._wrap_offset += half
        self._prev_centered = a
        return max(-half, min(half, a + self._wrap_offset))

    def _angle_to_t(self, r):
        """Normalize an OBB angle (radians) to a rotation position in [0, 1]."""
        span = self.angle_max - self.angle_min
        if abs(span) < 1e-9:
            return 0.0
        t = (r - self.angle_min) / span
        if self.invert_rotation:
            t = 1.0 - t
        return min(1.0, max(0.0, t))

    def _pitch_from_t(self, t):
        """Map rotation position ``t`` in [0, 1] to ``(midi_note, pitch_bend)``.

        In scale mode the bend is 0 (notes snap to the scale). In continuous
        mode the note is the nearest chromatic semitone and the bend carries the
        sub-semitone remainder for a smooth glide.
        """
        if not self.scale:
            return -1, 0

        if self.rotation_mode == "continuous":
            lo = int(self.scale[0])
            hi = int(self.scale[-1])
            if hi <= lo:
                hi = lo + 12
            exact = lo + t * (hi - lo)                 # semitones (float)
            midi_note = int(round(exact))
            midi_note = max(0, min(127, midi_note))
            cents = (exact - midi_note) * 100.0        # in [-50, 50]
            # Standard pitch-bend range is +/-2 semitones (+/-200 cents) across
            # the full +/-8192, so a cents offset maps as cents/200 * 8192.
            pitch_bend = int((cents / 200.0) * 8192)
            pitch_bend = max(-8192, min(8191, pitch_bend))
            return midi_note, pitch_bend

        # scale mode
        n = len(self.scale)
        idx = int(t * n)
        idx = max(0, min(n - 1, idx))
        return int(self.scale[idx]), 0

    def _process_right_hand(self, rotation, area):
        """Drive pitch (from rotation) + volume (from box area)."""
        t = self._angle_to_t(rotation)
        self.last_angle_t = t
        self.last_angle_deg = math.degrees(rotation)

        midi_note, pitch_bend = self._pitch_from_t(t)
        self.last_pitch_cents = midi_note * 100.0 + (pitch_bend / 8192.0) * 200.0

        # Volume: box AREA relative to a calibrated minimum.
        # vol = clip((area / area_min - 1) * gain, 0, 1) * 127
        self.last_right_area = area
        if self.area_min is None or self.area_min <= 0:
            # Seed from the first frame so volume is always defined (and we
            # never divide by zero). The Calibrate Min button overrides it.
            self.area_min = area

        vol_norm = np.clip((area / self.area_min - 1.0) * self.volume_gain,
                           0.0, 1.0)
        volume = int(vol_norm * 127)
        gate_open = volume > self.volume_gate

        if _ROT_DEBUG:
            print(f"[ROT] mode={self.rotation_mode} "
                  f"ang={self.last_angle_deg:+6.1f} t={t:.2f} "
                  f"note={midi_note} bend={pitch_bend} vol={volume} "
                  f"gate={'open' if gate_open else 'shut'} "
                  f"gesture={self.last_right_gesture}")

        # --- Note gating ---
        if not gate_open:
            volume = 0
            if self.current_pitch_note != -1:
                self.midi.send_note_off(self.current_pitch_note)
                if self.looper:
                    self.looper.record_event('note_off',
                                             note=self.current_pitch_note)
                self.current_pitch_note = -1
            self._set_bend(0)
            # Re-arm the bend throttle so the first bend after the gate reopens
            # is recorded cleanly rather than swallowed as "no change".
            self._last_recorded_pitch = -99999
        else:
            # Gate open. Pitch bend first so a re-triggered note lands already
            # bent (continuous mode); scale mode forces the bend back to 0.
            self._set_bend(pitch_bend if self.rotation_mode == "continuous" else 0)

            if midi_note != self.current_pitch_note:
                if self.current_pitch_note != -1:
                    self.midi.send_note_off(self.current_pitch_note)
                    if self.looper:
                        self.looper.record_event('note_off',
                                                 note=self.current_pitch_note)
                if midi_note != -1:
                    self.midi.send_note_on(midi_note)
                    if self.looper:
                        self.looper.record_event('note_on', note=midi_note)
                self.current_pitch_note = midi_note

        self.midi.send_cc(7, volume)
        if self.looper and abs(volume - self._last_recorded_volume) >= 5:
            self.looper.record_event('cc', control=7, value=volume)
            self._last_recorded_volume = volume

    def _set_bend(self, pitch_bend):
        """Send a pitch bend only when it actually changes (and record it)."""
        pitch_bend = max(-8192, min(8191, int(pitch_bend)))
        if pitch_bend == self._cur_bend:
            return
        self.midi.send_pitch_bend(pitch_bend)
        if self.looper and abs(pitch_bend - self._last_recorded_pitch) >= 100:
            self.looper.record_event('pitch_bend', pitch=pitch_bend)
            self._last_recorded_pitch = pitch_bend
        self._cur_bend = pitch_bend

    # ---- left hand / looper ----------------------------------------------

    def _process_left_hand(self, is_fist):
        """Toggle the looper on a SUSTAINED fist (``closedhand`` gesture).

        The fist must be held for ``_fist_confirm_frames`` consecutive frames
        before it counts, so a single-frame phantom can't toggle the looper.
        """
        self._fist_frames = self._fist_frames + 1 if is_fist else 0
        confirmed = self._fist_frames >= self._fist_confirm_frames
        now = time.time()

        # Rising edge of a CONFIRMED fist, with cooldown.
        if (confirmed and not self._prev_fist
                and (now - self._last_toggle_time) > self._toggle_cooldown):
            state = self.looper.toggle()
            self._last_toggle_time = now
            if state == "RECORDING":
                self._last_recorded_volume = -1
                self._last_recorded_pitch = -99999
            print(f"Looper: {state}")

        self._prev_fist = confirmed

    # ---- calibration ------------------------------------------------------

    def calibrate_min(self):
        """Capture the current right-hand box area as the volume-0 reference.

        Returns the captured area, or None if no right hand is currently
        visible (in which case nothing changes).
        """
        if self.last_right_area is not None and self.last_right_area > 0:
            self.area_min = self.last_right_area
            print(f"Calibrated area_min = {self.area_min:.5f}")
            return self.area_min
        return None

    def calibrate_rotation(self, which):
        """Capture the current right-hand angle as a pitch-range endpoint.

        ``which == "low"`` sets ``angle_min`` (the tilt that plays the lowest
        note, ``t == 0``); ``which == "high"`` sets ``angle_max`` (the highest
        note, ``t == 1``). Because ``_angle_to_t`` interpolates between the two
        endpoints, capturing them in either rotation direction Just Works — no
        separate invert flag. Returns the captured angle in **degrees**, or
        ``None`` if no right hand is currently visible (nothing changes).
        """
        if self.last_right_rotation is None:
            return None
        r = float(self.last_right_rotation)
        if which == "low":
            self.angle_min = r
        elif which == "high":
            self.angle_max = r
        else:
            return None
        print(f"Calibrated rotation {which} = {math.degrees(r):.1f} deg "
              f"(range {math.degrees(self.angle_min):.1f} .. "
              f"{math.degrees(self.angle_max):.1f})")
        return math.degrees(r)

    def reset_rotation_range(self):
        """Restore the default full-span angle window (undo a calibration)."""
        self.angle_min = self._default_angle_min
        self.angle_max = self._default_angle_max
        self.invert_rotation = False
        self._prev_centered = None
        self._wrap_offset = 0.0
        print(f"Rotation range reset to default "
              f"({math.degrees(self.angle_min):.0f} .. "
              f"{math.degrees(self.angle_max):.0f} deg)")
