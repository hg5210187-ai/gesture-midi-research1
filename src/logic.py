import os
import time
import numpy as np

# Set HAND_DEBUG=1 to print every RAW pose detection's confidence/area/center
# (before any filtering), to diagnose phantom "hand" detections on the
# background/face/shadow. Console line:  [HAND] det conf=.. area=.. cx=..
#   Run:  HAND_DEBUG=1 uv run python main.py
_HAND_DEBUG = os.environ.get("HAND_DEBUG") == "1"


class GestureLogic:
    def __init__(self, midi_engine, looper=None):
        self.midi = midi_engine
        self.looper = looper

        # Right hand state
        self.current_pitch_note = -1

        # Scale mapping (default Chromatic A2)
        self.scale = [45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57]
        self.scale_type = "chromatic"
        self.base_theremin_note = 60 # C4

        # Left hand looper control
        self._prev_fist = False        # confirmed (sustained) fist state
        self._last_toggle_time = 0
        self._toggle_cooldown = 1.0    # seconds between toggles
        # Require the fist to be held for several consecutive frames before it
        # counts, so a single-frame phantom (the pose model briefly mis-reading
        # background/face as a fist) can't toggle the looper.
        self._fist_frames = 0
        self._fist_confirm_frames = 3

        # Throttle recording of continuous events
        self._last_recorded_volume = -1
        self._last_recorded_pitch = -99999

        # Experimental mode (None = use legacy behavior)
        self.experiment_sensitivity = None
        self.experiment_scale_type = None

        # Last computed values for logging
        self.last_pitch_cents = None
        self.last_hand_y_raw = None
        self.last_hand_y_norm = None

        # Bounding-box driven right-hand state
        self.area_min = None          # calibrated min bbox area -> volume 0
        self.volume_gain = 2.0        # sensitivity gain G (set from UI slider)
        # Volume gate (0-127): below this the note is fully released (note_off),
        # not just made quiet, so rapid in/out hand movements switch the sound
        # cleanly on and off instead of leaving a held note ringing.
        self.volume_gate = 8
        self.last_right_area = None   # latest right-hand bbox area (for calibrate)
        self.last_right_bbox = None   # latest right-hand bbox (for overlay)
        self.last_left_bbox = None    # latest left-hand bbox (for overlay)

        # Minimum detection confidence. The hand-pose model has a single
        # 'hand' class and YOLO's default conf (0.25) is permissive enough to
        # hallucinate a "hand" from a face/shoulder/background. Such a phantom
        # box on the right half would be treated as the right hand and play a
        # note with no hand actually up, so weak detections are dropped below.
        self.min_confidence = 0.7

        # Playable dead-zone at each vertical edge. Pitch is read from the box
        # CENTER, which can't reach the frame edges (it's always >= half a box
        # height away), so the first/last scale notes are otherwise unreachable.
        # The reachable middle band [margin, 1-margin] is stretched back to the
        # full [0, 1] range (see _stretch_reach); the overlay uses the same
        # margin so the borderlines stay aligned with the box-center pitch line.
        self.play_margin = 0.15

        # Overlay toggles (set from UI checkboxes; read by render_overlay).
        # Pitch feedback is on by default so the readout, sampling line, and
        # active-region highlight are visible without hunting for a checkbox.
        self.show_borderlines = False
        self.show_pitch = True

        # --- Stationary-phantom rejection ---
        # The pose model can fire at HIGH confidence (0.95-1.0) on a fixed
        # background feature -- a picture frame, a light switch -- so a
        # confidence floor alone can't drop it. But such a phantom is perfectly
        # still frame-to-frame, while a real hand always jitters/drifts. We
        # track each detection's box center across frames and suppress any that
        # stays within `_static_move_eps` of its own anchor for `_static_frames`
        # consecutive sightings. Any real movement past the epsilon instantly
        # resets it, so a moving (or even naturally trembling) hand is never
        # suppressed -- only genuinely motionless boxes are. Unlike retraining,
        # this is background-agnostic: it works in any room, on any wall.
        self.reject_stationary = True
        self._static_move_eps = 0.012      # center movement (norm) below this = "still"
        self._static_match_radius = 0.025  # associate a detection with a track within this
        self._static_frames = 30           # sightings-still before a box is suppressed
        self._static_max_missed = 5        # drop a track after this many unseen frames
        self._static_tracks = []           # [{'cx','cy','still','missed'}]

    def process(self, yolo_result):
        """Process YOLOv8-pose results.

        Hand identification uses bounding-box center-x. The RIGHT hand is
        driven entirely by its bounding box (pitch from the box Y-midpoint,
        volume from box area). The LEFT hand still uses keypoints for fist
        detection (looper control).
        """
        boxes = yolo_result.boxes
        kpts_obj = yolo_result.keypoints

        if boxes is None or len(boxes) == 0:
            self._reset_no_hands()
            return

        bboxes = boxes.xyxyn.cpu().numpy()   # [N, 4] normalized x1, y1, x2, y2
        kpts = (kpts_obj.xyn.cpu().numpy()
                if kpts_obj is not None and len(kpts_obj) > 0 else None)

        # Drop low-confidence (phantom) detections before any hand is assigned
        # a role, so a hallucinated box can't make sound when no hand is shown.
        confs = (boxes.conf.cpu().numpy()
                 if getattr(boxes, "conf", None) is not None else None)

        # Diagnostic: log every raw detection (pre-filter) so a phantom's
        # confidence/area/position can be read off and the threshold tuned.
        if _HAND_DEBUG:
            for _i in range(len(bboxes)):
                _x1, _y1, _x2, _y2 = bboxes[_i]
                _c = float(confs[_i]) if confs is not None else 1.0
                print(f"[HAND] det conf={_c:.2f} "
                      f"area={float((_x2 - _x1) * (_y2 - _y1)):.3f} "
                      f"cx={float((_x1 + _x2) / 2.0):.2f} "
                      f"cy={float((_y1 + _y2) / 2.0):.2f}")

        if confs is not None:
            keep = confs >= self.min_confidence
            bboxes = bboxes[keep]
            if kpts is not None:
                # Keypoints are index-aligned with boxes; filter identically.
                # On a length mismatch we can't align, so drop keypoints.
                kpts = kpts[keep] if len(kpts) == len(keep) else None
            if len(bboxes) == 0:
                self._reset_no_hands()
                return

        # Drop fixed-background phantoms: high-confidence boxes that never move
        # (a picture frame / light switch the pose model mistakes for a hand).
        keep = self._reject_stationary(bboxes)
        if not keep.all():
            bboxes = bboxes[keep]
            if kpts is not None:
                kpts = kpts[keep] if len(kpts) == len(keep) else None
            if len(bboxes) == 0:
                self._reset_no_hands()
                return

        # Boxes and keypoints are index-aligned; guard counts defensively.
        n = len(bboxes) if kpts is None else min(len(bboxes), len(kpts))

        hands = []
        for i in range(n):
            x1, _, x2, _ = bboxes[i]
            hands.append({'cx': float((x1 + x2) / 2.0), 'bbox': bboxes[i], 'idx': i})
        hands.sort(key=lambda h: h['cx'])

        left_hand = None
        right_hand = None
        if len(hands) == 1:
            if hands[0]['cx'] >= 0.5:
                right_hand = hands[0]
            else:
                left_hand = hands[0]
        elif len(hands) >= 2:
            left_hand = hands[0]
            right_hand = hands[-1]

        # Right hand -> pitch + volume from bounding box only
        if right_hand is not None:
            self.last_right_bbox = right_hand['bbox']
            self._process_right_hand(right_hand['bbox'])
        else:
            self.last_right_bbox = None
            self.last_right_area = None
            self.last_hand_y_norm = None
            self.last_hand_y_raw = None
            self._silence_right_hand()

        # Left hand -> looper control (still uses keypoints for fist detection)
        if left_hand is not None:
            self.last_left_bbox = left_hand['bbox']
            if self.looper is not None and kpts is not None:
                self._process_left_hand(kpts[left_hand['idx']])
        else:
            self.last_left_bbox = None
            self._fist_frames = 0
            self._prev_fist = False

    def _reject_stationary(self, bboxes):
        """Return a boolean keep-mask that drops boxes held still for many
        frames (fixed-background phantoms). Real hands always move a little.

        Each detection center is matched to the nearest existing track within
        ``_static_match_radius``. A matched box that moved <= ``_static_move_eps``
        from its track anchor increments that track's ``still`` counter (the
        anchor is NOT updated, so slow drift eventually exceeds the epsilon and
        resets it); a larger move resets ``still`` and re-anchors. A track whose
        ``still`` count reaches ``_static_frames`` is a phantom -> its box is
        suppressed. Unmatched detections start fresh tracks (never suppressed on
        first sight); tracks unseen for ``_static_max_missed`` frames are pruned.
        """
        n = len(bboxes)
        keep = np.ones(n, dtype=bool)

        if not self.reject_stationary:
            return keep

        centers = [((float(b[0]) + float(b[2])) / 2.0,
                    (float(b[1]) + float(b[3])) / 2.0) for b in bboxes]

        matched = set()
        for i, (cx, cy) in enumerate(centers):
            # Nearest not-yet-claimed track within the match radius.
            best, best_d = None, self._static_match_radius
            for t in self._static_tracks:
                if id(t) in matched:
                    continue
                d = ((t['cx'] - cx) ** 2 + (t['cy'] - cy) ** 2) ** 0.5
                if d <= best_d:
                    best, best_d = t, d

            if best is None:
                # A newly-seen object: fresh track, never suppressed yet.
                self._static_tracks.append({'cx': cx, 'cy': cy,
                                            'still': 0, 'missed': 0})
                continue

            matched.add(id(best))
            best['missed'] = 0
            move = ((best['cx'] - cx) ** 2 + (best['cy'] - cy) ** 2) ** 0.5
            if move <= self._static_move_eps:
                best['still'] += 1               # kept its anchor -> still
            else:
                best['still'] = 0                # moved -> re-anchor
                best['cx'], best['cy'] = cx, cy

            if best['still'] >= self._static_frames:
                keep[i] = False
                if _HAND_DEBUG:
                    print(f"[HAND] suppress stationary box "
                          f"cx={cx:.2f} cy={cy:.2f} "
                          f"(still {best['still']} sightings)")

        # Age out tracks not seen this frame; prune the long-absent ones.
        for t in self._static_tracks:
            if id(t) not in matched:
                t['missed'] += 1
        self._static_tracks = [t for t in self._static_tracks
                               if t['missed'] <= self._static_max_missed]

        return keep

    def _reset_no_hands(self):
        """Clear per-frame hand state and silence the right hand.

        Used when no hands are detected, or when every detection was filtered
        out as a low-confidence phantom.
        """
        self._silence_right_hand()
        self.last_right_bbox = None
        self.last_left_bbox = None
        self.last_right_area = None
        self.last_hand_y_norm = None
        self.last_hand_y_raw = None
        self._fist_frames = 0
        self._prev_fist = False

    def _silence_right_hand(self):
        # current_pitch_note tracks whatever is sounding, including the theremin
        # base note (map_pitch returns it in theremin mode), so this single
        # guarded release covers every mode without a redundant extra note_off.
        if self.current_pitch_note != -1:
            self.midi.send_note_off(self.current_pitch_note)
            if self.looper:
                self.looper.record_event('note_off', note=self.current_pitch_note)
            self.current_pitch_note = -1

    # ---- pitch mapping ---------------------------------------------------

    def _stretch_reach(self, y):
        """Stretch the reachable middle band to the full [0, 1] range.

        Pitch is read from the bounding-box *center*, which can never reach the
        top/bottom edge of the frame, so the first and last scale notes would
        otherwise be unreachable. ``play_margin`` is the dead-zone fraction at
        each end; everything in between is remapped to [0, 1] (and clamped) so
        moving the hand to the top/bottom plays the extreme notes. The overlay
        uses the same margin so its borderlines stay aligned with the pitch.
        """
        m = self.play_margin
        span = 1.0 - 2.0 * m
        if span <= 0:
            return y
        return min(1.0, max(0.0, (y - m) / span))

    def map_pitch(self, hand_y, sensitivity=None, scale_type=None):
        """Map normalised hand Y (0 = bottom, 1 = top) to a MIDI note.

        Parameters
        ----------
        hand_y : float
            Vertical position in [0, 1], already flipped so up = higher.
        sensitivity : str or None
            Key into ``SENSITIVITY_LEVELS`` (``"low"``/``"medium"``/``"high"``).
            *None* keeps the legacy linear mapping.
        scale_type : str or None
            Key into experiment ``SCALE_TYPES``
            (``"chromatic"``/``"diatonic"``/``"pentatonic"``/``"continuous"``).
            *None* keeps the legacy behaviour driven by ``self.scale`` /
            ``self.scale_type``.

        Returns
        -------
        (midi_note, pitch_cents) : tuple[int, float]
            *midi_note* – quantised MIDI note number (0-127).
            *pitch_cents* – exact pitch in cents before snapping
            (e.g. 6000.0 = C4).
        """

        # --- Legacy path (matches original behaviour exactly) -------------
        if sensitivity is None and scale_type is None:
            if self.scale_type == "theremin":
                pitch_bend = int(hand_y * 16383) - 8192
                # ±2 semitones is the standard pitch-bend range
                bend_semitones = (pitch_bend / 8192) * 2
                pitch_cents = (self.base_theremin_note + bend_semitones) * 100
                return (self.base_theremin_note, pitch_cents)

            note_index = int(hand_y * len(self.scale))
            note_index = max(0, min(len(self.scale) - 1, note_index))
            midi_note = self.scale[note_index]
            return (midi_note, midi_note * 100.0)

        # --- Experimental path (cm_per_octave mapping) --------------------
        from experiment_config import (
            SENSITIVITY_LEVELS, SCALE_TYPES,
            BASE_OCTAVE, PITCH_RANGE_OCTAVES, ASSUMED_FRAME_HEIGHT_CM,
        )

        cm_per_oct = SENSITIVITY_LEVELS[sensitivity]["cm_per_octave"]
        scale_cfg = SCALE_TYPES[scale_type]

        # Y offset from frame centre → centimetres → octaves
        delta_cm = (hand_y - 0.5) * ASSUMED_FRAME_HEIGHT_CM
        delta_octaves = delta_cm / cm_per_oct

        center_midi = (BASE_OCTAVE + 1) * 12          # C4 = 60
        exact_semitones = center_midi + delta_octaves * 12

        # Clamp to the configured pitch range
        half_range = PITCH_RANGE_OCTAVES * 6           # in semitones
        exact_semitones = max(center_midi - half_range,
                              min(center_midi + half_range, exact_semitones))

        pitch_cents = exact_semitones * 100.0

        if scale_cfg["notes"] is None:
            # Continuous – nearest integer MIDI note (pitch bend handles rest)
            midi_note = int(round(exact_semitones))
        else:
            midi_note = self._snap_to_scale(exact_semitones, scale_cfg["notes"])

        midi_note = max(0, min(127, midi_note))
        return (midi_note, pitch_cents)

    @staticmethod
    def _snap_to_scale(exact_semitones, scale_degrees):
        """Return the MIDI note in *scale_degrees* nearest to *exact_semitones*.

        *scale_degrees* is a list of semitone offsets within one octave
        (e.g. ``[0, 2, 4, 5, 7, 9, 11]`` for diatonic).
        """
        center = int(round(exact_semitones))
        scale_set = set(scale_degrees)
        best_note = center
        best_dist = float("inf")

        for candidate in range(max(0, center - 12), min(128, center + 13)):
            if candidate % 12 in scale_set:
                dist = abs(exact_semitones - candidate)
                if dist < best_dist:
                    best_dist = dist
                    best_note = candidate

        return best_note

    # ---- left hand / looper -----------------------------------------------

    def _process_left_hand(self, landmarks):
        """Toggle the looper on a SUSTAINED fist.

        The fist must be held for ``_fist_confirm_frames`` consecutive frames
        before it counts, so a single-frame phantom (the pose model briefly
        mis-reading background/face as a fist) can't toggle the looper.
        """
        fist = self._is_fist(landmarks)
        self._fist_frames = self._fist_frames + 1 if fist else 0
        confirmed = self._fist_frames >= self._fist_confirm_frames
        now = time.time()

        # Rising edge of a CONFIRMED fist, with cooldown.
        if confirmed and not self._prev_fist and (now - self._last_toggle_time) > self._toggle_cooldown:
            state = self.looper.toggle()
            self._last_toggle_time = now
            # Reset recording throttles when starting a new recording
            if state == "RECORDING":
                self._last_recorded_volume = -1
                self._last_recorded_pitch = -99999
            print(f"Looper: {state}")

        self._prev_fist = confirmed

    def _is_fist(self, landmarks):
        """Detect fist: fingertips close to palm center relative to hand size."""
        wrist = landmarks[0]
        middle_knuckle = landmarks[9]
        hand_size = max(0.01, np.linalg.norm(wrist - middle_knuckle))

        palm_center = (wrist + middle_knuckle) / 2
        fingertips = [landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
        avg_dist = np.mean([np.linalg.norm(tip - palm_center) for tip in fingertips])

        return (avg_dist / hand_size) < 0.65

    def _process_right_hand(self, bbox):
        x1, y1, x2, y2 = bbox

        # Pitch: midpoint of the bounding box's Y-axis.
        # Note: 1.0 - y because image origin is top-left (so up = higher).
        y_mid = (y1 + y2) / 2.0
        y = 1.0 - y_mid

        # Compute pitch via map_pitch (works for both legacy and experimental)
        sens = self.experiment_sensitivity
        stype = self.experiment_scale_type
        use_experiment = sens is not None or stype is not None

        # In the live (legacy) path, stretch the reachable middle band so the
        # top/bottom notes are reachable. The experiment/benchmark path keeps
        # the raw position (its own cm-per-octave mapping assumes 0.5 = centre).
        if not use_experiment:
            y = self._stretch_reach(y)

        midi_note, pitch_cents = self.map_pitch(y, sensitivity=sens, scale_type=stype)

        # Load-bearing for run_benchmark.py (reads last_hand_y_norm) + overlays.
        self.last_pitch_cents = pitch_cents
        self.last_hand_y_raw = float(y_mid)
        self.last_hand_y_norm = float(y)

        # Volume: bounding-box AREA relative to a calibrated minimum.
        # vol = clip((area / area_min - 1) * gain, 0, 1) * 127
        area = float((x2 - x1) * (y2 - y1))
        self.last_right_area = area

        # Seed the minimum from the first frame so volume is always defined
        # (and we never divide by zero). The Calibrate Min button overrides it.
        if self.area_min is None or self.area_min <= 0:
            self.area_min = area

        vol_norm = np.clip((area / self.area_min - 1.0) * self.volume_gain, 0.0, 1.0)
        volume = int(vol_norm * 127)

        # Gate state: below the gate no note sounds (see the gating block below).
        # Computed up front so the pitch bend can be suppressed while closed.
        gate_open = volume > self.volume_gate

        # --- Pitch bend (continuous modes only, and only while a note sounds) ---
        # A pitch bend is meaningless with nothing playing, so it is suppressed
        # (and not recorded into the loop) while the gate is closed; otherwise a
        # hand sweep with no note would record stray bends that corrupt any loop.
        if gate_open and use_experiment and (stype or "chromatic") == "continuous":
            # Experimental continuous: sub-note pitch bend for smooth glissando
            cents_offset = pitch_cents - (midi_note * 100)
            pitch_bend = int((cents_offset / 200) * 8192)
            pitch_bend = max(-8192, min(8191, pitch_bend))
            self.midi.send_pitch_bend(pitch_bend)
            if self.looper and abs(pitch_bend - self._last_recorded_pitch) >= 200:
                self.looper.record_event('pitch_bend', pitch=pitch_bend)
                self._last_recorded_pitch = pitch_bend
        elif gate_open and not use_experiment and self.scale_type == "theremin":
            # Legacy theremin: full-range pitch bend (original behaviour)
            pitch_bend = int(y * 16383) - 8192
            self.midi.send_pitch_bend(pitch_bend)
            if self.looper and abs(pitch_bend - self._last_recorded_pitch) >= 200:
                self.looper.record_event('pitch_bend', pitch=pitch_bend)
                self._last_recorded_pitch = pitch_bend

        # --- Note gating ---
        # Below the gate the note is released entirely (not just made quiet),
        # so shrinking the hand cuts the sound instantly and fast in/out
        # movements switch notes on and off cleanly instead of blurring.
        if not gate_open:
            volume = 0
            if self.current_pitch_note != -1:
                self.midi.send_note_off(self.current_pitch_note)
                if self.looper:
                    self.looper.record_event('note_off', note=self.current_pitch_note)
                self.current_pitch_note = -1
            # Re-arm the bend throttle so the first bend after the gate reopens
            # is recorded cleanly rather than being swallowed as "no change".
            self._last_recorded_pitch = -99999
        elif midi_note != self.current_pitch_note:
            # Gate open: (re)trigger the note when the pitch changes.
            if self.current_pitch_note != -1:
                self.midi.send_note_off(self.current_pitch_note)
                if self.looper:
                    self.looper.record_event('note_off', note=self.current_pitch_note)
            self.midi.send_note_on(midi_note)
            if self.looper:
                self.looper.record_event('note_on', note=midi_note)
            self.current_pitch_note = midi_note

        self.midi.send_cc(7, volume)
        if self.looper and abs(volume - self._last_recorded_volume) >= 5:
            self.looper.record_event('cc', control=7, value=volume)
            self._last_recorded_volume = volume

    def calibrate_min(self):
        """Capture the current right-hand bbox area as the volume-0 reference.

        Returns the captured area, or None if no right hand is currently
        visible (in which case nothing changes).
        """
        if self.last_right_area is not None and self.last_right_area > 0:
            self.area_min = self.last_right_area
            print(f"Calibrated area_min = {self.area_min:.5f}")
            return self.area_min
        return None
