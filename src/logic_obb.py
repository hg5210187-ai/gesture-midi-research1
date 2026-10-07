import os
import time
import numpy as np

# Set OBB_DEBUG=1 in the environment to print each right-hand OBB detection's
# confidence / area / center, so the phantom-rejection gates below can be tuned
# to your room.  Run:  OBB_DEBUG=1 uv run python main_obb.py
_OBB_DEBUG = os.environ.get("OBB_DEBUG") == "1"


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
        self._toggle_cooldown = 1.0  # seconds between toggles

        # Require a sustained fist (held several consecutive frames) before it
        # counts, so a single-frame phantom fist can't toggle the looper.
        self._fist_frames = 0
        self._fist_confirm_frames = 5   # ~150ms at the 30fps vision loop

        # Fist-release debounce. The pose model occasionally drops the left hand
        # for a frame or two (a held fist is a hard pose to keep above
        # min_confidence). Releasing the fist state on a single dropout would
        # read the re-detection as a fresh grip and fire a spurious looper
        # toggle, so the fist is only considered released after the left hand
        # has been absent for several consecutive frames.
        self._left_absent_frames = 0
        self._fist_release_frames = 5   # ~150ms at the 30fps vision loop

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
        self.last_right_area = None   # latest right-hand box area (for calibrate)
        self.last_right_bbox = None   # latest right-hand axis-aligned bbox (overlay)
        self.last_right_obb = None    # latest right-hand OBB corners [4,2] norm (overlay)
        self.last_right_rotation = None  # latest OBB rotation, radians (future tilt param)
        self.last_left_bbox = None    # latest left-hand bbox (for overlay)

        # --- Phantom-rejection gates for the right-hand OBB model ---
        # The OBB model can fire on textured background (curtains/blinds/shelf),
        # playing a note with no hand up. Three cheap gates reject those:
        #   * min_confidence - drop weak detections. Kept only modestly high
        #     because the model under-detects real hands, so a high value would
        #     also hurt true detections.
        #   * max_box_area   - a real hand never fills this fraction of the
        #     frame at playing distance (calibrated min area is ~0.04), so a
        #     huge box is background, not a hand. Set to None to disable.
        #   * the cx >= 0.5 (right-half) check in _extract_right_obb.
        # Tune these from the OBB_DEBUG output for your room/lighting.
        self.min_confidence = 0.6
        self.max_box_area = 0.30

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

    def process(self, results):
        """Process the two-model results for the current frame.

        ``results`` is a ``(pose_result, obb_result)`` pair from
        ``UltralyticsVision.get_frame``:

          * **right hand** comes from ``obb_result`` (the self-trained YOLO-OBB
            model): pitch from the box Y-center, volume from the box AREA
            (``w*h``, rotation-invariant), and the OBB rotation angle is stored
            for a future tilt parameter.
          * **left hand** comes from ``pose_result`` (the YOLOv8 hand-pose
            model): its keypoints drive fist detection for looper control.

        Left/right roles still follow the bounding-box center-x convention (the
        right hand sits on the right half of the mirrored frame).

        A bare single result is the one-model path (``HbbVision``): a gesture
        detector whose plain boxes carry both hands, with the left hand's
        ``closedhand`` class standing in for the keypoint fist test.
        """
        if isinstance(results, (tuple, list)):
            pose_result, obb_result = (list(results) + [None, None])[:2]
            right = self._extract_right_obb(obb_result)
            left = self._extract_left_pose(pose_result)
        else:
            right, left = self._extract_gesture_boxes(results)

        # ---- Right hand: pitch from the box Y-center, volume from its area ----
        if right is not None:
            self.last_right_bbox = right["bbox"]
            self.last_right_obb = right["poly"]
            self.last_right_rotation = right["rotation"]
            self._process_right_hand(right["cy"], right["area"])
        else:
            self.last_right_bbox = None
            self.last_right_obb = None
            self.last_right_rotation = None
            self.last_right_area = None
            self.last_hand_y_norm = None
            self.last_hand_y_raw = None
            self._silence_right_hand()

        # ---- Left hand: fist (keypoints or gesture class) -> looper control ----
        if left is not None:
            self._left_absent_frames = 0
            self.last_left_bbox = left["bbox"]
            if self.looper is not None:
                if left.get("fist") is not None:
                    self._process_left_hand(left["fist"])
                elif left["kpts"] is not None:
                    self._process_left_hand(self._is_fist(left["kpts"]))
        else:
            self.last_left_bbox = None
            self._fist_frames = 0
            # Debounce: only release the held-fist state after sustained
            # absence, so a transient detection dropout doesn't re-trigger.
            self._left_absent_frames += 1
            if self._left_absent_frames >= self._fist_release_frames:
                self._prev_fist = False

    def _extract_right_obb(self, obb_result):
        """Return the right-hand OBB as a dict, or ``None``.

        The OBB model is right-hand-only, so any detection above
        ``min_confidence`` is a right-hand candidate; we keep the **rightmost**
        one (largest center-x) and require it to sit on the right half of the
        mirrored frame (``cx >= 0.5``), mirroring the ``_extract_left_pose``
        guard so a left-half detection (e.g. the model firing on the left hand)
        can't assume the right-hand role and play pitch. Keys:
          ``cx``/``cy`` - normalized box center,
          ``area``      - normalized ``w*h`` (rotation-invariant, for volume),
          ``rotation``  - OBB angle in radians (stored for a future tilt param),
          ``bbox``      - axis-aligned ``[x1,y1,x2,y2]`` normalized (for overlay
                          + the mid-Y pitch guide; the AABB shares the OBB's
                          center, so the pitch line stays aligned),
          ``poly``      - ``[4,2]`` normalized corners (for the rotated overlay).
        """
        if obb_result is None:
            return None
        obb = getattr(obb_result, "obb", None)
        if obb is None or len(obb) == 0:
            return None

        h, w = obb_result.orig_shape            # (height, width) in pixels
        xywhr = obb.xywhr.cpu().numpy()         # [N,5] px: cx, cy, w, h, r(rad)
        polys = obb.xyxyxyxyn.cpu().numpy()     # [N,4,2] normalized corners
        confs = (obb.conf.cpu().numpy()
                 if getattr(obb, "conf", None) is not None else None)

        best = None
        for i in range(len(xywhr)):
            conf = float(confs[i]) if confs is not None else 1.0
            cx_px, cy_px, bw_px, bh_px, r = xywhr[i]
            cx = float(cx_px / w)
            cy = float(cy_px / h)
            area = float((bw_px * bh_px) / (w * h))
            if _OBB_DEBUG:
                print(f"[OBB] det conf={conf:.2f} area={area:.3f} "
                      f"cx={cx:.2f} cy={cy:.2f}")

            # Phantom rejection: weak confidence, a left-half box (not the right
            # hand), or a box too large to be a real hand (background texture).
            if conf < self.min_confidence:
                continue
            if cx < 0.5:
                continue
            if self.max_box_area is not None and area > self.max_box_area:
                continue

            poly = polys[i]                      # [4,2] normalized
            x1 = float(poly[:, 0].min()); x2 = float(poly[:, 0].max())
            y1 = float(poly[:, 1].min()); y2 = float(poly[:, 1].max())
            cand = {
                "cx": cx,
                "cy": cy,
                "area": area,
                "rotation": float(r),
                "bbox": np.array([x1, y1, x2, y2], dtype=float),
                "poly": poly,
            }
            if best is None or cand["cx"] > best["cx"]:
                best = cand

        return best

    def _extract_gesture_boxes(self, result):
        """Split a gesture-detector result into ``(right, left)`` dicts.

        The detector sees both hands as plain boxes with a gesture class. After
        the ``min_confidence`` gate, the **rightmost** box on the right half is
        the right hand (same keys as ``_extract_right_obb``, with no ``poly`` /
        ``rotation``; ``max_box_area`` rejects background-sized boxes) and the
        **leftmost** box on the left half is the left hand (``bbox`` plus
        ``fist``: whether its class is ``closedhand``). Either may be ``None``.
        """
        boxes = getattr(result, "boxes", None) if result is not None else None
        if boxes is None or len(boxes) == 0:
            return None, None

        bboxes = boxes.xyxyn.cpu().numpy()       # [N,4] normalized x1,y1,x2,y2
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        names = result.names

        right = left = None
        for i in range(len(bboxes)):
            x1, y1, x2, y2 = (float(v) for v in bboxes[i])
            cx = (x1 + x2) / 2.0
            area = (x2 - x1) * (y2 - y1)
            if _OBB_DEBUG:
                print(f"[HBB] det {names[classes[i]]} conf={confs[i]:.2f} "
                      f"area={area:.3f} cx={cx:.2f} cy={(y1 + y2) / 2.0:.2f}")
            if confs[i] < self.min_confidence:
                continue

            if cx >= 0.5:
                if self.max_box_area is not None and area > self.max_box_area:
                    continue
                if right is None or cx > right["cx"]:
                    right = {
                        "cx": cx,
                        "cy": (y1 + y2) / 2.0,
                        "area": area,
                        "rotation": None,
                        "bbox": np.array([x1, y1, x2, y2], dtype=float),
                        "poly": None,
                    }
            elif left is None or cx < left["cx"]:
                left = {
                    "cx": cx,
                    "bbox": bboxes[i],
                    "kpts": None,
                    "fist": names[classes[i]] == "closedhand",
                }

        return right, left

    def _extract_left_pose(self, pose_result):
        """Return the left-hand pose detection as a dict, or ``None``.

        The pose model detects both hands; the LEFT hand is the **leftmost**
        detection above ``min_confidence`` that sits on the left half of the
        mirrored frame (the right half belongs to the OBB model). The
        confidence gate drops phantom boxes (a hallucinated "hand" from a
        face/shoulder) before a role is assigned. Keys: ``bbox`` (normalized
        ``[x1,y1,x2,y2]``) and ``kpts`` (normalized keypoints, or ``None``).
        """
        if pose_result is None:
            return None
        boxes = getattr(pose_result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return None

        bboxes = boxes.xyxyn.cpu().numpy()       # [N,4] normalized x1,y1,x2,y2
        kpts_obj = getattr(pose_result, "keypoints", None)
        kpts = (kpts_obj.xyn.cpu().numpy()
                if kpts_obj is not None and len(kpts_obj) > 0 else None)

        confs = (boxes.conf.cpu().numpy()
                 if getattr(boxes, "conf", None) is not None else None)
        if confs is not None:
            keep = confs >= self.min_confidence
            bboxes = bboxes[keep]
            if kpts is not None:
                # Keypoints are index-aligned with boxes; filter identically.
                # On a length mismatch we can't align, so drop keypoints.
                kpts = kpts[keep] if len(kpts) == len(keep) else None
            if len(bboxes) == 0:
                return None

        # Boxes and keypoints are index-aligned; guard counts defensively.
        n = len(bboxes) if kpts is None else min(len(bboxes), len(kpts))
        if n == 0:
            return None

        # Leftmost detection is the left-hand candidate.
        best_i, best_cx = 0, 2.0
        for i in range(n):
            x1, _, x2, _ = bboxes[i]
            cx = float((x1 + x2) / 2.0)
            if cx < best_cx:
                best_cx, best_i = cx, i

        # Only treat it as the left hand if it's on the left half; otherwise it
        # is the right hand (driven by the OBB model) and there is no left hand.
        if best_cx >= 0.5:
            return None
        return {
            "bbox": bboxes[best_i],
            "kpts": kpts[best_i] if kpts is not None else None,
        }

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

    def _process_left_hand(self, fist):
        """Toggle the looper on a SUSTAINED fist.

        The fist must be held for ``_fist_confirm_frames`` consecutive frames
        before it counts, so a single-frame phantom (the pose model briefly
        mis-reading background/face as a fist) can't toggle the looper.
        """
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

    def _process_right_hand(self, y_center, area):
        """Drive pitch + volume from the right-hand OBB.

        ``y_center`` is the box center Y in image space (0 = top, 1 = bottom);
        ``area`` is the box area normalized to the frame (``w*h``, which is
        rotation-invariant, so tilting the hand no longer changes the volume).
        """
        # Pitch: box Y-center.
        # Note: 1.0 - y because image origin is top-left (so up = higher).
        y_mid = y_center
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

        # Volume: box AREA relative to a calibrated minimum.
        # vol = clip((area / area_min - 1) * gain, 0, 1) * 127
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
