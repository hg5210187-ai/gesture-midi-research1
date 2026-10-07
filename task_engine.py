import math
import random
import time

from experiment_config import (
    SENSITIVITY_LEVELS, SCALE_TYPES,
    BASE_OCTAVE, PITCH_RANGE_OCTAVES, ASSUMED_FRAME_HEIGHT_CM,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

_TARGET_TRIALS = 10
_SUSTAIN_TRIALS = 5
_TARGET_HOLD_SEC = 0.5
_SUSTAIN_HOLD_SEC = 3.0
_MELODY_HOLD_SEC = 0.3
_ON_TARGET_CENTS = 50

# Scale-degree offsets from the centre of the playable range.
# Each melody is a 4-note sequence; offset 0 = centre note, +1 = one scale
# step up, -2 = two scale steps down, etc.
_MELODIES = [
    [0, 2, 4, 2],
    [4, 2, 0, -2],
    [-2, 0, 2, 4],
]


# ---------------------------------------------------------------------------
# TaskEngine
# ---------------------------------------------------------------------------

class TaskEngine:
    """Manages the three experimental task types (target, sustain, melody).

    Instantiate once per experimental condition, call ``start_task`` to begin
    a block of trials, then call ``update`` every frame.
    """

    def __init__(self, sensitivity, scale_type, logger, trial_counts=None):
        self._sensitivity = sensitivity
        self._scale_type = scale_type
        self._logger = logger
        self._trial_count_overrides = trial_counts or {}

        self._cm_per_oct = SENSITIVITY_LEVELS[sensitivity]["cm_per_octave"]
        scale_cfg = SCALE_TYPES[scale_type]
        self._scale_degrees = scale_cfg["notes"]          # None for continuous

        self._scale_notes = self._build_scale_notes()

        # --- shared state ---
        self._task_type = None
        self._total_trials = 0
        self._trial_index = 0
        self._trial_id = 0
        self._complete = False

        # per-trial
        self._target_note = None
        self._last_target_note = None
        self._on_target_start = None
        self._first_reach_logged = False

        # sustain-specific
        self._sustain_phase = None                        # "reach" | "hold"
        self._hold_start = None
        self._hold_pitches = []

        # melody-specific
        self._melody_notes = []
        self._melody_note_idx = 0
        self._melody_start = None
        self._note_errors = []

    # ---- public API -------------------------------------------------------

    def start_task(self, task_type):
        """Begin a block of trials for *task_type* (``"target"`` / ``"sustain"`` / ``"melody"``)."""
        self._task_type = task_type
        self._trial_index = 0
        self._trial_id = 0
        self._complete = False

        if task_type == "target":
            self._total_trials = self._trial_count_overrides.get("target", _TARGET_TRIALS)
        elif task_type == "sustain":
            self._total_trials = self._trial_count_overrides.get("sustain", _SUSTAIN_TRIALS)
        elif task_type == "melody":
            self._total_trials = self._trial_count_overrides.get("melody", len(_MELODIES))

        self._begin_trial()

    def update(self, hand_y_normalized):
        """Process one frame.  Returns a display-info dict (see module docstring)."""
        if self._complete:
            return self._result(0.0, 0.0, True, 1.0, "完了!")

        midi_note, pitch_cents = self._y_to_pitch(hand_y_normalized)
        target_cents = self._target_note * 100.0
        error_cents = pitch_cents - target_cents           # +ve = too high
        on_target = abs(error_cents) <= _ON_TARGET_CENTS

        # Log every frame
        hand_y_raw = 1.0 - hand_y_normalized
        self._logger.log_frame(
            self._trial_id, self._task_type,
            self._target_note, midi_note,
            hand_y_raw, hand_y_normalized, on_target,
        )

        if self._task_type == "target":
            return self._update_target(pitch_cents, error_cents, on_target)
        if self._task_type == "sustain":
            return self._update_sustain(pitch_cents, error_cents, on_target)
        return self._update_melody(pitch_cents, error_cents, on_target)

    def is_complete(self):
        return self._complete

    # ---- per-task update ---------------------------------------------------

    def _update_target(self, pitch_cents, error_cents, on_target):
        now = time.time()

        if on_target:
            if self._on_target_start is None:
                self._on_target_start = now
            if not self._first_reach_logged:
                self._logger.mark_on_target(self._trial_id)
                self._first_reach_logged = True
            held = now - self._on_target_start
        else:
            self._on_target_start = None
            held = 0.0

        progress = min(1.0, held / _TARGET_HOLD_SEC)

        if held >= _TARGET_HOLD_SEC:
            self._advance_trial()
            instr = "全試行完了！" if self._complete else "正解！次へ"
            return self._result(pitch_cents, 0.0, True, 1.0, instr)

        return self._result(pitch_cents, error_cents, on_target, progress,
                            self._direction_hint(error_cents, on_target))

    def _update_sustain(self, pitch_cents, error_cents, on_target):
        now = time.time()

        # -- reach phase: get to the target note ----------------------------
        if self._sustain_phase == "reach":
            if on_target:
                if self._on_target_start is None:
                    self._on_target_start = now
                if not self._first_reach_logged:
                    self._logger.mark_on_target(self._trial_id)
                    self._first_reach_logged = True
                self._sustain_phase = "hold"
                self._hold_start = now
                self._hold_pitches = [pitch_cents]
                return self._result(pitch_cents, error_cents, True, 0.0,
                                    "キープしてください")

            self._on_target_start = None
            return self._result(pitch_cents, error_cents, False, 0.0,
                                self._direction_hint(error_cents, False))

        # -- hold phase: stay on target for 3 s -----------------------------
        if not on_target:
            self._sustain_phase = "reach"
            self._hold_start = None
            self._hold_pitches = []
            self._on_target_start = None
            return self._result(pitch_cents, error_cents, False, 0.0,
                                "外れました！もう一度")

        self._hold_pitches.append(pitch_cents)
        held = now - self._hold_start
        progress = min(1.0, held / _SUSTAIN_HOLD_SEC)

        if held >= _SUSTAIN_HOLD_SEC:
            self._print_sustain_stats()
            self._advance_trial()
            instr = "全試行完了！" if self._complete else "素晴らしい！次へ"
            return self._result(pitch_cents, error_cents, True, 1.0, instr)

        return self._result(pitch_cents, error_cents, True, progress,
                            "そのまま！")

    def _update_melody(self, pitch_cents, error_cents, on_target):
        now = time.time()

        if on_target:
            if self._on_target_start is None:
                self._on_target_start = now
            if not self._first_reach_logged:
                self._logger.mark_on_target(self._trial_id)
                self._first_reach_logged = True

            self._note_errors.append(error_cents)
            held = now - self._on_target_start

            if held >= _MELODY_HOLD_SEC:
                avg_err = (sum(abs(e) for e in self._note_errors)
                           / max(1, len(self._note_errors)))
                print(f"  Melody note {self._melody_note_idx}: "
                      f"avg |error| = {avg_err:.1f} cents")

                self._melody_note_idx += 1
                self._note_errors = []
                self._on_target_start = None
                self._first_reach_logged = False

                if self._melody_note_idx >= len(self._melody_notes):
                    elapsed = now - self._melody_start
                    print(f"  Melody {self._trial_index} complete in {elapsed:.2f}s")
                    self._advance_trial()
                    instr = "全曲完了！" if self._complete else "次のメロディへ"
                    return self._result(pitch_cents, 0.0, True, 1.0, instr)

                # Advance to next note within the melody
                self._target_note = self._melody_notes[self._melody_note_idx]
                self._trial_id += 1
                self._logger.mark_trial_start(self._trial_id, self._target_note)

                new_error = pitch_cents - self._target_note * 100.0
                new_on = abs(new_error) <= _ON_TARGET_CENTS
                name = self._note_name(self._target_note)
                return self._result(pitch_cents, new_error, new_on,
                                    self._melody_progress(),
                                    f"次の音: {name}")
        else:
            self._on_target_start = None

        progress = self._melody_progress()
        if self._on_target_start is not None:
            partial = min(1.0, (now - self._on_target_start) / _MELODY_HOLD_SEC)
            progress += partial / len(self._melody_notes)

        return self._result(pitch_cents, error_cents, on_target, progress,
                            self._direction_hint(error_cents, on_target))

    # ---- trial lifecycle ---------------------------------------------------

    def _pick_note(self):
        """Pick a random scale note, avoiding the previous target."""
        pool = [n for n in self._scale_notes if n != self._last_target_note]
        if not pool:
            pool = self._scale_notes
        note = random.choice(pool)
        self._last_target_note = note
        return note

    def _begin_trial(self):
        self._on_target_start = None
        self._first_reach_logged = False
        self._trial_id += 1

        if self._task_type == "target":
            self._target_note = self._pick_note()
            self._logger.mark_trial_start(self._trial_id, self._target_note)

        elif self._task_type == "sustain":
            self._target_note = self._pick_note()
            self._sustain_phase = "reach"
            self._hold_start = None
            self._hold_pitches = []
            self._logger.mark_trial_start(self._trial_id, self._target_note)

        elif self._task_type == "melody":
            offsets = _MELODIES[self._trial_index]
            center = len(self._scale_notes) // 2
            self._melody_notes = [
                self._scale_notes[
                    max(0, min(len(self._scale_notes) - 1, center + o))
                ]
                for o in offsets
            ]
            # Avoid starting on the same note as the previous trial's target
            if (self._melody_notes[0] == self._last_target_note
                    and len(self._scale_notes) > 1):
                # Shift melody center by +1 scale step
                self._melody_notes = [
                    self._scale_notes[
                        max(0, min(len(self._scale_notes) - 1, center + 1 + o))
                    ]
                    for o in offsets
                ]
            self._melody_note_idx = 0
            self._target_note = self._melody_notes[0]
            self._last_target_note = self._target_note
            self._melody_start = time.time()
            self._note_errors = []
            self._logger.mark_trial_start(self._trial_id, self._target_note)

    def _advance_trial(self):
        self._trial_index += 1
        if self._trial_index >= self._total_trials:
            self._complete = True
            return
        self._begin_trial()

    # ---- result / display helpers ------------------------------------------

    def _result(self, pitch_cents, error_cents, on_target, progress, instruction):
        current_semitones = pitch_cents / 100.0 if pitch_cents else 0.0
        r = {
            "task_type": self._task_type,
            "target_note": self._target_note,
            "target_note_name": (self._note_name(self._target_note)
                                 if self._target_note is not None else ""),
            "current_pitch": round(current_semitones, 2),
            "error_cents": round(error_cents, 1),
            "on_target": on_target,
            "progress": round(max(0.0, min(1.0, progress)), 3),
            "trial_index": self._trial_index,
            "total_trials": self._total_trials,
            "instruction": instruction,
        }
        if self._task_type == "melody" and self._melody_notes:
            r["melody_sequence"] = [self._note_name(n) for n in self._melody_notes]
            r["melody_note_index"] = min(self._melody_note_idx,
                                         len(self._melody_notes) - 1)
        return r

    def _melody_progress(self):
        if not self._melody_notes:
            return 0.0
        return self._melody_note_idx / len(self._melody_notes)

    def _print_sustain_stats(self):
        if not self._hold_pitches:
            return
        target_cents = self._target_note * 100.0
        deviations = [p - target_cents for p in self._hold_pitches]
        n = len(deviations)
        mean_d = sum(deviations) / n
        variance = sum((d - mean_d) ** 2 for d in deviations) / n
        sd = math.sqrt(variance)
        print(f"  Sustain trial {self._trial_index}: "
              f"SD = {sd:.1f} cents  (n={n} frames)")

    @staticmethod
    def _direction_hint(error_cents, on_target):
        if on_target:
            return "そのまま！"
        if error_cents > 0:
            return "もう少し下げてください"
        return "もう少し上げてください"

    @staticmethod
    def _note_name(midi_note):
        if midi_note is None:
            return ""
        return f"{_NOTE_NAMES[midi_note % 12]}{(midi_note // 12) - 1}"

    # ---- pitch / scale helpers ---------------------------------------------

    def _build_scale_notes(self):
        """Return sorted list of MIDI notes reachable at this sensitivity.

        A 10 % safety margin is applied so notes near the physical
        boundary of the camera frame are excluded.
        """
        center = (BASE_OCTAVE + 1) * 12                   # 60 = C4
        max_offset_cm = ASSUMED_FRAME_HEIGHT_CM / 2        # ±20 cm
        max_semitones = max_offset_cm / self._cm_per_oct * 12 * 0.9
        lo = math.ceil(center - max_semitones)
        hi = math.floor(center + max_semitones)
        if self._scale_degrees is None:
            notes = list(range(lo, hi + 1))
        else:
            deg = set(self._scale_degrees)
            notes = [n for n in range(lo, hi + 1) if n % 12 in deg]
        print(f"  到達可能範囲: {self._note_name(notes[0])} ({notes[0]}) "
              f"〜 {self._note_name(notes[-1])} ({notes[-1]})")
        return notes

    def _y_to_pitch(self, hand_y):
        """Convert normalised hand-Y to ``(midi_note, pitch_cents)``.

        Uses the same cm_per_octave mapping as
        ``GestureLogic.map_pitch`` (experimental path).
        """
        delta_cm = (hand_y - 0.5) * ASSUMED_FRAME_HEIGHT_CM
        delta_oct = delta_cm / self._cm_per_oct
        center = (BASE_OCTAVE + 1) * 12
        exact = center + delta_oct * 12
        half = PITCH_RANGE_OCTAVES * 6
        exact = max(center - half, min(center + half, exact))
        pitch_cents = exact * 100.0
        if self._scale_degrees is None:
            midi_note = int(round(exact))
        else:
            midi_note = self._snap(exact)
        return max(0, min(127, midi_note)), pitch_cents

    def _snap(self, exact_semitones):
        """Snap *exact_semitones* to the nearest note in ``_scale_degrees``."""
        center = int(round(exact_semitones))
        deg = set(self._scale_degrees)
        best, best_d = center, float("inf")
        for c in range(max(0, center - 12), min(128, center + 13)):
            if c % 12 in deg:
                d = abs(exact_semitones - c)
                if d < best_d:
                    best_d = d
                    best = c
        return best


# ---------------------------------------------------------------------------
# Dry-run simulation
# ---------------------------------------------------------------------------

def _dry_run():
    """Run one full condition without a camera.

    Simulates hand_y with an exponential-convergence sine wave that homes in
    on each target note, holds it for the required duration, then moves to
    the next trial.  Writes a real CSV to ``logs/`` and prints progress to
    the terminal.
    """
    import time as _time
    from experiment_logger import ExperimentLogger

    FPS = 30
    DT = 1.0 / FPS
    SENSITIVITY = "high"
    SCALE = "pentatonic"

    random.seed(42)

    print(f"=== DRY-RUN ===")
    print(f"  sensitivity : {SENSITIVITY}")
    print(f"  scale       : {SCALE}")
    print(f"  sim FPS     : {FPS}")
    print(f"  tasks       : 10 target + 5 sustain + 1 melody\n")

    logger = ExperimentLogger()
    logger.start_session(SENSITIVITY, SCALE)

    cm_per_oct = SENSITIVITY_LEVELS[SENSITIVITY]["cm_per_octave"]
    center_midi = (BASE_OCTAVE + 1) * 12          # 60

    # -- helpers --

    def note_to_y(midi_note):
        """Inverse of _y_to_pitch: MIDI note -> normalised hand Y."""
        return 0.5 + (midi_note - center_midi) * cm_per_oct / (
            ASSUMED_FRAME_HEIGHT_CM * 12
        )

    def sine_y(target_y, elapsed):
        """Exponential approach with decaying vibrato."""
        blend = 1.0 - math.exp(-4.0 * elapsed)
        base = 0.5 + (target_y - 0.5) * blend
        vib_amp = 0.03 * (1.0 - blend) + 0.002
        vib = vib_amp * math.sin(2.0 * math.pi * 2.0 * elapsed)
        return max(0.0, min(1.0, base + vib))

    # -- run each task block --

    frame_global = 0
    tasks = [("target", "TARGET", None),
             ("sustain", "SUSTAIN", None),
             ("melody", "MELODY", 1)]        # melody: stop after 1

    for task_type, label, max_trials in tasks:
        hdr = (f"{'frm':>5} {'trl':>4} {'target':>6} {'pitch':>6} "
               f"{'err_c':>7} {'on':>3} {'prog':>5}  instruction")
        print(f"\n{'=' * len(hdr)}")
        print(f"  {label} TASK")
        print(f"{'=' * len(hdr)}")
        print(hdr)
        print("-" * len(hdr))

        te = TaskEngine(SENSITIVITY, SCALE, logger)
        te.start_task(task_type)

        target_y = 0.5
        approach_start = _time.time()
        prev_target_note = None

        while not te.is_complete():
            elapsed = _time.time() - approach_start
            y = sine_y(target_y, elapsed)

            info = te.update(y)
            frame_global += 1

            # Reset approach clock when target note changes
            cur_target = info["target_note"]
            if cur_target != prev_target_note:
                prev_target_note = cur_target
                if cur_target is not None:
                    target_y = note_to_y(cur_target)
                approach_start = _time.time()

            # Decide whether to print this frame
            instr = info["instruction"]
            is_event = any(k in instr for k in
                           ("次へ", "完了", "次の音", "もう一度", "キープ",
                            "素晴らしい"))
            if frame_global % 10 == 0 or is_event:
                tn = info.get("target_note_name", "")
                print(
                    f"{frame_global:5d} "
                    f"{info['trial_index']:4d} "
                    f"{tn:>6} "
                    f"{info['current_pitch']:6.1f} "
                    f"{info['error_cents']:7.1f} "
                    f"{'Y' if info['on_target'] else '.':>3} "
                    f"{info['progress']:5.3f}  "
                    f"{instr}"
                )

            # Stop melody early (1 melody only)
            if max_trials is not None and info["trial_index"] >= max_trials:
                break

            _time.sleep(DT)

    logger.end_session()

    # Report CSV location
    import glob as _glob
    logs = sorted(_glob.glob("logs/session_*.csv"))
    if logs:
        csv_path = logs[-1]
        with open(csv_path) as f:
            total_lines = sum(1 for _ in f) - 1        # minus header
        print(f"\nCSV saved : {csv_path}")
        print(f"Total rows: {total_lines}")


if __name__ == "__main__":
    import sys
    if "--dry-run" in sys.argv:
        _dry_run()
    else:
        print("Usage: python task_engine.py --dry-run")
        print("  Simulates one full experimental condition without a camera.")
