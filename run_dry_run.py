#!/usr/bin/env python3
"""Dry-run: simulate one condition without camera, MIDI, or UI.

Generates a sine-wave hand_y with Gaussian noise, feeds it through
TaskEngine (target task only, 10 trials), logs every frame to CSV,
and prints summary statistics at the end.

The raw sine (amplitude 0.4) sweeps ~12 semitones in 4 seconds, far too
fast for the 0.5 s on-target hold the task requires.  So each trial
starts with the free sine, then a convergence bias ramps up over ~1.5 s
to pull the signal toward the target — mimicking a player's approach.
The sine oscillation and noise remain visible throughout.
"""

import csv
import math
import random
import time
from collections import Counter

from experiment_config import (
    SENSITIVITY_LEVELS, SCALE_TYPES,
    BASE_OCTAVE, PITCH_RANGE_OCTAVES, ASSUMED_FRAME_HEIGHT_CM,
)
from experiment_logger import ExperimentLogger
from task_engine import TaskEngine

# -- config -----------------------------------------------------------------

SENSITIVITY = "high"
SCALE_TYPE = "diatonic"
FPS = 30
DT = 1.0 / FPS
SINE_PERIOD = 4.0                       # seconds per full oscillation
NOISE_SD = 0.005                        # small enough to hold within ±50 cents


def main():
    random.seed(42)

    cm_per_oct = SENSITIVITY_LEVELS[SENSITIVITY]["cm_per_octave"]
    center_midi = (BASE_OCTAVE + 1) * 12          # 60

    def note_to_y(midi_note):
        return 0.5 + (midi_note - center_midi) * cm_per_oct / (
            ASSUMED_FRAME_HEIGHT_CM * 12)

    print("=== DRY RUN ===")
    print(f"  condition : {SENSITIVITY} / {SCALE_TYPE}")
    print(f"  task      : target (10 trials)")
    print(f"  sine      : 0.5 +/- 0.4, period {SINE_PERIOD}s, noise SD {NOISE_SD}")
    print(f"  note      : high sens (15cm/oct) so all diatonic notes 48-72 are reachable")
    print()

    logger = ExperimentLogger()
    logger.start_session(SENSITIVITY, SCALE_TYPE)

    engine = TaskEngine(SENSITIVITY, SCALE_TYPE, logger)
    engine.start_task("target")

    t0 = time.time()
    frame = 0
    trial_start = t0
    prev_trial_idx = -1

    while not engine.is_complete():
        t = time.time() - t0

        # Base sine wave (user spec)
        base_sine = 0.5 + 0.4 * math.sin(2 * math.pi * t / SINE_PERIOD)

        # Convergence bias: first 1 s is free sine (shows the sweep),
        # then bias ramps to 0.998 over ~0.5 s so the hold can complete.
        info_peek = engine._target_note          # current target MIDI
        target_y = note_to_y(info_peek) if info_peek is not None else 0.5
        trial_t = time.time() - trial_start
        if trial_t < 1.0:
            bias = 0.0                           # free sine phase
        else:
            bias = min(0.998, (trial_t - 1.0) * 2.0)
        hand_y = base_sine + bias * (target_y - base_sine)

        # Gaussian noise
        hand_y += random.gauss(0, NOISE_SD)
        hand_y = max(0.0, min(1.0, hand_y))

        info = engine.update(hand_y)
        frame += 1

        # Detect trial advance → reset convergence clock
        if info["trial_index"] != prev_trial_idx:
            prev_trial_idx = info["trial_index"]
            trial_start = time.time()

        pitch = info["current_pitch"]
        note_int = int(round(pitch))
        note_name = _note_name(note_int)
        err = info["error_cents"]
        on = info["on_target"]

        print(
            f"[t={t:5.2f}s] "
            f"hand_y={hand_y:.3f} "
            f"\u2192 note={note_int}({note_name}) "
            f"error={err:+.0f}\u00a2 "
            f"on_target={on}"
        )

        time.sleep(DT)

    elapsed = time.time() - t0
    logger.end_session()

    # -- read CSV back and print stats --------------------------------------

    import glob
    csv_path = sorted(glob.glob("logs/session_*.csv"))[-1]

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"\n{'=' * 60}")
    print(f"  CSV: {csv_path}")
    print(f"{'=' * 60}")
    print(f"\nFirst 20 data rows:")

    header = list(rows[0].keys())
    print(f"  {', '.join(header)}")
    for row in rows[:20]:
        print(f"  {', '.join(row[k] for k in header)}")

    # -- summary stats ------------------------------------------------------

    total_frames = len(rows)
    on_target_count = sum(1 for r in rows if r["is_on_target"] == "1")
    on_target_pct = 100.0 * on_target_count / total_frames if total_frames else 0

    seen_trials = set()
    unique_reach = []
    for r in rows:
        tid = r["trial_id"]
        val = r["time_to_reach_ms"]
        if val not in ("", None) and tid not in seen_trials:
            seen_trials.add(tid)
            unique_reach.append(int(val))
    mean_reach = sum(unique_reach) / len(unique_reach) if unique_reach else 0

    note_counts = Counter(r["actual_note_midi"] for r in rows)

    print(f"\n{'─' * 40}")
    print(f"  Total frames logged : {total_frames}")
    print(f"  Elapsed time        : {elapsed:.1f}s")
    print(f"  On-target rate      : {on_target_pct:.1f}%")
    print(f"  Mean time_to_reach  : {mean_reach:.0f} ms  "
          f"(n={len(unique_reach)} trials)")
    print(f"\n  Note distribution:")
    for note_str, count in sorted(note_counts.items(), key=lambda x: int(x[0])):
        pct = 100.0 * count / total_frames
        name = _note_name(int(note_str))
        print(f"    MIDI {note_str:>3} ({name:>3}): {count:4d} frames ({pct:5.1f}%)")
    print()


def _note_name(midi_note):
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[midi_note % 12]}{(midi_note // 12) - 1}"


if __name__ == "__main__":
    main()
