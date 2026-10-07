#!/usr/bin/env python3
"""Single-user benchmark: rank all 12 sensitivity x scale conditions.

Usage:
    python run_benchmark.py --test --participant test01
    python run_benchmark.py --participant P01 --music-experience yes
"""

import argparse
import csv
import os
import random
import time
from datetime import datetime

import cv2

from experiment_config import (
    SENSITIVITY_LEVELS, SCALE_TYPES,
    BASE_OCTAVE, PITCH_RANGE_OCTAVES, ASSUMED_FRAME_HEIGHT_CM,
)
from experiment_logger import ExperimentLogger
from src.vision_ultralytics import UltralyticsVision
from src.midi_engine import MidiEngine
from src.logic import GestureLogic

# ── constants ────────────────────────────────────────────────────────────────

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
               "F#", "G", "G#", "A", "A#", "B"]

_TRIALS = 10
_ON_TARGET_CENTS = 50
_HOLD_SEC = 0.5
_TIMEOUT_SEC = 30

_CONDITIONS = [
    (sens, scale)
    for sens in SENSITIVITY_LEVELS
    for scale in SCALE_TYPES
]


# ── null MIDI (lets GestureLogic run without a real port) ────────────────────

class _NullMidi:
    def send_note_on(self, *a, **kw): pass
    def send_note_off(self, *a, **kw): pass
    def send_cc(self, *a, **kw): pass
    def send_pitch_bend(self, *a, **kw): pass
    def close(self): pass


# ── pitch helpers (same maths as TaskEngine) ─────────────────────────────────

def _build_scale_notes(scale_type, cm_per_oct):
    """Sorted MIDI notes reachable at *cm_per_oct* for *scale_type* (10 % margin)."""
    import math as _math
    notes = SCALE_TYPES[scale_type]["notes"]
    center = (BASE_OCTAVE + 1) * 12
    max_offset_cm = ASSUMED_FRAME_HEIGHT_CM / 2
    max_semitones = max_offset_cm / cm_per_oct * 12 * 0.9
    lo = _math.ceil(center - max_semitones)
    hi = _math.floor(center + max_semitones)
    if notes is None:
        result = list(range(lo, hi + 1))
    else:
        deg = set(notes)
        result = [n for n in range(lo, hi + 1) if n % 12 in deg]
    print(f"  到達可能範囲: {_note_name(result[0])} ({result[0]}) "
          f"〜 {_note_name(result[-1])} ({result[-1]})")
    return result


def _snap(exact, scale_degrees):
    center = int(round(exact))
    deg = set(scale_degrees)
    best, best_d = center, float("inf")
    for c in range(max(0, center - 12), min(128, center + 13)):
        if c % 12 in deg:
            d = abs(exact - c)
            if d < best_d:
                best_d, best = d, c
    return best


def _y_to_pitch(hand_y, cm_per_oct, scale_degrees):
    """Convert normalised hand-Y (0 = bottom, 1 = top) to (midi_note, pitch_cents)."""
    delta_cm = (hand_y - 0.5) * ASSUMED_FRAME_HEIGHT_CM
    delta_oct = delta_cm / cm_per_oct
    center = (BASE_OCTAVE + 1) * 12
    exact = center + delta_oct * 12
    half = PITCH_RANGE_OCTAVES * 6
    exact = max(center - half, min(center + half, exact))
    pitch_cents = exact * 100.0
    if scale_degrees is None:
        midi_note = int(round(exact))
    else:
        midi_note = _snap(exact, scale_degrees)
    return max(0, min(127, midi_note)), pitch_cents


def _note_name(midi_note):
    return f"{_NOTE_NAMES[midi_note % 12]}{(midi_note // 12) - 1}"


# ── cv2 overlay ──────────────────────────────────────────────────────────────

def _midi_to_frame_y(midi_note, cm_per_oct, frame_h):
    """Convert a MIDI note (float or int) to a frame-Y pixel coordinate."""
    center = (BASE_OCTAVE + 1) * 12
    delta_cm = (midi_note - center) / 12.0 * cm_per_oct
    hand_y = 0.5 + delta_cm / ASSUMED_FRAME_HEIGHT_CM
    return int(frame_h * (1.0 - hand_y))


def _draw_overlay(frame, target_name, trial_label,
                  error_cents, on_target, hold_progress,
                  scale_notes=None, cm_per_oct=None,
                  target_midi=None, current_midi=None):
    """Draw scale boundaries, target/current note, error, and hold-bar."""
    h, w = frame.shape[:2]

    # ── scale bar + note boundaries ──────────────────────────────────────
    if scale_notes and cm_per_oct is not None:
        # Compute boundary Y between each pair of adjacent notes
        boundaries = []
        for i in range(len(scale_notes) - 1):
            mid = (scale_notes[i] + scale_notes[i + 1]) / 2.0
            boundaries.append(_midi_to_frame_y(mid, cm_per_oct, h))

        # Draw each note zone
        for i, note in enumerate(scale_notes):
            zone_top = boundaries[i] if i < len(boundaries) else 0
            zone_bot = boundaries[i - 1] if i > 0 else h
            zone_top = max(0, min(h, zone_top))
            zone_bot = max(0, min(h, zone_bot))
            if zone_bot <= zone_top:
                continue

            # Green tint on the target zone
            if target_midi is not None and note == target_midi:
                roi = frame[zone_top:zone_bot, 0:w]
                green = roi.copy()
                green[:] = (0, 80, 0)
                cv2.addWeighted(green, 0.25, roi, 0.75, 0, roi)

            # Note label on the right margin (skip if zone too narrow)
            zone_h = zone_bot - zone_top
            if zone_h >= 18:
                label_y = (zone_top + zone_bot) // 2 + 4
                name = _note_name(note)
                colour = (0, 255, 0) if note == target_midi else (180, 180, 180)
                cv2.putText(frame, name, (w - 55, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)

        # Boundary lines across the full width
        for by in boundaries:
            if 0 < by < h:
                cv2.line(frame, (0, by), (w - 60, by), (100, 100, 100), 1)

    # ── text HUD ─────────────────────────────────────────────────────────
    # Trial counter (top-left)
    cv2.putText(frame, trial_label, (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # Target note (top-centre)
    cv2.putText(frame, f"Target: {target_name}",
                (w // 2 - 140, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)

    # Current note (large, centred, right below target)
    if current_midi is not None:
        now_name = _note_name(current_midi)
        is_correct = (target_midi is not None and current_midi == target_midi)
        col = (0, 255, 0) if is_correct else (0, 160, 255)
        cv2.putText(frame, f"You: {now_name}",
                    (w // 2 - 120, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 3)
    elif error_cents is None:
        cv2.putText(frame, "No hand", (w // 2 - 80, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (128, 128, 128), 2)

    # Error readout (below the note names)
    if error_cents is not None:
        colour = (0, 255, 0) if on_target else (0, 0, 255)
        cv2.putText(frame, f"{error_cents:+.0f} cents",
                    (w // 2 - 80, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)

    # Hold progress bar (bottom) with current note inside
    bar_x, bar_y, bar_w, bar_h = 40, h - 60, w - 80, 36
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
    fill = int(bar_w * min(1.0, hold_progress))
    if fill > 0:
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + fill, bar_y + bar_h), (0, 255, 0), -1)
    # Note name inside the bar
    if current_midi is not None:
        bar_text = _note_name(current_midi)
        cv2.putText(frame, bar_text, (bar_x + 8, bar_y + bar_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


# ── per-condition runner ─────────────────────────────────────────────────────

def _check_quit(key):
    """Return True if the user pressed q or ESC."""
    return key & 0xFF in (ord("q"), 27)


def _run_condition(vision, logic, sensitivity, scale_type,
                   num_trials=_TRIALS):
    """Run target-reach trials.  Returns ``(trial_rows, stopped)``."""

    cm_per_oct = SENSITIVITY_LEVELS[sensitivity]["cm_per_octave"]
    scale_degrees = SCALE_TYPES[scale_type]["notes"]
    scale_notes = _build_scale_notes(scale_type, cm_per_oct)

    logic.experiment_sensitivity = sensitivity
    logic.experiment_scale_type = scale_type

    # ExperimentLogger writes a per-frame CSV (used as-is)
    logger = ExperimentLogger()
    logger.start_session(sensitivity, scale_type)

    trial_rows = []
    trial_id_counter = 0
    stopped = False
    last_target = None

    for trial_idx in range(num_trials):
        if stopped:
            break

        trial_id_counter += 1
        pool = [n for n in scale_notes if n != last_target] or scale_notes
        target_note = random.choice(pool)
        last_target = target_note
        target_name = _note_name(target_note)
        target_cents = target_note * 100.0

        logger.mark_trial_start(trial_id_counter, target_note)

        print(f"    trial {trial_idx + 1:2d}/{num_trials}  "
              f"target: {target_name:<5}", end="", flush=True)

        trial_label = f"{trial_idx + 1}/{num_trials}"

        # Brief pause so the participant can see the new target
        gap_end = time.time() + 0.6
        while time.time() < gap_end:
            ok, frame, result = vision.get_frame()
            if ok:
                _draw_overlay(frame, target_name, trial_label,
                              None, False, 0.0,
                              scale_notes, cm_per_oct,
                              target_note, None)
                cv2.imshow("Benchmark", frame)
                if _check_quit(cv2.waitKey(1)):
                    stopped = True
                    print("  STOPPED")
                    break
        if stopped:
            break

        trial_start = time.time()
        on_target_start = None
        first_reach_logged = False
        trial_frames = 0
        trial_on_frames = 0

        while True:
            now = time.time()
            elapsed = now - trial_start

            if elapsed > _TIMEOUT_SEC:
                print(f"  TIMEOUT ({_TIMEOUT_SEC}s)")
                on_rate = (trial_on_frames / trial_frames * 100
                           if trial_frames else 0.0)
                trial_rows.append({
                    "trial_id": trial_idx + 1,
                    "target_note_midi": target_note,
                    "time_to_reach_ms": "",
                    "on_target_rate": round(on_rate, 1),
                    "is_timeout": "true",
                })
                break

            ok, frame, result = vision.get_frame()
            if not ok:
                continue

            # --- hand extraction via GestureLogic ---
            logic.last_hand_y_norm = None
            logic.last_hand_y_raw = None
            logic.process(result)
            hand_y = logic.last_hand_y_norm

            if hand_y is None:
                trial_frames += 1
                _draw_overlay(frame, target_name, trial_label,
                              None, False, 0.0,
                              scale_notes, cm_per_oct,
                              target_note, None)
                cv2.imshow("Benchmark", frame)
                if _check_quit(cv2.waitKey(1)):
                    stopped = True
                    print("  STOPPED")
                    break
                continue

            # --- pitch mapping ---
            midi_note, pitch_cents = _y_to_pitch(hand_y, cm_per_oct,
                                                 scale_degrees)
            error_cents = pitch_cents - target_cents
            on_target = abs(error_cents) <= _ON_TARGET_CENTS

            hand_y_raw = 1.0 - hand_y
            logger.log_frame(trial_id_counter, "target",
                             target_note, midi_note,
                             hand_y_raw, hand_y, on_target)

            trial_frames += 1
            if on_target:
                trial_on_frames += 1

            # First-reach marker (same as TaskEngine)
            if on_target and not first_reach_logged:
                logger.mark_on_target(trial_id_counter)
                first_reach_logged = True

            # Hold tracking
            if on_target:
                if on_target_start is None:
                    on_target_start = now
                held = now - on_target_start
            else:
                on_target_start = None
                held = 0.0

            hold_progress = min(1.0, held / _HOLD_SEC)
            _draw_overlay(frame, target_name, trial_label,
                          error_cents, on_target, hold_progress,
                          scale_notes, cm_per_oct,
                          target_note, midi_note)
            cv2.imshow("Benchmark", frame)
            if _check_quit(cv2.waitKey(1)):
                stopped = True
                print("  STOPPED")
                break

            if held >= _HOLD_SEC:
                reach_ms = (logger._trial_reach_times.get(trial_id_counter)
                            or (now - trial_start) * 1000)
                on_rate = (trial_on_frames / trial_frames * 100
                           if trial_frames else 0.0)
                trial_rows.append({
                    "trial_id": trial_idx + 1,
                    "target_note_midi": target_note,
                    "time_to_reach_ms": round(reach_ms, 1),
                    "on_target_rate": round(on_rate, 1),
                    "is_timeout": "false",
                })
                print(f"  reach {reach_ms:6.0f} ms  "
                      f"on-target {on_rate:.0f}%")
                break

    logger.end_session()
    return trial_rows, stopped


# ── main ─────────────────────────────────────────────────────────────────────

def run_benchmark(participant_id, music_experience):
    stamp = datetime.now()

    print("Initialising camera + MIDI...")
    vision = UltralyticsVision()
    midi = MidiEngine()
    logic = GestureLogic(midi)

    # Warm-up frames
    for _ in range(30):
        vision.get_frame()

    n_cond = len(_CONDITIONS)

    print(f"\n{'=' * 58}")
    print(f"  BENCHMARK")
    print(f"  participant          : {participant_id}")
    print(f"  music experience     : {music_experience}")
    print(f"  conditions           : {n_cond}")
    print(f"  trials / condition   : {_TRIALS}")
    print(f"  hold threshold       : ±{_ON_TARGET_CENTS} cents × {_HOLD_SEC}s")
    print(f"{'=' * 58}")

    input("\n  Press Enter to start...")

    all_trial_rows = []   # flat list for CSV
    cond_summaries = []   # one dict per condition for the ranked table

    for cond_idx, (sensitivity, scale_type) in enumerate(_CONDITIONS):
        sens_lbl = SENSITIVITY_LEVELS[sensitivity]["label"]
        scale_lbl = SCALE_TYPES[scale_type]["label"]

        print(f"\n{'─' * 58}")
        print(f"  条件 {cond_idx + 1}/{n_cond} | "
              f"感度: {sens_lbl} | "
              f"スケール: {scale_lbl}")
        print(f"{'─' * 58}")

        if cond_idx > 0:
            input("  Press Enter to continue...")

        trial_rows, stopped = _run_condition(vision, logic,
                                                sensitivity, scale_type)

        # Attach participant metadata + condition to every row
        for row in trial_rows:
            row["participant_id"] = participant_id
            row["music_experience"] = music_experience
            row["condition_sensitivity"] = sensitivity
            row["condition_scale"] = scale_type
        all_trial_rows.extend(trial_rows)

        # Per-condition aggregate (exclude timeouts)
        valid = [r for r in trial_rows if r["is_timeout"] == "false"]
        reaches = [r["time_to_reach_ms"] for r in valid]
        rates = [r["on_target_rate"] for r in valid]

        avg_reach = sum(reaches) / len(reaches) if reaches else float("inf")
        avg_rate = sum(rates) / len(rates) if rates else 0.0
        n_total = len(trial_rows)
        n_valid = len(valid)

        cond_summaries.append({
            "sensitivity": sensitivity,
            "scale_type": scale_type,
            "avg_reach_ms": avg_reach,
            "on_target_rate": avg_rate,
            "valid": f"{n_valid}/{n_total}",
        })

        reach_str = (f"{avg_reach:.0f} ms"
                     if avg_reach != float("inf") else "n/a")
        print(f"\n  >> avg reach: {reach_str}  |  "
              f"on-target: {avg_rate:.0f}%  |  "
              f"valid: {n_valid}/{n_total}")

        if stopped:
            print("\n  Benchmark stopped early by user.")
            break

    vision.release()
    midi.close()
    cv2.destroyAllWindows()

    # ── ranked table ─────────────────────────────────────────────────────
    cond_summaries.sort(key=lambda r: r["avg_reach_ms"])

    hdr = (f"  {'Rank':>4} | {'Sensitivity':<12} | {'Scale':<12} | "
           f"{'Avg reach (ms)':>15} | {'On-target rate':>14} | {'Valid':>5}")
    sep = (f"  {'─' * 4}-+-{'─' * 12}-+-{'─' * 12}-+-"
           f"{'─' * 15}-+-{'─' * 14}-+-{'─' * 5}")

    print(f"\n{'=' * 76}")
    print(f"  BENCHMARK RESULTS  ({stamp:%Y-%m-%d %H:%M})")
    print(f"  Participant: {participant_id}  "
          f"| Music experience: {music_experience}")
    print(f"{'=' * 76}")
    print(hdr)
    print(sep)

    for rank, r in enumerate(cond_summaries, 1):
        if r["avg_reach_ms"] == float("inf"):
            reach_str = "          n/a  "
        else:
            reach_str = f"{r['avg_reach_ms']:>11.0f} ms "
        print(f"  {rank:4d} | {r['sensitivity']:<12} | "
              f"{r['scale_type']:<12} | "
              f"{reach_str} | "
              f"{r['on_target_rate']:>13.0f}% | "
              f"{r['valid']:>5}")

    print(f"{'=' * 76}")

    # ── save CSV ─────────────────────────────────────────────────────────
    csv_path = _save_csv(all_trial_rows, participant_id, stamp)
    print(f"\n  Saved to {csv_path}\n")


# ── test mode ────────────────────────────────────────────────────────────────

def run_test(participant_id):
    """Quick smoke-test: 1 condition (medium/pentatonic), 3 trials."""
    stamp = datetime.now()

    print("Initialising camera + MIDI...")
    vision = UltralyticsVision()
    midi = MidiEngine()
    logic = GestureLogic(midi)

    for _ in range(30):
        vision.get_frame()

    sensitivity, scale_type = "medium", "pentatonic"
    sens_lbl = SENSITIVITY_LEVELS[sensitivity]["label"]
    scale_lbl = SCALE_TYPES[scale_type]["label"]

    print(f"\n{'=' * 58}")
    print(f"  TEST MODE")
    print(f"  participant : {participant_id}")
    print(f"  condition   : {sensitivity} / {scale_type}"
          f"  ({sens_lbl} x {scale_lbl})")
    print(f"  trials      : 3")
    print(f"{'=' * 58}\n")

    trial_rows, stopped = _run_condition(vision, logic, sensitivity,
                                           scale_type, num_trials=3)

    for row in trial_rows:
        row["participant_id"] = participant_id
        row["music_experience"] = "n/a"
        row["condition_sensitivity"] = sensitivity
        row["condition_scale"] = scale_type

    vision.release()
    midi.close()
    cv2.destroyAllWindows()

    # Summary (exclude timeouts)
    valid = [r for r in trial_rows if r["is_timeout"] == "false"]
    if valid:
        avg = sum(r["time_to_reach_ms"] for r in valid) / len(valid)
        print(f"\n  avg reach: {avg:.0f} ms  "
              f"(valid: {len(valid)}/3)")
    else:
        print("\n  no valid trials (all timed out)")

    csv_path = _save_csv(trial_rows, participant_id, stamp)
    print(f"  Saved to {csv_path}\n")


# ── CSV ──────────────────────────────────────────────────────────────────────

_CSV_COLUMNS = [
    "participant_id",
    "music_experience",
    "condition_sensitivity",
    "condition_scale",
    "trial_id",
    "target_note_midi",
    "time_to_reach_ms",
    "on_target_rate",
    "is_timeout",
]


def _save_csv(rows, participant_id, stamp):
    os.makedirs("logs", exist_ok=True)
    fname = f"benchmark_{stamp:%Y%m%d_%H%M%S}_{participant_id}.csv"
    path = os.path.join("logs", fname)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return path


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single-user benchmark across 12 conditions")
    parser.add_argument("--participant", required=True,
                        help="Participant ID (e.g. P01)")
    parser.add_argument("--test", action="store_true",
                        help="Quick test: 1 condition (medium/pentatonic), "
                             "3 trials, no --music-experience needed")
    parser.add_argument("--music-experience",
                        choices=["yes", "no"],
                        help="Music experience (required for full benchmark)")
    args = parser.parse_args()

    if args.test:
        run_test(args.participant)
    else:
        if args.music_experience is None:
            parser.error("--music-experience is required for full benchmark")
        run_benchmark(args.participant, args.music_experience)
