import csv
import os
import time
from datetime import datetime


class ExperimentLogger:
    """Logs per-frame experiment data to a timestamped CSV for offline analysis."""

    HEADER = [
        "timestamp_ms",
        "trial_id",
        "condition_sensitivity",
        "condition_scale",
        "task_type",
        "target_note_midi",
        "actual_note_midi",
        "hand_y_raw",
        "hand_y_normalized",
        "is_on_target",
        "time_to_reach_ms",
    ]

    def __init__(self):
        self._file = None
        self._writer = None
        self._sensitivity = None
        self._scale_type = None
        self._session_start_ms = None
        # trial_id -> timestamp_ms when trial started
        self._trial_starts = {}
        # trial_id -> time_to_reach_ms (None until first on-target)
        self._trial_reach_times = {}

    def start_session(self, sensitivity, scale_type):
        """Open a new CSV log file and write the header row."""
        os.makedirs("logs", exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join("logs", f"session_{stamp}.csv")

        self._file = open(path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.HEADER)

        self._sensitivity = sensitivity
        self._scale_type = scale_type
        self._session_start_ms = self._now_ms()
        self._trial_starts = {}
        self._trial_reach_times = {}

        print(f"[ExperimentLogger] Session started -> {path}")

    def log_frame(self, trial_id, task_type, target_note, actual_note,
                  hand_y_raw, hand_y_norm, is_on_target):
        """Write one row of per-frame data."""
        if self._writer is None:
            return

        reach_time = self._trial_reach_times.get(trial_id)

        self._writer.writerow([
            self._now_ms(),
            trial_id,
            self._sensitivity,
            self._scale_type,
            task_type,
            target_note,
            actual_note,
            round(hand_y_raw, 6),
            round(hand_y_norm, 6),
            int(is_on_target),
            reach_time if reach_time is not None else "",
        ])

    def mark_trial_start(self, trial_id, target_note):
        """Record the moment a new trial begins (for time-to-reach calculation)."""
        now = self._now_ms()
        self._trial_starts[trial_id] = now
        self._trial_reach_times[trial_id] = None
        print(f"[ExperimentLogger] Trial {trial_id} started | target MIDI {target_note}")

    def mark_on_target(self, trial_id):
        """Record when the player first reaches within +/-50 cents of the target.

        Only records the first hit per trial. Returns the time_to_reach_ms or
        None if the trial was never started via mark_trial_start.
        """
        if trial_id not in self._trial_starts:
            return None
        # Already recorded for this trial
        if self._trial_reach_times.get(trial_id) is not None:
            return self._trial_reach_times[trial_id]

        now = self._now_ms()
        elapsed = now - self._trial_starts[trial_id]
        self._trial_reach_times[trial_id] = elapsed
        print(f"[ExperimentLogger] Trial {trial_id} on-target in {elapsed} ms")
        return elapsed

    def end_session(self):
        """Close the CSV and print summary stats."""
        if self._file is None:
            return

        self._file.close()

        total_trials = len(self._trial_starts)
        reached = {tid: t for tid, t in self._trial_reach_times.items()
                   if t is not None}
        reached_count = len(reached)

        print("\n--- Experiment Session Summary ---")
        print(f"  Sensitivity : {self._sensitivity}")
        print(f"  Scale       : {self._scale_type}")
        print(f"  Trials      : {total_trials}")
        print(f"  On-target   : {reached_count}/{total_trials}")

        if reached:
            times = list(reached.values())
            print(f"  Reach time  : min={min(times)} ms, "
                  f"max={max(times)} ms, "
                  f"mean={sum(times) / len(times):.0f} ms")
        else:
            print("  Reach time  : no on-target events recorded")

        print("---------------------------------\n")

        self._file = None
        self._writer = None

    @staticmethod
    def _now_ms():
        return int(time.time() * 1000)
