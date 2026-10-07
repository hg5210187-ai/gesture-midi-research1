import math
import time

from experiment_config import (
    SENSITIVITY_LEVELS, SCALE_TYPES,
    BASE_OCTAVE, PITCH_RANGE_OCTAVES, ASSUMED_FRAME_HEIGHT_CM,
)
from experiment_logger import ExperimentLogger
from task_engine import TaskEngine

# All 12 conditions (order matches dict insertion order, Python 3.7+).
_ALL_CONDITIONS = [
    (sens, scale)
    for sens in SENSITIVITY_LEVELS
    for scale in SCALE_TYPES
]

_TASK_ORDER = ["target", "sustain", "melody"]
_TASK_LABELS = {
    "target":  "ターゲット (Target)",
    "sustain": "サステイン (Sustain)",
    "melody":  "メロディ (Melody)",
}

_PILOT_CONDITIONS = [
    ("low", "pentatonic"),
    ("medium", "diatonic"),
    ("high", "chromatic"),
]
_PILOT_TRIAL_COUNTS = {"target": 5, "sustain": 3, "melody": 2}

_SIM_FPS = 30
_TASK_TIMEOUT_SEC = 60


class ExperimentSession:
    """Orchestrates a full within-subjects experiment.

    Manages 12 conditions (3 sensitivities x 4 scale types) in a
    counterbalanced order derived from a cyclic Latin square.
    """

    def __init__(self, participant_id, no_pause=False, pilot=False):
        self._pid = str(participant_id)
        self._no_pause = no_pause
        self._pilot = pilot
        self._offset = self._compute_offset(self._pid)
        self._conditions = (
            list(_PILOT_CONDITIONS) if pilot else self._build_condition_order()
        )

    # ---- public API -------------------------------------------------------

    @property
    def conditions(self):
        """Return the counterbalanced list of ``(sensitivity, scale_type)`` pairs."""
        return list(self._conditions)

    def run_session(self):
        """Run the full experiment (or pilot subset)."""
        n = len(self._conditions)

        if self._pilot:
            mode = "パイロット (Pilot)"
        else:
            mode = "実験セッション (Experiment Session)"

        print(f"\n{'=' * 54}")
        print(f"  {mode}")
        print(f"  参加者 (Participant): {self._pid}")
        print(f"  条件数 (Conditions) : {n}")
        if not self._pilot:
            print(f"  Latin-square offset : {self._offset}")
        print(f"{'=' * 54}")
        print("\n  Condition order:")
        for i, (s, sc) in enumerate(self._conditions):
            sl = SENSITIVITY_LEVELS[s]["label"]
            scl = SCALE_TYPES[sc]["label"]
            print(f"    {i + 1:2d}. {sl} x {scl}  ({s}/{sc})")

        self._pause("\n  準備ができたらEnterを押してください (Press Enter to start)...")

        for i, (sens, scale) in enumerate(self._conditions):
            sens_lbl = SENSITIVITY_LEVELS[sens]["label"]
            scale_lbl = SCALE_TYPES[scale]["label"]

            print(f"\n{'─' * 54}")
            print(f"  条件 {i + 1}/{n}: {sens_lbl} x {scale_lbl}  ({sens}/{scale})")
            print(f"{'─' * 54}")

            if i > 0:
                self._pause(
                    "\n  休憩してください。準備ができたらEnterを押してください...\n"
                    "  (Take a break. Press Enter when ready...)\n"
                )

            self._run_condition(sens, scale)

        print(f"\n{'=' * 54}")
        print(f"  セッション完了！ (Session complete!)")
        print(f"  参加者: {self._pid}")
        print(f"  ログは logs/ に保存されました")
        print(f"{'=' * 54}\n")

    def run_practice(self):
        """Run a single practice condition (low / pentatonic)."""
        print(f"\n{'=' * 54}")
        print(f"  練習セッション (Practice)")
        print(f"  参加者 (Participant): {self._pid}")
        print(f"  条件: 低感度 x ペンタトニック  (low / pentatonic)")
        print(f"{'=' * 54}")

        self._pause("\n  Enterを押して練習を開始... (Press Enter to start practice...)\n")

        self._run_condition("low", "pentatonic")

        print("\n  練習完了！ (Practice complete!)\n")

    # ---- condition / task runner -------------------------------------------

    def _run_condition(self, sensitivity, scale_type):
        """Run all three tasks for one experimental condition."""
        logger = ExperimentLogger()
        logger.start_session(sensitivity, scale_type)

        trial_counts = _PILOT_TRIAL_COUNTS if self._pilot else None
        engine = TaskEngine(sensitivity, scale_type, logger,
                            trial_counts=trial_counts)

        task_stats = {}

        for task_idx, task_type in enumerate(_TASK_ORDER):
            label = _TASK_LABELS[task_type]
            if task_idx > 0:
                self._pause(f"\n  次のタスク: {label}\n  Press Enter to continue...\n")

            print(f"\n  --- {label} ---")

            before_ids = set(logger._trial_starts.keys())
            stats = self._simulate_task(engine, task_type, sensitivity)
            after_ids = set(logger._trial_starts.keys())

            task_trial_ids = after_ids - before_ids
            reach_times = [
                logger._trial_reach_times[tid]
                for tid in task_trial_ids
                if logger._trial_reach_times.get(tid) is not None
            ]
            stats["reach_times"] = reach_times
            stats["trials_total"] = (
                trial_counts[task_type] if trial_counts else
                {"target": 10, "sustain": 5, "melody": 3}[task_type]
            )
            task_stats[task_type] = stats

        logger.end_session()

        if self._pilot:
            self._print_pilot_summary(sensitivity, scale_type, task_stats)

    def _simulate_task(self, engine, task_type, sensitivity):
        """Drive one task with a simulated sine-wave hand input.

        The sine wave uses exponential convergence from the current hand
        position toward each new target, with decaying vibrato.  A per-task
        timeout prevents hangs when the target is physically unreachable at
        the given sensitivity.

        Returns a dict with ``total_frames`` and ``on_target_frames``.
        """
        engine.start_task(task_type)

        cm_per_oct = SENSITIVITY_LEVELS[sensitivity]["cm_per_octave"]
        center_midi = (BASE_OCTAVE + 1) * 12
        dt = 1.0 / _SIM_FPS

        current_y = 0.5
        target_y = 0.5
        start_y = 0.5
        approach_t = time.time()
        task_t = time.time()
        prev_target = None
        frame = 0
        total_frames = 0
        on_target_frames = 0

        while not engine.is_complete():
            if time.time() - task_t > _TASK_TIMEOUT_SEC:
                print(f"    [TIMEOUT] {task_type} exceeded {_TASK_TIMEOUT_SEC}s, "
                      f"moving on")
                break

            # Exponential approach from start_y toward target_y
            elapsed = time.time() - approach_t
            blend = 1.0 - math.exp(-4.0 * elapsed)
            base = start_y + (target_y - start_y) * blend
            vib = ((0.03 * (1.0 - blend) + 0.002)
                   * math.sin(2.0 * math.pi * 2.0 * elapsed))
            y = max(0.0, min(1.0, base + vib))
            current_y = y

            info = engine.update(y)
            frame += 1
            total_frames += 1
            if info["on_target"]:
                on_target_frames += 1

            # Reset approach curve when the target note changes
            cur_target = info["target_note"]
            if cur_target != prev_target:
                prev_target = cur_target
                if cur_target is not None:
                    target_y = (0.5 + (cur_target - center_midi)
                                * cm_per_oct / (ASSUMED_FRAME_HEIGHT_CM * 12))
                start_y = current_y
                approach_t = time.time()

            # Print status once per second
            if frame % _SIM_FPS == 0:
                tn = info.get("target_note_name", "")
                print(
                    f"    trial {info['trial_index'] + 1:2d}"
                    f"/{info['total_trials']} "
                    f"| {tn:>4} "
                    f"| err={info['error_cents']:+6.0f}c "
                    f"| {'ON' if info['on_target'] else '--'} "
                    f"| {info['instruction']}"
                )

            time.sleep(dt)

        return {
            "total_frames": total_frames,
            "on_target_frames": on_target_frames,
        }

    # ---- pilot summary -----------------------------------------------------

    def _print_pilot_summary(self, sensitivity, scale_type, task_stats):
        """Print a per-metric summary after a pilot condition."""
        sens_lbl = SENSITIVITY_LEVELS[sensitivity]["label"]
        scale_lbl = SCALE_TYPES[scale_type]["label"]

        print(f"\n{'━' * 54}")
        print(f"  PILOT SUMMARY: {sensitivity}/{scale_type}"
              f"  ({sens_lbl} x {scale_lbl})")
        print(f"{'━' * 54}")

        for task_type in _TASK_ORDER:
            stats = task_stats.get(task_type, {})
            total_f = stats.get("total_frames", 0)
            on_f = stats.get("on_target_frames", 0)
            pct = (on_f / total_f * 100) if total_f > 0 else 0
            trials = stats.get("trials_total", 0)

            reach = stats.get("reach_times", [])
            if reach:
                mean_r = sum(reach) / len(reach)
                if len(reach) > 1:
                    var = sum((r - mean_r) ** 2 for r in reach) / (len(reach) - 1)
                    sd_r = math.sqrt(var)
                    reach_str = f"{mean_r:.0f} ± {sd_r:.0f} ms"
                else:
                    reach_str = f"{mean_r:.0f} ms"
            else:
                reach_str = "n/a"

            label = _TASK_LABELS[task_type]
            print(f"  {label}")
            print(f"    trials: {trials}  |  on-target: {pct:.0f}%"
                  f"  |  reach: {reach_str}")

        print(f"{'━' * 54}")

    # ---- counterbalancing --------------------------------------------------

    @staticmethod
    def _compute_offset(participant_id):
        """Derive a Latin-square row offset from the participant ID."""
        try:
            return int(participant_id) % 12
        except ValueError:
            return hash(participant_id) % 12

    def _build_condition_order(self):
        """Return the 12 conditions in counterbalanced order.

        Uses a cyclic Latin square: participant *r* sees condition
        ``(r + j) % 12`` at position *j*.  This guarantees that across
        12 participants every condition appears in every ordinal position
        exactly once.
        """
        n = len(_ALL_CONDITIONS)               # 12
        indices = [(self._offset + j) % n for j in range(n)]
        return [_ALL_CONDITIONS[i] for i in indices]

    # ---- helpers -----------------------------------------------------------

    def _pause(self, prompt):
        """Print *prompt* and wait for Enter, unless ``no_pause`` is set."""
        if self._no_pause:
            return
        input(prompt)


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print("Usage: python experiment_session.py <participant_id> [options]")
        print()
        print("Options:")
        print("  --practice   Run one practice condition only (low/pentatonic)")
        print("  --pilot      Run 3 conditions with halved trials + per-condition summary")
        print("  --no-pause   Skip all 'Press Enter' prompts (automated testing)")
        print()
        print("Examples:")
        print("  python experiment_session.py P01              # full session")
        print("  python experiment_session.py P01 --practice   # practice only")
        print("  python experiment_session.py P01 --pilot      # pilot dry-run")
        print("  python experiment_session.py 3 --no-pause     # automated test")
        sys.exit(0)

    pid = args[0]
    no_pause = "--no-pause" in args
    practice = "--practice" in args
    pilot = "--pilot" in args

    session = ExperimentSession(pid, no_pause=no_pause, pilot=pilot)

    if practice:
        session.run_practice()
    else:
        session.run_session()
