# Session Summary — Bounding-Box Pitch/Volume Refactor

_Date: 2026-06-16_

## Goal

Move the **right hand** off of per-frame keypoints and drive it entirely from its
YOLO **bounding box**, add a volume-calibration flow, add two toggleable video
overlays, and trim keypoint usage to save processing — while keeping the
**left-hand** fist gesture for looper control.

## Original request

> Currently the hand recognition depends on the 20 points on my hand. Change how
> the pitch is determined to use the midpoint of the Y-axis of the bounding box.
> Determine volume relative to the initial size. Create a calibration function so
> the user can show the minimum size of their hand and set that to zero; the area
> of the bounding box corresponds to the relative change in volume. Add a
> tick-box-toggled function to show the borderline of each note of the scale. Add
> a tick-box-toggled function that shows the current pitch. Remove the keypoint
> detection completely except the function of the left hand, to save processing.

## Design decisions (confirmed with the user)

| Topic | Decision |
|---|---|
| **Volume mapping** | `volume = clip((area / area_min − 1) · gain, 0, 1) · 127`. Calibrate min → `area_min` (volume 0); a **Volume Sensitivity** slider sets `gain` (range 0.5–8.0, default 2.0). |
| **Calibration trigger** | "Calibrate Min" button captures the latest right-hand box area instantly (hold + click). No-op with on-screen feedback if no right hand is visible. |
| **Pitch readout** | Note name + octave + frequency, e.g. `A4 (440 Hz)`. |
| **Uncalibrated volume** | `area_min` lazily seeded from the first right-hand frame (volume always defined, never divides by zero). |
| **Borderlines** | Horizontal lines at band boundaries `screen_y = (1 − i/len(scale)) · H`; hidden in theremin mode; thinned to ≤24 lines for dense scales; octave boundaries emphasized; per-band labels only when `len(scale) ≤ 24`. |
| **Always-on feedback** | Since `result.plot()` is removed, the overlay always draws lightweight hand boxes (right = green, left = orange). |

## Important caveat — "remove keypoint detection to save processing"

Verified against the installed stack (**ultralytics 8.4.45**, **cv2 4.13.0**):
`result.boxes` and `result.keypoints` come from the **same single pose
inference** and are index-aligned (`boxes[i] ↔ keypoints[i]`). Because the
left-hand fist gesture still needs keypoints, the model must still run its
keypoint head every frame — so **inference cost is unchanged**. The real CPU
savings come from:

1. **Dropping `result.plot()`** — it rendered ~42 keypoint circles + ~40 skeleton
   segments + boxes + text on an image copy every frame.
2. **Eliminating all right-hand keypoint math** (thumb/knuckle distances,
   landmark-9 Y).

> Future option (not done): if the left-hand gesture is ever dropped too,
> switching to a detection-only hand model would give a genuine inference speedup.

## Files changed

| File | Change |
|---|---|
| `src/vision_ultralytics.py` | `get_frame()` returns the **raw** flipped frame (no `result.plot()`); 3-tuple `(success, frame, result)` contract preserved. |
| `src/logic.py` | Hand ID by **box center-x**; new `_process_right_hand(bbox)` (pitch from box Y-midpoint, volume from box area + `area_min`/`volume_gain`); new `calibrate_min()`; new state (`area_min`, `volume_gain`, `last_right_area`, `last_right_bbox`, `last_left_bbox`, `show_borderlines`, `show_pitch`). Left hand still uses `kpts[left_idx]` for fist detection. `map_pitch`, `_snap_to_scale`, `_process_left_hand`, `_is_fist` unchanged. The `run_benchmark.py` contract (`last_hand_y_norm`) preserved. |
| `src/scales.py` | Added `midi_to_name_octave(n)` (69 → `"A4"`) and `midi_to_freq(n)` (69 → 440.0). |
| `src/overlay.py` | **New.** `render_overlay(frame, logic)` — hand boxes always; toggleable note borderlines (theremin-hidden, dense-scale thinning) and `NoteName+Octave (Freq Hz)` readout. |
| `src/ui.py` | Added Calibrate Min button (+ feedback label), Volume Sensitivity slider, and the two checkboxes; extended the settings dict; added `_calibrate` / `set_calib_status`. Constructor takes `on_calibrate`. |
| `main.py` | Imports/calls `render_overlay` in `vision_loop` after `process()`; wires `on_calibrate` → `logic.calibrate_min()` with UI feedback; copies `volume_gain` / `show_borderlines` / `show_pitch` onto the logic in `update_settings`. |
| `implementation_details.md`, `README.md` / `readme.md` | Updated to describe bbox pitch/volume, calibration, overlays, and the new module. |

## Out of scope (audited as unaffected)

- **Synthetic experiment harness** — `experiment_config.py`, `experiment_logger.py`,
  `experiment_session.py`, `task_engine.py`, `run_dry_run.py`: zero coupling to
  the live code; no changes.
- **`run_benchmark.py`** — only consumes `logic.last_hand_y_norm`, which is
  preserved; no changes needed. (Note: the keypoint-9 → bbox-midpoint switch
  changes the *meaning* of that Y signal, so post-refactor benchmark CSVs aren't
  directly comparable to pre-refactor runs.)
- **`src/vision.py`** (legacy MediaPipe, unused) — left as-is.

## Verification

Done headlessly (no camera/GUI in the environment):

- `python -m py_compile` on all changed files — OK.
- `import main` — OK (no camera opened at import).
- Functional tests with synthetic YOLO results:
  - Pitch: hand at top → high note, hand at bottom → lower note.
  - Volume: `calibrate_min()` then a larger box → CC rises with gain.
  - No-detection guard → silence + `None` state, no crash.
  - Two hands → leftmost=left (keypoints/fist), rightmost=right (bbox).
  - Left-hand fist → looper toggles once.
  - Overlay renders in scale mode, thins a 60-note chromatic scale, and hides
    borderlines in theremin mode.
  - `midi_to_name_octave` / `midi_to_freq` correct (69 → `A4` / 440 Hz).

## Known behavioral notes / live-check items

- **Pitch feel changes** — box center moves differently than a single knuckle,
  and the box center can't reach the very top/bottom of the frame (compressed
  playable range).
- **Hand assignment** now uses box-center-x instead of wrist-x — verify
  near-frame-center edge cases live.
- **Theremin pitch label** shows the base note (C4) by design (theremin holds a
  base note + pitch bend); most meaningful in scale mode.
- **`README.md` and `readme.md` share one inode** on this case-insensitive
  filesystem (editing one updates both) — consider removing the duplicate path
  from git.

## Process note

This work was completed in the **local** session. A parallel cloud `ultraplan`
session (`session_01CDPf3KXQgqDjnT2CLyvumD`) failed to reach `ExitPlanMode`
(remote container/session-ID issue) — that failure was independent and did not
affect the local plan or implementation.

## Possible next steps

- Launch the app to confirm the live camera path (pitch tracking, calibration,
  overlays, looper).
- Commit the changes.
- Optionally tune the Volume Sensitivity range against real calibrated areas.

---

# Follow-up — Phantom-detection fix (right hand makes sound when not shown)

_Date: 2026-06-17_

## Symptom

Sound was produced even when the user's right hand was **not** in frame.

## Root cause

The hand-pose model (`yolov8n-hand-pose.pt`) has a **single class** (`{0: 'hand'}`),
so `GestureLogic` assigns hand roles purely by box center-x: any single box with
`cx ≥ 0.5` is treated as the right hand and immediately plays a note. Inference ran
at YOLO's **default confidence (0.25)**, which is permissive enough to hallucinate a
"hand" from a face/shoulder/background. A phantom box on the right half → note on,
with no hand actually up.

## Fix

`src/logic.py` — added a `min_confidence = 0.5` threshold and filter detections by
`boxes.conf` at the top of `process()` **before** any hand is assigned a role.
Boxes below threshold are dropped; if all are dropped the frame is treated as
"no hands". Keypoints are filtered with the same mask to stay index-aligned.
Refactored the duplicated no-hands reset into `_reset_no_hands()`.

- Guarded with `getattr(boxes, "conf", None)` so mock results without `.conf`
  (and the synthetic harness) are unaffected.
- `run_benchmark.py` feeds real YOLO results (carry `.conf`) → benefits from the
  gate; `run_dry_run.py` never calls `process()` → unaffected.

## Verification

Headless test feeding fake YOLO results through `process()`:
- Low-conf (0.30) right-side box → **no** note, `current_pitch_note == -1`.
- High-conf (0.90) right-side box → note plays as before.
- `python -m py_compile src/logic.py` — OK.

## Notes / possible follow-ups

- `min_confidence` is a tunable `GestureLogic` attribute (default 0.5). If a real
  hand is occasionally missed, lower it; if phantoms persist, raise it. Could be
  exposed in the UI later.
- This does **not** address a *real* left hand drifting onto the right half being
  read as the right hand — unavoidable without true handedness (the model lost the
  left/right label the legacy MediaPipe backend had). Mitigated in practice by the
  mirror view (left hand naturally sits on the left).

---

# Follow-up — Visible pitch overlay (readout box + sampling line + region label)

_Date: 2026-06-17_

## Request

The pitch readout was small text tucked in the top-left corner. Make it visible;
show it in a box; draw the mid-Y line where pitch is determined; and label the
region so the user knows where to put their hand.

## Changes

`src/overlay.py` — replaced the corner text with a **current-pitch feedback
suite**, drawn when `show_pitch` is on:

1. **Active region** — highlights the scale band the hand is in
   (`idx = int(last_hand_y_norm * len(scale))`) with a translucent green fill and
   a note label (`> A4`). Scale mode only. Helps dense scales too (the borderline
   labels only render for ≤24-note scales; this always labels the active band).
2. **Mid-Y sampling line** — yellow full-width line at the right box's Y-midpoint
   (`(y1+y2)/2`), the exact point pitch is read from, with a `pitch` tag.
3. **Pitch readout box** — large, top-centre, translucent bg + green border/text,
   `NoteName+Octave (Freq Hz)`.

Added helpers `_fill_band` and `_draw_label_box` (translucent, on-screen-clamped).

`src/logic.py` — `show_pitch` now defaults to **True** so the suite is visible
out of the box. `src/ui.py` — `show_pitch_check.select()` so the checkbox matches.

## Verification

Headless render onto a synthetic 480×640 frame:
- Mid-Y line: 640 yellow px across row 240 (= 0.5·H, the box midpoint). ✓
- Active band: ~11.6k green-tint px in the expected band rows. ✓
- Pitch box: green border/text px in the top-centre strip. ✓
- Theremin mode, no-hand frame, and `show_pitch=False` all render without error.
- `py_compile` on `overlay.py`, `ui.py`, `logic.py` — OK.

## Notes

- The whole suite is gated by **Show Current Pitch**; **Show Note Borderlines**
  (all band boundaries) remains a separate toggle.
- In Theremin mode the region highlight is skipped (continuous pitch, no bands)
  and the readout box still shows the held base note (C4) by design.

---

# Follow-up — Looper status on the video overlay

_Date: 2026-06-17_

## Request

Show the looper status on the video feed (it was only in the settings panel,
but the user watches the video while gesturing).

## Change

`src/overlay.py` — added an always-on **looper status badge** at the top-left:
a solid colour-filled box (`_draw_status_badge`) reading `LOOP: <state>`,
colour-coded to match the UI label — gray IDLE, red RECORDING, green PLAYING
(`_LOOP_COLORS`). State is read from `logic.looper.state`; guarded with
`getattr` so `looper=None` (e.g. `run_benchmark.py`) is safe. Placed in the
top-left spot freed when the pitch readout moved to top-centre — no conflict
with the pitch box (centre), mid-Y line, or region label.

## Verification

Headless render onto a 480×640 frame:
- IDLE / RECORDING / PLAYING each draw the badge in the correct colour. ✓
- `looper=None` renders without error. ✓
- `py_compile src/overlay.py` — OK.

---

# Follow-up — Highest/lowest notes were unreachable (no sound at extremes)

_Date: 2026-06-17_

## Symptom

The top and bottom scale notes (and the borderline regions near the frame
edges) never made sound.

## Root cause

Pitch is read from the bounding-box **center**, which can't reach the frame
edges — it's always ≥ half a box-height (`~0.15`) from top/bottom. So the
outer `1/n` screen strips (the first/last notes) are unreachable. Confirmed
geometrically: for a 0.30-tall box the center only spans `y ∈ [0.15, 0.85]`,
but the top note needs `y ≥ (n-1)/n` and the bottom note `y < 1/n` — both
outside that range, and worse for denser scales (n=13 → need 0.923 / 0.077).

## Fix

`src/logic.py` — added `play_margin = 0.15` and `_stretch_reach(y)`, which
remaps the reachable middle band `[margin, 1-margin]` back to the full `[0, 1]`
range (clamped). Applied in the live (legacy) path only — the
experiment/benchmark path keeps the raw position (its cm-per-octave mapping
assumes 0.5 = centre). Restructured `_process_right_hand` to compute
`use_experiment` once, up front.

`src/overlay.py` — added `_band_screen_y(y_play, margin, h)` and routed the
borderlines + active-region highlight through it with the same `play_margin`,
so the drawn bands line up exactly with where notes change. The box-center
"pitch" line aligns with the bands because the mapping is linear in screen
position (margin only relocates the dead zones to the very top/bottom, where no
notes live). `margin == 0` reduces to the old `(1 - y_play) * h`.

## Verification

Headless, default chromatic scale (13 notes, [45..57]):
- Hand at TOP → note 57 (highest); hand at BOTTOM → note 45 (lowest). ✓
- Box-center pitch line falls inside the highlighted band at 5 sampled
  positions (overlay/sound stay aligned). ✓
- Theremin: full pitch bend now reachable at the extremes (+8191 / −8192). ✓
- Experiment/benchmark path keeps the raw `last_hand_y_norm` (no stretch). ✓
- `py_compile src/logic.py src/overlay.py` — OK.

## Notes

- `play_margin` (0.15) is tunable. It must be ≥ the typical box half-height for
  the extremes to be reachable; the playable region is the central
  `(1 - 2·margin)` of the frame. Very dense scales with a large (close) hand box
  may still not reach the literal first/last note — raise `play_margin` or use a
  narrower pitch range if needed.

---

# Session — 2026-06-20: Volume gate, UI features, performance

## 1. Volume gate (crisp note on/off)

`src/logic.py` — added `self.volume_gate` (0–127, default 8). In
`_process_right_hand` the area→volume is now computed *before* the note logic;
when `volume <= volume_gate` the note is fully released (`note_off`,
`current_pitch_note = -1`, CC7→0) instead of just made quiet, so fast in/out
hand moves switch the sound cleanly on/off. Exposed as a **Volume Gate** slider
in the UI; applied via `main.update_settings`.

Adversarial-review fixes (multi-agent review, 3/10 findings confirmed):
- Pitch bend is now gated on `gate_open` so theremin/continuous bends aren't
  sent/recorded into a loop while no note sounds; `_last_recorded_pitch` re-arms
  on gate close.
- Removed the redundant unconditional theremin `note_off` in
  `_silence_right_hand` (the guarded release already covers the base note).

## 2. UI (`src/ui.py`)

- **Fullscreen**: window opens maximised; `⛶ Fullscreen` button / **F11** toggle
  true fullscreen, **Esc** exits. `update_image` now scales the feed to the
  video panel (was fixed 700px).
- **Sidebar toggle**: top bar with `☰ Hide/Show Settings` and an "Always show
  settings" pin checkbox (pinned by default = legacy behaviour). Sidebar is now
  a `CTkScrollableFrame`.
- **Instrument**: GM instrument dropdown → `MidiEngine.send_program_change`
  (new), sent on channel 0 and the looper channel only on change.
  ⚠️ GarageBand ignores Program Change for instrument switching (non-standard
  MIDI impl); works with GM destinations via IAC.

## 3. Performance — make it lighter (Apple Silicon)

`src/vision_ultralytics.py` — Ultralytics defaulted to **CPU @ imgsz=640**,
pinning multiple cores. Now: auto-select **MPS** (GPU) with CPU fallback,
**imgsz=384**, camera capture **640×480**, one-time warmup. `main.py` vision
loop paced to `self.target_fps` (default **20**) so it stops producing frames
the UI never shows.

Measured (headless vision loop, `(user+sys)/real` cores):

| Config | avg cores | CPU/frame |
|---|---|---|
| Old: CPU / 640 / uncapped | 1.70 | ~272 ms |
| New: MPS / 384 / 20 fps | 0.74 | ~62 ms |

Full app incl. UI ≈ **1.15 cores** (was ~2). `self.target_fps` is the single
knob: raise toward 30 for snappier response, lower for less power.

## 4. Off-thread settings handler (dropdown lag)

`src/ui.py` — settings-change callbacks (Instrument dropdown, sliders) ran
synchronously on the Tk main thread, so each change blocked the UI on the MIDI
program change + scale regeneration. Added a daemon **worker thread**
(`_settings_worker`) fed by a `queue.Queue`: `_update_settings` still reads
widget values on the main thread (Tk isn't thread-safe) and updates the gate
label, then enqueues a full settings snapshot; the worker runs the non-UI
`on_change_callback` off-thread. Rapid changes **coalesce to the newest** queued
snapshot (each dict is a complete state snapshot, so newest-wins never drops a
setting). `on_calibrate` still runs on the main thread (it touches Tk via
`set_calib_status`).

---

# Session — Point GUI app at Homebrew Python 3.12

_Date: 2026-06-20_

## Request

> I want to point the GUI application in this project to the homebrew version of
> python. (Follow-up to wanting to remove the anaconda install.)

## Context found

- The project's `.venv` was **not** anaconda — it ran on a uv-managed standalone
  CPython 3.12.12 (`~/.local/share/uv/python/...`), so `uv run python main.py`
  was already anaconda-free. The "on anaconda" feeling came from the interactive
  shell `python` resolving to `/opt/anaconda3/bin/python` (conda base is
  auto-activated in `.zshrc`/`.bash_profile`).
- Homebrew `python@3.12`/`@3.14` ship **without `_tkinter`**, which the
  customtkinter GUI needs. Only `python-tk@3.14` was installed (covers 3.14).
  The project is pinned to 3.12 (`.python-version`, `requires-python>=3.12`) and
  mediapipe/opencv/python-rtmidi have no 3.14 wheels, so 3.12 is required.

## Change

- `brew install python-tk@3.12` — gives Homebrew's `python@3.12` its `_tkinter`.
- `pyproject.toml`: added `[tool.uv] python-preference = "only-system"` so uv
  uses Homebrew's `python@3.12` (per `.python-version = 3.12`) instead of a
  downloaded interpreter — durable across future `uv venv`/`uv sync`.
- Recreated the venv on Homebrew 3.12:
  `uv venv --python /opt/homebrew/bin/python3.12 --clear` then `uv sync`.

## Verification

`.venv/bin/python` → `base_prefix /opt/homebrew/opt/python@3.12/...` (3.12.13).
All deps import (tkinter Tk 9.0, customtkinter 5.2.2, cv2 4.13.0, mediapipe
0.10.32, mido, rtmidi 1.5.8, ultralytics 8.4.7, torch 2.9.1, PIL 12.1.0). The
project `main` import graph loads; a `customtkinter.CTk()` root was created +
destroyed OK. `uv run python …` resolves to the Homebrew venv.

## Open item

Anaconda removal (`/opt/anaconda3`, 6.9 GB; envs base/examplegame/virtualhand;
conda init blocks in `.zshrc`/`.bash_profile`) was **not** performed — awaiting
confirmation. This project no longer depends on it.

---

# Session — Integrate self-trained right-hand YOLO-OBB model (two-model setup)

_Date: 2026-06-28_

## Request

> I have trained a YOLO model that can detect my right hand. I am going to
> import that. (See `SESSION_SUMMARY_EN.md` for the training write-up.)

Implements the "Next Step — Mac App Integration" from `SESSION_SUMMARY_EN.md`:
right hand → self-trained **YOLO11n-OBB** model; left hand stays on the existing
`yolov8n-hand-pose.pt`. Both models run on the same mirrored frame.

## Changes

- **`src/vision_ultralytics.py`** — loads **two** models (pose + OBB).
  `get_frame()` now returns `(success, frame, (pose_result, obb_result))`.
  Graceful degradation: if `right_hand_obb.pt` is missing, the app still runs
  (left hand/looper) with a loud warning and `obb_result=None`. MPS→CPU
  fallback applies per model. New `pose_every_n` ctor arg (default **1** =
  both-every-frame) throttles the pose model with last-result reuse to claw
  back the doubled inference cost when needed.
- **`src/logic.py`** — `process()` consumes the `(pose, obb)` pair.
  `_extract_right_obb` reads the right hand from `obb.xywhr`/`xyxyxyxyn`
  (normalized via `orig_shape`): pitch ← center-Y, volume ← **rotation-
  invariant** area `w*h`, rotation (radians) stored in `last_right_rotation`
  for a future tilt param. `_extract_left_pose` reads the left hand/keypoints.
  `_process_right_hand(y_center, area)` replaces the old bbox-based signature.
- **`src/overlay.py`** — draws the **rotated** OBB polygon for the right hand
  (`_draw_obb`), falling back to the axis-aligned box.

## Model file

Default path `right_hand_obb.pt` in the project root (resolved relative to repo
root). User copies it from Drive `MyDrive/midi_hand_model/right_hand_obb.pt`.
Not yet present in the repo at time of writing → right hand latent until added.

## Verification

- Synthetic end-to-end test (no camera/model): pitch, rotation-invariant
  volume, rotation capture, fist→looper, right-side-box rejection, phantom
  drop, note release, overlay render — **all pass**.
- Adversarial multi-agent review (4 confirmed findings, all fixed):
  1. **Spurious looper toggle** on a 1-frame fist-detection dropout (also a
     latent pre-existing bug) → fist-release **debounce**
     (`_fist_release_frames=3`).
  2. **Missing `cx≥0.5` guard** on the OBB right hand (asymmetric with the left
     extractor) → mirrored the guard so a left-half OBB can't play pitch.
  3. **One per-frame exception bricked the vision thread** (whole loop in one
     try/except with `cleanup()` in `finally`) → per-frame try/continue, bail
     only after 60 consecutive errors.
  4. **Two inferences/frame > 50 ms budget** (single-model baseline already
     ~62 ms) → `pose_every_n` cadence knob; recommend setting it to 2–3 once
     the model is in and FPS is measured.
- Post-fix synthetic suite (incl. debounce + cx-guard tests) — **all pass**.

## Open / next

- Drop `right_hand_obb.pt` into the project root and run `uv run python main.py`.
- Wire the stored `last_right_rotation` to a MIDI parameter (tilt) — deferred.
- Consider `pose_every_n=2` if effective FPS is below the 20 fps target.
