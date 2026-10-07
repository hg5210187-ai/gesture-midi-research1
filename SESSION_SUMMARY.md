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

- Drop `right_hand_obb.pt` into the project root and run `uv run python main_obb.py`.
- Wire the stored `last_right_rotation` to a MIDI parameter (tilt) — deferred.
- Consider `pose_every_n=2` if effective FPS is below the 20 fps target.

---

# Session — Split into two independent programs (working vs OBB)

_Date: 2026-06-28_

## Request

> The [OBB] hand is not detected enough. Although I want to improve, I need a
> working program right now. Therefore can you separate the files? The one that
> imported my trained model and the previous one which worked. Make the python
> code to start each program different.

The trained OBB model under-detects, so the two pipelines are split into two
fully independent, separately-launched programs that share no vision/logic code
(improving one can't break the other). MIDI, looper, UI, scales, and overlay
stay shared.

## Layout

| Program | Entry | Vision | Logic | Hands |
|---|---|---|---|---|
| **Working** (reliable) | `main.py` | `src/vision_ultralytics.py` | `src/logic.py` | one pose model → both hands |
| **OBB** (experimental) | `main_obb.py` | `src/vision_obb.py` | `src/logic_obb.py` | trained OBB → right; pose → left |

Start commands (different per program, as requested):
- Working: `uv run python main.py`
- OBB:     `uv run python main_obb.py`

## How it was done

- `src/logic.py` + `src/vision_ultralytics.py` were **restored to the original
  single-pose-model code** (the version that worked before the OBB session).
- The OBB two-model code was copied verbatim into `src/logic_obb.py` +
  `src/vision_obb.py`; `main_obb.py` (a copy of the OBB `main`) imports those.
- `src/overlay.py` stays shared — it draws the rotated OBB when
  `last_right_obb` is present (OBB program) and falls back to the axis-aligned
  box otherwise (working program), so one overlay serves both.
- `run_benchmark.py` imports `src.vision_ultralytics`/`src.logic`, so it now
  rides the **working** single-model path again (its original design).

## Verification

- All entry points + modules compile.
- Wiring asserted: `main.py`→`src.vision_ultralytics`/`src.logic`;
  `main_obb.py`→`src.vision_obb`/`src.logic_obb`. Constructor signatures differ
  (working `model_variant`; OBB `pose_model`/`obb_model`/`pose_every_n`).
- Behavioral synthetic tests pass for BOTH: working (single result → right hand
  from bbox plays a note, left fist toggles looper) and OBB (`(pose,obb)` tuple
  → note + rotation stored).
- `run_benchmark.py` imports resolve to the working module. Reference map:
  working modules ← `main.py`+`run_benchmark.py`; OBB modules ← `main_obb.py`.

## Follow-up — OBB phantom rejection (curtains detected as a hand)

`main_obb.py` fired on textured background (curtains/blinds) with no hand up,
playing a note. Added phantom-rejection gates in `src/logic_obb.py`
`_extract_right_obb` (OBB program only — `src/logic.py` untouched):
- `min_confidence` 0.5 → **0.6** (kept modest; the model under-detects real
  hands, so a high value hurts true positives).
- new `max_box_area` = **0.30** — reject boxes too large to be a real hand at
  playing distance (calibrated min area ≈ 0.04). Doesn't hurt recall (real
  hands are small). `None` disables.
- `OBB_DEBUG=1` env var prints `conf/area/cx/cy` per detection for tuning:
  `OBB_DEBUG=1 uv run python main_obb.py`.

Defaults are a first cut — exact thresholds need the OBB_DEBUG numbers for the
phantom vs a real hand. If they overlap on both conf and area, the real fix is
more negative-sample training data (already planned).

## Follow-up 2 — phantom also on main.py (pose model); live confidence + sustained fist

User confirmed the phantom also occurs on `main.py` (the WORKING program), so the
`yolov8n-hand-pose.pt` model itself hallucinates a hand on background/face. A
multi-agent investigation (5 agents) confirmed: the box is a **model false
positive, not an assignment/overlay bug**; the stuck `LOOP: RECORDING` **is** a
bug — a one-frame phantom fist tripped the looper toggle. Fixes applied to BOTH
programs:
- **Live "Detection Confidence" slider** (`src/ui.py`, default 0.60, range
  0.30–0.90) → `settings["detection_confidence"]` → `logic.min_confidence` in
  both `main.py` and `main_obb.py` `update_settings`. The user drags it up until
  the phantom disappears, in real time, no restart. Shared UI; both logic
  modules already have `min_confidence`, so no guard needed.
- **Sustained-fist gate** (`src/logic.py` + `src/logic_obb.py`
  `_process_left_hand`): the fist must be held `_fist_confirm_frames`=3
  consecutive frames before it toggles the looper, so a one-frame phantom can't.
  `_fist_frames` resets on open hand / hand-absent / no-hands.
- **`HAND_DEBUG=1`** env logging added to `src/logic.py` (mirrors OBB_DEBUG):
  prints every raw pose detection's conf/area/cx/cy for tuning.

Verified: compile; sustained-fist (1–2 frames ignored, 3rd toggles, hold doesn't
re-fire) in both modules; confidence wiring; working-version note regression.
Honest limit: if the phantom and a real hand only separate at a confidence so
high that the real hand stops registering, the model needs more negative-sample
training data.

---

# Session — Third program: MediaPipe detector (phantom-resistant)

_Date: 2026-06-28_

## Request

> I think it is a risk to change it to media pipe. So, can you make a file that
> uses media pipe? Leaving a file of also a YOLOv8 hand pose.

The YOLOv8 hand-pose model hallucinates hands on background/face (root cause of
the phantom notes). MediaPipe's palm-detector pipeline is far more robust. Added
as a THIRD, fully separate program; the YOLO version is untouched.

## Layout (now three programs)

| Program | Entry | Detector | Vision | Logic |
|---|---|---|---|---|
| YOLO hand-pose | `main.py` | `yolov8n-hand-pose.pt` | `src/vision_ultralytics.py` | `src/logic.py` |
| Trained OBB | `main_obb.py` | `right_hand_obb.pt` + pose | `src/vision_obb.py` | `src/logic_obb.py` |
| **MediaPipe** (new) | `main_mediapipe.py` | MediaPipe HandLandmarker | `src/vision_mediapipe.py` | reuses `src/logic.py` |

Run: `uv run python main_mediapipe.py`

## How it was done

- `src/vision_mediapipe.py` wraps the MediaPipe **Tasks** `HandLandmarker` and
  emits a **duck-typed Ultralytics Results** (`_Boxes.xyxyn/.conf`,
  `_Keypoints.xyn`, `.cpu().numpy()`), so the EXISTING, tested `GestureLogic` +
  overlay run unchanged — only the detector differs (no new logic file).
- Each hand -> bbox from landmark extents (pitch/volume) + 21 landmarks (fist).
  Frame mirrored; left/right still by center-x.
- `main_mediapipe.py` = copy of `main.py` with the vision import/instantiation
  swapped; reuses `from src.logic import GestureLogic`. Zero changes to the YOLO
  files.

## Environment note (important)

This build of `mediapipe==0.10.32` ships **only the Tasks API** — the legacy
`mediapipe.solutions.hands` is absent (so the old `src/vision.py` would fail
here too; that explains earlier confusion about the "older version"). The Tasks
HandLandmarker needs a model bundle: **`hand_landmarker.task`** (7.8 MB,
downloaded to the project root from
`storage.googleapis.com/mediapipe-models/.../hand_landmarker.task`). The vision
module raises a clear error with the curl command if it's missing.

## Verification

- Compile; HandLandmarker VIDEO pipeline creates + processes frames; **0 hands
  on a blank frame** (no false positive). MediaPipe duck-typed result plays a
  note through `GestureLogic`; fist (held 3 frames) toggles looper. YOLO files
  unchanged.

## Follow-up — removed the Detection Confidence slider

Per user request, the live "Detection Confidence" slider (added in "Follow-up 2"
above) was **removed**: the label/slider/hint from `src/ui.py`, the
`detection_confidence` key from its settings dict, and the
`min_confidence = settings.get(...)` wiring from `main.py`/`main_obb.py`/
`main_mediapipe.py`. The internal `min_confidence` defaults stay (0.5 YOLO / 0.6
OBB) — just no longer live-tunable. (MediaPipe is now the phantom-resistant
path, so live confidence tuning was no longer needed.) The sustained-fist gate
and `HAND_DEBUG`/`OBB_DEBUG` logging remain.

---

# Session — Cubase output + low-latency vision loop (2026-10-05)

## Request

Switch the MIDI destination from GarageBand to Cubase, check latency and side
effects, then cut camera + inference + loop latency (goal stated: < 10 ms whole
system). Entry point in use: `main_obb.py`.

## Findings

- The app was already sending to `IAC Driver Bus 1` (only port present; the
  "GarageBand" name match never fired). Cubase LE AI Elements 13 + HALion Sonic
  are installed and can listen to the same bus. IAC loopback: 0.15 ms median.
- **< 10 ms is not reachable with a camera**: the camera delivers 30 fps
  (33.3 ms/frame; a 60 fps request is refused). Options given: A software only,
  B + 120 fps USB camera (est. ~12 ms avg), C non-camera sensor (only way under
  10 ms). **A was implemented**; `CAMERA_INDEX`/`CAMERA_FPS` make B a config change.

## Changes

- `src/midi_engine.py`: `MIDI_PORT` env override; IAC -> virtual -> first port;
  GarageBand branch removed.
- `src/vision_obb.py`: capture thread keeps only the newest frame; `get_frame()`
  blocks for a NEW frame and runs only the right-hand OBB; left-hand pose runs
  in its own worker thread (`pose_every_n` removed). OBB on **CPU**, pose on
  **MPS** — sharing one device doubled OBB time (~14 -> ~28 ms), and OBB on MPS
  stalled when the UI redrew. `OBB_IMGSZ` env; `last_capture_ts`/`last_infer_ms`.
- `main_obb.py`: 20 FPS cap removed (loop runs at camera rate);
  `MIDI_LATENCY_DEBUG=1` readout; vision thread joined before exit (exiting
  mid-inference crashed torch).
- `src/logic_obb.py`: fist confirm/release 3 -> 5 frames (same ~150 ms at 30 fps).

## Measured (M4 Air, built-in camera)

| | Before | After (384) | After (`OBB_IMGSZ=256`) |
|---|---|---|---|
| Loop rate | 16.5 fps | 30 fps | 30 fps |
| Frame delivered -> MIDI | ~56 ms (incl. wait for frame) | ~18 ms mean | ~12 ms mean |

Occasional single-frame spikes of 50-100 ms remain. Figures exclude the
camera's internal delay and Cubase's audio buffer.

## Not verified

- No sound test in Cubase and no hands-in-frame test of gestures/looper were
  run in this session; 256 px accuracy for pitch/volume is unchecked.
- Cubase side effects to expect: Program Change now works (HALion Sonic);
  looper is on channel 2 (track channel `1` merges, `Any` needs a slot-2 sound);
  CC7 every frame fills recordings; ±2 semitone bend range assumed.

## Follow-up — deployed `hbb-s-fold1-320.pt` in `main_obb.py`

- The model is a YOLO26s **detect** model (plain boxes, trained at 320) with
  classes `thumbout` / `openhand` / `closedhand`, so ONE model now covers both
  hands: right-hand box -> pitch (Y-center) + volume (area); left-hand
  `closedhand` -> looper toggle. The pose model and its worker thread are no
  longer used by `main_obb.py`.
- New `src/vision_hbb.py` (`HbbVision`; env `HBB_MODEL`, `HBB_IMGSZ`,
  `HBB_DEVICE`). Camera thread moved to a shared `LatestFrameCamera` base in
  `src/vision_obb.py`; the two-model `UltralyticsVision` is kept there, unused.
- `src/logic_obb.py`: `process()` takes a bare detect result ->
  `_extract_gesture_boxes`; `_process_left_hand` now takes the fist bool.
- Measured in the real app on the M4 (PyTorch): MPS ~16-22 ms inference,
  ~18-24 ms frame->MIDI; CPU ~22-24 ms. **Not 7.93 ms** — `results.jsonl` shows
  that study ran on an RTX 4090. This is about the same as the previous
  two-model setup (~18 ms), not faster. CoreML export is the untried next step
  (`coremltools` is not installed).
- Verified with synthetic boxes (note on/off, CC7, 5-frame fist toggle,
  cooldown, confidence/area gates, overlay) and a 24 s app run per device. Not
  verified with real hands or sound. `min_confidence` 0.6 / `max_box_area` 0.30
  were tuned for the old OBB model; volume needs re-calibrating (box shape differs).

## Follow-up — Core ML export of `hbb-s-fold1-320` (now the default)

- `uv add coremltools "numpy<=2.3.5"`. NumPy 2.4.1 broke the coremltools
  export ("only 0-dimensional arrays can be converted") and made Ultralytics'
  Core ML loader attempt a pip AutoUpdate at startup; 2.3.5 fixes both.
- Export: `uv run yolo export model=hbb-s-fold1-320.pt format=coreml imgsz=320`
  -> `hbb-s-fold1-320.mlpackage` (18 MB). `HbbVision` uses it when present,
  else the `.pt`.
- Measured on the M4: model alone ~2.3 ms/frame (PyTorch 16-22 ms). In the real
  app at 30 fps: inference 6-7.6 ms mean, frame delivered -> MIDI 8-9.6 ms mean
  (was ~18 ms). Occasional single-frame spikes of 50-120 ms remain.
- `.pt` vs Core ML on 300 dataset images at conf >= 0.6: same count+classes on
  278; Core ML found fewer boxes (44 vs 53) and confidences differ by 0.07 on
  average (max 0.24), box coordinates by <= 0.025. If hands drop out, lower
  `min_confidence` (0.6) in `logic_obb.py`; `OBB_DEBUG=1` prints each detection.
- Still not verified with real hands or sound.


---

# Session — New instrument repo + research repo rename (2026-10-07)

## Request

Does `midi-gesture-detection` on GitHub include the process of creating the
instrument from scratch? If not, create a repo for that named
`gesture-midi-instrument`, and rename the other to `midi-gesture-research`.

## Finding

No. That repo is the detector study only (dataset protocol, training drivers,
evaluation, latency, figures). It has no camera, gesture->MIDI, looper or UI
code. The instrument existed only in this folder (and in the private
`Virtual_Hand-` remote, last pushed 2026-03-29).

## Done

- **New repo `hg5210187-ai/gesture-midi-instrument`** — created **PRIVATE**,
  one commit, pushed. Local folder: `~/Documents/gesture-midi-instrument`
  (a curated copy; this folder and its `Virtual_Hand-` remote were not touched).
  - Code copied byte-for-byte from here: the four `main*.py`, all of `src/`,
    the benchmark harness, `YOLO26*-ver`, `LICENSE`, `.python-version`.
  - `pyproject.toml`: name/description changed; `uv.lock` re-locked (only the
    root package name moved, no version changed).
  - New docs: `README.md`, `docs/BUILD_FROM_SCRATCH.md` (11 dated stages from
    the 2026-03-29 commit to the Core ML path, plus a from-zero build order),
    `docs/ARCHITECTURE.md`, `docs/MODELS.md`.
  - New `tests/check_headless.py`: synthetic detections through all three
    logic modules, no camera/MIDI/UI. Passes.
  - Left out on purpose: all weights (`*.pt`, `*.mlpackage`, `*.task`),
    `logs/` (participant data), `runs/`, the screenshot, `Indigo Goals.pdf`,
    and the raw session logs.
- **Renamed** `midi-gesture-detection` -> `midi-gesture-research` (still
  public; the old URL redirects). Updated `origin` in its local clone,
  `~/Documents/MIDI-Dataset/v2-study`.

## Open

- The new repo is private until reviewed. To publish:
  `gh repo edit hg5210187-ai/gesture-midi-instrument --visibility public --accept-visibility-change-consequences`
- No weights are published, so only `main_mediapipe.py` runs from a fresh
  clone (public Google model). Decide whether to attach `hbb-s-fold1-320`
  (and the rotation models) to a GitHub release.
- The research README does not link to the instrument repo yet.
- Code changed here from now on has to be copied to the new repo by hand.
