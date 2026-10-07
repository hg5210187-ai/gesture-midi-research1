# Architecture

How the parts fit, what each one promises the next, and every number that can be tuned.

## The pipeline

```
camera ──► vision_*.py ──► logic*.py ──► midi_engine.py ──► MIDI port ──► DAW / synth
            (frame, result)     │   ▲
                 │              ▼   │
                 │           looper.py        records live events, replays them on channel 2
                 ▼
            overlay.py ──► frame queue (size 2) ──► ui.py
```

One frame goes through it like this: the vision module returns a mirrored frame and a
detection result; the logic turns the result into MIDI messages and remembers what it saw;
the overlay draws that memory onto the frame; the frame is queued for the window.

## Which files make each program

| Program | Vision class | Logic module | What the logic receives |
|---|---|---|---|
| `main_mediapipe.py` | `MediaPipeVision` (`src/vision_mediapipe.py`) | `src/logic.py` | An object shaped like an Ultralytics pose result |
| `main.py` | `UltralyticsVision` (`src/vision_ultralytics.py`) | `src/logic.py` | Ultralytics pose result: boxes + 21 keypoints per hand |
| `main_rotate.py` | `Yolo26Vision` (`src/vision_yolo26.py`) | `src/logic_rotate.py` | Ultralytics OBB result: oriented boxes + class |
| `main_obb.py` | `HbbVision` (`src/vision_hbb.py`) | `src/logic_obb.py` | Ultralytics detect result: boxes + class |

`src/midi_engine.py`, `src/looper.py`, `src/scales.py`, `src/ui.py` and `src/overlay.py` are
shared by all four.

### The two contracts

**Vision.** `get_frame()` returns `(success, frame, result)`. `frame` is BGR, mirrored, and
not drawn on. `release()` closes the camera.

**Logic.** `process(result)` sends MIDI through the engine it was constructed with and leaves
its view of the frame in attributes that the overlay reads:

| Attribute | Meaning |
|---|---|
| `last_right_bbox`, `last_left_bbox` | Normalised `[x1, y1, x2, y2]`, or `None` |
| `last_right_obb` | Four normalised corners of a rotated box, where the model gives one |
| `last_hand_y_norm` | Right-hand height after the reach stretch, 0 = bottom, 1 = top |
| `current_pitch_note` | The MIDI note sounding now, or `-1` |
| `scale`, `scale_type`, `play_margin` | What the pitch bands are drawn from |
| `show_borderlines`, `show_pitch` | The two overlay checkboxes |
| `pitch_from_rotation`, `last_angle_t`, `last_angle_deg`, `rotation_mode` | Rotation instrument only: draw the dial instead of the height guides |

A new detector needs one new vision file that honours the first contract. A new way of
playing needs one new logic file that honours the second. `src/vision_mediapipe.py` shows the
cheapest form of the first: it wraps MediaPipe's output in four tiny classes so the existing
logic cannot tell the difference.

## Threads

| Thread | What runs in it |
|---|---|
| Main | The Tk event loop. Every 40 ms `update_ui_video` takes one frame from the queue, shows it, and refreshes the looper label. Only this thread touches widgets |
| Vision (daemon) | `MIDIApp.vision_loop`: `get_frame` → `logic.process` → `render_overlay` → queue. Each frame is inside its own `try`; the loop stops after 60 consecutive errors |
| Settings worker (daemon) | Applies settings from the UI to the logic and the MIDI engine. If several changes are queued, only the newest is applied |
| Looper playback (daemon) | Exists only while the looper is `PLAYING` |
| Camera capture (daemon) | `main_obb.py` only. Keeps the newest mirrored frame; `get_frame` blocks until it is newer than the last one returned |

The frame queue holds two frames. When it is full the vision thread drops the frame rather
than wait, so the picture cannot fall behind the hands.

`main.py`, `main_mediapipe.py` and `main_rotate.py` pace the vision loop to 20 fps
(`target_fps`). `main_obb.py` has no pacing and runs at the camera's rate.

## Telling the hands apart

No program uses true handedness. The frame is mirrored, so the player's right hand appears on
the right.

- `src/logic.py`: detections are sorted by box centre X. With two or more, the leftmost is
  the left hand and the rightmost the right. With one, the half of the picture decides.
- `src/logic_obb.py`, `src/logic_rotate.py`: the rightmost box **on the right half** is the
  right hand and the leftmost box **on the left half** is the left hand. A box on the wrong
  half never takes the other hand's role.

## Right hand → pitch and volume

Height programs (`logic.py`, `logic_obb.py`):

```
y      = 1 − box_centre_y                                    up = higher
y      = clamp((y − play_margin) / (1 − 2·play_margin))      stretch the reachable band
note   = scale[int(y · len(scale))]                          scale mode
bend   = int(y · 16383) − 8192, held note C4                 Theremin mode

volume = clamp((area / area_min − 1) · gain) · 127           sent as CC 7
gate   = volume > volume_gate                                closed → note off
```

Rotation program (`logic_rotate.py`):

```
a      = recentre and unwrap the box angle                   vertical hand = 0, range ±90°
t      = clamp((a − angle_min) / (angle_max − angle_min))
note   = scale[int(t · len(scale))]                          Scale mode
exact  = scale[0] + t · (scale[−1] − scale[0])               Continuous mode:
note   = round(exact), bend = (exact − note) · 100 / 200 · 8192
```

Volume and the gate are the same in every program.

## MIDI that leaves the program

| Message | Sent when | Channel |
|---|---|---|
| Note on, velocity 64 | The gate is open and the note changes | 1 |
| Note off | The note changes, the gate closes, or the right hand disappears | 1 |
| Control change 7 | Every frame that has a right hand, value 0–127 | 1 |
| Pitch bend | Theremin mode and rotation Continuous mode, while the gate is open. The rotation program also sends one bend back to centre when the note stops | 1 |
| Program change | The Instrument menu changes | 1 and 2 |
| Everything the looper recorded | During loop playback | 2 |

Loudness is carried by CC 7, not by velocity. The Theremin mode sends the full bend range, so
how far the pitch travels depends on the bend range set in the synth; at the common default
of ±2 semitones it covers C4 ± 2 semitones. The Continuous rotation mode assumes ±2
semitones.

**Port selection** (`MidiEngine.__init__`): a port whose name contains `MIDI_PORT`, if set;
else the first port containing `IAC`; else a new virtual port named
`Indigo Gesture Controller`; and if that fails, the first port available.

## Looper

`IDLE → RECORDING → PLAYING → IDLE`, one step per confirmed left-hand fist.

- Events are stored as `(seconds since recording began, type, data)`.
- Recorded types: note on, note off, CC, pitch bend.
- Playback sleeps on a `threading.Event` between events, so stopping is immediate.
- All sounding notes are turned off at each loop boundary and when playback stops.
- A recording with no events or shorter than 0.3 s goes back to `IDLE`.

## Constants

| | `logic.py` | `logic_obb.py` | `logic_rotate.py` |
|---|---|---|---|
| Used by | `main.py`, `main_mediapipe.py` | `main_obb.py` | `main_rotate.py` |
| `min_confidence` | 0.7 | 0.6 | 0.5 |
| `max_box_area` (fraction of frame) | none | 0.30 | 0.35 |
| `play_margin` | 0.15 | 0.15 | not used |
| `volume_gain` default | 2.0 | 2.0 | 2.0 |
| `volume_gate` default | 8 | 8 | 8 |
| Fist test | fingertip-to-palm ratio < 0.65 | class `closedhand` | class `closedhand` |
| Frames a fist must be held | 3 | 5 | 3 |
| Frames absent before the fist is released | 1 | 5 | 3 |
| Cooldown between looper toggles | 1.0 s | 1.0 s | 1.0 s |
| Stationary-box rejection | 30 sightings within 0.012 | none | none |
| Smallest bend change recorded in a loop | 200 | 200 | 100 |
| Smallest volume change recorded in a loop | 5 | 5 | 5 |

| Elsewhere | Value | Where |
|---|---|---|
| Camera capture size | 640 × 480 | every vision module in use |
| Inference size | 384 (`main.py`), 640 (`main_rotate.py`), 320 (`main_obb.py`) | vision modules |
| Vision loop pacing | 20 fps, or camera rate in `main_obb.py` | `main*.py` |
| Video refresh | every 40 ms | `main*.py` |
| Frame queue | 2 | `main*.py` |
| Consecutive frame errors before stopping | 60 | `main*.py` |
| Shortest loop kept | 0.3 s | `src/looper.py` |
| Most band lines drawn | 24 | `src/overlay.py` |

## Leftovers

Kept because removing them was never the task at hand:

- **Number of Tracks slider.** From the first version, which selected GarageBand tracks. Its
  value still enters the settings dictionary as `tracks`; nothing reads it.
- **`src/vision.py`.** The original MediaPipe backend. It uses `mediapipe.solutions`, which
  the pinned MediaPipe no longer has.
- **The two-model backend.** `UltralyticsVision` in `src/vision_obb.py` (a right-hand
  oriented-box model plus a left-hand pose model) and the matching `_extract_right_obb` and
  `_extract_left_pose` in `src/logic_obb.py`. No program creates it now; `main_obb.py` uses
  `HbbVision`, which reuses only the camera thread from that file.
- **The benchmark path in `map_pitch`.** `src/logic.py` and `src/logic_obb.py` can map height
  in centimetres per octave when `experiment_sensitivity` or `experiment_scale_type` is set.
  Only `run_benchmark.py` sets them.
