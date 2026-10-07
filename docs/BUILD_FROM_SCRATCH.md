# Building the instrument from scratch

How a webcam became a playable MIDI instrument, in the order it happened, and then the order
to build it in if you start from zero.

The first part is a record. Each stage says what was added, the file that holds it today,
and what went wrong. The training procedure of every model is set out separately, in
[MODEL_PROCEDURES.md](MODEL_PROCEDURES.md). Dates come from the commit history and the project log; where only a
file date survives, the stage says so. The second part is a recipe. The third lists what was
still open at the last entry.

| Stage | When | What changed |
|---|---|---|
| [1](#stage-1--the-first-complete-instrument) | 2026-03-29 | First complete instrument: keypoints in, MIDI out, GarageBand driven by keystrokes |
| [2](#stage-2--a-looper-inside-the-app) | April 2026 | Looper inside the app replaces the GarageBand control |
| [3](#stage-3--a-playability-benchmark) | 2026-04-30 | Benchmark of 3 pitch sensitivities × 4 scale types |
| [4](#stage-4--the-right-hand-moves-from-keypoints-to-its-box) | 2026-06-16 | Right hand read from its bounding box; volume calibration; own overlay |
| [5](#stage-5--phantom-hands-a-visible-pitch-reachable-extremes) | 2026-06-17 | Confidence gate, pitch feedback on the video, extreme notes made reachable |
| [6](#stage-6--a-volume-gate-a-better-window-a-lighter-loop) | 2026-06-20 | Volume gate, fullscreen UI, instrument menu, CPU load roughly halved |
| [7](#stage-7--the-first-self-trained-detector) | 2026-06-28 | A right-hand oriented-box model, trained on 209 images |
| [8](#stage-8--three-programs-instead-of-one) | 2026-06-28 | The pipelines split into separate programs; MediaPipe added |
| [9](#stage-9--one-model-for-both-hands-and-pitch-from-rotation) | 2026-07-17 to 08-02 | One model with gesture classes; the rotation instrument |
| [10](#stage-10--choosing-the-detector-by-measurement) | August 2026 | The detector is measured twice; the second, published study picks the model |
| [11](#stage-11--cut-the-latency) | 2026-10-05 | Latest-frame capture, one model, Core ML: about 56 ms down to under 10 ms |

---

## The idea

Camera in, MIDI out. The program never synthesises audio. It opens a MIDI port and sends
note, volume and pitch-bend messages; a DAW or synth turns them into sound. That one decision
keeps the instrument small: every sound a DAW has is available, and the program's whole job
is to turn two hands into a few numbers quickly and without mistakes.

Two hands, two jobs. The right hand is the voice: where it is decides the note, how close it
is decides the volume. The left hand is the transport: it starts and stops a loop.

---

## Part 1 — The build, as it happened

### Stage 1 — The first complete instrument

**2026-03-29.** The first committed code already had every part the instrument still has.

| Part | File | What it did |
|---|---|---|
| MIDI out | `src/midi_engine.py` | Opened a port (one named GarageBand, else the IAC Driver, else a new virtual port); sent note on / off, CC, pitch bend |
| Scales | `src/scales.py` | Built a list of MIDI notes from a root, a scale type, a start octave and a number of octaves. The UI offered Chromatic, Pentatonic and Theremin |
| Vision | `src/vision_ultralytics.py` | Read the webcam, mirrored the frame, ran a YOLOv8n hand-pose model (21 keypoints per hand) |
| Logic | `src/logic.py` | Turned keypoints into MIDI |
| UI | `src/ui.py`, `main.py` | A CustomTkinter window; vision in a background thread, Tk on the main thread, a two-frame queue between them |

`src/vision.py`, a MediaPipe backend, was already in that commit and already unused: the
detector had been swapped for the YOLO hand-pose model before the first commit. The file is
still in the repository as the starting point.

How the hands were read then:

- **Hands were told apart by position.** Sorted by wrist X; the leftmost is the left hand.
  With one hand in view, the half of the picture decides.
- **Pitch** came from one keypoint, the middle-finger knuckle (landmark 9). Its height
  indexed into the scale: `note = scale[int(y * len(scale))]`. A note was sent only when the
  index changed, with a note-off for the previous one.
- **Theremin mode** held C4 and sent the height as a 14-bit pitch bend instead.
- **Volume** came from a pinch: the distance from thumb tip to index knuckle, as CC 7.
- **The left hand drove GarageBand.** The picture was cut into three vertical zones, one per
  track. An open hand started recording, a fist stopped it, a pointing finger played, a
  victory sign deleted. Each gesture ran AppleScript that brought GarageBand to the front and
  typed a key (`r`, space, the arrow keys, delete).

The video showed the model's own rendering (`result.plot()`): boxes, 21 keypoint dots and the
skeleton.

### Stage 2 — A looper inside the app

**April 2026** (file date 2026-04-15). The GarageBand keystrokes were removed and
`src/looper.py` took their place.

**Decision: loop MIDI, not audio, and do it in the program.** The keystroke control worked
with one application and had to bring it to the front each time. A looper that records the
instrument's own MIDI messages works with any destination.

- Three states, cycled by one gesture: `IDLE → RECORDING → PLAYING → IDLE`.
- While recording, every note on / off, volume change and pitch bend is stored with its time
  since the recording began.
- Playback runs in its own thread and sends the stored events on a **separate MIDI channel**
  (channel 2), so the loop and the live hand do not cut each other's notes.
- At each loop boundary every sounding note is turned off, so nothing sticks.
- A recording shorter than 0.3 s is thrown away.
- Volume changes smaller than 5 and bend changes smaller than 200 are not recorded, or one
  second of hand movement would fill the loop with hundreds of events.

The left-hand gesture became a single **fist**, read from the keypoints: the mean distance of
the four fingertips from the palm centre, divided by the hand's size, below 0.65.

```python
hand_size = norm(wrist - middle_knuckle)            # landmarks 0 and 9
palm      = (wrist + middle_knuckle) / 2
fist      = mean(norm(tip - palm) for tip in fingertips) / hand_size < 0.65
```

It fires on the rising edge only, with a one-second cooldown.

### Stage 3 — A playability benchmark

**2026-04-30.** Before changing how pitch was mapped, the mapping was measured.

`run_benchmark.py` runs the real camera and logic through 12 conditions: three sensitivities
(60, 30 and 15 cm of hand travel per octave) times four scale types (chromatic, diatonic,
pentatonic, continuous). In each trial a target note is shown and the player has to reach it
and hold within ±50 cents for 0.5 s. `experiment_session.py`, `task_engine.py` and
`run_dry_run.py` are a fuller task set (target, sustain, melody) driven by a simulated hand,
which runs without a camera.

For this, `GestureLogic.map_pitch` gained a second path that maps height to pitch in
centimetres per octave. The live instrument keeps the original mapping; the benchmark sets
`experiment_sensitivity` and `experiment_scale_type` to switch paths.

The logs hold participant data and are not in this repository.

### Stage 4 — The right hand moves from keypoints to its box

**2026-06-16.** The right hand stopped using keypoints at all.

- **Pitch** from the vertical midpoint of the hand's bounding box.
- **Volume** from the box **area** relative to a calibrated minimum, replacing the pinch:

  ```python
  volume = int(clip((area / area_min - 1.0) * gain, 0.0, 1.0) * 127)    # CC 7
  ```

  Moving the hand toward the camera makes the box bigger and the sound louder. A **Calibrate
  Min** button captures the current area as silence; a **Volume Sensitivity** slider sets
  `gain`. Until the button is pressed, `area_min` is seeded from the first frame, so volume
  is always defined and nothing divides by zero.
- **An own overlay**, `src/overlay.py`, replaced `result.plot()`: a box per hand, the scale's
  band boundaries, the current note.

**Problem: "remove keypoints to save processing" did not save inference.** The box and the
keypoints come out of the same single forward pass, and the left-hand fist still needed
keypoints. Inference cost was unchanged. The saving came from elsewhere: `result.plot()` had
been drawing about 42 circles and 40 line segments on a copy of every frame.

This stage is what made everything after it possible. Once the right hand needed only a box,
it no longer needed a keypoint model.

### Stage 5 — Phantom hands, a visible pitch, reachable extremes

**2026-06-17.** Three faults found by playing.

**Problem: sound with no hand in view.** The hand-pose model has one class, so any box on the
right half of the picture was the right hand, and at YOLO's default confidence of 0.25 the
model found "hands" in a face, a shoulder, the background. Fix: drop detections below
`min_confidence` before any hand gets a role. It started at 0.5 and is 0.7 in `src/logic.py`
today.

**Problem: the pitch could not be seen.** The readout was small text in a corner. It became
three guides, drawn when *Show Current Pitch* is on: the band of the scale the hand is in,
tinted and named; a yellow line across the frame at the exact height where pitch is sampled;
and a large box with note and frequency, `A4 (440 Hz)`. A looper badge was added top-left,
grey / red / green, because the player watches the video, not the settings panel.

**Problem: the highest and lowest notes never sounded.** Pitch is read from the box
*centre*, and the centre of a box cannot reach the edge of the frame: it is always at least
half a box-height away. For a box 0.30 of the frame tall, the centre spans only 0.15 to 0.85,
so the top and bottom strips of the scale were out of reach. Fix: stretch the reachable
middle band back to the full range.

```python
y    = 1.0 - (y1 + y2) / 2.0                     # box centre, flipped so up = higher
y    = min(1.0, max(0.0, (y - 0.15) / 0.70))     # play_margin = 0.15 at each end
note = scale[min(len(scale) - 1, int(y * len(scale)))]
```

The overlay uses the same margin, so the drawn bands sit exactly where the notes change.

### Stage 6 — A volume gate, a better window, a lighter loop

**2026-06-20.**

**Volume gate.** Lowering the volume to zero left the note held and merely silent, so fast
in-and-out movements blurred. Now, when the volume is at or below the gate (default 8 of
127), the note is released: note off, and no pitch bend is sent or recorded while nothing
sounds. Pulling the hand back cuts the sound.

**Window.** Opens maximised; `F11` for fullscreen; a hideable settings sidebar; the video
scales to the panel. An **Instrument** menu sends a MIDI Program Change on both the live and
the looper channel. GarageBand ignores Program Change; General MIDI destinations follow it.

**Problem: the app pinned the CPU.** Ultralytics defaulted to the CPU at 640 px. Moving
inference to the GPU through MPS, shrinking it to 384 px, capturing at 640×480 and capping
the loop at 20 fps roughly halved the load:

| Configuration | Average cores | CPU per frame |
|---|---|---|
| CPU, 640 px, uncapped | 1.70 | ~272 ms |
| MPS, 384 px, 20 fps | 0.74 | ~62 ms |

**Problem: the Instrument menu lagged.** Settings callbacks ran on the Tk thread, so each
change blocked the window on a MIDI send and a scale rebuild. Widget values are still read
on the main thread, because Tk is not thread-safe, but the work now runs on a worker thread
fed by a queue. Rapid changes collapse to the newest one.

### Stage 7 — The first self-trained detector

**2026-06-28.** With the right hand needing only a box, a small detector could replace the
pose model for it. An **oriented** box was chosen over a plain one so the hand's tilt angle
would be available later as another control.

- 209 right-hand images, varied in distance, position, rotation, hand shape, background and
  lighting, with some hand-free frames as negatives.
- Annotated as polygons in Roboflow, exported as oriented boxes.
- Trained in Google Colab on a T4 from `yolo11n-obb.pt`:

  ```python
  model.train(data=..., epochs=80, imgsz=384, batch=16, patience=20,
              cos_lr=True, degrees=15.0, translate=0.1, scale=0.5,
              fliplr=0.0, mosaic=1.0, close_mosaic=10)
  ```

  `imgsz=384` matched the size the app ran at. `fliplr=0.0` because a mirrored right hand is
  a left hand, and this model was to see right hands only.

| Validation, 41 images | |
|---|---|
| Precision | 0.90 |
| Recall | 0.878 |
| mAP50 | 0.841 |
| mAP50-95 | 0.624 |

What went wrong on the way:

- The Roboflow export had no validation split; the data was split 80 / 20 by hand (168 / 41).
- Two labels had corners outside the image and were skipped by the trainer.
- A Colab runtime reset wiped the trained model. After that, `best.pt` was copied to Google
  Drive in the same cell that finished training.

In the app the right hand came from the new model and the left hand stayed on the pose model,
two inferences per frame. Volume used `w × h` of the oriented box, which does not change when
the hand tilts.

### Stage 8 — Three programs instead of one

**2026-06-28, the same day.**

**Problem: the new model missed the hand too often to play with.** A working instrument was
needed at once, and improving the model could not be allowed to break it.

**Decision: separate programs that share no vision or logic code.** `main.py` went back to
the single pose model. The two-model pipeline moved to `main_obb.py` with its own
`src/vision_obb.py` and `src/logic_obb.py`. MIDI, looper, scales, UI and overlay stay shared.
The overlay draws a rotated box when the logic offers one and a plain box otherwise, so one
overlay serves every program.

**Problem: phantoms in both programs.** The new model fired on curtains. Two more gates:
`min_confidence` 0.6 (kept modest, because the model already missed real hands) and
`max_box_area` 0.30, since a hand at playing distance never fills a third of the frame. Then
the pose model turned out to do the same thing in `main.py`, and a one-frame phantom fist had
left the looper stuck in RECORDING. Fixes:

- **Sustained fist.** The fist must be held for several consecutive frames before it toggles
  the looper.
- **Release debounce.** The fist counts as released only after the left hand has been gone
  for several frames, so a one-frame dropout is not read as a second grip.
- **One bad frame no longer kills the loop.** Each frame is handled inside its own
  `try`; the vision thread gives up only after 60 consecutive errors.
- `HAND_DEBUG=1` and `OBB_DEBUG=1` print every raw detection, so thresholds can be set from
  what the model actually reports in a given room.

`src/logic.py` also carries a filter for a phantom that confidence cannot remove: a picture
frame or a light switch that the model reports at 0.95 and above. Such a box never moves, and
a real hand always does, so a box whose centre stays within 0.012 of the frame for 30
sightings is suppressed until it moves.

**Decision: a third program on MediaPipe.** `main_mediapipe.py` uses MediaPipe's
HandLandmarker, which finds the palm first and was added as the detector less prone to
inventing hands. No new logic was written for it. `src/vision_mediapipe.py` wraps MediaPipe's
output in small objects that look like an Ultralytics result (`.boxes.xyxyn`, `.boxes.conf`,
`.keypoints.xyn`), so the existing `GestureLogic` and overlay run unchanged.

The installed MediaPipe (0.10.32) ships only the Tasks API. The legacy `mediapipe.solutions`
module is gone, which is why the original `src/vision.py` no longer runs and why a
`hand_landmarker.task` model file has to be downloaded.

### Stage 9 — One model for both hands, and pitch from rotation

**2026-07-17 to 2026-08-02.** The next model was trained to do both hands' jobs.

**Decision: detect gestures, not hands.** Three classes: `thumbout`, `openhand`,
`closedhand`. Every hand in the frame comes back as a box with a class, so the left-hand fist
is simply `closedhand` on the left half. No keypoints, no second model.

The models are YOLO26 oriented-box models in five sizes. Each has a `YOLO26<size>-ver` file
beside it recording where it came from, which `src/vision_yolo26.py` prints at startup:

| Size | Trained | mAP50-95 | Speed at 640 px on MPS |
|---|---|---|---|
| n | 2026-07-17, at 1024 px | 0.49 | ~90 fps |
| s | 2026-07-31, at 640 px | 0.77 | ~29 fps |
| m | 2026-07-31, at 640 px | 0.67 | ~25 fps |
| l | 2026-07-31, at 640 px | 0.80 | ~24 fps |
| x | 2026-07-31, at 640 px | 0.37 | ~11 fps |

Bigger was not better: the largest model scored lowest. The cause was the procedure, not
the architecture. Each was a single run with early stopping judged on 41 validation images,
and training stopped at epoch 45, 43, 23, 63 and 19. The table ranks which run got unlucky
first.

**The rotation instrument, `main_rotate.py`.** The oriented box finally gave the angle, and a
fourth program uses it: rotate the right hand to change pitch.

**Problem: the angle wraps exactly where the hand rests.** Ultralytics reports an oriented
box's angle in `[0, π/2)`, which puts the discontinuity at a vertical hand and gives only 90°
of range. `src/logic_rotate.py` recentres the angle so a vertical hand reads 0 and plays the
middle of the scale, then **unwraps** it across frames: a jump of more than 45° between two
frames is a wrap, not a movement, and 90° is added or subtracted to keep the angle
continuous. That recovers a full ±90° sweep.

Two modes, switched live: **Scale** snaps the angle to the scale's notes; **Continuous** maps
it to a chromatic note plus a pitch bend for the remainder, a glide. **Set Low Note** and
**Set High Note** capture two angles so the pitch range fits how far the player's wrist
turns. A dial on the video shows the angle.

### Stage 10 — Choosing the detector by measurement

**August 2026.** Until then each detector had been chosen by trying it. In August it was
measured, twice.

**First on the rotation models' own data, 2026-08-03 to 08-07.** More photos of the rare
gestures took the dataset from 203 to 390 images. The five sizes were retrained under one
fixed recipe with early stopping off, over five folds grouped by capture minute, and scored
on a held-out test set. The result: **model size made no measurable difference.** The gap
between the best and worst size was smaller than the gap between two folds of one size. The
smallest model was kept, `MIDI-gesture-model.pt`, at 0.748 mAP50-95 on 54 test images. That
study still had one hand per image and measured speed through PyTorch only.

**Then as a published study, from 2026-08-11:**
**[midi-gesture-research](https://github.com/hg5210187-ai/midi-gesture-research)**. A new
dataset of 60 photographs with both hands in every frame, shot in 12 places.

114 training runs: YOLO26 in 5 sizes × 2 box geometries × 3 resolutions × 3 folds, and
DEIMv2 in 8 variants × 3 folds. Accuracy by 3-fold cross-validation with folds disjoint by
place; latency measured on the MacBook Air M4 the instrument runs on.

What it decided for the instrument:

| | |
|---|---|
| Model | YOLO26-s, **axis-aligned** boxes, 320 px |
| Runtime | Core ML |
| Accuracy | 0.8056 ± 0.0437 mAP50-95 (3-fold CV), 0.9846 macro AUC |
| Model latency | 7.93 ms median, sustained for 15 minutes from cold |

Three of its findings changed how the instrument was built:

1. **Measure on the runtime you will ship.** Under PyTorch-MPS a fixed overhead of about
   18 ms hid every difference between models. Under Core ML the differences appeared.
2. **Measure sustained latency, not a burst.** The laptop has no fan. Several models passed a
   few-second test and failed after minutes.
3. **Capacity was not the limit.** Above a threshold, bigger models bought almost nothing on
   a dataset this small.

The chosen model is axis-aligned, so it carries no angle. The current program reads pitch
from height again, and the rotation instrument stays on the oriented-box models.

The instrument uses one checkpoint of that configuration, **`hbb-s-fold1-320`**, the
preferred model. [MODEL_PROCEDURES.md](MODEL_PROCEDURES.md) gives the procedure for it and
for every model before it.

### Stage 11 — Cut the latency

**2026-10-05.** The destination moved from GarageBand to Cubase and the goal became a
response under 10 ms.

**Finding: the camera sets the floor.** The built-in camera delivers 30 frames per second,
one every 33.3 ms, and refuses a 60 fps request. No amount of software makes a gesture
visible before the next frame arrives. What software can shrink is everything after the
frame: that is the number tracked below. `CAMERA_INDEX` and `CAMERA_FPS` were added so that a
faster camera is a configuration change.

What was done, in order:

1. **A capture thread that keeps only the newest frame** (`LatestFrameCamera` in
   `src/vision_obb.py`). `get_frame()` blocks until a frame newer than the last one exists.
   The loop never works on a stale buffered frame and never handles one twice.
2. **No frame-rate cap.** The loop runs at the camera's rate. Because `get_frame()` blocks,
   it does not spin a core while waiting.
3. **The two models on different devices.** Sharing one device doubled the right-hand
   model's time, about 14 ms to 28 ms. Right hand on the CPU and left hand on the GPU in its
   own worker thread did not.
4. **One model.** The study's `hbb-s-fold1-320` replaced both. `src/vision_hbb.py` runs it;
   `src/logic_obb.py` takes the rightmost box on the right half as the right hand and the
   leftmost on the left half as the left hand, and reads the fist from the `closedhand`
   class. The pose model and its thread are no longer used by this program.
5. **Core ML.** In PyTorch the single model was no faster than the two it replaced. Exported
   to Core ML it was:

   ```bash
   uv run yolo export model=hbb-s-fold1-320.pt format=coreml imgsz=320
   ```

   `src/vision_hbb.py` loads the `.mlpackage` when it exists and the `.pt` otherwise.

| Step | Camera frame delivered → MIDI sent |
|---|---|
| Start: two models, 20 fps cap, loop at 16.5 fps | ~56 ms, including the wait for a frame |
| Latest-frame thread, no cap, separate devices, 384 px | ~18 ms |
| The same with the right-hand model at 256 px | ~12 ms |
| One YOLO26-s model, PyTorch on MPS | 18–24 ms |
| The same model under Core ML | **8–9.6 ms** |

The model alone takes about 2.3 ms per frame under Core ML, against 16–22 ms in PyTorch.
Inside the running app, with the UI and overlay alongside, inference averages 6–7.6 ms.
Single-frame spikes of 50–120 ms still occur. The figures exclude the camera's internal delay
and the DAW's audio buffer. `MIDI_LATENCY_DEBUG=1` prints them every five seconds.

Things that broke:

- **NumPy 2.4.1** broke the Core ML export and made Ultralytics try to update itself at
  startup. `numpy<=2.3.5` is pinned in `pyproject.toml` for that reason.
- **Exiting during inference crashed PyTorch.** The vision thread is now joined before the
  interpreter exits.
- **Core ML is not identical to the `.pt`.** On 300 dataset images at confidence ≥ 0.6 the
  two agreed on box count and classes for 278. Confidences differed by 0.07 on average and
  0.24 at most, box coordinates by 0.025 at most, and Core ML tended to find fewer boxes. If
  hands drop out, lower `min_confidence` in `src/logic_obb.py`.
- At 30 fps the fist had to be held for 5 frames instead of 3 to keep the same 150 ms.

---

## Part 2 — Building it from zero

The order below is the one that lets you test each piece before the next depends on it. The
files named are the finished versions in this repository; each has a small self-test at the
bottom.

**1. Make a sound before anything else.** `src/midi_engine.py`. Open a port with `mido`,
send a note on, wait, send a note off. With a DAW listening on the IAC bus:

```bash
uv run python -m src.midi_engine      # middle C for one second
```

Nothing else in the project can be heard until this works, and it is the only part that
needs no camera.

**2. Build the note table.** `src/scales.py`. A scale is a list of semitone steps; walk it
from a root note for some octaves and you have the list of MIDI notes the hand will index.

```bash
uv run python -m src.scales
```

**3. Get a frame and a detection.** Pick a detector you do not have to train:
`src/vision_mediapipe.py`. Mirror the frame first, so moving right moves right on screen.
Settle one contract and keep it: `get_frame()` returns `(success, frame, result)`.

```bash
uv run python -m src.vision_mediapipe  # camera window, q to quit
```

It needs `hand_landmarker.task` in the project root; the download command is in the README.

**4. Write the smallest logic.** One hand. Height → index into the scale. Send a note on
only when the index changes, and a note off for the old note. When the hand disappears, send
a note off. That is a playable instrument. Give the logic a MIDI object in its constructor
instead of letting it open a port, and you can test it with a fake:

```bash
uv run python tests/check_headless.py
```

That script feeds invented detections through all three logic modules and checks the notes
that come out. Most of this project was verified that way, without a camera.

**5. Put it in a window.** `main.py`, `src/ui.py`. Tk must own the main thread. Run the
camera and the logic in a daemon thread, pass frames through a queue of size 2, and drop
frames when the queue is full so the video never falls behind the hands. Let the window poll
the queue with `after()`.

**6. Add volume, then a gate.** Box area over a calibrated minimum, sent as CC 7. Then
release the note below a threshold. Without the gate the instrument cannot play a rest.

**7. Draw what the instrument is reading.** `src/overlay.py`. The sampling line and the
active band did more for playability than any change to the mapping, because the player can
see where the next note is.

**8. Add the second hand and the looper.** `src/looper.py`. Record on the way out of the
logic, play back on another channel, and throttle continuous controllers.

**9. Harden it against the detector.** Every detector lies sometimes. A confidence floor, a
maximum box size, a fist that must be held, a release that must be sustained, a `try` around
each frame. Add a debug print of raw detections before tuning any of them.

**10. Only then replace the detector.** Collect images, train a model, compare it on the
machine it will run on, export it to the runtime it will run in. The procedures used here
are in [MODEL_PROCEDURES.md](MODEL_PROCEDURES.md); the protocol and the training drivers are
in [midi-gesture-research](https://github.com/hg5210187-ai/midi-gesture-research). Because the
logic only reads boxes, a class and a confidence, a new detector is one new `vision_*.py`
file.

---

## Part 3 — Open at the last entry

As of 2026-10-05:

- The Core ML program was checked with synthetic boxes and with timed runs of the real app.
  Playing it with real hands while listening in the DAW was not recorded as done.
- `min_confidence` 0.6 and `max_box_area` 0.30 were tuned for the earlier oriented-box model,
  and volume needs re-calibrating because the new boxes have a different shape.
- Accuracy of pitch and volume with the right-hand model at 256 px was not checked.
- A camera faster than 30 fps is the remaining lever on response time. A 120 fps USB camera
  was estimated, not measured, to bring the average to about 12 ms end to end.
- The hand's tilt angle as an extra MIDI parameter is stored by the oriented-box logic and
  not yet sent anywhere.
