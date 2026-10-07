# Model procedures

Every hand detector trained for this instrument, in the order it was made, with the procedure
that produced it.

> ## ⭐ The preferred model is `hbb-s-fold1-320`
>
> The research study compared 38 model configurations over 114 training runs and chose
> **YOLO26-s, axis-aligned boxes, 320 px, run under Core ML**. It is the most accurate model
> that holds a 10 ms budget on the target laptop for 15 minutes without a breach.
>
> | | |
> |---|---|
> | Accuracy, 3-fold cross-validation | 0.8056 ± 0.0437 mAP50-95, 0.9846 macro AUC |
> | Model latency on a MacBook Air M4 | 7.93 ms median, sustained for 15 minutes from cold |
> | Program that runs it | `main_obb.py` |
> | Files | `hbb-s-fold1-320.mlpackage` (Core ML, used first) and `hbb-s-fold1-320.pt` |
>
> **Why this model:** [the deployment choice, in the research report](https://github.com/hg5210187-ai/midi-gesture-research/blob/main/REPORT.md#9-the-deployment-choice).
> **How to get it:** [the `models-v1` release](../../../releases/tag/models-v1), or
> [Models](MODELS.md#download). **How it was made:** [step 7 below](#7--the-research-study-and-hbb-s-fold1-320).

Use the other models only for what they are still needed for: the `YOLO26<size>-MIDI.pt`
models give the rotation instrument its angle, which the preferred model does not have.

---

## The order they were made

| # | When | Model | What it is | Where trained | What became of it |
|---|---|---|---|---|---|
| [1](#1--right_hand_obbpt) | 2026-06-28 | `right_hand_obb.pt` | YOLO11n oriented boxes, one class (right hand) | Google Colab, T4 | Missed the hand too often. Retired |
| [2](#2--yolo26n-midipt) | 2026-07-17 | `YOLO26n-MIDI.pt` | YOLO26n oriented boxes, three gesture classes | MacBook Air M4 | Default of the rotation instrument |
| [3](#3--yolo26s-m-l-x-midipt) | 2026-07-31 | `YOLO26s-MIDI.pt`, `m`, `l`, `x` | The same model in four larger sizes | MacBook Air M4 | Rotation instrument. Their ranking made no sense, which started the studies |
| [4](#4--the-first-cross-validated-study-289-photos) | 2026-08-04 | 5-fold sweep, 289 photos | 5 sizes × 5 folds, oriented boxes | Rented 24 GB GPU | Showed model size did not matter. One class too small to measure |
| [5](#5--the-same-study-on-390-photos-and-midi-gesture-modelpt) | 2026-08-05 | 5-fold sweep, 390 photos, and `MIDI-gesture-model.pt` | YOLO26n oriented boxes | Rented 24 GB GPU | First model with an honest test score |
| [6](#6--the-ablation-pair-and-the-tuning-search) | 2026-08-06 to 08-07 | Ablation pair, tuning trials | YOLO26n oriented boxes | MacBook Air M4 | Analysis only. Found a data leak; tuning not adopted |
| [7](#7--the-research-study-and-hbb-s-fold1-320) | 2026-08-11 to 08-13 | **`hbb-s-fold1-320.pt`**, one of 114 runs | **YOLO26-s plain boxes, 320 px** | Kaggle T4, rented RTX 4090 | ⭐ **Preferred** |
| [8](#8--the-core-ml-export) | 2026-10-05 | **`hbb-s-fold1-320.mlpackage`** | Core ML export of the same weights | MacBook Air M4 | ⭐ **What the instrument runs** |

Models 2, 3, 7 and 8 are in the [`models-v1` release](../../../releases/tag/models-v1).
Models 1, 5 and 6 are not published.

All of them were fine-tuned from Ultralytics pretrained weights with the `ultralytics`
package, and all use the same three classes from model 2 onward:

| Index | Class | Gesture |
|---|---|---|
| 0 | `thumbout` | Closed fist, thumb extended sideways |
| 1 | `openhand` | Flat palm to the camera, fingers spread |
| 2 | `closedhand` | Closed fist, thumb tucked in |

---

## 1 — `right_hand_obb.pt`

**Goal.** Replace the keypoint model for the right hand with something small, and get the
hand's tilt angle for later use. One class: a right hand.

**Procedure.**

1. Photograph 209 right-hand images, varying distance, position, rotation, hand shape,
   background and lighting. Include some frames with no hand as negatives.
2. Annotate in Roboflow as polygons. Export in the "YOLOv8 Oriented Bounding Boxes" format,
   which converts the polygons to oriented boxes.
3. The export contained no validation split. Split it 80 / 20 by hand, moving each image
   with its label file: 168 train, 41 validation.
4. Rewrite `train`, `val` and `path` in `data.yaml` as absolute paths. The relative ones did
   not resolve.
5. Train in Google Colab on a T4, Ultralytics 8.4.80:

   ```python
   from ultralytics import YOLO

   model = YOLO("yolo11n-obb.pt")
   model.train(
       data=f"{dataset.location}/data.yaml",
       epochs=80, imgsz=384, batch=16, patience=20, device=0,
       project="midi_hand", name="yolo_obb_run",
       cos_lr=True, degrees=15.0, translate=0.1, scale=0.5,
       fliplr=0.0, mosaic=1.0, close_mosaic=10,
   )
   ```

   `imgsz=384` matched the size the app ran at. `fliplr=0.0` because a mirrored right hand is
   a left hand. `degrees=15.0` to reinforce tilt.
6. Copy `best.pt` to Google Drive **in the cell that finishes training**. A runtime reset
   wiped the first trained model.

**Result,** on the 41 validation images: precision 0.90, recall 0.878, mAP50 0.841,
mAP50-95 0.624.

**What became of it.** In the app it missed the real hand too often and fired on curtains.
It ran only in the two-model backend, which no program uses now.

---

## 2 — `YOLO26n-MIDI.pt`

**Goal.** One model for both hands. Detect the *gesture*, so a fist needs no keypoints, and
keep the oriented box for its angle.

**Procedure.**

1. **Photograph** 203 images, 2250 × 1500, one hand per image: 186 `thumbout`, 13
   `openhand`, 4 `closedhand`.
2. **Annotate in QuPath.** Draw around the hand and assign its class. A Groovy export script
   paints every annotation into an 8-bit mask the size of the photo, using one grey value
   per class: 200 `thumbout`, 100 `openhand`, 255 `closedhand`, 0 background.
3. **Convert masks to oriented-box labels.** For each class value, find the outer contours of
   that value, fit the tightest rotated rectangle to each, and write its four corners
   normalised to the image size:

   ```python
   CLASS_VALUE_TO_LABEL = {200: 0, 100: 1, 255: 2}

   for value, cls in CLASS_VALUE_TO_LABEL.items():
       blob = np.uint8(mask == value) * 255
       contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       for cnt in contours:
           corners = cv2.boxPoints(cv2.minAreaRect(cnt))       # 4 × (x, y) in pixels
           # write: cls x1 y1 x2 y2 x3 y3 x4 y4, each divided by width / height, clamped to [0, 1]
   ```

   Labels are normalised against the original image; Ultralytics does its own resizing.
   Three masks painted as `thumbout` showed open hands and were relabelled in the script.
4. **Split** 70 / 20 / 10 with seed 42. The validation set came to 41 images: 37 `thumbout`,
   3 `openhand`, 1 `closedhand`.
5. **Train** on the MacBook Air M4, Ultralytics 8.4.45:

   ```python
   from ultralytics import YOLO

   model = YOLO("yolo26n-obb.pt")
   model.train(
       data="data/yolo_obb_midi/data.yaml",
       epochs=100, patience=10, imgsz=1024, batch=8,
       device="mps", seed=42, deterministic=True,
   )
   ```

   Everything else was left at the Ultralytics defaults, including `fliplr=0.5`.
6. **Write the version record.** `YOLO26n-ver` in the project root holds the base model,
   dataset, epochs, sizes, date, mAP and measured speed. The app prints it at startup.

**Result.** Training stopped early at epoch 45 of 100, best epoch 35, after 34 minutes.

| Validation, 41 images | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| All classes | 0.651 | 0.629 | 0.660 | 0.492 |
| `thumbout` (37) | 0.952 | 0.919 | 0.984 | 0.738 |
| `openhand` (3) | 1.000 | 0.970 | 0.995 | 0.738 |
| `closedhand` (1) | 0 | 0 | 0 | 0 |

**Read this before using it.** The overall score is low because of one class. With a handful
of `closedhand` images to learn from, the model scored zero on the only fist it was tested
on. In the rotation instrument the fist is the looper gesture, so do not expect this model
to start the looper reliably.

---

## 3 — `YOLO26s`, `m`, `l`, `x`-`MIDI.pt`

**Goal.** Find out whether a larger model is better.

**Procedure.** The dataset, split and recipe of model 2, with two differences forced by the
16 GB laptop: the image size is 640 instead of 1024, and the batch shrinks as the model
grows. A driver script ran the four sizes in turn and retried a smaller batch on an
out-of-memory error.

```python
BATCH = {"s": 8, "m": 4, "l": 4, "x": 2}

for size, batch in BATCH.items():
    model = YOLO(f"yolo26{size}-obb.pt")
    model.train(
        data="data/yolo_obb_midi/data.yaml",
        epochs=100, patience=10, imgsz=640, batch=batch,
        device="mps", seed=42, deterministic=True,
    )
```

**Result.**

| Size | Batch | Image size | Stopped at epoch | Best epoch | Minutes | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|---|---|---|
| n | 8 | 1024 | 45 | 35 | 34 | 0.651 | 0.629 | 0.660 | 0.492 |
| s | 8 | 640 | 43 | 33 | 19 | 0.968 | 0.604 | 0.979 | 0.773 |
| m | 4 | 640 | 23 | 13 | 26 | 0.786 | 0.637 | 0.904 | 0.674 |
| l | 4 | 640 | 63 | 53 | 87 | 0.713 | 0.979 | 0.987 | 0.798 |
| x | 2 | 640 | 19 | 9 | 50 | 0.142 | 0.889 | 0.551 | 0.366 |

Measured speed at 640 px on the M4 through MPS: n ~90 fps, s ~29, m ~25, l ~24, x ~11.

**What this showed, and what it did not.** The order is l, s, m, n, x. Bigger models should
not be erratically worse. Two things in the procedure caused it:

- **Early stopping fired at random.** "Stop after 10 epochs without improvement", judged on
  41 images, stopped `m` at epoch 23 and `x` at 19 while `l` ran to 63. The table ranks which
  run got unlucky first.
- **One class had almost no data.** `closedhand` had about two training images.

So these five are usable models, not a comparison. By this table `l` and `s` scored
highest; `n` is the default of the rotation instrument because it is the fastest. Every step
from here on exists to replace this table with a real measurement.

---

## 4 — The first cross-validated study, 289 photos

**Goal.** Compare the five sizes fairly.

**What changed in the procedure.**

| Change | Why |
|---|---|
| 86 more `closedhand` photos (2026-08-03), 289 in total | The class had 4 |
| 21 blank masks and 4 mis-annotated ones excluded; 1 relabelled in code | An unannotated hand teaches that a hand is not a hand. One mis-click painted 97% of a frame |
| A held-out test set (41 images) and **5 folds** for the rest | One split gives one number; five give a spread |
| **Folds grouped by capture minute** | The new photos were webcam bursts. Frames from the same minute are near-duplicates and must not sit on both sides of a split |
| **Early stopping off**, a fixed 60 epochs | It caused the incoherent ranking above |
| **AdamW with an explicit learning rate** (0.001429) | In `optimizer=auto` Ultralytics ignores `lr0` and substitutes its own. Writing it out makes the setting real |
| Horizontal flip and rotation augmentation off | Every photo is a right hand; the hand angles already span ±90° |
| 640 px and batch 8 for every size, on a rented 24 GB GPU | The laptop's memory had forced the largest model down to batch 2, so sizes were being trained differently. About 3 hours instead of about 38 |

**Procedure.**

1. Build one pool from every annotated photo, with the exclusions above.
2. Search 200 candidate splits against a balance rule fixed in advance, using label counts
   only and never model scores. Record the seed.
3. Train 5 sizes × 5 folds = 25 runs with the one recipe.
4. Evaluate every run on the held-out test set.
5. Take the epoch count for a final model from the folds: 1.15 × the median best epoch.
6. Train the chosen size once on all non-test images, and read the test set once.

**Result.** Cross-validated mAP50-95 between 0.692 and 0.731 across the five sizes: no size
clearly ahead. The final nano model, trained on 248 images, scored 0.694 mAP50-95 on the 41
test images. But `openhand` still had 13 photos, 2 of them in the test set, so a third of
the score rested on two photographs.

---

## 5 — The same study on 390 photos, and `MIDI-gesture-model.pt`

**What changed.** 77 `openhand` and 24 `closedhand` photos were added on 2026-08-05. The
dataset became **390 images: 186 `thumbout`, 90 `openhand`, 114 `closedhand`**, with a
54-image test set and five folds of 64 to 72 images.

**Procedure.** As in step 4, re-run in full: 25 runs, 177 minutes of GPU.

**Result,** held-out test mAP50-95 averaged over 5 folds:

| Size | mAP50-95 | ± SD | mAP50 |
|---|---|---|---|
| n | 0.7372 | 0.021 | 0.9265 |
| s | 0.7421 | 0.027 | 0.9365 |
| m | 0.7431 | 0.036 | 0.9270 |
| l | 0.7254 | 0.036 | 0.9177 |
| x | 0.7512 | 0.013 | 0.9348 |

**The finding: model size makes no measurable difference.** The gap between the best and
worst size, 0.026, is smaller than the disagreement between folds of one size, 0.027. The
limit is the dataset, not the network.

Speed was then measured on the M4, one frame at a time through MPS: n 18.5 ms, s 20.0,
m 37.8, l 43.9, x 91.8. With accuracy tied, the smallest model was chosen.

**`MIDI-gesture-model.pt`** is that choice: YOLO26n, oriented boxes, trained on all 336
non-test images for 51 epochs.

| Test, 54 images, read once | AP50 | AP50-95 |
|---|---|---|
| `thumbout` | 0.894 | 0.689 |
| `openhand` | 0.995 | 0.785 |
| `closedhand` | 0.889 | 0.771 |
| **All** | **0.926** | **0.748** |

Precision 0.947, recall 0.804. It has the same three classes in the same order as the
rotation models and an oriented box, so `main_rotate.py --model MIDI-gesture-model.pt` loads
it. It is not in the release.

---

## 6 — The ablation pair and the tuning search

Two pieces of analysis that trained models but delivered none.

**A leak, found and measured.** The capture-minute grouping had been applied to the
`closedhand` photos but not to the `openhand` ones. Nine of the ten `openhand` test frames
shared a capture minute with a training frame, so the 0.995 above is inflated. Two nano
models were trained to measure the real effect: an identical 50-epoch recipe, differing only
in whether the new `openhand` photos were included, tested on 33 `openhand` frames from 7
capture minutes that neither had seen.

| Model | `openhand` training photos | `openhand` AP50, unseen minutes |
|---|---|---|
| Before | 13 | 0.47 |
| After | 57 | **0.77** |

The honest gain from the new data is +0.30, not the +0.43 the leaked figure implied. The
grouping was fixed.

**A hyperparameter search.** Optuna, 20 configurations of 10 hyperparameters on fold 0,
30-epoch trials, the best three re-run at 60 epochs. Against an untuned control at the same
60 epochs (0.7227 ± 0.0100 over three seeds) the best tuned configuration scored 0.7775,
+0.055. It was **not adopted**: it was measured on one fold, the fold that carried the leak.

**What these three steps left open.** One hand per image, when the instrument reads two.
Rare classes from a single sitting. Speed measured through MPS only. The next study differs
on each of these points.

---

## 7 — The research study and `hbb-s-fold1-320`

**Goal.** Answer the real question: is a camera-based gesture instrument deployable on a
laptop a musician owns? That needs accuracy and latency measured together, on the target
machine, in the runtime that will ship.

This study is published in full as
**[midi-gesture-research](https://github.com/hg5210187-ai/midi-gesture-research)**. The
procedure below names the file in that repository that carries each step.

**Procedure.**

1. **Write the shot list before taking a photo.** 60 photographs, **both hands in every
   frame**, across 12 places. Each place is shot as triplets of gesture pairs so that every
   fold gets exactly 10 instances of each class, and each gesture appears on the left hand as
   often as the right. Rotate the hands 0°, 30°, 60° within a triplet.
   → [`DATA_COLLECTION.md`](https://github.com/hg5210187-ai/midi-gesture-research/blob/main/DATA_COLLECTION.md)
2. **Assign places to folds.** Three folds and a test set, 15 photos and 30 annotations each,
   **no place in two groups**. Backgrounds dominate the frame, so a random split lets a model
   score by recognising a room.
   → `scripts/assign_places.py`, `scripts/split_folds.py`, `data/splits/folds.json`
3. **Annotate as grey-level masks**, the same three values as before. Masks were generated
   automatically, scored for how well their edge sits on the image gradient, inspected as
   overlays, and corrected by hand in QuPath where they were wrong. All four hands the
   automatic pass missed were backlit.
   → `scripts/make_masks_v2.py`, `scripts/mask_qa.py`, `scripts/install_manual_masks.py`
4. **Build the pool.** One script turns the masks into axis-aligned labels, oriented labels
   and COCO JSON, so both box geometries and both model families train on identical data.
   → `scripts/build_pool.py`
5. **Train the grid.** YOLO26 in 5 sizes × 2 box geometries × 3 resolutions (320, 416, 640)
   × 3 folds = 90 runs, and DEIMv2 in 8 variants × 3 folds = 24 runs. Each fold's model
   trains on the other two folds, 30 photos, and validates on its own 15.
   → [`kaggle/train_yolo.py`](https://github.com/hg5210187-ai/midi-gesture-research/blob/main/kaggle/train_yolo.py),
   `kaggle/setup_deimv2.py`, `kaggle/run_deim.py`

   The run that produced the preferred model is one cell of that grid:

   ```bash
   python kaggle/train_yolo.py --data <dataset> --out <workdir> \
       --geoms hbb --sizes s --folds 1 --imgsz 320
   ```

   which is this Ultralytics call:

   ```python
   from ultralytics import YOLO

   model = YOLO("yolo26s.pt")                    # detect model: plain boxes
   model.train(
       data="splits/hbb/fold1.yaml",             # train on folds 0 and 2, validate on fold 1
       epochs=100, imgsz=320, batch=8,
       patience=100,                             # = epochs, so early stopping can never fire
       seed=42, deterministic=True,
       cache="ram", workers=2, device=0,
   )
   ```

   Ultralytics 8.4.45 on a rented RTX 4090; the run took 2.1 minutes. `optimizer` was left on
   `auto`, which resolved to **AdamW at a learning rate of 0.001429**, not the 0.01 written
   in the checkpoint.
   → [`TRAINING_CONFIG.md`](https://github.com/hg5210187-ai/midi-gesture-research/blob/main/TRAINING_CONFIG.md)
6. **Evaluate.** Confusion matrices pooled over the three folds, one-vs-rest AUC, IoU
   computed exactly for every architecture. The test set is read once, at the end, for all
   38 configurations.
   → `scripts/pool_metrics.py`, `scripts/test_eval.py`
7. **Measure latency on the laptop, under Core ML.** Export each configuration and time
   `predict()` end to end. Then run the candidates near the budget for 15 continuous
   minutes, because the laptop has no fan and a few-second test describes a cold machine.
   → `scripts/export_coreml.py`, `scripts/sustained.py`
8. **Choose.** Among the configurations that hold 10 ms in every one-minute window, take the
   most accurate.

**Result.** Two configurations hold the budget. YOLO26-s at 320 px is the more accurate.

| Configuration | CV mAP50-95 | Sustained latency | |
|---|---|---|---|
| YOLO26-n, plain boxes, 320 px | 0.6902 | 7.22 ms | holds |
| **YOLO26-s, plain boxes, 320 px** | **0.8056 ± 0.0437** | **7.93 ms** | **holds, chosen** |

Its three fold models, on their own validation folds:

| Checkpoint | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| `hbb-s-fold0-320.pt` | 0.920 | 0.933 | 0.967 | 0.802 |
| **`hbb-s-fold1-320.pt`** | **0.992** | **0.954** | **0.981** | **0.837** |
| `hbb-s-fold2-320.pt` | 0.893 | 0.884 | 0.925 | 0.781 |

These are Ultralytics' own validation figures; the 0.8056 above is the study's exact-IoU
recomputation of the same three models. The instrument uses the fold-1 checkpoint, the
highest-scoring of the three.

Three things the study found that the earlier steps could not:

- **Oriented boxes did not help, and 320 px was both faster and more accurate than 640 px**
  on this close-range data (the report attaches a software-version caveat to the 640 px
  comparison). The preferred model therefore has no angle, and the rotation instrument stays
  on models 2 and 3.
- **The runtime decides the latency.** Through PyTorch-MPS a fixed overhead of about 18 ms
  hid every difference. Under Core ML the same models spread out.
- **A short benchmark recommends the wrong model.** Several configurations passed a burst
  test and failed after minutes of continuous use.

---

## 8 — The Core ML export

**Goal.** Run the preferred model at the speed the study measured. In PyTorch on the M4 the
`.pt` takes 16–22 ms per frame.

**Procedure,** in this project's own environment (coremltools 9.0, `numpy<=2.3.5`):

```bash
uv run yolo export model=hbb-s-fold1-320.pt format=coreml imgsz=320
```

This writes `hbb-s-fold1-320.mlpackage` beside the `.pt`. `main_obb.py` loads the package
when it is there and the `.pt` when it is not.

**Result.**

| | PyTorch `.pt` | Core ML `.mlpackage` |
|---|---|---|
| Model alone | 16–22 ms per frame | about 2.3 ms |
| Inference inside the running app | 16–22 ms | 6–7.6 ms |
| Camera frame delivered → MIDI sent | 18–24 ms | **8–9.6 ms** |

**Check the export against the original.** On 300 images at confidence ≥ 0.6 the two agreed
on box count and classes for 278. Confidences differed by 0.07 on average and box
coordinates by 0.025 at most, and Core ML tended to find fewer boxes.

---

## Where to read more: a guide to the research repository

**[github.com/hg5210187-ai/midi-gesture-research](https://github.com/hg5210187-ai/midi-gesture-research)**

| If you want to know | Read |
|---|---|
| The answer in one page | [README](https://github.com/hg5210187-ai/midi-gesture-research#readme) |
| **Why `hbb-s@320` was chosen** | [REPORT §9, The deployment choice](https://github.com/hg5210187-ai/midi-gesture-research/blob/main/REPORT.md#9-the-deployment-choice) |
| The one case where another model looks better | [REPORT §9, An honest tension in that choice](https://github.com/hg5210187-ai/midi-gesture-research/blob/main/REPORT.md#an-honest-tension-in-that-choice) |
| How every model ranks, on accuracy and on latency | [results/RANKINGS.md](https://github.com/hg5210187-ai/midi-gesture-research/blob/main/results/RANKINGS.md) |
| Accuracy against latency in one picture | [REPORT, Figure 5](https://github.com/hg5210187-ai/midi-gesture-research/blob/main/REPORT.md#figure-5--accuracy-versus-latency) |
| Why latency was measured for 15 minutes, and what failed | [REPORT §6, Latency](https://github.com/hg5210187-ai/midi-gesture-research/blob/main/REPORT.md#6-latency) and [results/LATENCY.md](https://github.com/hg5210187-ai/midi-gesture-research/blob/main/results/LATENCY.md) |
| How to photograph an equivalent dataset | [DATA_COLLECTION.md](https://github.com/hg5210187-ai/midi-gesture-research/blob/main/DATA_COLLECTION.md) |
| The hyperparameters that actually ran | [TRAINING_CONFIG.md](https://github.com/hg5210187-ai/midi-gesture-research/blob/main/TRAINING_CONFIG.md) |
| The training script | [kaggle/train_yolo.py](https://github.com/hg5210187-ai/midi-gesture-research/blob/main/kaggle/train_yolo.py) |
| What the study does not establish | [REPORT §10](https://github.com/hg5210187-ai/midi-gesture-research/blob/main/REPORT.md#10-what-is-not-established) |

The photographs are not distributed in either repository: they show an identifiable person.
The research repository holds the fold assignments, labels, prediction dumps and metrics, so
every number in its report can be recomputed without them.
