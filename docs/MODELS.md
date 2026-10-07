# Models

> **⭐ The preferred model is `hbb-s-fold1-320`**, the one the
> [research study](https://github.com/hg5210187-ai/midi-gesture-research/blob/main/REPORT.md#9-the-deployment-choice)
> chose: YOLO26-s, plain boxes, 320 px, 7.93 ms sustained on a MacBook Air M4 under Core ML.
> `main_obb.py` runs it. How every model was made, in order, is in
> [MODEL_PROCEDURES.md](MODEL_PROCEDURES.md).

| Program | File it loads | What the file is | Where it comes from |
|---|---|---|---|
| ⭐ `main_obb.py` | `hbb-s-fold1-320.mlpackage`, else `hbb-s-fold1-320.pt` | YOLO26-s detect model, 3 gesture classes, 320 px | [`models-v1` release](../../../releases/tag/models-v1) |
| `main_rotate.py` | `YOLO26<size>-MIDI.pt` | YOLO26 oriented-box model, 3 gesture classes, five sizes | [`models-v1` release](../../../releases/tag/models-v1) |
| `main_mediapipe.py` | `hand_landmarker.task` | MediaPipe HandLandmarker bundle, 7.8 MB | Public download from Google, below |
| `main.py` | `yolov8n-hand-pose.pt` | YOLOv8n pose model, one class `hand`, 21 keypoints, 7 MB | A third-party checkpoint, not redistributed here |

The gesture classes are `thumbout`, `openhand`, `closedhand`.

## Download

The trained weights are attached to this repository's
**[`models-v1` release](../../../releases/tag/models-v1)**, not stored in git. One of them is
126 MB, over GitHub's 100 MB limit for a file in a repository, and binaries in git would make
every clone carry all of them. `.gitignore` excludes `*.pt`, `*.mlpackage`, `*.task` and
`*.onnx`.

| Release file | Size | For |
|---|---|---|
| ⭐ `hbb-s-fold1-320.mlpackage.zip` | 18 MB | `main_obb.py`, Core ML. Unzip it |
| ⭐ `hbb-s-fold1-320.pt` | 20 MB | `main_obb.py`, PyTorch. Also the source for a new export |
| `YOLO26n-MIDI.pt` | 5.8 MB | `main_rotate.py` (default) |
| `YOLO26s-MIDI.pt` | 21 MB | `main_rotate.py --model s` |
| `YOLO26m-MIDI.pt` | 47 MB | `main_rotate.py --model m` |
| `YOLO26l-MIDI.pt` | 56 MB | `main_rotate.py --model l` |
| `YOLO26x-MIDI.pt` | 126 MB | `main_rotate.py --model x` |
| `SHA256SUMS.txt` | | Checksums of the seven files above |

With the [GitHub CLI](https://cli.github.com/), from the project root:

```bash
# the preferred model
gh release download models-v1 --pattern 'hbb-s-fold1-320.*'
unzip hbb-s-fold1-320.mlpackage.zip && rm hbb-s-fold1-320.mlpackage.zip

# one rotation model, or all five (257 MB)
gh release download models-v1 --pattern 'YOLO26s-MIDI.pt'
gh release download models-v1 --pattern 'YOLO26*-MIDI.pt'
```

To check what arrived, before deleting the zip:

```bash
gh release download models-v1 --pattern SHA256SUMS.txt
shasum -a 256 --ignore-missing -c SHA256SUMS.txt
```

Without the CLI, open the release page in a browser, download the files, and put them in the
project root.

Model files go in the **project root**. Three of the programs find that folder themselves;
`main.py` looks in the current directory, so start it from the root.

These weights were fine-tuned from Ultralytics YOLO26 pretrained weights and carry the
AGPL-3.0 license recorded inside each file.

## ⭐ The preferred model, for `main_obb.py`

`hbb-s-fold1-320` is fold 1 of the cross-validation in
[midi-gesture-research](https://github.com/hg5210187-ai/midi-gesture-research): YOLO26-s,
axis-aligned boxes, trained at 320 px for 100 epochs. On its own validation fold it scores
0.992 precision, 0.954 recall and 0.837 mAP50-95.

`main_obb.py` loads the `.mlpackage` when it is in the project root and the `.pt` when it is
not. The `.pt` runs in PyTorch at 16–22 ms per frame on the M4; the Core ML package at about
2.3 ms.

A model you train yourself works here if it is an Ultralytics **detect** model with a class
named `closedhand`. `src/logic_obb.py` uses the right-hand box whatever its class, and reads
the left-hand fist from that class name.

```bash
HBB_MODEL=my-model.pt HBB_IMGSZ=320 uv run python main_obb.py
```

### Exporting to Core ML

The release already contains the export. To make it again, or to export another model:

```bash
uv run yolo export model=hbb-s-fold1-320.pt format=coreml imgsz=320
```

This writes `hbb-s-fold1-320.mlpackage` next to the `.pt`, and `main_obb.py` picks it up on
the next start. Two things to know:

- `numpy` is pinned to `<=2.3.5` in `pyproject.toml`. NumPy 2.4.1 broke this export.
- The export is close to the original, not identical. Compared on 300 images at confidence
  ≥ 0.6, the two agreed on box count and classes for 278, and Core ML tended to find fewer
  boxes. `OBB_DEBUG=1` prints every detection if hands start dropping out.

`HBB_DEVICE=mps|cpu` applies to PyTorch models only.

## The rotation models, for `main_rotate.py`

Five sizes of the same model. `--model n|s|m|l|x` picks `YOLO26<size>-MIDI.pt`; a path to any
other `.pt` also works.

| Size | File size | mAP50-95 | Speed at 640 px on MPS |
|---|---|---|---|
| n (default) | 5.8 MB | 0.49 | ~90 fps |
| s | 21 MB | 0.77 | ~29 fps |
| m | 47 MB | 0.67 | ~25 fps |
| l | 56 MB | 0.80 | ~24 fps |
| x | 126 MB | 0.37 | ~11 fps |

These scores come from single training runs that stopped early at different points, so they
do not rank the sizes; see
[MODEL_PROCEDURES.md](MODEL_PROCEDURES.md#3--yolo26s-m-l-x-midipt). The `n` model scored
zero on the only fist in its validation set, so with it the left-hand looper gesture may not
register.

The figures are read from the `YOLO26<size>-ver` files in the project root, one per model.
Each is a small JSON record: base model, dataset, epochs, training size, training date,
Ultralytics version, mAP and measured speed. `src/vision_yolo26.py` derives the file name
from the model name and prints the record at startup; a missing record is skipped, not an
error. Update the record when a model is retrained.

The rotation logic identifies the fist by class index 2, so a replacement model must keep the
class order `thumbout`, `openhand`, `closedhand`.

## MediaPipe, for `main_mediapipe.py`

```bash
curl -L -o hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

`src/vision_mediapipe.py` prints this command if the file is missing.

## The hand-pose model, for `main.py`

`yolov8n-hand-pose.pt` was not trained in this project. Its metadata records Ultralytics
8.2.15, May 2024, a 224 px training size and the AGPL-3.0 license.

Any Ultralytics pose model with a single hand class and the standard 21-point hand layout can
take its place. `src/logic.py` reads the box of every hand and six keypoints of the left
hand: 0 (wrist), 9 (middle-finger knuckle) and the fingertips 8, 12, 16 and 20. One way to
get such a model is to train a pose model on Ultralytics' hand-keypoints dataset; that was
not tried here. Pass another file with
`UltralyticsVision(model_variant="your-model.pt")` in `main.py`.

## Models that are not published

- `right_hand_obb.pt`, the first self-trained model (YOLO11n oriented boxes, one class,
  June 2026). Only the two-model backend in `src/vision_obb.py` loads it, and no program uses
  that backend now.
- `MIDI-gesture-model.pt`, the nano model delivered by the first cross-validated study in
  August 2026.

Both are described in [MODEL_PROCEDURES.md](MODEL_PROCEDURES.md).
