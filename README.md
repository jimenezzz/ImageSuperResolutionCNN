# Computational Photography Final Project
Final project for computational photography: single-image and video super-resolution using SRCNN and FSRCNN.


### Contributors:
- jesusaj2@illinois.edu
- ahmedha3@illinois.edu
- asingla4@illinois.edu


Models trained and used:
- `SRCNN`
- `FSRCNN`


## 1) Setup (create venv and install requirements)

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## 2) Dataset Layout

The dataset used to train the models in this repo is available on Google Drive (e.g. `datasets.zip` and related archives). Download and extract as needed to match the layout below:

[ImageSuperResolutionCNN — dataset and related files (Google Drive)](https://drive.google.com/drive/u/0/folders/1D5SDws1CWnbIKBmxEGNaG4QFpYLUfOMW)

The training/evaluation scripts expect DIV2K-style folders:

```text
datasets/DIV2K/
  train/
    HR/
    LR_bicubic/X2/
  val/
    HR/
    LR_bicubic/X2/
  test/
    HR/
    LR_bicubic/X2/
```

If you train/evaluate at other scales (`X3`, `X4`, `X8`), the corresponding LR folder should exist for that scale.

## 3) Validate Dataset

Quick sanity check before training:

```bash
python3 validate_dataset.py \
  --dataset-root datasets/DIV2K \
  --scale 2
```

## 4) Pretrained Models + Training

This repository already includes trained checkpoints under `artifacts/experiments`, so you can skip training and go straight to evaluation/inference.

Available `best.pth` checkpoints:

- `artifacts/experiments/srcnn/x2/best.pth`
- `artifacts/experiments/srcnn/x4/best.pth`
- `artifacts/experiments/srcnn/x8/best.pth`
- `artifacts/experiments/fsrcnn/x2/best.pth`
- `artifacts/experiments/fsrcnn/x4/best.pth`
- `artifacts/experiments/fsrcnn/x8/best.pth`

Each experiment directory also includes per-epoch checkpoints and metric/history JSON files.

If you want to train your own model, you can run:

Example runs:

```bash
# SRCNN
python3 train_model.py \
  --model srcnn \
  --scale 2 \
  --patches-per-image 128 \
  --epochs 30

# FSRCNN
python3 train_model.py \
  --model fsrcnn \
  --scale 2 \
  --patches-per-image 128 \
  --fsrcnn-augment
```

Checkpoints are written under `artifacts/experiments/...`.

## 5) Evaluate

Validation split:

```bash
python3 evaluate_model.py \
  --model srcnn \
  --scale 2 \
  --checkpoint artifacts/experiments/srcnn/x2/best.pth \
  --dataset-root datasets/DIV2K
```

Test split:

```bash
python3 evaluate_model.py \
  --model srcnn \
  --scale 2 \
  --checkpoint artifacts/experiments/srcnn/x2/best.pth \
  --dataset-root datasets/DIV2K \
  --split test
```

## 6) Run on Images

LR input with paired HR ground truth (for metrics/comparison):

```bash
python3 run_model.py \
  --model fsrcnn \
  --scale 2 \
  --checkpoint artifacts/experiments/fsrcnn/x2/best.pth \
  --input datasets/DIV2K/val/LR_bicubic/X2 \
  --paired-hr-dir datasets/DIV2K/val/HR
```

LR image(s) to super-resolved outputs:

```bash
python3 run_model.py \
  --model srcnn \
  --scale 2 \
  --checkpoint artifacts/experiments/srcnn/x2/best.pth \
  --input path/to/lr_images_dir \
  --output-dir path/to/output
```

HR input degraded first (for qualitative experiments):

```bash
python3 run_model.py \
  --model fsrcnn \
  --scale 2 \
  --checkpoint artifacts/experiments/fsrcnn/x2/best.pth \
  --input datasets/DIV2K/val/HR/0801.png \
  --degrade-hr-first
```

## 7) Run on Videos (Upscaling)

Use `run_model_on_video.py` to:
1. extract frames from an input video,
2. run SRCNN/FSRCNN on each frame, 
3. rebuild the frames into an output video.

Intermediate frames are written to a temporary directory and removed when the script finishes.

Note: videos with audio will be written without audio in the output.

### Prerequisite

Install `ffmpeg` and `ffprobe` so they are available in your shell:

```bash
ffmpeg -version
ffprobe -version
```

### Basic command

```bash
python3 run_model_on_video.py \
  --model srcnn \
  --scale 2 \
  --checkpoint artifacts/experiments/srcnn/x2/best.pth \
  --input-video path/to/input.mp4 \
  --output-video artifacts/predictions/video_output_srcnn_x2.mp4
```

### FSRCNN example

```bash
python3 run_model_on_video.py \
  --model fsrcnn \
  --scale 2 \
  --checkpoint artifacts/experiments/fsrcnn/x2/best.pth \
  --input-video path/to/input.mp4 \
  --output-video artifacts/predictions/video_output_fsrcnn_x2.mp4
```

## Notes

- Make sure your `--scale` matches the checkpoint you are using.
- If `libx264` is not available in your ffmpeg build, the script automatically falls back to `mpeg4`.

## References and acknowledgments

**Papers**

- **FSRCNN** — Chao Dong, Chen Change Loy, Xiaoou Tang, *Accelerating the Super-Resolution Convolutional Neural Network* ([arXiv](https://arxiv.org/abs/1608.00367)).
- **SRCNN** — Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang, *Image Super-Resolution Using Deep Convolutional Networks* ([arXiv](https://arxiv.org/abs/1501.00092), [PDF](https://arxiv.org/pdf/1501.00092)).

This project implements both methods; FSRCNN is the faster variant, trained in the same overall pipeline as SRCNN.

**Dataset**

- Training data and related archives (`datasets.zip`, etc.): [Google Drive folder](https://drive.google.com/drive/u/0/folders/1D5SDws1CWnbIKBmxEGNaG4QFpYLUfOMW).

**Code (reference implementations)**

- [yjn870/FSRCNN-pytorch](https://github.com/yjn870/FSRCNN-pytorch)
- [yjn870/SRCNN-pytorch](https://github.com/yjn870/SRCNN-pytorch)

**Tools**

- **Cursor** was used to help with understanding parts of the codebase and with writing and editing code and documentation.
