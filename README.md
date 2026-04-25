# Computational Photography Final Project

Image super-resolution with SRCNN and FSRCNN.

Models:

- SRCNN
- FSRCNN

Python 3.12.

## Dataset

The code expects:

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

## Install

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python3.12 -m pip install -r requirements.txt
```

## Validate

```bash
python3.12 validate_dataset.py \
  --dataset-root datasets/DIV2K \
  --scale 2
```

## Train

```bash
python3.12 train_model.py --model srcnn --scale 2 --patches-per-image 128 --epochs 30
python3.12 train_model.py --model fsrcnn --scale 2 --patches-per-image 128 --fsrcnn-augment
```

## Evaluate

Validation:

```bash
python3.12 evaluate_model.py \
  --model srcnn \
  --scale 2 \
  --checkpoint artifacts/experiments/srcnn/x2/best.pth \
  --dataset-root datasets/DIV2K
```

Test:

```bash
python3.12 evaluate_model.py \
  --model srcnn \
  --scale 2 \
  --checkpoint artifacts/experiments/srcnn/x2/best.pth \
  --dataset-root datasets/DIV2K \
  --split test
```

## Run

For LR input and paired HR input:

```bash
python3.12 run_model.py \
  --model fsrcnn \
  --scale 2 \
  --checkpoint artifacts/experiments/fsrcnn/x2/best.pth \
  --input datasets/DIV2K/val/LR_bicubic/X2 \
  --paired-hr-dir datasets/DIV2K/val/HR
```

For LR input to create HR images:

```bash
python3.12 run_model.py \
  --model srcnn \
  --scale 2 \
  --checkpoint artifacts/experiments/srcnn/x2/best.pth \
  --input path/to/lr_images_dir \
  --output-dir path/to/output
```

For HR input to degrade:

```bash
python3.12 run_model.py \
  --model fsrcnn \
  --scale 2 \
  --checkpoint artifacts/experiments/fsrcnn/x2/best.pth \
  --input datasets/DIV2K/val/HR/0801.png \
  --degrade-hr-first
```

