# PhenoBench Weed Segmentation

This repository contains a compact semantic segmentation pipeline for experimenting with the [PhenoBench](https://www.phenobench.org/) dataset, focused on weed detection in agricultural field images.

It supports two training targets:

- `weed_binary`: weed vs. non-weed
- `multiclass`: soil, crop, weed

The codebase is intended to be easy to read and modify. It includes dataset loading, model definition, training, evaluation, and tiled inference for larger images.

## Disclaimer

- this repository does not provide, redistribute, or own the PhenoBench dataset
- PhenoBench is an external dataset created and maintained by its original authors
- this project is only a testbed for semantic segmentation experiments and model comparisons on that dataset
- you must obtain the dataset from the official PhenoBench website and follow its license terms

## What This Project Does

- loads RGB images and semantic masks from PhenoBench
- maps raw dataset labels into training labels
- trains a lightweight `DeepLabV3 + MobileNetV3` baseline
- evaluates `IoU` per class and `mIoU`
- runs tiled inference and saves overlay predictions

## Dataset Summary

PhenoBench provides:

- `1407` training images
- `772` validation images
- `693` test images
- image size `1024 x 1024`

This code expects a local copy of:

- `PhenoBench/train/images`
- `PhenoBench/train/semantics`
- `PhenoBench/val/images`
- `PhenoBench/val/semantics`

## Label Mapping

Raw semantic labels in PhenoBench:

- `0`: background / soil
- `1`: crop
- `2`: weed
- `3`: partial-crop
- `4`: partial-weed

Mapping used in this repository:

- `multiclass`: `0=soil`, `1=crop`, `2=weed`
- `weed_binary`: `0=non_weed`, `1=weed`

Conversion rules:

- `3 -> 1` in `multiclass`
- `4 -> 2` in `multiclass`
- `{0,1,3} -> 0` in `weed_binary`
- `{2,4} -> 1` in `weed_binary`

This makes the binary task suitable for weed-only segmentation, where crop and soil are treated as the negative class.

## Repository Structure

- [src/data_phenobench.py](/home/vahab/dev/phenobench/src/data_phenobench.py): dataset loader and label mapping
- [src/models.py](/home/vahab/dev/phenobench/src/models.py): DeepLabV3-MobileNetV3 model builder
- [src/train_semseg.py](/home/vahab/dev/phenobench/src/train_semseg.py): training loop and checkpointing
- [src/eval_semseg.py](/home/vahab/dev/phenobench/src/eval_semseg.py): evaluation on train or validation split
- [src/infer_tile_5mp.py](/home/vahab/dev/phenobench/src/infer_tile_5mp.py): tiled inference and overlay export
- [src/metrics.py](/home/vahab/dev/phenobench/src/metrics.py): IoU and mIoU computation
- [docker/Dockerfile](/home/vahab/dev/phenobench/docker/Dockerfile): container image for reproducible runs
- [docker/run.sh](/home/vahab/dev/phenobench/docker/run.sh): helper wrapper for Docker execution

## Local Python Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Train weed-only segmentation:

```bash
python -m src.train_semseg \
  --data_root ./PhenoBench \
  --task weed_binary \
  --epochs 5 \
  --image_size 512 \
  --batch_size 4 \
  --amp
```

Evaluate the best checkpoint:

```bash
python -m src.eval_semseg \
  --data_root ./PhenoBench \
  --task weed_binary \
  --split val \
  --ckpt runs/ckpt_best.pt \
  --image_size 512
```

Run the original 3-class setup:

```bash
python -m src.train_semseg --data_root ./PhenoBench --task multiclass --epochs 5 --image_size 512 --batch_size 4 --amp
python -m src.eval_semseg --data_root ./PhenoBench --task multiclass --split val --ckpt runs/ckpt_best.pt --image_size 512
```

## Docker Usage

Build the image:

```bash
docker build -t phenobench-semseg -f docker/Dockerfile .
```

### Quick Linux Commands

These are the direct Docker commands for building, training, and evaluation:

```bash
# build
docker build -t phenobench-semseg -f docker/Dockerfile .

# train
docker run --rm -it --gpus all -v "$PWD":/workspace -w /workspace phenobench-semseg \
  python -m src.train_semseg --data_root /workspace/PhenoBench --epochs 5 --image_size 512 --batch_size 4 --amp

# eval
docker run --rm -it --gpus all -v "$PWD":/workspace -w /workspace phenobench-semseg \
  python -m src.eval_semseg --data_root /workspace/PhenoBench --split val --ckpt runs/ckpt_best.pt --image_size 512
```

Explanation:

- `docker build -t phenobench-semseg -f docker/Dockerfile .`
  creates the Docker image named `phenobench-semseg`
- `docker run`
  starts a container from that image
- `--rm`
  removes the container after the command finishes
- `-it`
  keeps the run interactive and attaches a terminal
- `--gpus all`
  exposes all visible NVIDIA GPUs to the container
- `-v "$PWD":/workspace`
  mounts your current repository into the container at `/workspace`
- `-w /workspace`
  sets the container working directory to the mounted repository
- `python -m src.train_semseg`
  runs the training entry point
- `python -m src.eval_semseg`
  runs the evaluation entry point
- `--data_root /workspace/PhenoBench`
  tells the code where the dataset is inside the container
- `--epochs 5`
  trains for 5 passes over the training set
- `--image_size 512`
  resizes images and masks to `512 x 512` during training and evaluation
- `--batch_size 4`
  uses 4 images per training step
- `--amp`
  enables automatic mixed precision, which is useful on CUDA GPUs
- `--ckpt runs/ckpt_best.pt`
  evaluates the checkpoint with the best validation score saved during training

Important:

- because `--task` is not provided in these commands, they run the default task: `multiclass`
- if you want weed-only segmentation, add `--task weed_binary`

Weed-only versions of the same commands:

```bash
docker run --rm -it --gpus all -v "$PWD":/workspace -w /workspace phenobench-semseg \
  python -m src.train_semseg --data_root /workspace/PhenoBench --task weed_binary --epochs 5 --image_size 512 --batch_size 4 --amp

docker run --rm -it --gpus all -v "$PWD":/workspace -w /workspace phenobench-semseg \
  python -m src.eval_semseg --data_root /workspace/PhenoBench --task weed_binary --split val --ckpt runs/ckpt_best.pt --image_size 512
```

### Helper Script

The helper script wraps the Docker build and run steps:

```bash
chmod +x docker/run.sh
./docker/run.sh python -m src.train_semseg \
  --data_root /workspace/PhenoBench --epochs 5 --image_size 512 --batch_size 4 --amp

./docker/run.sh python -m src.eval_semseg \
  --data_root /workspace/PhenoBench --split val --ckpt runs/ckpt_best.pt --image_size 512
```

Your `docker/run.sh` script does two things:

- builds the `phenobench-semseg` image first
- runs the container with `--gpus all` and mounts the repo at `/workspace`

That means your helper-script commands are equivalent to the direct Docker commands above, with one difference:

- the script rebuilds the Docker image every time you call it

If you want weed-only segmentation through the helper script, use:

```bash
./docker/run.sh python -m src.train_semseg \
  --data_root /workspace/PhenoBench --task weed_binary --epochs 5 --image_size 512 --batch_size 4 --amp

./docker/run.sh python -m src.eval_semseg \
  --data_root /workspace/PhenoBench --task weed_binary --split val --ckpt runs/ckpt_best.pt --image_size 512
```

### CUDA Notes

- `--gpus all` only works if the host machine has a working NVIDIA driver and Docker GPU runtime
- `--amp` is most useful when CUDA is available
- if CUDA is not available, remove `--amp` and run on CPU
- if needed, you can also remove `--gpus all` from the Docker command for CPU-only runs

## Inference

Run tiled inference on a single image and save an overlay:

```bash
python -m src.infer_tile_5mp \
  --image ./PhenoBench/val/images/<image_name>.png \
  --ckpt runs/ckpt_best.pt \
  --task weed_binary \
  --out_png pred_overlay.png
```

## Notes

- the dataset is not included in this repository
- weed pixels are a very small fraction of the dataset, so class imbalance is significant
- the current baseline uses standard cross-entropy loss without class weighting
- the validation and test images come from the official PhenoBench split structure

## Reference

PhenoBench dataset website:

- https://www.phenobench.org/
- https://www.phenobench.org/dataset.html
