# Quick Start

This guide will help you get started with MM-GRPO by training a Stable Diffusion 3.5 model on an OCR task using Flow-GRPO.

## Prerequisites

- Completed [Installation](installation.md)
- Access to NVIDIA GPUs (8 GPUs recommended for the examples)
- Dataset prepared (see below)

## Step 1: Prepare Your Dataset

Download the OCR dataset from the [Flow-GRPO repository](https://github.com/yifan123/flow_grpo/tree/main/dataset/ocr) and place it in a `dataset` folder.

The dataset should contain:
- Training file: `dataset/ocr/train.txt`
- Validation file: `dataset/ocr/test.txt`

Each line in these files should contain a text prompt for image generation.

## Step 2: Configure Training Paths

Update the dataset paths in your training script. The example scripts use environment variables:

```bash
export TRAIN_FILE=$HOME/dataset/ocr/train.txt
export VAL_FILE=$HOME/dataset/ocr/test.txt
```

Or modify the scripts directly to use your paths.

## Step 3: Run Flow-GRPO Training

### Standard Flow-GRPO

```bash
bash examples/flowgrpo_trainer/run_sd3.sh
```

This script trains SD3.5-M with:

- Flow-GRPO algorithm

- PaddleOCR reward

- 8 GPUs

- Batch size 64

- Learning rate 3e-4

### Flow-GRPO-Fast

For faster training with improved sampling efficiency:

```bash
bash examples/flowgrpo_trainer/run_sd3_fast.sh
```

## Monitoring Training

### Console Output

The training process prints:

- Training metrics (loss, reward, etc.)

- Validation metrics

- Checkpoint saves

### WandB Integration

If WandB is configured (`trainer.logger='["console", "wandb"]'`), you can monitor:

- Training curves

- Validation rewards

- Generated samples

Access your dashboard at [wandb.ai](https://wandb.ai).

## Next Steps

- Learn more about [Algorithms](user-guide/algorithms.md)
- Understand [Configuration](user-guide/configuration.md)
