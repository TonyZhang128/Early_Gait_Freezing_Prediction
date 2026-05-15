# Early Gait Freezing Prediction

A PyTorch-based framework for early freezing of gait (FOG) prediction using self-supervised contrastive learning and fine-tuning classification.

## Pipeline

Two-stage training pipeline:

1. **Self-supervised Contrastive Pre-training** (`train_selfsup.py`)
   - SimCLR framework: learns gait representations from unlabeled data via NT-Xent loss
   - Data augmentation: frequency dropout, channel shuffle, time reversal, random crop/erase
   - LR schedule: linear warmup + cosine annealing
   - Saves encoder checkpoint for downstream fine-tuning

2. **Fine-tune Classification** (`train_finetune.py`)
   - Loads pretrained encoder, attaches a classifier head
   - Optionally freezes encoder to train classifier only
   - Metrics: accuracy, precision, recall, F1, AUC
   - Saves best model (per test accuracy), final model, and confusion matrix

## Project Structure

```
├── datasets/
│   ├── Contrastive_dataset.py    # SimCLR data loading + augmentation
│   ├── Gait_dataset_new.py       # Finetune data loading + augmentation
│   ├── Gait_dataset_old.py       # Legacy dataset (deprecated)
│   └── data_10000/               # .mat data files
│       ├── all_data.mat          # Full dataset (self-supervised training)
│       ├── sub_data.mat          # Labeled subset (fine-tuning)
│       └── sub_label.mat         # Labels for sub_data
├── models/
│   ├── GSDNN_new.py              # Default encoder: multi-scale 1D CNN + SE + spatial attention
│   ├── resnet.py                 # ResNet18/34/50/101 (2D conv adapted for gait)
│   ├── Conformer.py              # Transformer + CNN hybrid
│   ├── EEGNet.py                 # EEGNet architecture
│   └── DNN.py                    # Baseline DNN
├── utils/
│   ├── analysis.py               # set_seed(), save_checkpoint()
│   ├── calculate.py              # FLOPs computation (via thop)
│   └── plt_curves.py             # Training curves + confusion matrix
├── scripts/
│   ├── train.sh                  # Full pipeline (self-sup -> finetune)
│   ├── train_selfsup.sh          # Self-supervised only
│   └── train_finetune.sh         # Fine-tune only
├── train_selfsup.py              # Entrypoint: self-supervised training
├── train_finetune.py             # Entrypoint: fine-tuning classification
├── save_models/                  # Saved checkpoints (auto-generated)
├── runs/                         # TensorBoard logs (auto-generated)
├── train_logs/                   # Training log files (auto-generated)
└── pyproject.toml                # Project config and dependencies
```

## Requirements

- Python >= 3.12
- PyTorch (with CUDA recommended)
- See `pyproject.toml` for core dependencies

## Installation

```bash
uv sync
```

## Usage

### Self-supervised Pre-training

```bash
bash scripts/train_selfsup.sh
```

### Fine-tune Classification

```bash
bash scripts/train_finetune.sh
```

### Full Pipeline

```bash
bash scripts/train.sh
```

### Debug Mode

Add `--mode debug` to run a single batch per epoch for quick validation.

## Models

| Model | Description |
|-------|-------------|
| **GSDNN_new** | Default. Multi-branch 1D CNN with SE block + spatial attention. Processes gait channels in high/mid/low frequency bands |
| **ResNet18** | Standard ResNet18 adapted for 1-channel gait input |
| **Conformer** | CNN + Transformer hybrid (originally for EEG) |
| **EEGNet** | Compact CNN originally for EEG |
| **DNN** | Baseline deep 1D CNN with residual blocks |

## Data Format

- Input: `[batch, 18, 101]` — 18 sensor channels, 101 time steps
- Labels: Subject IDs 1-27 (converted to 0-based indices internally)
- Fine-tuning: 70/30 train/test split by default

## Monitoring

```bash
tensorboard --logdir=./runs
```

TensorBoard tracks loss, accuracy, precision, recall, F1, AUC, and learning rate.
