# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Gait freezing prediction using self-supervised contrastive learning (SimCLR) + fine-tuning classification. The pipeline: (1) pre-train an encoder via contrastive learning on unlabeled gait data, (2) fine-tune with a classifier head on labeled data.

## Commands

```bash
# Self-supervised contrastive pre-training
uv run python train_selfsup.py [args]

# Fine-tune classification training
uv run python train_finetune.py [args]

# Full pipeline (self-sup + finetune)
bash scripts/train.sh
```

Key flags: `--mode debug` for quick iteration, `--model_type GSDNN|ResNet|Conformer`, `--device cuda|cpu`.

## Project Structure

```
├── datasets/
│   ├── Contrastive_dataset.py    # SimCLR data loading + augmentation
│   ├── Gait_dataset_new.py       # Finetune data loading + augmentation
│   └── data_10000/               # .mat data files (all_data, sub_data, labels)
├── models/
│   ├── GSDNN_new.py              # Custom 1D CNN with SE + spatial attention
│   ├── resnet.py                 # ResNet18/34/50/101 (2D conv adapted for gait)
│   ├── Conformer.py              # Transformer + CNN hybrid for EEG/gait
│   ├── EEGNet.py                 # EEGNet architecture
│   └── DNN.py                    # Baseline DNN
├── utils/
│   ├── analysis.py               # set_seed(), save_checkpoint()
│   ├── calculate.py              # FLOPs computation (via thop)
│   └── plt_curves.py             # Training curves + confusion matrix
├── scripts/
│   ├── train.sh                  # Full pipeline (self-sup → finetune)
│   ├── train_selfsup.sh          # Self-supervised only
│   └── train_finetune.sh         # Fine-tune only
├── train_selfsup.py              # SimCLR contrastive learning entrypoint
├── train_finetune.py             # Classification fine-tuning entrypoint
└── main.py                       # Placeholder
```

## Training Pipeline

### Stage 1: Self-supervised (SimCLR)
- `train_selfsup.py` → `SimCLRModel` (encoder + projector) + `ContrastiveLoss` (NT-Xent)
- Data: `all_data.mat` (unlabeled), augmentation via `GaitAugmentation` (freq dropout, channel shuffle, time reversal, random crop/erase)
- LR schedule: linear warmup → cosine annealing
- Saves: checkpoints + `best_model.pth`

### Stage 2: Fine-tune Classification
- `train_finetune.py` → `SimCLREncoder` + `ClassificationModel` (classifier head)
- Loads pretrained encoder weights, optionally freezes encoder
- Metrics: accuracy, precision, recall, F1, AUC
- Saves: best model per accuracy, final model, confusion matrix

## Data Format

- Input shape: `[batch, 18, 101]` (18 channels, 101 time steps)
- Labels: Subject IDs (1-27), converted to 0-based indices
- Self-sup uses `all_data.mat`; finetune uses `sub_data.mat` + `sub_label.mat`

## Architecture Notes

- **GSDNN_new**: Default encoder. Multi-scale 1D conv branches, SE block + spatial attention, residual connections with stride/padding matching
- **ResNet18**: 2D conv adapted for gait (1 input channel), standard residual blocks
- When used as encoder (SimCLR), the classifier FC layer is removed and a projection head is attached
