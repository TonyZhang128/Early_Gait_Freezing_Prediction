#!/bin/bash
set -e
source "$(dirname "$0")/common.sh"

activate_venv

export CUDA_VISIBLE_DEVICES="0"
MODEL_TYPE="ResNet"
LOG_DIR="./train_logs"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)

# ---------- Self-supervised stage ----------
SELF_NAME="Gait_selfsup_GSDNN_baseline"
SELF_SAVE="./save_models/${SELF_NAME}_${TS}"

SELF_CMD="python train_selfsup.py \
    --exp_name $SELF_NAME --mode normal \
    --data_path datasets/data_10000/all_data.mat \
    --batch_size 256 --num_workers 0 --views 2 \
    --model_type $MODEL_TYPE --num_classes 1 \
    --block_n 8 --init_channels 18 --growth_rate 12 \
    --base_channels 48 --stride 2 --dropout_GSDNN 0.2 \
    --out_dim 512 --proj_out_dim 1024 --contrastive_dim 128 \
    --dropout 0.1 --epochs 200 --base_lr 0.3 \
    --momentum 0.9 --weight_decay 0.0001 --warmup_ratio 0.05 \
    --temperature 0.1 --freq_keep_ratio 0.6 \
    --save_dir $SELF_SAVE --log_dir ./runs/${SELF_NAME}_${TS} 
    "

run_train "Self-supervised Training" "$SELF_CMD" "$LOG_DIR/${SELF_NAME}_${TS}.log"

PRETRAINED="$SELF_SAVE/best_model.pth"
[ -f "$PRETRAINED" ] || { echo "Pretrained model not found: $PRETRAINED"; exit 1; }

# ---------- Fine-tune stage ----------
FINETUNE_NAME="Gait_finetune_GSDNN_baseline"
FINETUNE_SAVE="./save_models/${FINETUNE_NAME}_${TS}"

FINETUNE_CMD="python train_finetune.py \
    --exp_name $FINETUNE_NAME --mode normal \
    --data_path ./datasets/data_10000/ --train_ratio 0.7 \
    --batch_size 64 --num_workers 0 --num_epochs 200 \
    --learning_rate 3e-4 --num_classes 27 --model_type $MODEL_TYPE \
    --pretrained_model $PRETRAINED \
    --out_dim 512 --proj_out_dim 1024 --contrastive_dim 128 \
    --dropout 0.1 --augmentation_prob 0.5 --freq_keep_ratio 0.6 \
    --log_dir ./runs/${FINETUNE_NAME}_${TS} \
    --save_dir $FINETUNE_SAVE"

run_train "Fine-tune Training" "$FINETUNE_CMD" "$LOG_DIR/${FINETUNE_NAME}_${TS}.log"
