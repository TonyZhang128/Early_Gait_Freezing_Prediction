#!/bin/bash
set -e
source "$(dirname "$0")/common.sh"

activate_venv 

export CUDA_VISIBLE_DEVICES="0"

LOG_DIR="./train_logs"
mkdir -p "$LOG_DIR"

EXP_NAME="Gait_finetune_ResNet_baseline"
MODEL_TYPE="ResNet"
LOG_FILE="$LOG_DIR/${EXP_NAME}_${MODEL_TYPE}_$(date +%Y%m%d_%H%M%S).log"
pretrained_model="./save_models/Gait_selfsup_GSDNN_baseline_20260521_184607/best_model.pth"

CMD="python train_finetune.py \
    --exp_name $EXP_NAME --mode normal \
    --data_path ./datasets/data_10000/ --train_ratio 0.7 \
    --batch_size 64 --num_workers 0 --num_epochs 200 \
    --learning_rate 3e-2 --num_classes 27 --model_type $MODEL_TYPE \
    --pretrained_model $pretrained_model --freeze_encoder \
    --out_dim 512 --proj_out_dim 1024 --contrastive_dim 128 \
    --dropout 0.5 --augmentation_prob 0.5 --freq_keep_ratio 0.6 \
    --device cuda --log_dir ./runs/${EXP_NAME}_$(date +%Y%m%d_%H%M%S) \
    --save_dir ./save_models/${EXP_NAME}_$(date +%Y%m%d_%H%M%S) --seed 42"

run_train "Fine-tune Training" "$CMD" "$LOG_FILE"
