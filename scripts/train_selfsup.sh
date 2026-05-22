#!/bin/bash
set -e
source "$(dirname "$0")/common.sh"

activate_venv

export CUDA_VISIBLE_DEVICES="0"

LOG_DIR="./train_logs"
mkdir -p "$LOG_DIR"

EXP_NAME="Gait_selfsup_GSDNN_baseline"
MODEL_TYPE="ResNet"
LOG_FILE="$LOG_DIR/${EXP_NAME}_${MODEL_TYPE}_$(date +%Y%m%d_%H%M%S).log"

CMD="python train_selfsup.py \
    --exp_name $EXP_NAME --mode normal \
    --data_path datasets/data_10000/all_data.mat \
    --batch_size 256 --num_workers 0 --views 2 \
    --model_type $MODEL_TYPE --num_classes 1 \
    --block_n 8 --init_channels 18 --growth_rate 12 \
    --base_channels 48 --stride 2 --dropout_GSDNN 0.2 \
    --out_dim 512 --proj_out_dim 1024 --contrastive_dim 128 \
    --dropout 0.1 --epochs 200 --base_lr 0.3 \
    --momentum 0.9 --weight_decay 0.0001 --warmup_ratio 0.05 \
    --temperature 0.1 --freq_keep_ratio 0.6 \
    --save_dir ./save_models "

run_train "Self-supervised Training" "$CMD" "$LOG_FILE"
