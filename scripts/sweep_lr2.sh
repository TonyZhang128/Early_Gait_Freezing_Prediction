#!/bin/bash
set -e
source "$(dirname "$0")/common.sh"

activate_venv

export CUDA_VISIBLE_DEVICES="0"

LOG_DIR="./train_logs"
mkdir -p "$LOG_DIR"

PRETRAINED_MODEL="./save_models/Gait_selfsup_GSDNN_baseline_20260521_184607/best_model.pth"

# LR sweep: 3e-2 ~ 1e-1 already done, now extending upward with dropout=0
LRS=("3e-2" "5e-2" "7e-2" "1e-1" "3e-1" "5e-1")

for LR in "${LRS[@]}"; do
    EXP_NAME="Gait_finetune_ResNet_lr${LR}_e200"
    MODEL_TYPE="ResNet"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$LOG_DIR/${EXP_NAME}_${MODEL_TYPE}_${TIMESTAMP}.log"

    echo ""
    echo "============================================================"
    echo "  Sweep: learning_rate=$LR, dropout=0.0, freeze_encoder"
    echo "============================================================"
    echo ""

    CMD="python train_finetune.py \
        --exp_name $EXP_NAME --mode normal \
        --data_path ./datasets/data_10000/ --train_ratio 0.7 \
        --batch_size 64 --num_workers 0 --num_epochs 200 \
        --learning_rate $LR --num_classes 27 --model_type $MODEL_TYPE \
        --pretrained_model $PRETRAINED_MODEL --freeze_encoder \
        --out_dim 512 --proj_out_dim 1024 --contrastive_dim 128 \
        --dropout 0.0 --augmentation_prob 0.5 --freq_keep_ratio 0.6 \
        --device cuda --log_dir ./runs/${EXP_NAME}_${TIMESTAMP} \
        --save_dir ./save_models/${EXP_NAME}_${TIMESTAMP} --seed 42"

    run_train "Fine-tune LR=$LR" "$CMD" "$LOG_FILE"
done

echo ""
echo "============================================================"
echo "  Sweep complete. Summary:"
echo "============================================================"
for LR in "${LRS[@]}"; do
    BEST=$(grep "Best test accuracy" "$LOG_DIR"/Gait_finetune_ResNet_lr${LR}_nodrop_ResNet_*.log 2>/dev/null | tail -1 | awk '{print $NF}')
    echo "  lr=$LR dropout=0 -> best_test_acc=$BEST"
done
