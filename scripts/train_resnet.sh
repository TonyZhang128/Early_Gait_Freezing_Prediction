#!/bin/bash
# 步态识别合并训练脚本：先执行对比学习训练 → 再执行微调分类训练
# 适配GSDNN/ResNet101/MSDNN等多模型，保留双日志输出、GPU选择、虚拟环境激活

set -e  # 遇到错误立即退出，保证训练稳定性

# ======================== 全局基础配置 ========================
# 虚拟环境激活路径（两个任务共用）
VENV_PATH="/root/miniconda3/envs/d2_cuda118/bin/activate"
TIME_STAMP=$(date +%m%d_%H%M)
# 日志根目录（自动创建）
LOG_ROOT_DIR="./train_logs"
mkdir -p ${LOG_ROOT_DIR}
# GPU配置（两个任务共用）
export CUDA_VISIBLE_DEVICES="0"
MODE="normal" # "debug" or "normal"
# ======================== 【任务1：对比学习训练】参数配置 ========================
# 基础配置
SELFEXP_NAME="resnet_slefsup"
# SELF_MODE="debug"
SELF_SEED=42
SELF_DEVICE="cuda"

# 数据参数
SELF_DATA_PATH="datasets/data_10000/all_data.mat"
SELF_BATCH_SIZE=256
SELF_NUM_WORKERS=4
SELF_VIEWS=2

# 模型参数（GSDNN专属）
SELF_MODEL_TYPE="ResNet"
SELF_NUM_CLASSES=1
SELF_BLOCK_N=8
SELF_INIT_CHANNELS=18
SELF_GROWTH_RATE=12
SELF_BASE_CHANNELS=48
SELF_STRIDE=2
SELF_DROPOUT_GSDNN=0.2

# 投影头参数
SELF_OUT_DIM=512
SELF_PROJ_OUT_DIM=1024
SELF_CONTRASTIVE_DIM=128
SELF_DROPOUT=0.1

# 训练参数
SELF_EPOCHS=200
SELF_LR=0.3
SELF_TEMPERATURE=0.07
SELF_MOMENTUM=0.9
SELF_WEIGHT_DECAY=0.0001
SELF_WARMUP_RATIO=0.1

# 数据增强
SELF_FREQ_KEEP_RATIO=0.6

# 保存和日志
SELF_SAVE_DIR="./save_models"/${SELFEXP_NAME}_${TIME_STAMP} # 独立目录避免覆盖
SELF_LOG_DIR_TB="./runs"/${SELFEXP_NAME}_${TIME_STAMP}
SELF_SAVE_FREQ=10
SELF_PRINT_PARAMS=True
SELF_TRAIN_SCRIPT="train_selfsup.py"

# ======================== 【任务2：微调分类训练】参数配置 ========================
# 基础配置
FINETUNE_NAME="resnet_finetune"
# FINETUNE_MODE="normal"
FINETUNE_SEED=42
FINETUNE_DEVICE="cuda"

# 数据参数
FINETUNE_DATA_PATH="./datasets/data_10000/"
FINETUNE_TRAIN_RATIO=0.7
FINETUNE_BATCH_SIZE=64
FINETUNE_NUM_WORKERS=4

# 模型参数
FINETUNE_MODEL_TYPE="ResNet"
FINETUNE_NUM_CLASSES=27
# 自动关联对比学习的最佳模型输出路径
FINETUNE_PRETRAINED_MODEL=${SELF_SAVE_DIR}/"best_model.pth"
FINETUNE_FREEZE_ENCODER="False"

# 投影头参数
FINETUNE_OUT_DIM=512
FINETUNE_PROJ_OUT_DIM=1024
FINETUNE_CONTRASTIVE_DIM=128
FINETUNE_DROPOUT=0.1

# 训练参数
FINETUNE_NUM_EPOCHS=200
FINETUNE_LEARNING_RATE=3e-4

# 数据增强
FINETUNE_AUGMENTATION_PROB=0.5
FINETUNE_FREQ_KEEP_RATIO=0.6

# 保存和日志
FINETUNE_SAVE_DIR="./save_models"/${FINETUNE_NAME}_${TIME_STAMP}  
FINETUNE_LOG_DIR_TB="./runs"/${FINETUNE_NAME}_${TIME_STAMP}
FINETUNE_TRAIN_SCRIPT="train_finetune.py"

# ======================== 激活虚拟环境 ========================
if [ -f "${VENV_PATH}" ]; then
    echo "激活虚拟环境: ${VENV_PATH}"
    source ${VENV_PATH}
else
    echo "未找到虚拟环境，使用系统Python环境"
fi

# ======================== 工具函数：打印任务分隔符 ========================
print_task_header() {
    echo -e "\n=============================================================="
    echo "============== 开始执行：$1 =============="
    echo "=============================================================="
}

# ======================== 任务1：执行对比学习训练 ========================
print_task_header "对比学习训练 (Self-supervised Training)"
SELF_LOG_FILE="${LOG_ROOT_DIR}/${SELFEXP_NAME}_${TIME_STAMP}.log"

# 拼接对比学习训练命令
SELF_TRAIN_CMD="python ${SELF_TRAIN_SCRIPT} \
    --exp_name ${SELFEXP_NAME} \
    --mode ${MODE} \
    --data_path ${SELF_DATA_PATH} \
    --batch_size ${SELF_BATCH_SIZE} \
    --num_workers ${SELF_NUM_WORKERS} \
    --views ${SELF_VIEWS} \
    --model_type ${SELF_MODEL_TYPE} \
    --num_classes ${SELF_NUM_CLASSES} \
    --block_n ${SELF_BLOCK_N} \
    --init_channels ${SELF_INIT_CHANNELS} \
    --growth_rate ${SELF_GROWTH_RATE} \
    --base_channels ${SELF_BASE_CHANNELS} \
    --stride ${SELF_STRIDE} \
    --dropout_GSDNN ${SELF_DROPOUT_GSDNN} \
    --out_dim ${SELF_OUT_DIM} \
    --proj_out_dim ${SELF_PROJ_OUT_DIM} \
    --contrastive_dim ${SELF_CONTRASTIVE_DIM} \
    --dropout ${SELF_DROPOUT} \
    --epochs ${SELF_EPOCHS} \
    --base_lr ${SELF_LR} \
    --momentum ${SELF_MOMENTUM} \
    --weight_decay ${SELF_WEIGHT_DECAY} \
    --warmup_ratio ${SELF_WARMUP_RATIO} \
    --temperature ${SELF_TEMPERATURE} \
    --freq_keep_ratio ${SELF_FREQ_KEEP_RATIO} \
    --save_dir ${SELF_SAVE_DIR} \
    --log_dir ${SELF_LOG_DIR_TB} \
    --save_freq ${SELF_SAVE_FREQ} \
    --print_params ${SELF_PRINT_PARAMS} \
    --device ${SELF_DEVICE} \
    --seed ${SELF_SEED}"

# 输出并执行对比学习命令
echo -e "\n【对比学习训练命令】:"
echo "=============================================================="
echo ${SELF_TRAIN_CMD}
echo "=============================================================="
${SELF_TRAIN_CMD} 2>&1 | tee ${SELF_LOG_FILE}

# 验证对比学习模型是否生成
if [ ! -f "${FINETUNE_PRETRAINED_MODEL}" ]; then
    echo -e "\n警告：对比学习训练未生成预期的预训练模型文件！"
    echo "路径：${FINETUNE_PRETRAINED_MODEL}"
    echo "请检查对比学习训练是否正常完成，微调训练将跳过！"
    exit 1
fi

# ======================== 任务2：执行微调分类训练 ========================
print_task_header "微调分类训练 (Finetune Classification Training)"
FINETUNE_LOG_FILE="${LOG_ROOT_DIR}/${FINETUNE_NAME}_${TIME_STAMP}.log"

# 拼接微调训练命令
FINETUNE_TRAIN_CMD="python ${FINETUNE_TRAIN_SCRIPT} \
    --exp_name ${FINETUNE_NAME} \
    --mode ${MODE} \
    --data_path ${FINETUNE_DATA_PATH} \
    --train_ratio ${FINETUNE_TRAIN_RATIO} \
    --batch_size ${FINETUNE_BATCH_SIZE} \
    --num_workers ${FINETUNE_NUM_WORKERS} \
    --num_epochs ${FINETUNE_NUM_EPOCHS} \
    --learning_rate ${FINETUNE_LEARNING_RATE} \
    --num_classes ${FINETUNE_NUM_CLASSES} \
    --model_type ${FINETUNE_MODEL_TYPE} \
    --pretrained_model ${FINETUNE_PRETRAINED_MODEL} \
    --out_dim ${FINETUNE_OUT_DIM} \
    --proj_out_dim ${FINETUNE_PROJ_OUT_DIM} \
    --contrastive_dim ${FINETUNE_CONTRASTIVE_DIM} \
    --dropout ${FINETUNE_DROPOUT} \
    --augmentation_prob ${FINETUNE_AUGMENTATION_PROB} \
    --freq_keep_ratio ${FINETUNE_FREQ_KEEP_RATIO} \
    --device ${FINETUNE_DEVICE} \
    --log_dir ${FINETUNE_LOG_DIR_TB} \
    --save_dir ${FINETUNE_SAVE_DIR} \
    --seed ${FINETUNE_SEED}"

# 处理冻结编码器布尔参数
if [ "${FINETUNE_FREEZE_ENCODER}" = "True" ]; then
    FINETUNE_TRAIN_CMD="${FINETUNE_TRAIN_CMD} --freeze_encoder"
fi

# 输出并执行微调命令
echo -e "\n【微调分类训练命令】:"
echo "=============================================================="
echo ${FINETUNE_TRAIN_CMD}
echo "=============================================================="
${FINETUNE_TRAIN_CMD} 2>&1 | tee ${FINETUNE_LOG_FILE}

# ======================== 训练完成 ========================
echo -e "\n=============================================================="
echo "============== 所有训练任务执行完成 =============="
echo "=============================================================="
echo "对比学习日志: ${SELF_LOG_FILE}"
echo "对比学习模型: ${SELF_SAVE_DIR}"
echo "微调分类日志: ${FINETUNE_LOG_FILE}"
echo "微调分类模型: ${FINETUNE_SAVE_DIR}"
echo "完成时间: $(date)"
echo "=============================================================="