#!/bin/bash
# 步态识别对比学习训练脚本（适配GSDNN/ResNet/EEGNet/Conformer）
# 支持多模型类型、GSDNN专属参数配置、日志记录、GPU选择

set -e  # 遇到错误立即退出，保证训练稳定性

# ======================== 基础环境配置 ========================
# 替换为你的Python虚拟环境激活路径（无需则注释）
VENV_PATH="/root/miniconda3/envs/d2_cuda118/bin/activate"
# 训练主脚本名称（如train.py/main.py，需替换为你的实际脚本名）
TRAIN_SCRIPT="train_selfsup.py"
# 日志目录（自动创建，避免手动操作）
LOG_DIR="./train_logs"
mkdir -p ${LOG_DIR}

# ======================== 核心训练参数配置 ========================
# 基础配置
EXP_NAME="Gait_selfsup_GSDNN_baseline"
MODE="normal"  # debug/normal，debug模式建议减小batch_size和epochs
SEED=42
DEVICE="cuda"  # cuda/cpu，自动检测可设为None

# 数据参数
DATA_PATH="datasets/data_10000/all_data.mat"
BATCH_SIZE=64
NUM_WORKERS=4  # 建议设为CPU核心数（如4/8），比0效率更高
VIEWS=2  # 对比学习视图数量

# 模型参数（核心：支持多模型类型，适配GSDNN专属参数）
MODEL_TYPE="GSDNN"  # 可选：GSDNN/ResNet/EEGNet/Conformer

## GSDNN专属参数（MODEL_TYPE=GSDNN时生效）
NUM_CLASSES=1
BLOCK_N=8
INIT_CHANNELS=18
GROWTH_RATE=12
BASE_CHANNELS=48
STRIDE=2
DROPOUT_GSDNN=0.2

## 投影头通用参数
OUT_DIM=132
PROJ_OUT_DIM=128
CONTRASTIVE_DIM=256
DROPOUT=0.5

# 训练参数
EPOCHS=200
LR=3e-4
TEMPERATURE=0.5

# 数据增强参数
FREQ_KEEP_RATIO=0.6

# 保存和日志参数
SAVE_DIR="./save_models/${MODEL_TYPE}_exp"  # 按模型类型分目录
LOG_DIR_TB="./runs/${MODEL_TYPE}_exp"
SAVE_FREQ=5

# ======================== GPU配置 ========================
# 指定GPU（单卡：0，多卡：0,1,2），注释则自动使用所有可用GPU
export CUDA_VISIBLE_DEVICES="0"

# ======================== 激活虚拟环境（可选） ========================
if [ -f "${VENV_PATH}" ]; then
    echo "激活虚拟环境: ${VENV_PATH}"
    source ${VENV_PATH}
else
    echo "未找到虚拟环境，使用系统Python环境"
fi

# ======================== 开始训练 ========================
echo "========================================"
echo "训练启动时间: $(date)"
echo "实验名称: ${EXP_NAME}"
echo "模型类型: ${MODEL_TYPE}"
echo "训练模式: ${MODE}"
echo "使用设备: ${DEVICE}"
echo "日志保存路径: ${LOG_DIR}/${EXP_NAME}_${MODEL_TYPE}_$(date +%Y%m%d_%H%M%S).log"
echo "模型保存路径: ${SAVE_DIR}"
echo "========================================"

# 拼接完整训练命令（严格匹配parse_args的参数名）
TRAIN_CMD="python ${TRAIN_SCRIPT} \
    --exp_name ${EXP_NAME} \
    --mode ${MODE} \
    --data_path ${DATA_PATH} \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --views ${VIEWS} \
    --model_type ${MODEL_TYPE} \
    --num_classes ${NUM_CLASSES} \
    --block_n ${BLOCK_N} \
    --init_channels ${INIT_CHANNELS} \
    --growth_rate ${GROWTH_RATE} \
    --base_channels ${BASE_CHANNELS} \
    --stride ${STRIDE} \
    --dropout_GSDNN ${DROPOUT_GSDNN} \
    --out_dim ${OUT_DIM} \
    --proj_out_dim ${PROJ_OUT_DIM} \
    --contrastive_dim ${CONTRASTIVE_DIM} \
    --dropout ${DROPOUT} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --temperature ${TEMPERATURE} \
    --freq_keep_ratio ${FREQ_KEEP_RATIO} \
    --save_dir ${SAVE_DIR} \
    --log_dir ${LOG_DIR_TB} \
    --save_freq ${SAVE_FREQ} \
    --device ${DEVICE} \
    --seed ${SEED}"

# ======================== 新增：输出完整的TRAIN_CMD ========================
echo -e "\n【即将执行的训练命令】:"
echo "=============================================================="
echo ${TRAIN_CMD}  # 单行输出（便于复制执行）
echo "=============================================================="

# 执行训练并记录日志（终端+文件双输出）
${TRAIN_CMD} 2>&1 | tee ${LOG_DIR}/${EXP_NAME}_${MODEL_TYPE}_$(date +%Y%m%d_%H%M%S).log

# ======================== 训练结束 ========================
echo "========================================"
echo "训练结束时间: $(date)"
echo "日志文件: ${LOG_DIR}/${EXP_NAME}_${MODEL_TYPE}_$(date +%Y%m%d_%H%M%S).log"
echo "模型文件: ${SAVE_DIR}"
echo "========================================"