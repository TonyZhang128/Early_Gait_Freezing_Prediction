#!/bin/bash
# 步态识别分类训练脚本（适配GSDNN/ResNet101/MSDNN等多模型）
# 支持分类训练专属参数、日志双输出、GPU选择、虚拟环境激活

set -e  # 遇到错误立即退出，保证训练稳定性

# ======================== 基础环境配置 ========================
# 替换为你的Python虚拟环境激活路径（无需则注释）
VENV_PATH="/home/user/miniconda3/envs/gait_env/bin/activate"
# 训练主脚本名称（如train.py/main.py，需替换为你的实际脚本名）
TRAIN_SCRIPT="train.py"
# 日志目录（自动创建，避免手动操作）
LOG_DIR="./train_logs"
mkdir -p ${LOG_DIR}

# ======================== 核心训练参数配置 ========================
# 基础配置
EXP_NAME="Gait_finetune_GSDNN_final"
MODE="debug"  # normal/debug，debug模式建议减小batch_size和epochs
SEED=42
DEVICE="cuda"  # cuda/cpu，自动检测可设为None

# 数据参数
DATA_PATH="./datasets/data_10000/"
TRAIN_RATIO=0.7  # 训练集比例
BATCH_SIZE=64
NUM_WORKERS=4  # 建议设为CPU核心数（如4/8），比0效率更高

# 模型参数（分类训练专属，适配多模型类型）
MODEL_TYPE="GSDNN"  # 可选：DNN/GSDNN/GSDNN2/GSDNN_new/MSDNN/ResNet101
NUM_CLASSES=27      # 分类数量（分类训练核心参数）
PRETRAINED_MODEL="./save_models/Gait_self_supervised_training/best_model.pth"  # 预训练模型路径
FREEZE_ENCODER="False"  # True/False，是否冻结编码器参数

## 投影头通用参数（分类训练保留投影头配置）
OUT_DIM=132
PROJ_OUT_DIM=128
CONTRASTIVE_DIM=256
DROPOUT=0.5

# 训练参数
NUM_EPOCHS=20
LEARNING_RATE=3e-4

# 数据增强参数
AUGMENTATION_PROB=0.5  # 数据增强概率
FREQ_KEEP_RATIO=0.6    # 频率成分保留比例

# 保存和日志参数
SAVE_DIR="./save_models/${MODEL_TYPE}_finetune_exp"  # 按模型类型分目录
LOG_DIR_TB="./runs/${MODEL_TYPE}_finetune_exp"       # TensorBoard日志目录

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
    --train_ratio ${TRAIN_RATIO} \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --num_epochs ${NUM_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --num_classes ${NUM_CLASSES} \
    --model_type ${MODEL_TYPE} \
    --pretrained_model ${PRETRAINED_MODEL} \
    --out_dim ${OUT_DIM} \
    --proj_out_dim ${PROJ_OUT_DIM} \
    --contrastive_dim ${CONTRASTIVE_DIM} \
    --dropout ${DROPOUT} \
    --augmentation_prob ${AUGMENTATION_PROB} \
    --freq_keep_ratio ${FREQ_KEEP_RATIO} \
    --device ${DEVICE} \
    --log_dir ${LOG_DIR_TB} \
    --save_dir ${SAVE_DIR} \
    --seed ${SEED}"

# 处理布尔参数：冻结编码器（仅当FREEZE_ENCODER为True时添加）
if [ "${FREEZE_ENCODER}" = "True" ]; then
    TRAIN_CMD="${TRAIN_CMD} --freeze_encoder"
fi

# ======================== 输出完整的TRAIN_CMD ========================
echo -e "\n【即将执行的训练命令】:"
echo "=============================================================="
echo ${TRAIN_CMD}  # 单行输出（便于复制执行）
echo "=============================================================="
echo -e "注：以上命令会同时输出到终端和日志文件\n"

# 执行训练并记录日志（终端+文件双输出）
${TRAIN_CMD} 2>&1 | tee ${LOG_DIR}/${EXP_NAME}_${MODEL_TYPE}_$(date +%Y%m%d_%H%M%S).log

# ======================== 训练结束 ========================
echo "========================================"
echo "训练结束时间: $(date)"
echo "日志文件: ${LOG_DIR}/${EXP_NAME}_${MODEL_TYPE}_$(date +%Y%m%d_%H%M%S).log"
echo "模型文件: ${SAVE_DIR}"
echo "========================================"