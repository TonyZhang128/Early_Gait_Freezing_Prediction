#!/bin/bash
# 步态识别对比学习训练脚本
# 支持自定义参数、GPU选择、日志记录、后台运行等功能

set -e  # 遇到错误立即退出，避免脚本继续执行

# ======================== 基础配置 ========================
# 激活虚拟环境（根据你的实际环境修改）
VENV_PATH="/home/user/miniconda3/envs/gait_env/bin/activate"
# 训练主脚本名称（假设你的训练代码在 train.py 中）
TRAIN_SCRIPT="train_selfsup.py"
# 日志文件保存路径
LOG_DIR="./train_logs"
mkdir -p ${LOG_DIR}  # 确保日志目录存在

# ======================== 训练参数配置 ========================
# 可根据需求修改以下参数，与parse_args中的参数一一对应
EXP_NAME="Gait_self_supervised_GSDNN_baseline"
MODE="debug"  # debug/normal
DATA_PATH="datasets/data_10000/all_data.mat"
BATCH_SIZE=64
NUM_WORKERS=4  # 建议设置为CPU核心数，比0效率更高
VIEWS=2

# 模型参数
MODEL_TYPE=GSDNN
block_n
growth_rate
base_channels
stride
dropout_GSDNN

OUT_DIM=32
PROJ_OUT_DIM=128
DROPOUT=0.5

# 训练参数
EPOCHS=40
LR=3e-4
TEMPERATURE=0.5
FREQ_KEEP_RATIO=0.6

# 保存和日志参数
SAVE_DIR="./save_models/final_exp"
LOG_DIR_TB="./runs/final_exp"
SAVE_FREQ=5

# 设备参数
DEVICE="cuda"  # cuda/cpu
SEED=42

# ======================== GPU配置（可选） ========================
# 如果有多卡，指定使用的GPU（例如：使用0号和1号GPU）
# export CUDA_VISIBLE_DEVICES="0,1"
export CUDA_VISIBLE_DEVICES="0"  # 单卡推荐

# ======================== 开始训练 ========================
echo "========================================"
echo "开始训练：$(date)"
echo "实验名称：${EXP_NAME}"
echo "训练模式：${MODE}"
echo "使用设备：${DEVICE}"
echo "日志保存：${LOG_DIR}/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).log"
echo "========================================"

# 拼接训练命令
TRAIN_CMD="python ${TRAIN_SCRIPT} \
    --exp_name ${EXP_NAME} \
    --mode ${MODE} \
    --data_path ${DATA_PATH} \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --views ${VIEWS} \
    --out_dim ${OUT_DIM} \
    --proj_out_dim ${PROJ_OUT_DIM} \
    --dropout ${DROPOUT} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --temperature ${TEMPERATURE} \
    --freq_keep_ratio ${FREQ_KEEP_RATIO} \
    --save_dir ${SAVE_DIR} \
    --log_dir ${LOG_DIR_TB} \
    --experiment_name ${EXPERIMENT_NAME} \
    --save_freq ${SAVE_FREQ} \
    --device ${DEVICE} \
    --seed ${SEED}"

# 执行训练命令并记录日志
${TRAIN_CMD} 2>&1 | tee ${LOG_DIR}/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).log

# ======================== 训练结束 ========================
echo "========================================"
echo "$(TRAIN_CMD)"
echo "训练结束：$(date)"
echo "日志文件：${LOG_DIR}/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).log"
echo "模型保存：${SAVE_DIR}"
echo "========================================"