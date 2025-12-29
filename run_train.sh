#!/bin/bash
# 步态识别训练脚本使用示例
# 显示TensorBoard: tensorboard --logdir=./runs

echo "=========================================="
echo "步态识别对比学习训练示例"
echo "=========================================="
echo ""

# 基础训练（使用默认参数）
echo "1. 基础训练（使用默认参数）："
echo "python train_refactor.py"
echo ""

# 自定义实验名称
echo "2. 自定义实验名称："
echo "python train_refactor.py --experiment_name exp_baseline"
echo ""

# 调整训练参数
echo "3. 调整训练参数（更多epoch、更大batch size）："
echo "python train_refactor.py --epochs 100 --batch_size 128 --lr 1e-4"
echo ""

# 调整数据增强
echo "4. 调整数据增强强度（保留更多频率成分）："
echo "python train_refactor.py --freq_keep_ratio 0.8"
echo ""

# 调整模型参数
echo "5. 调整模型参数（更大的投影头、更低dropout）："
echo "python train_refactor.py --proj_out_dim 256 --dropout 0.3"
echo ""

# 完整配置示例
echo "6. 完整配置示例："
echo "python train_refactor.py \\"
echo "  --data_path data2/all_data.mat \\"
echo "  --batch_size 64 \\"
echo "  --epochs 40 \\"
echo "  --lr 3e-4 \\"
echo "  --temperature 0.5 \\"
echo "  --freq_keep_ratio 0.6 \\"
echo "  --proj_out_dim 128 \\"
echo "  --dropout 0.5 \\"
echo "  --save_dir ./save_model \\"
echo "  --log_dir ./runs \\"
echo "  --experiment_name my_experiment \\"
echo "  --save_freq 5 \\"
echo "  --seed 42"
echo ""

# 查看所有参数
echo "7. 查看所有可用参数："
echo "python train_refactor.py --help"
echo ""

echo "=========================================="
echo "TensorBoard查看训练进度："
echo "tensorboard --logdir=./runs"
echo "=========================================="
