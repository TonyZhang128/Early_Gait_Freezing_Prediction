"""
步态识别对比学习训练脚本（重构版）
功能：使用SimCLR框架进行对比学习，支持TensorBoard监控、参数解析、模块化设计
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import scipy.io as sio
import argparse
import os
from datetime import datetime
from pathlib import Path

# 导入模型
from models.resnet import ResNet101
from models.DNN import DNN
from models.MSDNN import MSDNN
from models.GSDNN import GSDNN

# TensorBoard支持
from torch.utils.tensorboard import SummaryWriter

# ==================== 配置管理 ====================

class TrainingConfig:
    """训练配置类，集中管理所有超参数"""
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.learning_rate = args.learning_rate
        self.temperature = args.temperature
        self.num_views = args.num_views  # 对比视图数量
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 数据路径
        self.data_path = args.data_path
        self.checkpoint_dir = args.checkpoint_dir
        self.log_dir = args.log_dir

        # 模型配置
        self.model_type = args.model_type
        self.proj_out_dim = args.proj_out_dim
        self.dropout_rate = args.dropout_rate

        # 数据增强参数
        self.crop_scale = tuple(map(float, args.crop_scale.split(',')))
        self.erasing_scale = tuple(map(float, args.erasing_scale.split(',')))
        self.freq_keep_ratio = args.freq_keep_ratio

        # 其他
        self.save_freq = args.save_freq
        self.log_freq = args.log_freq


# ==================== 数据增强模块 ====================

class GaitAugmentation:
    """步态数据增强类"""

    @staticmethod
    def reverse_time_series(data):
        """时间序列反转"""
        return -data

    @staticmethod
    def random_channel_shuffle(data):
        """随机通道打乱"""
        assert data.dim() == 3, "Input data must be 3D with channels as the second dimension"
        num_channels = data.size(1)
        shuffled_indices = torch.randperm(num_channels)
        return data[:, shuffled_indices, :]

    @staticmethod
    def random_frequency_dropout(img, keep_ratio=0.6):
        """随机频率成分丢弃"""
        fft_img = torch.fft.fftn(img, dim=2)
        magnitude = torch.abs(fft_img)
        num_freqs = magnitude.shape[2]
        keep_indices = np.random.choice(num_freqs, int(num_freqs * keep_ratio), replace=False)
        mask = torch.zeros_like(magnitude, dtype=torch.bool)
        mask[:, :, keep_indices] = 1
        fft_img = fft_img * mask
        img = torch.fft.ifftn(fft_img, dim=2)
        return torch.real(img)

    @classmethod
    def get_transforms(cls, config):
        """获取数据增强变换组合"""
        return transforms.Compose([
            transforms.RandomApply([
                # 弱增强操作
                transforms.RandomResizedCrop(
                    (18, 101),
                    scale=config.crop_scale,
                    antialias=True
                ),
                transforms.RandomErasing(
                    p=1.0,
                    scale=config.erasing_scale,
                    ratio=(0.3, 3.3),
                    value=0
                ),
                transforms.Lambda(
                    lambda x: cls.random_frequency_dropout(x, config.freq_keep_ratio)
                ),
                # 强增强操作
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.Lambda(cls.random_channel_shuffle),
                transforms.Lambda(cls.reverse_time_series)
            ], p=1.0)
        ])


# ==================== 数据集定义 ====================

class ContrastiveDataset(torch.utils.data.Dataset):
    """对比学习数据集"""

    def __init__(self, data_array, data_transform, num_views=2):
        """
        参数:
            data_array: 数据数组
            data_transform: 数据变换
            num_views: 每个样本生成的视图数量
        """
        self.transform = data_transform
        self.data_array = data_array
        self.num_views = num_views

    def __len__(self):
        return len(self.data_array)

    def __getitem__(self, idx):
        img = self.data_array[idx]
        img = torch.tensor(np.expand_dims(img, axis=0))
        imgs = []
        for _ in range(self.num_views):
            img_transformed = self.transform(img)
            imgs.append(img_transformed)
        return imgs


# ==================== 模型定义 ====================

class SimCLRModel(nn.Module):
    """SimCLR对比学习模型"""

    def __init__(self, base_model, config):
        """
        参数:
            base_model: 基础编码器模型
            config: 训练配置对象
        """
        super(SimCLRModel, self).__init__()
        self.encoder = base_model
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])  # 移除分类层
        self.dropout = nn.Dropout(p=config.dropout_rate)

        # SimCLR v2风格的投影头
        self.projector = nn.Sequential(
            nn.Linear(132, 132),
            nn.BatchNorm1d(132),
            nn.ReLU(inplace=True),
            nn.Linear(132, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, config.proj_out_dim)
        )

    def forward(self, x):
        """前向传播"""
        h = self.encoder(x)
        h = self.dropout(h)
        h = h.view(h.size(0), -1)
        h = self.projector(h)
        return h


# ==================== 损失函数 ====================

class ContrastiveLoss(nn.Module):
    """对比损失（NT-Xent损失）"""

    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        参数:
            z_i: 第一个视图的特征
            z_j: 第二个视图的特征
        返回:
            loss: 对比损失值
        """
        batch_size = z_i.shape[0]

        # L2归一化
        z_i = nn.functional.normalize(z_i, dim=1)
        z_j = nn.functional.normalize(z_j, dim=1)

        # 计算相似度矩阵
        sim_matrix = torch.matmul(z_i, z_j.T) / self.temperature
        sim_labels = torch.arange(batch_size).to(z_i.device)

        loss = nn.CrossEntropyLoss()(sim_matrix, sim_labels)
        return loss


# ==================== 数据加载模块 ====================

def load_data(config):
    """
    加载并预处理数据

    参数:
        config: 训练配置对象
    返回:
        dataloader: 数据加载器
    """
    print(f"正在加载数据: {config.data_path}")
    gait_data = sio.loadmat(config.data_path)['all_data']
    print(f"数据形状: {gait_data.shape}")

    # 创建数据增强变换
    data_transforms = GaitAugmentation.get_transforms(config)

    # 创建数据集和加载器
    dataset = ContrastiveDataset(
        data_array=gait_data,
        data_transform=data_transforms,
        num_views=config.num_views
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Windows下建议设为0
        pin_memory=True if config.device == "cuda" else False
    )

    print(f"数据集大小: {len(dataset)}")
    print(f"批次数: {len(dataloader)}")

    return dataloader


# ==================== 模型初始化模块 ====================

def create_model(config):
    """
    创建模型实例

    参数:
        config: 训练配置对象
    返回:
        model: SimCLR模型
    """
    print(f"正在初始化模型: {config.model_type}")

    # 根据配置选择基础模型
    model_mapping = {
        'resnet101': ResNet101(),
        'dnn': DNN(),
        'msdnn': MSDNN(),
        'gsdnn': GSDNN()
    }

    base_model = model_mapping.get(config.model_type.lower(), GSDNN())

    model = SimCLRModel(base_model, config).to(config.device)

    print(f"模型已创建，设备: {config.device}")
    return model


def create_optimizer_and_scheduler(model, config):
    """
    创建优化器和学习率调度器

    参数:
        model: 模型实例
        config: 训练配置对象
    返回:
        optimizer: 优化器
        scheduler: 学习率调度器（可选）
    """
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate
    )

    # 可选：添加学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs
    )

    return optimizer, scheduler


# ==================== 训练模块 ====================

def train_one_epoch(model, dataloader, criterion, optimizer, config, epoch, writer):
    """
    训练一个epoch

    参数:
        model: 模型实例
        dataloader: 数据加载器
        criterion: 损失函数
        optimizer: 优化器
        config: 训练配置对象
        epoch: 当前epoch数
        writer: TensorBoard写入器
    返回:
        avg_loss: 平均损失
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (img1, img2) in enumerate(dataloader):
        # 数据移至设备
        img1 = torch.squeeze(img1, dim=1).to(device=config.device, dtype=torch.float32)
        img2 = torch.squeeze(img2, dim=1).to(device=config.device, dtype=torch.float32)

        # 前向传播
        z_i = model(img1)
        z_j = model(img2)

        # 计算损失
        loss = criterion(z_i, z_j)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        num_batches += 1

        # 日志记录
        if (batch_idx + 1) % config.log_freq == 0:
            avg_batch_loss = total_loss / num_batches
            print(f"Epoch [{epoch+1}/{config.num_epochs}], "
                  f"Batch [{batch_idx+1}/{len(dataloader)}], "
                  f"Loss: {loss.item():.4f}, "
                  f"Avg Loss: {avg_batch_loss:.4f}")

            # TensorBoard记录
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Loss/batch_loss', loss.item(), global_step)
            writer.add_scalar('Loss/avg_batch_loss', avg_batch_loss, global_step)
        # break

    return total_loss / num_batches


def save_checkpoint(model, optimizer, epoch, loss, config, is_best=False):
    """
    保存模型检查点

    参数:
        model: 模型实例
        optimizer: 优化器
        epoch: 当前epoch数
        loss: 当前损失
        config: 训练配置对象
        is_best: 是否为最佳模型
    """
    # 创建保存目录
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config.__dict__
    }

    # 保存最新模型
    latest_path = os.path.join(config.checkpoint_dir, 'latest_model.pth')
    torch.save(checkpoint, latest_path)

    # 保存最佳模型
    if is_best:
        best_path = os.path.join(config.checkpoint_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"✓ 最佳模型已保存: {best_path}")

    # 定期保存
    if (epoch + 1) % config.save_freq == 0:
        epoch_path = os.path.join(config.checkpoint_dir, f'model_epoch_{epoch+1}.pth')
        torch.save(checkpoint, epoch_path)
        print(f"✓ 检查点已保存: {epoch_path}")


# ==================== 主训练函数 ====================

def train(config):
    """
    主训练函数

    参数:
        config: 训练配置对象
    """
    print("="*60)
    print("开始训练 - 对比学习")
    print("="*60)
    print(f"设备: {config.device}")
    print(f"批大小: {config.batch_size}")
    print(f"训练轮数: {config.num_epochs}")
    print(f"学习率: {config.learning_rate}")
    print(f"温度参数: {config.temperature}")
    print(f"模型类型: {config.model_type}")
    print("="*60)

    # 初始化TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(config.log_dir, f"run_{timestamp}")
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard日志目录: {log_dir}")

    # 加载数据
    dataloader = load_data(config)

    # 创建模型
    model = create_model(config)

    # 创建损失函数
    criterion = ContrastiveLoss(temperature=config.temperature)

    # 创建优化器
    optimizer, scheduler = create_optimizer_and_scheduler(model, config)

    # 训练循环
    min_loss = float('inf')
    loss_history = []

    for epoch in range(config.num_epochs):
        print(f"\n{'='*40}")
        print(f"Epoch {epoch+1}/{config.num_epochs}")
        print(f"{'='*40}")

        # 训练一个epoch
        avg_loss = train_one_epoch(
            model, dataloader, criterion, optimizer,
            config, epoch, writer
        )

        # 更新学习率
        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Learning_Rate', current_lr, epoch)

        # 记录损失
        loss_history.append(avg_loss)
        writer.add_scalar('Loss/epoch_loss', avg_loss, epoch)

        # 打印epoch信息
        print(f"\nEpoch [{epoch+1}/{config.num_epochs}] 完成")
        print(f"平均损失: {avg_loss:.4f}")
        print(f"学习率: {optimizer.param_groups[0]['lr']:.6f}")

        # 保存模型
        is_best = avg_loss < min_loss
        if is_best:
            min_loss = avg_loss
            print(f"✓ 新的最佳损失: {min_loss:.4f}")

        save_checkpoint(model, optimizer, epoch, avg_loss, config, is_best)

    # 训练完成
    print("\n"+"="*60)
    print("训练完成!")
    print(f"最佳损失: {min_loss:.4f}")
    print(f"模型保存在: {config.checkpoint_dir}")
    print(f"TensorBoard查看: tensorboard --logdir={log_dir}")
    print("="*60)

    writer.close()

    return model, loss_history


# ==================== 参数解析 ====================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='步态识别对比学习训练脚本')

    # 基础训练参数
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批大小 (默认: 64)')
    parser.add_argument('--num_epochs', type=int, default=40,
                        help='训练轮数 (默认: 40)')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='学习率 (默认: 3e-4)')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='对比学习温度参数 (默认: 0.07)')
    parser.add_argument('--num_views', type=int, default=2,
                        help='每个样本生成的视图数量 (默认: 2)')

    # 数据路径
    parser.add_argument('--data_path', type=str, default='data4/sub_all_data.mat',
                        help='数据文件路径')
    parser.add_argument('--checkpoint_dir', type=str, default='./save_model_refactor',
                        help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='./runs',
                        help='TensorBoard日志目录')

    # 模型参数
    parser.add_argument('--model_type', type=str, default='gsdnn',
                        choices=['resnet101', 'dnn', 'msdnn', 'gsdnn'],
                        help='基础模型类型 (默认: gsdnn)')
    parser.add_argument('--proj_out_dim', type=int, default=128,
                        help='投影头输出维度 (默认: 128)')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                        help='Dropout率 (默认: 0.2)')

    # 数据增强参数
    parser.add_argument('--crop_scale', type=str, default='0.3,0.8',
                        help='随机裁剪范围 (默认: 0.3,0.8)')
    parser.add_argument('--erasing_scale', type=str, default='0.2,0.4',
                        help='随机擦除范围 (默认: 0.2,0.4)')
    parser.add_argument('--freq_keep_ratio', type=float, default=0.6,
                        help='频率成分保留比例 (默认: 0.6)')

    # 其他参数
    parser.add_argument('--save_freq', type=int, default=5,
                        help='模型保存频率（epoch间隔） (默认: 5)')
    parser.add_argument('--log_freq', type=int, default=10,
                        help='日志打印频率（batch间隔） (默认: 10)')

    return parser.parse_args()


# ==================== 主入口 ====================

def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    # 创建配置对象
    config = TrainingConfig(args)

    # 创建必要目录
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    # 开始训练
    model, loss_history = train(config)

    print("\n训练完成!")


if __name__ == "__main__":
    main()
