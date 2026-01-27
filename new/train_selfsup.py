"""
步态识别对比学习训练脚本
功能：使用SimCLR框架进行对比学习，支持数据增强、TensorBoard监控、参数化配置
"""

import os
import argparse
import random
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from models.resnet import ResNet18
from models.GSDNN_new import GSDNN_new
from models.Conformer import Conformer


# ============== 配置和参数解析模块 ==============

def parse_args():
    """解析命令行参数，遵循KISS原则：配置集中管理"""
    parser = argparse.ArgumentParser(description='步态识别对比学习训练')
    parser.add_argument('--exp_name', type=str, default='Gait_self_supervised_training')
    parser.add_argument('--mode', type=str, default='debug', help='normal, debug')

    # 数据参数
    parser.add_argument('--data_path', type=str, default='datasets/data_10000/all_data.mat', help='数据路径')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载线程数')
    parser.add_argument('--views', type=int, default=2, help='对比学习的视图数量')

    # 模型参数
    parser.add_argument("--model_type", type=str, default="GSDNN", help="模型类型, 可以选GSDNN,ResNet,EEGNet,Conformer")
    ## parameters for GSDNN
    parser.add_argument('--num_classes', type=int, default=1, help='输出类别数')
    parser.add_argument('--block_n', type=int, default=8, help='模块堆叠次数')
    parser.add_argument('--init_channels', type=int, default=18, help='数据输入维度')
    parser.add_argument('--growth_rate', type=int, default=12, help='模块每叠一次，维度提升多少')
    parser.add_argument('--base_channels', type=int, default=48, help='初始特征维度')
    parser.add_argument('--stride', type=int, default=2, help='卷积步长')
    parser.add_argument('--dropout_GSDNN', type=float, default=0.2, help='GSDNN丢失概率')
    
    ## parameters for projection head
    ### GSDNN [132 128 256]
    ### ResNet18 [64 128 256]
    parser.add_argument('--out_dim', type=int, default=132, help='编码器输出维度')
    parser.add_argument('--proj_out_dim', type=int, default=128, help='投影头中间层维度')
    parser.add_argument('--contrastive_dim', type=int, default=256, help='进行对比学习的特征空间维度')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout概率')
    

    # 训练参数
    parser.add_argument('--epochs', type=int, default=40, help='训练轮数')
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
    parser.add_argument('--temperature', type=float, default=0.5, help='对比学习温度参数')

    # 数据增强参数
    parser.add_argument('--freq_keep_ratio', type=float, default=0.6, help='频率dropout保留比例')

    # 保存和日志参数
    parser.add_argument('--save_dir', type=str, default='./save_models', help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='./runs', help='TensorBoard日志目录')
    parser.add_argument('--save_freq', type=int, default=5, help='模型保存频率（每n个epoch）')

    # 设备参数
    parser.add_argument('--device', type=str, default=None, help='训练设备（cuda/cpu，默认自动检测）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    return parser.parse_args()


def set_seed(seed):
    """设置随机种子，确保实验可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============== 数据增强模块 ==============

class GaitAugmentation:
    """步态数据增强类，遵循单一职责原则"""

    @staticmethod
    def reverse_time_series(data):
        """时间序列反转（加负号）"""
        return -data

    @staticmethod
    def random_channel_shuffle(data):
        """随机通道打乱"""
        assert data.dim() == 3, "Input data must be 3D with channels as the second dimension"
        num_channels = data.size(1)
        shuffled_indices = torch.randperm(num_channels)
        return data[:, shuffled_indices, :]

    @staticmethod
    def random_frequency_dropout(data, keep_ratio=0.6):
        """随机频率dropout"""
        fft_img = torch.fft.fftn(data, dim=2)
        magnitude = torch.abs(fft_img)
        num_freqs = magnitude.shape[2]
        keep_indices = np.random.choice(num_freqs, int(num_freqs * keep_ratio), replace=False)
        mask = torch.zeros_like(magnitude, dtype=torch.bool)
        # keep_indices = torch.from_numpy(keep_indices).to(data.device)  # 转torch张量+对齐设备
        mask[:, :, keep_indices] = 1
        fft_img = fft_img * mask
        img = torch.fft.ifftn(fft_img, dim=2)
        return torch.real(img)

    @classmethod
    def get_transforms(cls, freq_keep_ratio=0.6):
        """获取数据增强变换组合"""
        return transforms.Compose([
            transforms.RandomApply([
                # 弱增强操作
                transforms.RandomResizedCrop((18, 101), scale=(0.3, 0.8)), # , antialias=True
                transforms.RandomErasing(p=0.5, scale=(0.2, 0.4), ratio=(0.3, 3.3), value=0),
                transforms.Lambda(lambda x: cls.random_frequency_dropout(x, freq_keep_ratio)),
                # 强增强操作
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Lambda(cls.random_channel_shuffle),
                transforms.Lambda(cls.reverse_time_series)
            ], p=1.0)
        ]) # trasform 执行的概率是一个值得关注的超参数


# ============== 数据集模块 ==============

class ContrastiveDataset(Dataset):
    """对比学习数据集，遵循单一职责原则"""

    def __init__(self, data_array, data_transform, views=2):
        """
        Args:
            data_array: numpy数组格式的步态数据
            data_transform: 数据增强变换
            views: 对比学习的视图数量
        """
        self.transform = data_transform
        self.data_array = data_array
        self.views = views

    def __len__(self):
        return len(self.data_array)

    def __getitem__(self, idx):
        """获取增强后的样本对"""
        img = self.data_array[idx]
        img = torch.tensor(np.expand_dims(img, axis=0))
        imgs = []
        for _ in range(self.views):
            img_aug = self.transform(img)
            imgs.append(img_aug)
        return imgs


def load_data(data_path, batch_size, views, num_workers, transform):
    """加载数据并创建DataLoader"""
    gait = sio.loadmat(data_path)['all_data']
    dataset = ContrastiveDataset(data_array=gait, data_transform=transform, views=views)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader


# ============== 模型定义模块 ==============

class SimCLRModel(nn.Module):
    """SimCLR对比学习模型，遵循单一职责原则"""

    def __init__(self, base_model, out_dim=32, proj_out_dim=128, contrastive_dim=256, dropout=0.5):
        """
        Args:
            base_model: 基础编码器（如ResNet）
            out_dim: 编码器输出维度
            proj_out_dim: 投影头中间层维度
            dropout: Dropout概率
        """
        super(SimCLRModel, self).__init__()
        self.encoder = nn.Sequential(*list(base_model.children())[:-1])
        self.dropout = nn.Dropout(p=dropout)

        # 投影头
        self.projector = nn.Sequential(
            nn.Linear(out_dim, proj_out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_out_dim, contrastive_dim)
        )

    def forward(self, x):
        """前向传播"""
        h = self.encoder(x)
        h = self.dropout(h)
        h = h.view(h.size(0), -1)
        h = self.projector(h)
        return h


class ContrastiveLoss(nn.Module):
    """对比损失函数（NT-Xent损失）"""

    def __init__(self, temperature=0.5):
        """
        Args:
            temperature: 温度参数，控制分布的平滑度
        """
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        Args:
            z_i: 第一个视图的投影向量
            z_j: 第二个视图的投影向量
        Returns:
            对比损失值
        """
        batch_size = z_i.shape[0]
        z_i = nn.functional.normalize(z_i, dim=1)
        z_j = nn.functional.normalize(z_j, dim=1)

        # 计算相似度矩阵
        sim_matrix = torch.matmul(z_i, z_j.T) / self.temperature
        sim_labels = torch.arange(batch_size).to(z_i.device)
        loss = nn.CrossEntropyLoss()(sim_matrix, sim_labels)
        return loss


def create_model(device, model_type, args):
    """创建模型实例"""
    if model_type == 'GSDNN':
        encoder = GSDNN_new(args.num_classes, args.block_n, args.init_channels, 
                            args.growth_rate, args.base_channels, args.stride, args.dropout_GSDNN)
    elif model_type == 'ResNet':
        encoder = ResNet18()
    elif model_type == 'Conformer':
        encoder = Conformer(emb_size=40, depth=6, n_classes=4)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    model = SimCLRModel(encoder, args.out_dim, args.proj_out_dim, args.contrastive_dim, args.dropout).to(device)
    return model


def create_optimizer(model, lr=3e-4):
    """创建优化器"""
    return torch.optim.Adam(model.parameters(), lr=lr)


# ============== 训练和验证模块 ==============

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer, args):
    """训练一个epoch

    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        epoch: 当前epoch
        writer: TensorBoard writer

    Returns:
        平均损失值
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (img1, img2) in enumerate(dataloader):
        optimizer.zero_grad()
        if args.model_type == 'GSDNN':
            img1 = img1.squeeze(1)
            img2 = img2.squeeze(1)
        
        # 前向传播
        z_i = model(img1.to(device=device, dtype=torch.float32))
        z_j = model(img2.to(device=device, dtype=torch.float32))

        # 计算损失
        loss = criterion(z_i, z_j)

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # 记录到TensorBoard（每10个batch）
        if batch_idx % 10 == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Loss/train_batch', loss.item(), global_step)

        if args.mode == 'debug':
            break
    avg_loss = total_loss / num_batches
    return avg_loss


def save_checkpoint(model, save_dir, filename, epoch, loss, is_best=False):
    """保存模型检查点

    Args:
        model: 模型
        save_dir: 保存目录
        filename: 文件名
        epoch: 当前epoch
        loss: 当前损失
        is_best: 是否为最佳模型
    """
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
    }

    # 保存检查点
    save_path = os.path.join(save_dir, filename)
    torch.save(checkpoint, save_path)

    # 如果是最佳模型，额外保存
    if is_best:
        best_path = os.path.join(save_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f'Epoch {epoch + 1}, New minimum loss: {loss:.6f}, Best model saved.')


def plot_loss_curve(loss_values, save_dir):
    """绘制损失曲线

    Args:
        loss_values: 损失值列表
        save_dir: 保存目录
    """
    plt.figure(figsize=(10, 5))
    plt.plot(loss_values, label='Training Loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 保存图像
    save_path = os.path.join(save_dir, 'loss_curve.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Loss curve saved to {save_path}')
    plt.close()


# ============== 主训练流程 ==============

def train(args):
    """主训练函数，整合所有模块

    Args:
        args: 命令行参数
    """
    # 1. 设置随机种子
    set_seed(args.seed)

    # 2. 设置设备
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f'Using device: {device}')

    # 3. 创建实验目录
    if args.exp_name is None:
        args.exp_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    # else:
    #     args.exp_name = args.exp_name + datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.log_dir, args.exp_name)
    save_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # 4. 创建TensorBoard writer
    writer = SummaryWriter(log_dir)
    print(f'TensorBoard logs will be saved to: {log_dir}')
    print(f'Models will be saved to: {save_dir}')

    # 5. 准备数据
    print('Loading data...')
    transform = GaitAugmentation.get_transforms(args.freq_keep_ratio)
    dataloader = load_data(
        data_path=args.data_path,
        batch_size=args.batch_size,
        views=args.views,
        num_workers=args.num_workers,
        transform=transform
    )
    print(f'Data loaded. Total batches: {len(dataloader)}. Total train data: {len(dataloader)*args.batch_size}')

    # 6. 创建模型和优化器
    print('Creating model...')
    model = create_model(device, args.model_type, args)
    criterion = ContrastiveLoss(temperature=args.temperature)
    optimizer = create_optimizer(model, args.lr)

    # 7. 训练循环
    print('Starting training...')
    loss_values = []
    min_loss = float('inf')

    for epoch in range(args.epochs):
        # 训练一个epoch
        avg_loss = train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer, args)
        loss_values.append(avg_loss)

        # 记录到TensorBoard
        writer.add_scalar('Loss/train_epoch', avg_loss, epoch)

        # 打印训练信息
        print(f'Epoch {epoch + 1}/{args.epochs}, Average Loss: {avg_loss:.6f}')

        # 检查是否为最佳模型
        is_best = avg_loss < min_loss
        if is_best:
            min_loss = avg_loss

        # 保存检查点
        if (epoch + 1) % args.save_freq == 0 or is_best:
            filename = f'checkpoint_epoch_{epoch + 1}.pth'
            save_checkpoint(model, save_dir, filename, epoch, avg_loss, is_best)

    # 8. 训练完成
    print('Training completed!')
    print(f'Minimum loss: {min_loss:.6f}')

    # 9. 绘制损失曲线
    # plot_loss_curve(loss_values, save_dir)

    # 10. 关闭TensorBoard writer
    writer.close()

    print(f'All results saved to: {save_dir}')
    print(f"View TensorBoard logs with: tensorboard --logdir={log_dir}")


if __name__ == '__main__':
    # 解析参数
    args = parse_args()

    # 开始训练
    train(args)
