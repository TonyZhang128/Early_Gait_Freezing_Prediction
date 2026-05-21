"""
步态识别对比学习训练脚本
功能：使用SimCLR框架进行对比学习，支持数据增强、TensorBoard监控、参数化配置
"""

import os
import argparse
from datetime import datetime
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from datasets.Contrastive_dataset import load_data

from models.resnet import ResNet18
from models.GSDNN_new import GSDNN_new
from models.Conformer import Conformer

from utils.analysis import set_seed, save_checkpoint
from utils.calculate import FLOPs_calculat




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
        h = torch.flatten(h, start_dim=1) # h.view(h.size(0), -1)
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
        self.eps = 1e-8

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


# 3. 定义预热调度器（线性上升）
def get_warmup_lr_lambda(warmup_steps):
    def warmup_lr_lambda(step):
        return step / warmup_steps if step < warmup_steps else 1.0
    return warmup_lr_lambda

def create_optimizer(model, args, total_steps):
    """创建优化器"""
    batch_size = args.batch_size  # 实际训练批次
    base_lr = args.base_lr     # 基础批次256对应的初始学习率
    total_steps = total_steps  # 总训练步数
    warmup_steps = int(total_steps * args.warmup_ratio)  # 预热步数：5%总步数
    min_lr = base_lr * (batch_size / 256) * 1e-3  # 最小学习率
    optimizer = SGD(model.parameters(), 
                lr=base_lr * (batch_size / 256),  # 线性缩放学习率
                momentum=args.momentum, 
                weight_decay=args.weight_decay)
    
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=get_warmup_lr_lambda(warmup_steps))

    # 4. 定义余弦退火调度器（预热后执行）
    cos_scheduler = CosineAnnealingLR(optimizer, 
                                  T_max=total_steps - warmup_steps,  # 退火总步数
                                  eta_min=min_lr)  # 最小学习率
    # torch.optim.Adam(model.parameters(), lr=lr)
    return optimizer, warmup_scheduler, cos_scheduler


# ============== 训练和验证模块 ==============

def train_epoch(model, dataloader, criterion, optimizer, warmup_scheduler, 
                cos_scheduler, warmup_steps, device, epoch, writer, args):
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
    steps_one_epoch = len(dataloader)
    
    for batch_idx, (img1, img2) in enumerate(dataloader):
        optimizer.zero_grad()
        if args.model_type == 'GSDNN':
            img1 = img1.squeeze(1)
            img2 = img2.squeeze(1)
        
        img1, img2 = img1.to(device, dtype=torch.float32), img2.to(device, dtype=torch.float32)
        # 前向传播
        z_i = model(img1)
        z_j = model(img2)

        # 计算损失
        loss = criterion(z_i, z_j)

        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度范数最大为1.0，可微调
        optimizer.step()
        
         
        step = epoch * steps_one_epoch + batch_idx + 1 
        if step >= warmup_steps: # 预热结束后，启动余弦退火
            cos_scheduler.step()
        else:
            warmup_scheduler.step() 

        total_loss += loss.item()
        num_batches += 1

        # 记录到TensorBoard（每10个batch）
        if batch_idx % 50 == 0:
            global_step = epoch * len(dataloader) + batch_idx
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('LearningRate/train', current_lr, global_step)
            writer.add_scalar('Loss/train_batch', loss.item(), global_step)

        if args.mode == 'debug':
            break
    avg_loss = total_loss / num_batches
    return avg_loss


# ============== 主训练流程 ==============

def train(args):
    """主训练函数，整合所有模块

    Args:
        args: 命令行参数
    """
    # 1. 设置随机种子
    set_seed(args.seed)
    # 性能优化
    torch.backends.cudnn.benchmark = True  # 针对固定输入尺寸，加速卷积计算
    torch.backends.cudnn.deterministic = False  # 关闭确定性，提升速度（若需严格复现，设为True）
    torch.cuda.empty_cache()  # 清空GPU缓存

    # 2. 设置设备
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f'Using device: {device}')

    # 3. 创建实验目录        
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # log_dir = os.path.join(args.log_dir, args.exp_name + '_' + timestamp)
    # save_dir = os.path.join(args.save_dir, args.exp_name + '_' + timestamp)
    log_dir = args.log_dir
    save_dir = args.save_dir
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # 4. 创建TensorBoard writer
    writer = SummaryWriter(log_dir)
    # print(f'TensorBoard logs will be saved to: {log_dir}')
    # print(f'Models will be saved to: {save_dir}')

    # 5. 准备数据
    print('Loading data...')
    
    dataloader = load_data(
        data_path=args.data_path,
        batch_size=args.batch_size,
        views=args.views,
        num_workers=args.num_workers,
        args=args
    )
    print(f'Data loaded. Total batches: {len(dataloader)}. Total train data: {len(dataloader.dataset)}')
     
    # 6. 创建模型和优化器
    print(f'Creating model...Encoder use {args.model_type}')
    model = create_model(device, args.model_type, args)
    data_shape = [1, 1, 18, 101]
    FLOPs_calculat(model, device, data_shape)
    criterion = ContrastiveLoss(temperature=args.temperature).to(device)
    
    total_steps = args.epochs * len(dataloader) 
    warmup_steps = int(total_steps * args.warmup_ratio)
    optimizer, warmup_scheduler, cos_scheduler = create_optimizer(model, args, total_steps)

    # 7. 训练循环
    print('Starting training...')
    loss_values = []
    min_loss = float('inf')

    for epoch in range(args.epochs):
        # 训练一个epoch
        avg_loss = train_epoch(model, dataloader, criterion, optimizer, warmup_scheduler, 
                               cos_scheduler, warmup_steps, device, epoch, writer, args)
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

# ============== 配置和参数解析模块 ==============

def parse_args():
    """解析命令行参数，遵循KISS原则：配置集中管理"""
    parser = argparse.ArgumentParser(description='步态识别对比学习预训练')
    parser.add_argument('--exp_name', type=str, default='Gait_selfsup_GSDNN_baseline')
    parser.add_argument('--mode', type=str, default='debug', help='normal, debug')
    parser.add_argument('--print_params', type=bool, default=True, help='打印参数')

    # 数据参数
    parser.add_argument('--data_path', type=str, default='datasets/data_10000/all_data.mat', help='数据路径')
    parser.add_argument('--batch_size', type=int, default=256, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载线程数')
    parser.add_argument('--views', type=int, default=2, help='对比学习的视图数量')

    # 模型参数
    parser.add_argument("--model_type", type=str, default="ResNet", help="模型类型, 可以选GSDNN,ResNet,EEGNet,Conformer")
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
    ### ResNet18 [512 1024 128]
    parser.add_argument('--out_dim', type=int, default=512, help='编码器输出维度')
    parser.add_argument('--proj_out_dim', type=int, default=1024, help='投影头中间层维度')
    parser.add_argument('--contrastive_dim', type=int, default=128, help='进行对比学习的特征空间维度')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout概率')
    

    # 训练参数
    parser.add_argument('--epochs', type=int, default=40, help='训练轮数')
    parser.add_argument('--base_lr', type=float, default=0.3, help='学习率')
    parser.add_argument('--warmup_ratio', type=float, default=0.05, help='学习率预热步数（比例）')
    parser.add_argument('--temperature', type=float, default=0.1, help='对比学习温度参数')
    
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD动量')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减系数')

    # 数据增强参数
    parser.add_argument('--freq_keep_ratio', type=float, default=0.6, help='频率dropout保留比例')

    # 保存和日志参数
    parser.add_argument('--save_dir', type=str, default='./save_models', help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='./runs', help='TensorBoard日志目录')
    parser.add_argument('--save_freq', type=int, default=10, help='模型保存频率（每n个epoch）')

    # 设备参数
    parser.add_argument('--device', type=str, default=None, help='训练设备（cuda/cpu，默认自动检测）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    args = parser.parse_args()
    
    # 自动检测设备（如果未指定）
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if args.print_params:
        print("="*70)
        print("步态识别对比学习训练配置信息")
        print("="*70)
        for key, value in sorted(vars(args).items()):
            print(f"  {key.ljust(30)}: {value}")
        print("="*70)
    
    return args

if __name__ == '__main__':
    # 解析参数
    args = parse_args()

    # 开始训练
    train(args)
