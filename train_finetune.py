"""
步态识别分类训练脚本 - 重构版
包含TensorBoard监控、参数解析、模块化设计
"""

import argparse
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score,recall_score, auc, roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import scipy.io as sio



from datasets.Gait_dataset_new import GaitDataModule
# 导入模型
from models.DNN import DNN
from models.GSDNN_new import GSDNN_new
from models.resnet import ResNet18
from models.Conformer import Conformer
from utils.analysis import set_seed
from utils.calculate import FLOPs_calculat
from utils.plt_curves import plot_confusion_matrix

# # ================================ 数据增强模块 ================================

# def reverse_time_series(data):
#     """时间序列反转"""
#     return -data


# def random_channel_shuffle(data):
#     """随机通道打乱"""
#     assert data.dim() == 3, "Input data must be 3D with channels as the second dimension"
#     num_channels = data.size(1)
#     shuffled_indices = torch.randperm(num_channels)
#     return data[:, shuffled_indices, :]


# def random_frequency_dropout(img, keep_ratio=0.6):
#     """随机频率成分丢弃"""
#     fft_img = torch.fft.fftn(img, dim=2)
#     magnitude = torch.abs(fft_img)
#     num_freqs = magnitude.shape[2]
#     keep_indices = np.random.choice(num_freqs, int(num_freqs * keep_ratio), replace=False)
#     mask = torch.zeros_like(magnitude, dtype=torch.bool)
#     mask[:, :, keep_indices] = 1
#     fft_img = fft_img * mask
#     img = torch.fft.ifftn(fft_img, dim=2)
#     return torch.real(img)


# def get_data_transforms(augmentation_prob=0.5, freq_keep_ratio=0.6):
#     """构建数据增强变换组合"""
#     return transforms.Compose([
#         transforms.RandomApply([
#             transforms.Lambda(lambda x: random_frequency_dropout(x, freq_keep_ratio)),
#             transforms.Lambda(reverse_time_series),
#         ], p=augmentation_prob)
#     ])


# # ================================ 数据集类 ================================

# class GaitDataset(Dataset):
#     """步态数据集类"""

#     def __init__(self, data_array, data_label, data_transform=None, views=2):
#         """
#         Args:
#             data_array: 数据数组
#             data_label: 标签数组
#             data_transform: 数据变换
#             views: 视角数量
#         """
#         self.transform = data_transform
#         self.data_array = data_array
#         self.data_label = data_label
#         self.views = views

#     def __len__(self):
#         return len(self.data_array)

#     def __getitem__(self, idx):
#         img = self.data_array[idx]
#         if self.transform:
#             img = self.transform(torch.tensor(np.expand_dims(img, axis=0)))
#         return img, self.data_label[idx]


# # ================================ 数据加载模块 ================================

# def load_and_split_data(data_path, train_ratio=0.8, random_seed=42):
#     """
#     加载并划分训练集和测试集

#     Args:
#         data_path: 数据文件路径（不含后缀）
#         train_ratio: 训练集比例
#         random_seed: 随机种子

#     Returns:
#         train_data, test_data, train_label, test_label
#     """
#     np.random.seed(random_seed)

#     # 加载数据
#     # data_finetue = sio.loadmat(f'{data_path}/sub_train_data.mat')['sub_train_data']
#     # labels_finetue = sio.loadmat(f'{data_path}/sub_train_label.mat')['sub_train_label'][0]
    
#     # data_test = sio.loadmat(f'{data_path}/sub_test_data.mat')['sub_data']
#     # labels_test = sio.loadmat(f'{data_path}/sub_test_label.mat')['sub_label'][0]
    
#     # train_data = data_finetue
#     # train_label = labels_finetue - 1
#     # test_data = data_test
#     # test_label = labels_test - 1
    
#     data = sio.loadmat(f'{data_path}/sub_data.mat')['sub_data']
#     labels = sio.loadmat(f'{data_path}/sub_label.mat')['sub_label'][0]

#     # 打乱索引
#     random_index = np.array(range(len(data)))
#     np.random.shuffle(random_index)

#     # 应用打乱
#     data = data[random_index]
#     labels = labels[random_index]

#     # 划分数据集
#     train_len = int(len(data) * train_ratio)

#     train_data = data[:train_len]
#     test_data = data[train_len:]
#     train_label = labels[:train_len] - 1  # 标签从0开始
#     test_label = labels[train_len:] - 1

#     return train_data, test_data, train_label, test_label


# def create_dataloaders(args):
#     """
#     创建数据加载器

#     Args:
#         args: 参数对象

#     Returns:
#         train_loader, test_loader
#     """
#     # 加载数据
#     train_data, test_data, train_label, test_label = load_and_split_data(
#         args.data_path, args.train_ratio
#     )

#     # 创建数据增强
#     data_transforms = get_data_transforms(
#         args.augmentation_prob,
#         args.freq_keep_ratio
#     )

#     # 创建数据集
#     train_dataset = GaitDataset(
#         data_array=train_data,
#         data_label=train_label,
#         data_transform=data_transforms,
#         views=2
#     )

#     test_dataset = GaitDataset(
#         data_array=test_data,
#         data_label=test_label,
#         data_transform=data_transforms,
#         views=2
#     )

#     # 创建数据加载器
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=args.batch_size,
#         shuffle=True,
#         num_workers=args.num_workers
#     )

#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=args.num_workers
#     )

#     return train_loader, test_loader

# ================================ 模型定义模块 ================================

class SimCLREncoder(nn.Module):
    """SimCLR风格的编码器"""

    def __init__(self, base_model, out_dim=132, proj_out_dim=128, contrastive_dim=256, dropout=0.5):
        """
        Args:
            base_model: 基础模型
            out_dim: 输出特征维度
            proj_out_dim: 投影头输出维度
            dropout: dropout概率
        """
        super(SimCLREncoder, self).__init__()
        self.encoder = nn.Sequential(*list(base_model.children())[:-1])
        self.dropout = nn.Dropout(p=dropout)

        # 投影头（可选）
        self.projector = nn.Sequential(
            nn.Linear(out_dim, proj_out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_out_dim, contrastive_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        h = self.dropout(h)
        h = torch.flatten(h, start_dim=1)
        return h


class ClassificationModel(nn.Module):
    """分类模型"""

    def __init__(self, encoder, num_features=132, num_classes=27):
        """
        Args:
            encoder: 编码器
            num_features: 特征维度
            num_classes: 分类数量
        """
        super(ClassificationModel, self).__init__()
        self.encoder = encoder
        
        # 复杂的多层分类器
        self.classifier = nn.Sequential(
            # 第四层：进一步降维
            nn.Linear(num_features, num_features // 2),
            nn.BatchNorm1d(num_features // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            
            # 输出层
            nn.Linear(num_features // 2, num_classes)
        )

    def forward(self, x):
        h = self.encoder(x)
        return self.classifier(h)


def get_model(model_type, args, num_classes=27, device='cpu'):
    """创建模型实例"""
    if model_type == 'GSDNN':
        base_model = GSDNN_new(args.num_classes, args.block_n, args.init_channels, 
                            args.growth_rate, args.base_channels, args.stride, args.dropout_GSDNN)
    elif model_type == 'ResNet':
        base_model = ResNet18()
    elif model_type == 'Conformer':
        base_model = Conformer(emb_size=40, depth=6, n_classes=4)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    encoder = SimCLREncoder(base_model, args.out_dim, args.proj_out_dim, args.contrastive_dim, args.dropout)
    model = ClassificationModel(encoder, num_features=args.out_dim, num_classes=num_classes)

    return model.to(device)



# ================================ 训练和评估模块 ================================

def train_one_epoch(model, dataloader, optimizer, criterion, device, args):
    """
    训练一个epoch

    Args:
        model: 模型
        dataloader: 数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 设备

    Returns:
        avg_loss: 平均损失
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for x, label in dataloader:
        optimizer.zero_grad()

        # 数据预处理
        if args.model_type == 'GSDNN':
            x = torch.squeeze(x, dim=1)
        x = x.to(device=device, dtype=torch.float32)
        label = label.to(device=device)

        # 前向传播
        y_pred = model(x)
        loss = criterion(y_pred, label)

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        if args.mode == 'debug':
            break

    return total_loss / num_batches


def evaluate(model, dataloader, device):
    """
    评估模型

    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备

    Returns:
        dict: 包含各项评估指标的字典
    """
    model.eval()
    y_preds = []
    y_labels = []
    y_probs = []  # 存储概率值

    with torch.no_grad():
        for x, y in dataloader:
            if x.ndim != 4:
                x = torch.squeeze(x, dim=1)
            x = x.to(device=device, dtype=torch.float32)

            # 获取模型输出（logits）
            logits = model(x)
            # 获取预测类别
            y_pred = torch.argmax(logits, dim=1).cpu().numpy()
            # 获取概率值（使用softmax）
            y_prob = torch.softmax(logits, dim=1).cpu().numpy()
            y_label = y.cpu().numpy()

            y_preds.append(y_pred)
            y_labels.append(y_label)
            y_probs.append(y_prob)

    y_preds = np.concatenate(y_preds, axis=0)
    y_labels = np.concatenate(y_labels, axis=0)
    y_probs = np.concatenate(y_probs, axis=0)

    # 计算指标
    acc = np.mean(np.equal(y_preds, y_labels))
    precision = precision_score(y_labels, y_preds, average='macro', zero_division=0)
    recall = recall_score(y_labels, y_preds, average='macro', zero_division=0)
    f1 = f1_score(y_labels, y_preds, average='macro', zero_division=0)
    
    # 正确计算AUC（多分类情况）
    try:
        auc_score = roc_auc_score(y_labels, y_probs, multi_class='ovr', average='macro')
    except ValueError:
        # 如果只有一个类别，设置AUC为0
        auc_score = 0.0

    return {
        'predictions': y_preds,
        'labels': y_labels,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc_score
    }




def log_metrics_to_tensorboard(writer, metrics, epoch, phase='train'):
    """
    将指标记录到TensorBoard

    Args:
        writer: TensorBoard writer
        metrics: 指标字典
        epoch: 当前epoch
        phase: 阶段（train/test）
    """
    prefix = f'{phase}/'

    # 仅训练集记录Loss
    if phase == 'train' and 'loss' in metrics:
        writer.add_scalar(f'{prefix}Loss', metrics['loss'], epoch)
        
    writer.add_scalar(f'{prefix}Loss', metrics.get('loss', 0), epoch)
    writer.add_scalar(f'{prefix}Accuracy', metrics['accuracy'], epoch)
    writer.add_scalar(f'{prefix}Precision', metrics['precision'], epoch)
    writer.add_scalar(f'{prefix}Recall', metrics['recall'], epoch)
    writer.add_scalar(f'{prefix}F1', metrics['f1'], epoch)
    writer.add_scalar(f'{prefix}AUC', metrics['auc'], epoch)

        
# ================================ 主训练流程 ================================

def train(args):
    """
    主训练函数

    Args:
        args: 参数对象
    """
    set_seed(args.seed)
    
    # 设置device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 创建数据加载器
    print("Creating dataloaders...")
    # train_loader, test_loader = create_dataloaders(args)
    data_module = GaitDataModule(args)
    train_loader, test_loader = data_module.setup()
    print(f"Train batches: {len(train_loader)}, Total train data:{len(train_loader.dataset)} \
          Test batches: {len(test_loader)}, Total test data:{len(test_loader.dataset)}")

    # 创建模型
    print(f"Creating model: {args.model_type}")
    model = get_model(args.model_type, args, args.num_classes, device)

    # 加载预训练权重
    if args.pretrained_model and os.path.exists(args.pretrained_model):
        print(f"Loading pretrained model from {args.pretrained_model}")
        # state_dict = torch.load(args.pretrained_model, weights_only=True)
        if torch.__version__ >= "1.13.0":
            state_dict = torch.load(args.pretrained_model, weights_only=True)
        else:
            # 低版本移除 weights_only 参数
            state_dict = torch.load(args.pretrained_model)
        # 重大bug，之前的代码没有成功加载预训练权重！
        model.encoder.load_state_dict(state_dict['model_state_dict'], strict=False)

    # 冻结编码器参数
    if args.freeze_encoder:
        print("Freezing encoder parameters")
        for param in model.encoder.parameters():
            param.requires_grad = False

        # 初始化分类头参数
        print("Initializing classifier parameters...")
        for m in model.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                
    # 计算模型参数量与FLOPS
    data_shape = [1,1, 18, 101]
    FLOPs_calculat(model, device, data_shape)
    
    # 创建优化器和损失函数
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate
    )
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, 
    #                                             gamma=args.lr_gamma)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.num_epochs,  # 余弦周期：学习率从初始值衰减到最小值的轮数，设为总训练轮数最佳
                eta_min=1e-6        # 学习率衰减的最小值（避免衰减到0导致训练停滞，可根据需求调整）
            )

    criterion = nn.CrossEntropyLoss()

    # 创建TensorBoard writer
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # log_dir = os.path.join(args.log_dir, args.exp_name + '_' + timestamp)
    # save_dir = os.path.join(args.save_dir, args.exp_name + '_' + timestamp)
    log_dir = args.log_dir
    save_dir = args.save_dir
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to {log_dir}")
    print(f'Models will be saved to: {save_dir}')

    # 训练历史记录
    history = {
        'losses': [],
        'train_accs': [],
        'train_pres': [],
        'train_recs': [],
        'train_f1s': [],
        'train_aucs': [],
        'test_accs': [],
        'test_pres': [],
        'test_recs': [],
        'test_f1s': [],
        'test_aucs': []
    }

    best_test_acc = 0.0

    # 训练循环
    print(f"\nStarting training for {args.num_epochs} epochs...")
    for epoch in range(args.num_epochs):
        # 训练一个epoch
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, args
        )
        history['losses'].append(train_loss)

        # 评估
        train_metrics = evaluate(model, train_loader, device)
        test_metrics = evaluate(model, test_loader, device)

        # 记录历史
        history['train_accs'].append(train_metrics['accuracy'])
        history['train_pres'].append(train_metrics['precision'])
        history['train_recs'].append(train_metrics['recall'])
        history['train_f1s'].append(train_metrics['f1'])
        history['train_aucs'].append(train_metrics['auc'])

        history['test_accs'].append(test_metrics['accuracy'])
        history['test_pres'].append(test_metrics['precision'])
        history['test_recs'].append(test_metrics['recall'])
        history['test_f1s'].append(test_metrics['f1'])
        history['test_aucs'].append(test_metrics['auc'])

        # 记录到TensorBoard
        log_metrics_to_tensorboard(
            writer,
            {**train_metrics, 'loss': train_loss},
            epoch,
            'train'
        )
        log_metrics_to_tensorboard(writer, test_metrics, epoch, 'test')

        # 打印进度
        print(f'Epoch [{epoch+1}/{args.num_epochs}], '
              f'Loss: {train_loss:.5f}, '
              f'Train Acc: {train_metrics["accuracy"]:.5f}, '
              f'Test Acc: {test_metrics["accuracy"]:.5f}')

        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('train/LearningRate', current_lr, epoch)
        scheduler.step()
        
        # 保存最佳模型
        if test_metrics['accuracy'] > best_test_acc:
            best_test_acc = test_metrics['accuracy']
            best_model_path = os.path.join(
                save_dir,
                f'best_model_{args.model_type}.pth'
            )
            torch.save(model.state_dict(), best_model_path)
            print(f'New best model saved with accuracy: {best_test_acc:.5f}')

    # 保存最终模型
    final_model_path = os.path.join(
        save_dir,
        f'final_acc_model_{args.model_type}.pth'
    )
    torch.save(model.state_dict(), final_model_path)
    print(f'Final model saved to {final_model_path}')

    writer.close()

    # 计算最后10个epoch的平均指标
    compute_final_metrics(history)
    
    print(f'All results saved to: {save_dir}')
    print(f"View TensorBoard logs with: tensorboard --logdir={log_dir}")

    # 绘制训练曲线
    # plot_training_curves(
    #     history['losses'],
    #     history['train_accs'],
    #     history['test_accs'],
    #     save_path=os.path.join(args.save_dir, 'training_curves.png')
    # )

    # # 绘制混淆矩阵
    plot_confusion_matrix(
        test_metrics['labels'],
        test_metrics['predictions'],
        save_path=os.path.join(args.save_dir, 'confusion_matrix.png')
    )

    return history, model


def compute_final_metrics(history):
    """
    计算并打印最后N个epoch的平均指标

    Args:
        history: 训练历史字典
    """
    last_n = 20
    num_epochs = len(history['test_accs'])

    if num_epochs < last_n:
        print(f"\nWarning: Only {num_epochs} epochs trained, showing all epochs")
        last_n = num_epochs

    print(f"\n{'='*60}")
    print(f"Average metrics of last {last_n} epochs:")
    print(f"{'='*60}")
    print(f"Accuracy:  {np.mean(history['test_accs'][-last_n:]):.5f}")
    print(f"Precision: {np.mean(history['test_pres'][-last_n:]):.5f}")
    print(f"Recall:    {np.mean(history['test_recs'][-last_n:]):.5f}")
    print(f"F1:        {np.mean(history['test_f1s'][-last_n:]):.5f}")
    print(f"AUC:       {np.mean(history['test_aucs'][-last_n:]):.5f}")
    print(f"{'='*60}\n")


# ================================ 参数解析 ================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='步态识别分类训练')
    parser.add_argument('--exp_name', type=str, default='Gait_finetune_training')
    parser.add_argument('--mode', type=str, default='debug', help='normal, debug')
    parser.add_argument('--print_params', type=bool, default=True, help='打印参数')

    # 数据相关
    parser.add_argument('--data_path', type=str, default='./datasets/data_10000/', help='数据文件路径')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='训练集比例')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载线程数')

    # 训练相关
    parser.add_argument('--num_epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='学习率')
    parser.add_argument('--num_classes', type=int, default=27, help='分类数量')
    parser.add_argument('--lr_step_size', type=int, default=10, help='StepLR学习率衰减步长')
    parser.add_argument('--lr_gamma', type=float, default=0.5, help='StepLR学习率衰减系数')

    # 模型相关
    parser.add_argument('--model_type', type=str, default='ResNet',
                       choices=['DNN', 'GSDNN', 'GSDNN2', 'GSDNN_new', 'MSDNN', 'ResNet101'],
                       help='模型类型')
    parser.add_argument('--pretrained_model', type=str,
                       default='./save_models/Gait_selfsup_GSDNN_baseline_20260131_192518/best_model.pth',
                       help='预训练模型路径') # './save_model/best_modelGSDNNk3_27class_aug123.pth'
    parser.add_argument('--freeze_encoder', action='store_true', help='是否冻结编码器参数')

    ## parameters for GSDNN
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
    
    # 数据增强相关
    parser.add_argument('--augmentation_prob', type=float, default=0.5, help='数据增强概率')
    parser.add_argument('--freq_keep_ratio', type=float, default=0.6, help='频率成分保留比例')

    # 设备和路径
    parser.add_argument('--device', type=str, default='cuda', help='设备（cuda/cpu）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--log_dir', type=str, default='./runs', help='TensorBoard日志目录')
    parser.add_argument('--save_dir', type=str, default='./save_models',
                       help='模型保存目录')

    args = parser.parse_args()
    
    # 打印所有配置信息
    if args.print_params:
        print("="*70)
        print("步态识别分类训练配置信息")
        print("="*70)
        for key, value in sorted(vars(args).items()):
            print(f"  {key.ljust(30)}: {value}")
        print("="*70)
    
    return args



# ================================ 主入口 ================================

def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    # 开始训练
    history, model = train(args)

    print("\nTraining completed!")
    print(f"Best test accuracy: {max(history['test_accs']):.5f}")


if __name__ == '__main__':
    main()
