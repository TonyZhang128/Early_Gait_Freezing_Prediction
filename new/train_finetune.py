"""
步态识别分类训练脚本 - 重构版
包含TensorBoard监控、参数解析、模块化设计
"""

import argparse
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, auc, roc_auc_score)
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# 导入模型
from models.DNN import DNN
from models.GSDNN import GSDNN
from models.GSDNN2 import GSDNN2
from models.GSDNN_new import GSDNN_new
from models.MSDNN import MSDNN
from models.resnet import ResNet101


# ================================ 配置和常量 ================================

class Config:
    """配置类，集中管理所有超参数"""
    def __init__(self, args):
        self.data_path = args.data_path
        self.train_ratio = args.train_ratio
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.learning_rate = args.learning_rate
        self.num_classes = args.num_classes
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.log_dir = args.log_dir
        self.model_save_dir = args.model_save_dir
        self.pretrained_model = args.pretrained_model
        self.model_type = args.model_type
        self.freeze_encoder = args.freeze_encoder
        self.augmentation_prob = args.augmentation_prob
        self.freq_keep_ratio = args.freq_keep_ratio

        # 创建必要的目录
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.model_save_dir).mkdir(parents=True, exist_ok=True)


# ================================ 数据增强模块 ================================

def reverse_time_series(data):
    """时间序列反转"""
    return -data


def random_channel_shuffle(data):
    """随机通道打乱"""
    assert data.dim() == 3, "Input data must be 3D with channels as the second dimension"
    num_channels = data.size(1)
    shuffled_indices = torch.randperm(num_channels)
    return data[:, shuffled_indices, :]


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


def get_data_transforms(augmentation_prob=0.5, freq_keep_ratio=0.6):
    """构建数据增强变换组合"""
    return transforms.Compose([
        transforms.RandomApply([
            transforms.Lambda(lambda x: random_frequency_dropout(x, freq_keep_ratio)),
            transforms.Lambda(reverse_time_series),
        ], p=augmentation_prob)
    ])


# ================================ 数据集类 ================================

class GaitDataset(Dataset):
    """步态数据集类"""

    def __init__(self, data_array, data_label, data_transform=None, views=2):
        """
        Args:
            data_array: 数据数组
            data_label: 标签数组
            data_transform: 数据变换
            views: 视角数量
        """
        self.transform = data_transform
        self.data_array = data_array
        self.data_label = data_label
        self.views = views

    def __len__(self):
        return len(self.data_array)

    def __getitem__(self, idx):
        img = self.data_array[idx]
        if self.transform:
            img = self.transform(torch.tensor(np.expand_dims(img, axis=0)))
        return img, self.data_label[idx]


# ================================ 数据加载模块 ================================

def load_and_split_data(data_path, train_ratio=0.8, random_seed=42):
    """
    加载并划分训练集和测试集

    Args:
        data_path: 数据文件路径（不含后缀）
        train_ratio: 训练集比例
        random_seed: 随机种子

    Returns:
        train_data, test_data, train_label, test_label
    """
    np.random.seed(random_seed)

    # 加载数据
    # data_finetue = sio.loadmat(f'{data_path}/sub_train_data.mat')['sub_train_data']
    # labels_finetue = sio.loadmat(f'{data_path}/sub_train_label.mat')['sub_train_label'][0]
    
    # data_test = sio.loadmat(f'{data_path}/sub_test_data.mat')['sub_data']
    # labels_test = sio.loadmat(f'{data_path}/sub_test_label.mat')['sub_label'][0]
    
    # train_data = data_finetue
    # train_label = labels_finetue - 1
    # test_data = data_test
    # test_label = labels_test - 1
    
    data = sio.loadmat(f'{data_path}/sub_test_data.mat')['sub_data']
    labels = sio.loadmat(f'{data_path}/sub_test_label.mat')['sub_label'][0]

    # 打乱索引
    random_index = np.array(range(len(data)))
    np.random.shuffle(random_index)

    # 应用打乱
    data = data[random_index]
    labels = labels[random_index]

    # 划分数据集
    train_len = int(len(data) * train_ratio)

    train_data = data[:train_len]
    test_data = data[train_len:]
    train_label = labels[:train_len] - 1  # 标签从0开始
    test_label = labels[train_len:] - 1

    return train_data, test_data, train_label, test_label


def create_dataloaders(config):
    """
    创建数据加载器

    Args:
        config: 配置对象

    Returns:
        train_loader, test_loader
    """
    # 加载数据
    train_data, test_data, train_label, test_label = load_and_split_data(
        config.data_path, config.train_ratio
    )

    # 创建数据增强
    data_transforms = get_data_transforms(
        config.augmentation_prob,
        config.freq_keep_ratio
    )

    # 创建数据集
    train_dataset = GaitDataset(
        data_array=train_data,
        data_label=train_label,
        data_transform=data_transforms,
        views=2
    )

    test_dataset = GaitDataset(
        data_array=test_data,
        data_label=test_label,
        data_transform=data_transforms,
        views=2
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )

    return train_loader, test_loader


# ================================ 模型定义模块 ================================

class SimCLREncoder(nn.Module):
    """SimCLR风格的编码器"""

    def __init__(self, base_model, out_dim=132, proj_out_dim=128, dropout=0.5):
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
            nn.Linear(proj_out_dim, out_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        h = self.dropout(h)
        h = h.view(h.size(0), -1)
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


def get_model(model_type, num_classes=27, device='cpu'):
    """
    根据类型获取模型

    Args:
        model_type: 模型类型
        num_classes: 分类数量
        device: 设备

    Returns:
        model: 模型实例
    """
    model_dict = {
        'DNN': DNN,
        'GSDNN': GSDNN,
        'GSDNN2': GSDNN2,
        'GSDNN_new': GSDNN_new,
        'MSDNN': MSDNN,
        'ResNet101': ResNet101
    }

    if model_type not in model_dict:
        raise ValueError(f"Unsupported model type: {model_type}")

    base_model = model_dict[model_type]()
    encoder = SimCLREncoder(base_model, out_dim=132, proj_out_dim=128)
    model = ClassificationModel(encoder, num_features=132, num_classes=num_classes)

    return model.to(device)


def init_weights(m):
    """Xavier初始化"""
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


def init_weights_normal(m):
    """正态分布初始化"""
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


# ================================ 训练和评估模块 ================================

def train_one_epoch(model, dataloader, optimizer, criterion, device):
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


# ================================ 可视化模块 ================================

def plot_training_curves(losses, train_accs, test_accs, save_path=None):
    """
    绘制训练曲线

    Args:
        losses: 损失列表
        train_accs: 训练准确率列表
        test_accs: 测试准确率列表
        save_path: 保存路径（可选）
    """
    plt.figure(figsize=(15, 5))

    # 绘制损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(losses, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 绘制准确率曲线
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(test_accs, label='Testing Accuracy')
    plt.title('Training and Testing Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Training curves saved to {save_path}")
    else:
        plt.show()


def plot_confusion_matrix(y_labels, y_preds, save_path=None):
    """
    绘制混淆矩阵

    Args:
        y_labels: 真实标签
        y_preds: 预测标签
        save_path: 保存路径（可选）
    """
    conf_matrix = confusion_matrix(y_labels, y_preds)

    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    num_classes = conf_matrix.shape[0]
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    thresh = conf_matrix.max() / 2.0
    for i, j in np.ndindex(conf_matrix.shape):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                horizontalalignment="center",
                color="white" if conf_matrix[i, j] > thresh else "black")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()


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

    writer.add_scalar(f'{prefix}Loss', metrics.get('loss', 0), epoch)
    writer.add_scalar(f'{prefix}Accuracy', metrics['accuracy'], epoch)
    writer.add_scalar(f'{prefix}Precision', metrics['precision'], epoch)
    writer.add_scalar(f'{prefix}Recall', metrics['recall'], epoch)
    writer.add_scalar(f'{prefix}F1', metrics['f1'], epoch)
    writer.add_scalar(f'{prefix}AUC', metrics['auc'], epoch)


# ================================ 主训练流程 ================================

def train(config):
    """
    主训练函数

    Args:
        config: 配置对象
    """
    # 创建数据加载器
    print("Creating dataloaders...")
    train_loader, test_loader = create_dataloaders(config)
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # 创建模型
    print(f"Creating model: {config.model_type}")
    model = get_model(config.model_type, config.num_classes, config.device)

    # 加载预训练权重
    if config.pretrained_model and os.path.exists(config.pretrained_model):
        print(f"Loading pretrained model from {config.pretrained_model}")
        state_dict = torch.load(config.pretrained_model, weights_only=True)
        model.encoder.load_state_dict(state_dict, strict=False)

    # 冻结编码器参数
    if config.freeze_encoder:
        print("Freezing encoder parameters")
        for param in model.encoder.parameters():
            param.requires_grad = False

    # 创建优化器和损失函数
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate
    )
    criterion = nn.CrossEntropyLoss()

    # 创建TensorBoard writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(config.log_dir, f"run_{timestamp}")
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to {log_dir}")

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
    print(f"\nStarting training for {config.num_epochs} epochs...")
    for epoch in range(config.num_epochs):
        # 训练一个epoch
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, config.device
        )
        history['losses'].append(train_loss)

        # 评估
        train_metrics = evaluate(model, train_loader, config.device)
        test_metrics = evaluate(model, test_loader, config.device)

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
        print(f'Epoch [{epoch+1}/{config.num_epochs}], '
              f'Loss: {train_loss:.5f}, '
              f'Train Acc: {train_metrics["accuracy"]:.5f}, '
              f'Test Acc: {test_metrics["accuracy"]:.5f}')

        # 保存最佳模型
        if test_metrics['accuracy'] > best_test_acc:
            best_test_acc = test_metrics['accuracy']
            best_model_path = os.path.join(
                config.model_save_dir,
                f'best_model_{config.model_type}.pth'
            )
            torch.save(model.state_dict(), best_model_path)
            print(f'New best model saved with accuracy: {best_test_acc:.5f}')

    # 保存最终模型
    final_model_path = os.path.join(
        config.model_save_dir,
        f'final_model_{config.model_type}.pth'
    )
    torch.save(model.state_dict(), final_model_path)
    print(f'Final model saved to {final_model_path}')

    writer.close()

    # 计算最后10个epoch的平均指标
    compute_final_metrics(history)

    # 绘制训练曲线
    plot_training_curves(
        history['losses'],
        history['train_accs'],
        history['test_accs'],
        save_path=os.path.join(config.model_save_dir, 'training_curves.png')
    )

    # 绘制混淆矩阵
    plot_confusion_matrix(
        test_metrics['labels'],
        test_metrics['predictions'],
        save_path=os.path.join(config.model_save_dir, 'confusion_matrix.png')
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

    # 数据相关
    parser.add_argument('--data_path', type=str, default='data3',
                       help='数据文件路径')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='训练集比例')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批次大小')

    # 训练相关
    parser.add_argument('--num_epochs', type=int, default=200,
                       help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help='学习率')
    parser.add_argument('--num_classes', type=int, default=27,
                       help='分类数量')

    # 模型相关
    parser.add_argument('--model_type', type=str, default='GSDNN',
                       choices=['DNN', 'GSDNN', 'GSDNN2', 'GSDNN_new', 'MSDNN', 'ResNet101'],
                       help='模型类型')
    parser.add_argument('--pretrained_model', type=str,
                       default='./save_model_refactor/best_model.pth',
                       help='预训练模型路径') # './save_model/best_modelGSDNNk3_27class_aug123.pth'
    parser.add_argument('--freeze_encoder', action='store_true',
                       help='是否冻结编码器参数')

    # 数据增强相关
    parser.add_argument('--augmentation_prob', type=float, default=0.5,
                       help='数据增强概率')
    parser.add_argument('--freq_keep_ratio', type=float, default=0.6,
                       help='频率成分保留比例')

    # 设备和路径
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备（cuda/cpu）')
    parser.add_argument('--log_dir', type=str, default='runs/classification',
                       help='TensorBoard日志目录')
    parser.add_argument('--model_save_dir', type=str, default='./save_model_finetue_frozen_refactor',
                       help='模型保存目录')

    return parser.parse_args()


# ================================ 主入口 ================================

def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    # 创建配置对象
    config = Config(args)

    # 打印配置信息
    print("="*60)
    print("Training Configuration:")
    print("="*60)
    print(f"Model: {config.model_type}")
    print(f"Device: {config.device}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Data Path: {config.data_path}")
    print(f"Log Dir: {config.log_dir}")
    print(f"Model Save Dir: {config.model_save_dir}")
    print("="*60 + "\n")

    # 开始训练
    history, model = train(config)

    print("\nTraining completed!")
    print(f"Best test accuracy: {max(history['test_accs']):.5f}")


if __name__ == '__main__':
    main()
