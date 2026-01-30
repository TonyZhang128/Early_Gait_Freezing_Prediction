import random
import numpy as np
import scipy.io as sio

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
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
    
    data = sio.loadmat(f'{data_path}/sub_data.mat')['sub_data']
    labels = sio.loadmat(f'{data_path}/sub_label.mat')['sub_label'][0]

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

def create_dataloaders(args):
    """
    创建数据加载器

    Args:
        args: 参数对象

    Returns:
        train_loader, test_loader
    """
    # 加载数据
    train_data, test_data, train_label, test_label = load_and_split_data(
        args.data_path, args.train_ratio
    )

    # 创建数据增强
    data_transforms = get_data_transforms(
        args.augmentation_prob,
        args.freq_keep_ratio
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
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    return train_loader, test_loader