import numpy as np
import scipy.io as sio

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

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


def load_data(data_path, batch_size, views, num_workers, args):
    """加载数据并创建DataLoader"""
    gait = sio.loadmat(data_path)['all_data']
    transform = GaitAugmentation.get_transforms(args.freq_keep_ratio)
    dataset = ContrastiveDataset(data_array=gait, data_transform=transform, views=views)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader