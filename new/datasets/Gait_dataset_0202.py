import torch
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random

class GaitDataset(Dataset):
    """步态数据集类"""
    def __init__(self, data_array, data_label, data_transform=None):
        self.data_array = data_array.astype(np.float32)
        self.data_label = data_label.astype(np.int64)
        self.transform = data_transform
        
    def __len__(self):
        return len(self.data_array)
        
    def __getitem__(self, idx):
        img = torch.tensor(self.data_array[idx])
        label = self.data_label[idx]
        
        # 确保正确的维度 [1, channels, time]
        if img.dim() == 2:
            img = img.unsqueeze(0)
            
        if self.transform:
            img = self.transform(img)
            
        return img, label


class GaitDataModule:
    """统一的数据管理类"""
    def __init__(self, args):
        self.args = args
        self.set_seed()
        
    def set_seed(self, seed=None):
        seed = seed or self.args.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
    def load_data(self, data_path):
        """加载数据"""
        data = sio.loadmat(f'{data_path}/sub_data.mat')['sub_data']
        labels = sio.loadmat(f'{data_path}/sub_label.mat')['sub_label'][0]
        return data, labels
        
    def split_data(self, data, labels):
        """划分数据"""
        indices = np.arange(len(data))
        np.random.shuffle(indices)
        
        split_idx = int(len(data) * self.args.train_ratio)
        train_idx, test_idx = indices[:split_idx], indices[split_idx:]
        
        train_data = data[train_idx]
        train_labels = labels[train_idx] - 1
        test_data = data[test_idx]
        test_labels = labels[test_idx] - 1
        
        return train_data, test_data, train_labels, test_labels
        
    def get_transforms(self, augment=False):
        """获取数据变换"""
        if augment:
            return transforms.Compose([
                transforms.RandomApply([
                    transforms.Lambda(lambda x: self.random_frequency_dropout(x, self.args.freq_keep_ratio)),
                    transforms.Lambda(self.reverse_time_series),
                ], p=self.args.augmentation_prob),
            ])
        return None
        
    def setup(self):
        """准备数据加载器"""
        # 加载并划分数据
        data, labels = self.load_data(self.args.data_path)
        train_data, test_data, train_labels, test_labels = self.split_data(data, labels)
        
        # 创建数据集
        train_dataset = GaitDataset(
            train_data, 
            train_labels, 
            self.get_transforms(augment=True)
        )
        test_dataset = GaitDataset(
            test_data, 
            test_labels, 
            self.get_transforms(augment=False)
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        return train_loader, test_loader
        
    @staticmethod
    def reverse_time_series(data):
        return -data
    
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