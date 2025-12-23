import torch
import torch.nn as nn
import torchvision.transforms as transforms
from models.resnet import ResNet101
from torch.utils.data import DataLoader, Dataset
import numpy as np
from models.DNN import DNN
from models.MSDNN import MSDNN
from models.GSDNN import GSDNN
import scipy.io as sio
import random
from scipy.fft import dct, idct
# 自定义数据集

device = "cuda" if torch.cuda.is_available() else "cpu"

class ContrastiveDataset(Dataset):
    def __init__(self, data_array, data_transform, views):
        self.transform = data_transform
        self.data_array = data_array
        self.views = views

    def __len__(self):
        return len(self.data_array)

    def __getitem__(self, idx):
        img = self.data_array[idx]
        img = torch.tensor(np.expand_dims(img, axis=0))
        imgs = []
        for _ in range(self.views):
            img2 = self.transform(img)
            imgs.append(img2)
        return imgs

# 自定义变换函数，对每个通道的时间序列进行反转
def reverse_time_series(data):
    # 翻转最后一个维度（时间点维度）
    return -data

# 使用 transforms.Lambda 将自定义函数包装起来
gait_reverse_transform = transforms.Lambda(reverse_time_series)

# 自定义变换函数，对通道进行随机打乱
def random_channel_shuffle(data):
    assert data.dim() == 3, "Input data must be 3D with channels as the second dimension"
    num_channels = data.size(1)
    shuffled_indices = torch.randperm(num_channels)
    shuffled_data = data[:, shuffled_indices, :]
    return shuffled_data

shuffle_channel = transforms.Lambda(random_channel_shuffle)

# 自定义变换函数，随机丢掉一些频率成分
def random_frequency_dropout(img):
    # 傅里叶变换，dim=2 表示在最后一个维度（时间点维度）上进行变换
    fft_img = torch.fft.fftn(img, dim=2)
    magnitude = torch.abs(fft_img)
    keep_ratio = 0.6  # 保留60%的频率成分
    num_freqs = magnitude.shape[2]
    keep_indices = np.random.choice(num_freqs, int(num_freqs * keep_ratio), replace=False)
    mask = torch.zeros_like(magnitude, dtype=torch.bool)
    mask[:, :, keep_indices] = 1
    fft_img = fft_img * mask
    img = torch.fft.ifftn(fft_img, dim=2)
    return torch.real(img)
# 使用 transforms.Lambda 将自定义函数包装起来
gait_frequency_dropout = transforms.Lambda(random_frequency_dropout)

# 数据增强变换
data_transforms = transforms.Compose([
    transforms.RandomApply([
        # 弱增强操作
        transforms.RandomResizedCrop((18, 101), scale=(0.3, 0.8), antialias=True),  #aug1 随机裁剪并恢复为18x101大小
        transforms.RandomErasing(p=1.0, scale=(0.2, 0.4), ratio=(0.3, 3.3), value=0),  #aug2 随机擦擦一部分用0填充
        gait_frequency_dropout,#aug3
        # 强增强操作
        transforms.RandomHorizontalFlip(p=1.0),  #1 水平翻转（每个沿y轴翻转）
        shuffle_channel,  #2 通道随机打乱
        gait_reverse_transform      #3将每个信号沿x轴翻转（加负号）
    ], p=1.0)  # p=1.0 表示总是随机应用其中一种方法
])



gait = sio.loadmat('data4/sub_all_data.mat')['all_data']
data_set = ContrastiveDataset(data_array=gait, data_transform=data_transforms, views=2)
dataloader = DataLoader(data_set, batch_size=64, shuffle=True)

# 使用ResNet作为编码器
class SimCLRModel(nn.Module):
    def __init__(self, base_model, out_dim=132, proj_out_dim=128):
        super(SimCLRModel, self).__init__()
        self.encoder = base_model
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])  # 移除最后的分类层
        self.dropout = nn.Dropout(p=0.2)
        # 添加投影头 v1
        #self.projector = nn.Sequential(
        #    nn.Linear(132, proj_out_dim),
        #    nn.ReLU(inplace=True),
        #    nn.Linear(proj_out_dim, 132)  # 投影回原始特征维度
        #)#176  132
        # 添加投影头（SimCLR v2 风格）
        self.projector = nn.Sequential(
            nn.Linear(132,132),  # 第一层投影
            nn.BatchNorm1d(132),      # 批量归一化
            nn.ReLU(inplace=True),    # 激活函数
            nn.Linear(132, 256),      # 第二层投影
            nn.BatchNorm1d(256),      # 批量归一化
            nn.ReLU(inplace=True),    # 激活函数
            nn.Linear( 256,128)  # 输出到目标维度
        )

    def forward(self, x):
        h = self.encoder(x)
        h = self.dropout(h)
        h = h.view(h.size(0), -1)
        # 使用投影头
        h = self.projector(h)
        return h
# 对比损失函数
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        z_i = nn.functional.normalize(z_i, dim=1)
        z_j = nn.functional.normalize(z_j, dim=1)

        # 计算余弦相似度
        sim_matrix = torch.matmul(z_i, z_j.T) / self.temperature
        sim_labels = torch.arange(batch_size).to(z_i.device)
        loss = nn.CrossEntropyLoss()(sim_matrix, sim_labels)
        return loss

# 数据加载

# 初始化模型和损失函数
model = SimCLRModel(GSDNN(), out_dim=128).to(device)
criterion = ContrastiveLoss()

import matplotlib.pyplot as plt

# 初始化损失值列表
loss_values = []
# 初始化最小损失值
min_loss = float('inf')
# 训练循环
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
epoch = 40



for epoch_index in range(epoch):
    for (img1, img2) in dataloader:
        optimizer.zero_grad()
        # 调整img1和img2的形状
        img1 = torch.squeeze(img1, dim=1)  # 将64*1*18*101变为64*18*101
        img2 = torch.squeeze(img2, dim=1)  # 将64*1*18*101变为64*18*101

        z_i = model(img1.to(device=device, dtype=torch.float32))
        z_j = model(img2.to(device=device, dtype=torch.float32))
        loss = criterion(z_i, z_j)
        loss.backward()
        optimizer.step()

    # 将损失值添加到列表中
    loss_values.append(loss.cpu().item())

    # 检查是否达到最小损失
    if loss.cpu().item() < min_loss:
        min_loss = loss.cpu().item()
        # 保存模型
        torch.save(model.state_dict(), "./save_model/best_model2.pth")  # 保存模型参数
        print(f'Epoch {epoch_index + 1}, New minimum loss: {min_loss}, Model saved.')
    else:
        print(f'Epoch {epoch_index + 1}, Loss: {loss.cpu().item()}')



# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(loss_values, label='Training Loss')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()