import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
# 假设你的输入是一个18x101的矩阵，这里我们用随机数来模拟这个输入
input_matrix = torch.randn(1, 18, 101)

# 自定义变换函数，对每个通道的时间序列进行反转
def reverse_time_series(data):
    # 翻转最后一个维度（时间点维度）
    return data

# 使用 transforms.Lambda 将自定义函数包装起来
reverse_transform = transforms.Lambda(reverse_time_series)


# 自定义变换函数，对通道进行随机打乱
def random_channel_shuffle(data):
    # 确保数据是三维的，其中第二维是通道维
    assert data.dim() == 3, "Input data must be 3D with channels as the second dimension"

    # 获取通道数
    num_channels = data.size(1)

    # 生成通道的随机索引
    shuffled_indices = torch.randperm(num_channels)

    # 根据随机索引打乱通道
    shuffled_data = data[:, shuffled_indices, :]

    return shuffled_data

shuffle_channel = transforms.Lambda(random_channel_shuffle)

# 自定义变换函数，随机丢掉一些频率成分
def random_frequency_dropout(img):
    # 傅里叶变换，dim=2 表示在最后一个维度（时间点维度）上进行变换
    fft_img = torch.fft.fftn(img, dim=2)
    # 获取频率成分的幅度
    magnitude = torch.abs(fft_img)
    # 确定要保留的频率成分比例
    keep_ratio = 0.6  # 保留60%的频率成分
    num_freqs = magnitude.shape[2]
    # 随机选择要保留的频率成分索引
    keep_indices = np.random.choice(num_freqs, int(num_freqs * keep_ratio), replace=False)
    # 创建一个全零的掩码，只在保留的频率成分上设为1
    mask = torch.zeros_like(magnitude, dtype=torch.bool)
    mask[:, :, keep_indices] = 1
    # 应用掩码，将不需要的频率成分设置为0
    fft_img = fft_img * mask
    # 逆傅里叶变换
    img = torch.fft.ifftn(fft_img, dim=2)
    # 取实部并返回
    return torch.real(img)
# 使用 transforms.Lambda 将自定义函数包装起来
gait_frequency_dropout = transforms.Lambda(random_frequency_dropout)

# 定义你的数据增强变换
data_transforms = transforms.Compose([
    transforms.RandomApply([
        # 弱增强操作
        #transforms.RandomResizedCrop((18, 101), scale=(0.3, 0.8), antialias=True),  # 随机裁剪并恢复为18x101大小
        #transforms.RandomErasing(p=1.0, scale=(0.2, 0.4), ratio=(0.3, 3.3), value=0),  # 随机擦擦一部分用0填充
        #gait_frequency_dropout,
        # 强增强操作
        #transforms.RandomHorizontalFlip(p=1.0),  # 水平翻转（每个沿y轴翻转）
        #transforms.RandomVerticalFlip(p=1.0),  # 翻转(通道翻转了，1通道变成18通道，18通道变成了1通道)
        reverse_transform,      #将每个信号沿x轴翻转（加负号）
        #shuffle_channel
    ], p=1.0)  # p=1.0 表示总是随机应用其中一种方法
])

# 应用变换
transformed_matrix = data_transforms(input_matrix)

# 选择第一个通道的数据
channel_index = 1  # 选择第一个通道
original_channel_data = input_matrix[0, channel_index, :].numpy()
transformed_channel_data = transformed_matrix[0, channel_index, :].numpy()

# 绘制原始数据
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(original_channel_data, label='Original Data')
plt.title(f'Original Channel {channel_index} Data')
plt.legend()

# 绘制变换后的数据
plt.subplot(2, 1, 2)
plt.plot(transformed_channel_data, label='Transformed Data', color='red')
plt.title(f'Transformed Channel {channel_index} Data (Random Horizontal Flip)')
plt.legend()

plt.tight_layout()
plt.show()
# 输出变换后的矩阵
print(transformed_matrix)