import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
from models.GSDNN import GSDNN
import scipy.io as sio
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from models.GSDNN_atten import GSDNN_atten

# 初始化模型和损失函数
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GSDNN_atten(num_classes=27)  # 确保你初始化了基础模型

model.load_state_dict(torch.load("./save_model/best_modelGSDNN_27class_aug123.pth", map_location=device), strict=False)  # 加载状态字典

# 将模型设置为评估模式
model.eval()

gait = sio.loadmat('data3/sub_test_data.mat')['sub_data']
#gaitlabel = sio.loadmat('data2/milabels.mat')['milabels'][0]

# 选择测试集中的第一个数据
test_data = gait[0,:,:]  # 假设test_dataset是你的测试数据集
input_data = torch.from_numpy(test_data)
input_tensor = torch.tensor(input_data).unsqueeze(0).float()
input_tensor.requires_grad = True  # 需要计算梯度
# 前向传播，获取特征图和分类结果
features = model.blocks[:-1](input_tensor) # 获取倒数第二层的特征图
pooled_features = model.blocks[-1:](features) # 获取最后的全局池化层输出
logits = model.fc(torch.squeeze(pooled_features)) # 获取分类结果
probs = F.softmax(logits, dim=0) # 获取概率分布
predicted_class = torch.argmax(probs) # 获取预测类别
# 获取目标类别的得分
target = logits[predicted_class]
# 获取权重
weights = model.fc.weight.data.cpu().numpy()


# 选择第i个类别的权重
i = predicted_class.item()
#weights_i = weights[:, :]
features = torch.squeeze(features, dim=0)
# 将权重转换为Tensor
weights_i_tensor = torch.tensor(weights, dtype=torch.float32)
weights_i_tensor = weights_i_tensor.unsqueeze(0)
#cam = torch.zeros(1, 101, dtype=torch.float32)
cam = torch.matmul( weights_i_tensor , features)
cam = cam.unsqueeze(0)
# 重采样到 (18, 101) 的大小
cam_resampled = F.interpolate(cam, size=(18, 101), mode='bilinear', align_corners=False)

# 将结果转换回 (18, 101) 的形状
cam_resampled = cam_resampled.squeeze(0).squeeze(0)
# 应用ReLU激活函数，去除负值
cam_resampled = torch.relu(cam_resampled)


# 标准化CAM
cam_resampled = cam_resampled - cam_resampled.min()
cam_resampled = cam_resampled / cam_resampled.max()

import matplotlib.pyplot as plt
import numpy as np

# 假设你已经有了原始信号和CAM结果
# original_signal: 原始信号，形状为 (18, 101)
# cam: CAM结果，形状为 (18, 101)

# 创建一个新的图形
plt.figure(figsize=(15, 30))

# 遍历所有通道
for i in range(input_data.shape[0]):
    plt.subplot(input_data.shape[0], 1, i + 1)

    # 绘制当前通道的原始信号
    plt.plot(input_data[i, :], color='blue')

    # 使用 PyTorch 计算均值
    mean_value = torch.mean(cam_resampled[i, :])

    # 将 PyTorch 张量转换为 NumPy 数组
    cam_resampled_np = cam_resampled[i, :].detach().numpy()

    # 绘制当前通道的CAM结果，使用红色突出显示高激活区域
    plt.fill_between(np.arange(input_data.shape[1]),
                     input_data[i, :],
                     where=cam_resampled_np > mean_value.item(),
                     color='red', alpha=0.5)

    # 添加图例和标签
    #plt.legend()
    #plt.xlabel('Time Points')
    #plt.ylabel('Amplitude')
    #plt.title(f'Channel {i} with CAM Activation Highlighted')

# 调整子图间距
plt.tight_layout()

# 显示图形
plt.show()

# 将 18 个通道重新组合为 6 个通道
combined_input_data = []
combined_cam_resampled = []

for i in range(0, 18, 3):
    combined_input_data.append(torch.sum(input_data[i:i+3, :], dim=0))
    combined_cam_resampled.append(torch.sum(cam_resampled[i:i+3, :], dim=0))

combined_input_data = torch.stack(combined_input_data, dim=0)  # 形状为 (6, 101)
combined_cam_resampled = torch.stack(combined_cam_resampled, dim=0)  # 形状为 (6, 101)

# 创建一个新的图形
plt.figure(figsize=(15, 10))

# 遍历所有通道
for i in range(combined_input_data.shape[0]):
    plt.subplot(combined_input_data.shape[0], 1, i + 1)

    # 绘制当前通道的原始信号
    plt.plot(combined_input_data[i, :], color='blue')

    # 使用 PyTorch 计算均值
    mean_value = torch.mean(combined_cam_resampled[i, :])

    # 将 PyTorch 张量转换为 NumPy 数组
    cam_resampled_np = combined_cam_resampled[i, :].detach().numpy()

    # 绘制当前通道的 CAM 结果，使用红色突出显示高激活区域
    plt.fill_between(np.arange(combined_input_data.shape[1]),
                     combined_input_data[i, :],
                     where=cam_resampled_np > mean_value.item(),
                     color='red', alpha=0.5)

    # 添加图例和标签
  #  plt.legend(['Original Signal', 'CAM Activation'])
  #  plt.xlabel('Time Points')
  #  plt.ylabel('Amplitude')
 #   plt.title(f'Channel {i+1} with CAM Activation Highlighted')

# 调整子图间距
plt.tight_layout()

# 显示图形
plt.show()