import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy.io as sio
from models.GSDNN import GSDNN
from models.resnet import ResNet101
from models.resnet import ResNet, Bottleneck
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# 自定义数据集
class GaitDataset(Dataset):
    def __init__(self, data_array, labels):
        self.data_array = data_array
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.data_array[idx]
        label = self.labels[idx]
        return img, label

# 加载数据
gait_data = sio.loadmat('C:/Users/shuli/Desktop/gait_DL4/gait_DL4/data3/sub_all_data.mat')['all_data']
gait_labels = sio.loadmat('C:/Users/shuli/Desktop/gait_DL4/gait_DL4/data3/sub_all_label.mat')['sub_all_label'].flatten()

# 划分数据集
train_data, test_data, train_labels, test_labels = train_test_split(
    gait_data, gait_labels, test_size=0.2, random_state=42, stratify=gait_labels
)

# 创建数据集和数据加载器
train_dataset = GaitDataset(train_data, train_labels)
test_dataset = GaitDataset(test_data, test_labels)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class ResNetModel(nn.Module):
    def __init__(self, base_model, num_classes=5):
        super(ResNetModel, self).__init__()
        self.encoder = base_model
        # 移除最后的分类层和平均池化层
        self.encoder.fc = nn.Identity()  # 使用Identity层代替原来的全连接层
        #self.dropout = nn.Dropout(0.1)  # 添加Dropout层
        # 添加新的全连接层
        self.linear = nn.Linear(132, 27)  # 移除最后的分类层# 根据ResNet的输出特征图大小调整

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # 展平特征图
       # x = self.dropout(x)  # 应用Dropout
        return x
# 初始化模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ResNetModel(ResNet101(), num_classes=5).to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)

# 初始化变量来跟踪最佳性能
best_loss = float('inf')
best_model_path = 'C:/Users/shuli/Desktop/gait_DL4/gait_DL4/gait_SL/save_model/best_model.pth'


# 训练循环
num_epochs = 40
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(num_epochs):
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    # 训练循环
    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        #inputs = inputs.squeeze(0).unsqueeze(1).to(device)  # 调整为[64, 1, 18, 101]
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs.to(device=device, dtype=torch.float32))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct_train / total_train
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # 测试循环
    model.eval()
    running_loss = 0.0
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            #inputs = inputs.squeeze(0).unsqueeze(1).to(device)  # 调整为[64, 1, 18, 101]
            labels = labels.to(device)
            outputs = model(inputs.to(device=device, dtype=torch.float32))
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    test_loss = running_loss / len(test_loader)
    test_accuracy = correct_test / total_test
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    # 检查是否是最佳模型
    if test_loss < best_loss:
        best_loss = test_loss
        # 保存模型
        torch.save(model.state_dict(), best_model_path)
        print(f'Best model saved with loss: {best_loss:.4f}')

    print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}')

print('Finished Training')

# 可视化结果
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy over epochs')
plt.legend()

plt.show()