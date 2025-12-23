from email.policy import strict

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from sympy import false

from models.resnet import ResNet101
from models.GSDNN import GSDNN

from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.nn.functional as F
import scipy
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score
# 自定义数据集

class MyDataset(Dataset):
    def __init__(self, data_array, data_label, data_transform, views):
        self.transform = data_transform
        self.data_array = data_array
        self.data_label = data_label
        self.views = views

    def __len__(self):
        return len(self.data_array)

    def __getitem__(self, idx):
        img = self.data_array[idx]
        return self.transform(torch.tensor(np.expand_dims(img, axis=0))), self.data_label[idx]


# 自定义变换函数，对每个通道的时间序列进行反转
def reverse_time_series(data):
    # 翻转最后一个维度（时间点维度）
    return -data

# 使用 transforms.Lambda 将自定义函数包装起来
reverse_transform = transforms.Lambda(reverse_time_series)

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

# 定义你的数据增强变换
data_transforms = transforms.Compose([
    transforms.RandomApply([
        # 弱增强操作
        #transforms.RandomResizedCrop((18, 101), scale=(0.3, 0.8), antialias=True),  # 随机裁剪并恢复为18x101大小
        #transforms.RandomErasing(p=1.0, scale=(0.2, 0.4), ratio=(0.3, 3.3), value=0),  # 随机擦擦一部分用0填充
        gait_frequency_dropout,
        # 强增强操作
        #transforms.RandomHorizontalFlip(p=1.0),  # 水平翻转（每个沿y轴翻转）
        #shuffle_channel,  # 通道翻转
        reverse_transform      #将每个信号沿x轴翻转（加负号）
    ], p=0)  # p=1.0 表示总是随机应用其中一种方法
])

random_index = np.array(range(0, len(scipy.io.loadmat('data2/midata.mat')['midata'])))  # 将数组打乱)
np.random.shuffle(random_index)
gait = sio.loadmat('data2/midata.mat')['midata'][random_index]
gaitlabel = sio.loadmat('data2/milabels.mat')['milabels'][0][random_index]
train_len = int(len(gait) * 0.8)
train_dataset = MyDataset(data_array=gait[: train_len], data_label=gaitlabel[: train_len] - 1, data_transform=data_transforms, views=2)
test_dataset = MyDataset(data_array=gait[train_len:], data_label=gaitlabel[train_len: ] - 1, data_transform=data_transforms, views=2)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# 使用ResNet作为编码器
class SimCLRModel(nn.Module):
    def __init__(self, base_model, out_dim=132, proj_out_dim=128):
        super(SimCLRModel, self).__init__()
        self.encoder = base_model
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])  # 移除最后的分类层
        self.dropout = nn.Dropout(p=0.5)
        # 添加投影头
        self.projector = nn.Sequential(
            nn.Linear(132, proj_out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_out_dim, 132)  # 投影回原始特征维度
        )

    def forward(self, x):
        h = self.encoder(x)
        h = self.dropout(h)
        h = h.view(h.size(0), -1)
        # 使用投影头
        #h = self.projector(h)
        return h


# 使用ResNet作为编码器
class ClassficationModel(nn.Module):
    def __init__(self, encoder):
        super(ClassficationModel, self).__init__()
        self.encoder = encoder
        self.linear = nn.Linear(132, 5)  # 移除最后的分类层
#MSDNN176,GSDNN132
    def forward(self, x):
        h = self.encoder(x)
        return self.linear(F.relu(h))


device = "cuda" if torch.cuda.is_available() else "cpu"
# 初始化模型和损失函数
base_model = GSDNN()  # 确保你初始化了基础模型
encoder = SimCLRModel(base_model, out_dim=32).to(device)  # 创建SimCLR模型
encoder.load_state_dict(torch.load("./save_model/best_modelGSDNNk357_5class_aug123.pth"))  # 加载状态字典
# 冻结编码器的参数
for param in encoder.encoder.parameters():
    param.requires_grad = False

# 冻结投影头的参数
#for param in encoder.projector.parameters():
#    param.requires_grad = False

model = ClassficationModel(encoder).to(device)

loss_fun = nn.CrossEntropyLoss()
# 训练循环
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)


# 定义specificity_score函数
def specificity_score(y_true, y_pred):
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def eval(model, dataloader):
    model = model.eval()
    with torch.no_grad():
        y_preds = []
        y_labels = []
        for x, y in dataloader:
            # 调整img1和img2的形状
            x = torch.squeeze(x, dim=1)  # 将64*1*18*101变为64*18*101
            y_pred = torch.argmax(model(x.to(device=device, dtype=torch.float32)), dim=1).cpu().numpy()
            y_label = y.cpu().numpy()
            y_preds.append(y_pred)
            y_labels.append(y_label)
    y_preds = np.concatenate(y_preds, axis=0)
    y_labels = np.concatenate(y_labels, axis=0)
    acc = np.mean(np.equal(y_preds, y_labels))
    # 计算评估指标
    precision = precision_score(y_labels, y_preds, average='macro')
    recall = recall_score(y_labels, y_preds, average='macro')
    f1 = f1_score(y_labels, y_preds, average='macro')
    specificity = specificity_score(y_labels, y_preds)  # 需要定义specificity_score函数
    return y_preds, y_labels, acc, precision, recall, f1, specificity



epoch = 200
losses = []
train_accs = []
train_pres = []
train_recs = []
train_f1s = []
train_spes = []
test_accs = []
test_pres = []
test_recs = []
test_f1s = []
test_spes = []

for epoch in range(epoch):
    for (x, label) in train_dataloader:

        optimizer.zero_grad()
        # 调整img1和img2的形状
        x = torch.squeeze(x, dim=1)  # 将64*1*18*101变为64*18*101
        label = label.to(device=device)
        y_pred = model(x.to(device=device, dtype=torch.float32))
        loss = loss_fun(y_pred, label)
        loss.backward()
        optimizer.step()
    losses.append(loss.item())
    y_preds, y_labels, train_acc, train_pre,train_rec, train_f1, train_spe  = eval(model, train_dataloader)
    test_preds, test_labels, test_acc ,test_pre,test_rec, test_f1, test_spe= eval(model, test_dataloader)
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    test_pres.append(test_pre)
    test_recs.append(test_rec)
    test_f1s.append(test_f1)
    test_spes.append(test_spe)
    print(f'Epoch {epoch + 1}, loss: {loss.item():.5f}, train_acc={train_acc:.5f}, test_acc={test_acc:.5f}')

# 保存模型
#if epoch + 1 == 200:  # 检查是否完成了200个epoch
#    model_path = "./save_model/final_modelGSDNNk357_pro.pth"  # 定义模型保存路径
#    torch.save(model.state_dict(), model_path)  # 保存模型状态字典#
#    print(f"Model saved to {model_path}")

# 绘制图表

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(losses, label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Training Accuracy')
plt.plot(test_accs, label='Testing Accuracy')
plt.title('Training and Testing Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# 生成混淆矩阵
conf_matrix = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(5)
plt.xticks(tick_marks, tick_marks)
plt.yticks(tick_marks, tick_marks)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
thresh = conf_matrix.max() / 2.
for i, j in np.ndindex(conf_matrix.shape):
    plt.text(j, i, format(conf_matrix[i, j], 'd'),
             horizontalalignment="center",
             color="white" if conf_matrix[i, j] > thresh else "black")
plt.show()

# 假设你已经有了所有指标的列表
last_10_accs = np.mean(test_accs[-10:])
last_10_precisions = np.mean(test_pres[-10:])
last_10_recalls = np.mean(test_recs[-10:])
last_10_f1s = np.mean(test_f1s[-10:])
last_10_specificities = np.mean(test_spes[-10:])

print(f'Average of last 10 epochs - Accuracy: {last_10_accs}, Precision: {last_10_precisions}, Recall: {last_10_recalls}, F1: {last_10_f1s}, Specificity: {last_10_specificities}')