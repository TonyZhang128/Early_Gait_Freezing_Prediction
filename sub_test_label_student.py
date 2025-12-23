from email.policy import strict

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from sympy import false
from models.DNN import DNN
from models.GSDNN_new import GSDNN_new
from models.MSDNN import MSDNN
from models.resnet import ResNet101
from models.GSDNN import GSDNN
from models.GSDNN2 import GSDNN2

from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.nn.functional as F
import scipy
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score
device = "cuda" if torch.cuda.is_available() else "cpu"

# 自定义数据集

# --------------------------  1. Dataset  --------------------------
class ContrastiveDataset(Dataset):
    def __init__(self, data_array, transform, views=2):
        self.data = data_array
        self.transform = transform
        self.views = views

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = torch.tensor(np.expand_dims(self.data[idx], 0))
        return [self.transform(img) for _ in range(self.views)]

class MyDataset(Dataset):
    def __init__(self, data_array, data_label, transform):
        self.data  = data_array
        self.label = data_label
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = torch.tensor(np.expand_dims(self.data[idx], 0))
        img = self.transform(img)
        return torch.squeeze(img, 1), self.label[idx]  # 返回 (18,101) 去掉通道维


# 自定义变换函数，对每个通道的时间序列进行反转
def reverse_time_series(data):
    # 翻转最后一个维度（时间点维度）
    return  -data

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
        #shuffle_channel,  # 通道打乱
        reverse_transform      #将每个信号沿x轴翻转（加负号）垂直翻转
    ], p=0)  # p=1.0 表示总是随机应用其中一种方法
])
data_transforms_train = transforms.Compose([])

# 使用ResNet作为编码器
class SimCLRModel(nn.Module):
    def __init__(self, base_model, out_dim=132, proj_out_dim=128):
        super(SimCLRModel, self).__init__()
        self.encoder = base_model
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])  # 移除最后的分类层
        self.dropout = nn.Dropout(p=0.5)
        # 添加投影头 v1
        # self.projector = nn.Sequential(
        #    nn.Linear(132, proj_out_dim),
        #    nn.ReLU(inplace=True),
        #    nn.Linear(proj_out_dim, 132)  # 投影回原始特征维度
        # )#176  132
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
        #h = self.projector(h)

        return h


class ClassficationModel(nn.Module):
    def __init__(self, encoder):
        super(ClassficationModel, self).__init__()
        self.encoder = encoder
        self.linear = nn.Linear( 132,27)  # 移除最后的分类层
#MSDNN176,GSDNN132
    def forward(self, x):
        h = self.encoder(x)
        h = h.squeeze(-1)
        return self.linear(F.relu(h))






def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

def init_weights_random(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        if m.bias is not None:
            m.bias.data.fill_(0.01)



encoder_t = SimCLRModel(GSDNN_new()).to(device)
encoder_t.load_state_dict(torch.load("./save_model/final_model_after_teacher_T.pth", weights_only=True),strict=False)
teacher = ClassficationModel(encoder_t.encoder).to(device)
teacher.eval()

gait_unlabel = sio.loadmat('data4/sub_all_data.mat')['all_data']
soft_ds = MyDataset(gait_unlabel, np.zeros(len(gait_unlabel)), data_transforms_train)
soft_ld = DataLoader(soft_ds, 64, shuffle=False, drop_last=False)

soft_labels = []
with torch.no_grad():
    for x, _ in soft_ld:
        x = x.to(device, dtype=torch.float32)
        x = torch.squeeze(x, dim=1)
        logits = teacher(x)
        soft_labels.append(logits.cpu())
soft_labels = torch.cat(soft_labels, 0)          # [N,27]
torch.save(soft_labels, './save_model/soft_labels.pt')
print("========== 3. 自蒸馏训练 ==========")
# 有标签数据
idx = np.arange(len(sio.loadmat('data4/sub_test_data.mat')['sub_data']))
np.random.shuffle(idx)
gait_x   = sio.loadmat('data4/sub_test_data.mat')['sub_data'][idx]
gait_y   = sio.loadmat('data4/sub_test_label.mat')['sub_label'][0][idx] - 1
train_len = int(0.8 * len(gait_x))
train_ds = MyDataset(gait_x[:train_len], gait_y[:train_len], data_transforms_train)
test_ds  = MyDataset(gait_x[train_len:], gait_y[train_len:], data_transforms_train)
train_ld = DataLoader(train_ds, 64, shuffle=True)
test_ld  = DataLoader(test_ds, 64, shuffle=False)

encoder_s = SimCLRModel(GSDNN_new()).to(device)
student = ClassficationModel(encoder_s.encoder).to(device)
opt_stu = torch.optim.Adam(student.parameters(), 3e-4)

# 软标签对齐索引
soft_idx = torch.from_numpy(idx)           # 与 gait_x 的 shuffle 保持一致
soft_labels = torch.load('./save_model/soft_labels.pt')
soft_labels = soft_labels.to(device)

# ----------- 超参数 -----------
T       = 4.0          # 蒸馏温度
alpha   = 0.7          # 交叉熵权重
epochs  = 200

def distill_loss(logits_s, y_true, logits_t, alpha, T):
    L_ce = F.cross_entropy(logits_s, y_true)
    L_kl = F.kl_div(
        F.log_softmax(logits_s/T, dim=1),
        F.softmax(logits_t/T, dim=1),
        reduction='batchmean'
    ) * (T * T)
    return alpha * L_ce + (1-alpha) * L_kl




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
    auc_sc = auc(y_labels, y_preds)
    return y_preds, y_labels, acc, precision, recall, f1, auc_sc


# ----------- 日志容器 -----------
losses      = []
train_accs, train_pres, train_recs, train_f1s, train_aucs = [], [], [], [], []
test_accs , test_pres , test_recs , test_f1s , test_aucs  = [], [], [], [], []


# ----------- 训练循环 -----------
for epoch in range(epochs):
    student.train()
    epoch_loss = 0.0
    # 与软标签索引对齐
    soft_idx = torch.arange(len(train_ds)).split(train_ld.batch_size)

    for (x, y), idx_b in zip(train_ld, soft_idx):
        x, y = x.to(device, dtype=torch.float32), y.to(device)
        soft_b = soft_labels[idx_b]

        opt_stu.zero_grad()
        x = torch.squeeze(x, dim=1)
        logits = student(x)

        # 蒸馏损失
        L_ce = F.cross_entropy(logits, y)
        L_kl = F.kl_div(
            F.log_softmax(logits / T, dim=1),
            F.softmax(soft_b / T, dim=1),
            reduction='batchmean'
        ) * (T * T)
        loss = alpha * L_ce + (1 - alpha) * L_kl

        loss.backward()
        opt_stu.step()
        epoch_loss += loss.item()

    losses.append(epoch_loss / len(train_ld))

    # ---------------- 评估 ----------------
    train_preds, train_labels, tr_acc, tr_pre, tr_rec, tr_f1, tr_auc = eval(student, train_ld)
    test_preds , test_labels , te_acc, te_pre, te_rec, te_f1, te_auc = eval(student, test_ld)

    # 记录
    train_accs.append(tr_acc); train_pres.append(tr_pre); train_recs.append(tr_rec)
    train_f1s.append(tr_f1);   train_aucs.append(tr_auc)
    test_accs .append(te_acc);  test_pres .append(te_pre);  test_recs .append(te_rec)
    test_f1s  .append(te_f1);   test_aucs .append(te_auc)

    print(f'Epoch {epoch+1:3d} | loss {losses[-1]:.4f} | '
          f'train_acc {tr_acc:.4f} | test_acc {te_acc:.4f}')

# ---------------- 画图 ----------------
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(losses, label='Training Loss')
plt.title('Training Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

plt.subplot(1,2,2)
plt.plot(train_accs, label='Train Acc'); plt.plot(test_accs, label='Test Acc')
plt.title('Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
plt.tight_layout(); plt.show()

# ---------------- 混淆矩阵 ----------------
conf_mat = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(8,6))
plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix'); plt.colorbar()
tick_marks = np.arange(27)
plt.xticks(tick_marks); plt.yticks(tick_marks)
plt.xlabel('Predicted'); plt.ylabel('True')
thresh = conf_mat.max() / 2.
for i, j in np.ndindex(conf_mat.shape):
    plt.text(j, i, format(conf_mat[i, j], 'd'),
             ha="center", va="center",
             color="white" if conf_mat[i, j] > thresh else "black")
plt.show()

# ---------------- 最后 10 个 epoch 平均 ----------------
last10 = lambda x: np.mean(x[-10:])
print(f'Last-10-epoch avg: '
      f'Acc={last10(test_accs):.4f}, '
      f'Precision={last10(test_pres):.4f}, '
      f'Recall={last10(test_recs):.4f}, '
      f'F1={last10(test_f1s):.4f}, '
      f'AUC={last10(test_aucs):.4f}')

# ---------------- 保存最终模型 ----------------
#torch.save(student.state_dict(), "./save_model/final_model_student.pth")