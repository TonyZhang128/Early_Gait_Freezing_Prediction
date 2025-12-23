import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from collections import Counter
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- 1. 加载带标签数据 ----------------
data = sio.loadmat('C:/Users/shuli/Desktop/gait_DL4/gait_DL4/gait_GSL/data_GCL2/gait_GCL_sub_data.mat')['sub_data']      # (887, 18, 18)
labels = sio.loadmat('C:/Users/shuli/Desktop/gait_DL4/gait_DL4/gait_GSL/data_GCL2/gait_GCL_sub_label.mat')['sub_label'][0]  # (887,)
# 调整标签范围为 0 到 26
labels = labels - 1

# ---------------- 2. 构建 Dataset / Dataloader ----------------
# 节点失活（随机移除部分节点的边，比例为50%）
def node_dropping(adj_matrix, drop_rate=1):
    num_nodes = adj_matrix.shape[0]
    drop_mask = np.random.choice([0, 1], size=num_nodes, p=[drop_rate, 1 - drop_rate])
    drop_mask = np.array(drop_mask, dtype=bool)
    adj_aug = adj_matrix.copy()
    adj_aug[drop_mask, :] = 0  # 移除节点的出边
    adj_aug[:, drop_mask] = 0  # 移除节点的入边
    return adj_aug

# 边扰动（随机移除部分边，删除概率为50%）
def edge_perturbation(adj_matrix, drop_rate=1):
    adj_aug = adj_matrix.copy()
    edges = np.argwhere(adj_aug > 0)
    num_edges = edges.shape[0]
    drop_indices = np.random.choice(num_edges, size=int(num_edges * drop_rate), replace=False)
    adj_aug[edges[drop_indices, 0], edges[drop_indices, 1]] = 0
    return adj_aug


class CustomGraphDataset(Dataset):
    def __init__(self, data_array, labels, drop_rate=0.5):
        self.data = data_array          # (N, 18, 18)
        self.labels = labels            # (N,)
        self.drop_rate = drop_rate

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        adj_matrix = self.data[idx]
        label = self.labels[idx]

        # 数据增强：两个视图
        adj_aug1 = node_dropping(adj_matrix, drop_rate=self.drop_rate)
        adj_aug1 = edge_perturbation(adj_aug1, drop_rate=self.drop_rate)

        adj_aug2 = node_dropping(adj_matrix, drop_rate=self.drop_rate)
        adj_aug2 = edge_perturbation(adj_aug2, drop_rate=self.drop_rate)

        data1 = self._convert_to_data(adj_aug1)
        data2 = self._convert_to_data(adj_aug2)
        #data2 = self._convert_to_data(adj_matrix)
        return data2, int(label)

    def _convert_to_data(self, adj_matrix):
        adj_tensor = torch.tensor(adj_matrix, dtype=torch.float)
        edge_index = torch.nonzero(adj_tensor).t()
        edge_attr = adj_tensor[edge_index[0], edge_index[1]]
        num_nodes = adj_tensor.shape[0]
        x = torch.arange(num_nodes, dtype=torch.float).view(-1, 1)  # 每个节点的特征是一个标量
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)



# 打乱并划分训练/测试
indices = np.arange(len(labels))
np.random.shuffle(indices)
split = int(0.8 * len(indices))
train_ds = CustomGraphDataset(data[indices[:split]], labels[indices[:split]])
test_ds = CustomGraphDataset(data[indices[split:]], labels[indices[split:]])

from torch_geometric.loader import DataLoader
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

# ---------------- 3. 加载预训练 Encoder ----------------

from torch_geometric.nn import GCNConv
class GCNEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=4, dropout=0.5):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        # 输入层
        self.layers.append(GCNConv(input_dim, hidden_dim))
        self.layers.append(nn.BatchNorm1d(hidden_dim))
        self.layers.append(nn.Dropout(dropout))

        # 隐藏层
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
            self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.Dropout(dropout))

    def forward(self, x, edge_index):
        z = x
        for layer in self.layers:
            if isinstance(layer, GCNConv):
                z = layer(z, edge_index)
                z = F.relu(z)
            else:
                z = layer(z)
        return z


encoder = GCNEncoder(input_dim=1, hidden_dim=256)
encoder.load_state_dict(torch.load('C:/Users/shuli/Desktop/gait_DL4/gait_DL4/gait_GSL/save_models/best_modelGCL.pth',weights_only=True, map_location=device),strict=False)

# 冻结 encoder
for param in encoder.parameters():
    param.requires_grad = False

# ---------------- 4. 分类头 ----------------
class Classifier(nn.Module):
    def __init__(self, encoder, hidden_dim=256, num_classes=27):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, data):
        x = self.encoder(data.x, data.edge_index)
        x = global_mean_pool(x, data.batch)
        return self.fc(x)

num_classes = len(np.unique(labels))
model = Classifier(encoder, hidden_dim=256, num_classes=num_classes).to(device)

# ---------------- 5. 训练配置 ----------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=3e-5,weight_decay=1e-5)

def evaluate(loader):
    model.eval()
    total_loss = 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for data1, label in loader:
            data1, label = data1.to(device), label.to(device)
            out = model(data1)
            loss = criterion(out, label)
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            y_true.extend(label.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
    avg_test_loss = total_loss / len(loader)
    test_losses.append(avg_test_loss)
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    return y_true, y_pred, acc

# ---------------- 6. 训练循环 ----------------
epochs = 100
train_accs, test_accs = [], []

# 初始化损失值列表
train_losses, test_losses = [], []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    y_true, y_pred = [], []
    for data1, label in train_loader:
        data1, label = data1.to(device), label.to(device)
        optimizer.zero_grad()
        out1 = model(data1)
        loss = criterion(out1, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = out1.argmax(dim=1)
        y_true.extend(label.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    train_acc = np.mean(np.array(y_true) == np.array(y_pred))
    train_accs.append(train_acc)

    _, _, test_acc = evaluate(test_loader)
    test_accs.append(test_acc)

    print(
        f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

# ---------------- 7. 可视化 ----------------
plt.figure(figsize=(15,4))
plt.subplot(1,3,1)
plt.plot(train_accs, label='Train Acc')
plt.plot(test_accs, label='Test Acc')
plt.legend(); plt.title("Accuracy")

plt.subplot(1, 3, 2)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.legend()
plt.title("Loss Over Epochs")

y_true, y_pred, _ = evaluate(test_loader)
cm = confusion_matrix(y_true, y_pred)
plt.subplot(1,3,3)
plt.imshow(cm, cmap='Blues'); plt.title("Confusion Matrix")
plt.colorbar()
plt.show()

# ---------------- 8. 最终指标 ----------------
print("Final Test Acc:", test_accs[-1])
print("Precision:", precision_score(y_true, y_pred, average='macro'))
print("Recall:", recall_score(y_true, y_pred, average='macro'))
print("F1:", f1_score(y_true, y_pred, average='macro'))