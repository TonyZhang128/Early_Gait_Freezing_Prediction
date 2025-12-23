import torch
import torch.nn as nn
import torchvision.transforms as transforms
from models.resnet import ResNet101
from torch.utils.data import DataLoader, Dataset
import numpy as np
from models.GSDNN import GSDNN
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import torch.nn.functional as F
import scipy.io as sio
import random
from scipy.fft import dct, idct
# 自定义数据集

device = "cuda" if torch.cuda.is_available() else "cpu"


class CustomGraphDataset(Dataset):
    def __init__(self, mat_file_path, drop_rate=1):
        self.data = sio.loadmat(mat_file_path)['all_data']
        self.drop_rate = drop_rate

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        adj_matrix = self.data[idx]

        # 数据增强
        adj_aug1 = node_dropping(adj_matrix, drop_rate=self.drop_rate)
        adj_aug1 = edge_perturbation(adj_aug1, drop_rate=self.drop_rate)

        adj_aug2 = node_dropping(adj_matrix, drop_rate=self.drop_rate)
        adj_aug2 = edge_perturbation(adj_aug2, drop_rate=self.drop_rate)

        # 转换为 Data 对象
        data1 = self._convert_to_data(adj_aug1)
        data2 = self._convert_to_data(adj_aug2)

        return data1, data2

    def _convert_to_data(self, adj_matrix):
        adj_matrix = torch.tensor(adj_matrix, dtype=torch.float)
        edge_index = torch.nonzero(adj_matrix).t()
        edge_attr = adj_matrix[edge_index[0], edge_index[1]]
        num_nodes = adj_matrix.shape[0]
        x = torch.arange(num_nodes, dtype=torch.float).view(-1, 1)  # 每个节点的特征是一个标量
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)



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

class GraphSimCLRTransform:
    def __init__(self, drop_rate=1):
        self.drop_rate = drop_rate

    def __call__(self, data):
        # 第一个增强视图
        data1 = node_dropping(data.clone(), drop_rate=self.drop_rate)
        data1 = edge_perturbation(data1, drop_rate=self.drop_rate)

        # 第二个增强视图
        data2 = node_dropping(data.clone(), drop_rate=self.drop_rate)
        data2 = edge_perturbation(data2, drop_rate=self.drop_rate)

        return data1, data2


# 数据集路径
dataset = CustomGraphDataset('C:/Users/shuli/Desktop/gait_DL4/gait_DL4/gait_GSL/data_GCL2/gait_GCL_all_data.mat')

# 数据增强
transform = GraphSimCLRTransform()

# 数据加载器
from torch_geometric.loader import DataLoader

dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

class GCNEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=3, dropout=0.5):
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


class GraphSimCLR(torch.nn.Module):
    def __init__(self, encoder, proj_dim=256):
        super().__init__()
        self.encoder = encoder

        # 投影头
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(encoder.layers[-3].out_channels, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, proj_dim)
        )

    def forward(self, data1, data2):
        # 获取两种增强视图的特征
        z1 = self.encoder(data1.x, data1.edge_index)
        z2 = self.encoder(data2.x, data2.edge_index)

        # 投影到低维空间
        p1 = self.projector(z1)
        p2 = self.projector(z2)

        return z1, z2, p1, p2
# 对比损失函数
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
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

# 训练循环
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 初始化模型和损失函数
encoder = GCNEncoder(input_dim=1, hidden_dim=256, num_layers=3, dropout=0.5)
model = GraphSimCLR(encoder, proj_dim=256).to(device)
criterion = ContrastiveLoss(temperature=0.5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)
# 初始化损失值列表
loss_values = []
min_loss = float('inf')

for epoch in range(60):
    model.train()
    total_loss = 0
    for batch in dataloader:
        data1, data2 = batch
        data1 = data1.to(device)
        data2 = data2.to(device)

       # print(f"Data1: Nodes: {data1.num_nodes}, Edges: {data1.edge_index.size(1)}")
       # print(f"Data2: Nodes: {data2.num_nodes}, Edges: {data2.edge_index.size(1)}")

        optimizer.zero_grad()
        _, _, p1, p2 = model(data1, data2)
        loss = criterion(p1, p2)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    # 将平均损失添加到列表中
    avg_loss = total_loss / len(dataloader)
    loss_values.append(avg_loss)

    # 检查是否达到最小损失
    if avg_loss < min_loss:
        min_loss = avg_loss
        # 保存模型
        torch.save(model.state_dict(), "C:/Users/shuli/Desktop/gait_DL4/gait_DL4/gait_GSL/save_models/best_modelGCL.pth")  # 保存模型参数
        print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}, New minimum loss: {min_loss:.4f}, Model saved.')
    else:
        print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}')


# 绘制损失曲线
import matplotlib.pyplot as plt
# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(loss_values, label='Training Loss')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()