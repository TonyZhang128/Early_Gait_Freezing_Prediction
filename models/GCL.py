import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.nn as nn

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation=F.relu, num_layers=2, dropout=0.5):
        super().__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList()
        # 输入层
        self.layers.append(GCNConv(input_dim, hidden_dim))
        # 隐藏层
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, edge_weight=None):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_weight)
            if i < len(self.layers) - 1:
                x = self.activation(x)
                x = self.dropout(x)
        return x



class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, proj_dim):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

        self.fc1 = torch.nn.Linear(hidden_dim, proj_dim)
        self.fc2 = torch.nn.Linear(proj_dim, hidden_dim)

    def forward(self, x, edge_index,edge_weight):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index,edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index,edge_weight)
        z = self.encoder(x, edge_index, edge_weight)
        z1 = self.encoder(x1, edge_index1, edge_weight1)
        z2 = self.encoder(x2, edge_index2, edge_weight2)
        return z, z1, z2

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)



# 生成随机的 64x18x18 矩阵
batch_size = 64
num_nodes = 18
features_batch = torch.rand(batch_size, num_nodes, num_nodes)

# 构建边索引（假设每个图的边索引相同）
edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))

# 初始化模型
input_dim = 18  # 每个节点的特征维度为18
hidden_dim = 16
output_dim = 8
model = GCN(input_dim, hidden_dim, output_dim)

# 将特征矩阵和边索引输入到模型中
# 由于 PyTorch Geometric 的 GCNConv 需要每个图单独处理，这里我们只处理第一个图作为示例
x = features_batch[0]  # 取第一个图的特征矩阵
output = model(x, edge_index)

print("GCN 模型的输出：")
print(output)