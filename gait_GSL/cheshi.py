import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx

# 随机生成一个 18x18 的有向加权图
def generate_random_graph(n=18):
    adj_matrix = np.random.rand(n, n)
    adj_matrix = np.triu(adj_matrix, k=1)  # 生成上三角矩阵
    adj_matrix += adj_matrix.T  # 转换为对称矩阵
    adj_matrix = np.round(adj_matrix, 2)  # 保留两位小数
    return adj_matrix

# 节点失活（随机移除部分节点的边，比例为50%）
def node_dropping(adj_matrix, drop_rate=0.5):
    num_nodes = adj_matrix.shape[0]
    drop_mask = np.random.choice([0, 1], size=num_nodes, p=[drop_rate, 1 - drop_rate])
    drop_mask = np.array(drop_mask, dtype=bool)
    adj_aug = adj_matrix.copy()
    adj_aug[drop_mask, :] = 0  # 移除节点的出边
    adj_aug[:, drop_mask] = 0  # 移除节点的入边
    return adj_aug

# 边扰动（随机移除部分边，删除概率为50%）
def edge_perturbation(adj_matrix, drop_rate=0.5):
    adj_aug = adj_matrix.copy()
    edges = np.argwhere(adj_aug > 0)
    num_edges = edges.shape[0]
    drop_indices = np.random.choice(num_edges, size=int(num_edges * drop_rate), replace=False)
    adj_aug[edges[drop_indices, 0], edges[drop_indices, 1]] = 0
    return adj_aug



# 生成随机图
adj_matrix = generate_random_graph()

# 应用数据增强
adj_aug1 = node_dropping(adj_matrix)
adj_aug2 = edge_perturbation(adj_matrix)


# 绘制原始图和增强后的图
def plot_graph(adj_matrix, title):
    G = nx.from_numpy_array(adj_matrix)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=100, node_color='skyblue', font_size=8)
    plt.title(title)
    plt.show()


print(adj_matrix)
plot_graph(adj_matrix, "原始图")

print(adj_aug1)
plot_graph(adj_aug1, "节点失活后的图")


print(adj_aug2)
plot_graph(adj_aug2, "边扰动后的图")


