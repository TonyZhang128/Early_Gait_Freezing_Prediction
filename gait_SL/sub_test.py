import torch
from torch.utils.data import DataLoader
from models.GSDNN import GSDNN
from models.resnet import ResNet101
import numpy as np
import torch.optim as optim
import scipy.io as sio
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score

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

# 加载保存的模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ResNetModel(GSDNN(), num_classes=5).to(device)
model.load_state_dict(torch.load('C:/Users/shuli/Desktop/gait_DL4/gait_DL4/save_model/.pth'),strict=False)
model.eval()

# 加载新数据集
new_gait_data = sio.loadmat('C:/Users/shuli/Desktop/gait_DL4/gait_DL4/data3/sub_test_data.mat')['sub_data']
new_gait_labels = sio.loadmat('C:/Users/shuli/Desktop/gait_DL4/gait_DL4/data3/sub_test_label.mat')['sub_label'].flatten()

# 创建新数据集的数据加载器
new_dataset = GaitDataset(new_gait_data, new_gait_labels)
new_loader = DataLoader(new_dataset, batch_size=64, shuffle=False)

# 损失函数和优化器
#criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(model.parameters(), lr=3e-4)

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

test_accs = []
test_pres = []
test_recs = []
test_f1s = []
test_spes = []


for epoch in range(epoch):
    for (x, label) in new_loader:
        model.eval()

        # 调整img1和img2的形状
        x = torch.squeeze(x, dim=1)  # 将64*1*18*101变为64*18*101
        label = label.to(device=device)
        y_pred = model(x.to(device=device, dtype=torch.float32))

    test_preds, test_labels, test_acc ,test_pre,test_rec, test_f1, test_spe= eval(model, new_loader)
    test_accs.append(test_acc)
    test_pres.append(test_pre)
    test_recs.append(test_rec)
    test_f1s.append(test_f1)
    test_spes.append(test_spe)
    print(f'Epoch {epoch + 1}, test_acc={test_acc:.5f}')

# 保存模型
#if epoch + 1 == 200:  # 检查是否完成了200个epoch
#    model_path = "./save_model/final_modelGSDNNk357_pro.pth"  # 定义模型保存路径
#    torch.save(model.state_dict(), model_path)  # 保存模型状态字典#
#    print(f"Model saved to {model_path}")

# 绘制图表

plt.subplot(1, 2, 2)
plt.plot(test_accs, label='Testing Accuracy')
plt.title('Testing Accuracy')
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