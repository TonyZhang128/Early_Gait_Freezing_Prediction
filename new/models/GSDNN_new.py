import torch
import torch.nn as nn
import torch.nn.functional as F

k1, p1 = 3, 1
k2, p2 = 5, 2
k3, p3 = 7, 3

class SELayer1D(nn.Module):
    def __init__(self, nChannels, reduction=16):
        super(SELayer1D, self).__init__()
        self.globalavgpool = nn.AdaptiveAvgPool1d(1)
        self.se_block = nn.Sequential(
            nn.Linear(nChannels, nChannels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(nChannels // reduction, nChannels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        alpha = torch.squeeze(self.globalavgpool(x))
        alpha = self.se_block(alpha)
        alpha = torch.unsqueeze(alpha, -1)
        out = torch.mul(x, alpha)
        return out

class SpatialAttention1D(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention1D, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd."
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        x = torch.cat([max_out, avg_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class BranchConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BranchConv1D, self).__init__()
        OC = out_channels // 3
        OD = in_channels // 3
        self.b1 = nn.Conv1d(OD, OC, k1, stride, p1, bias=False)
        self.b2 = nn.Conv1d(OD, OC, k2, stride, p2, bias=False)
        self.b3 = nn.Conv1d(OD, OC, k3, stride, p3, bias=False)

    def forward(self, x):
        i = x.size(1)
        high_freq_indices = list(range(0, i, 3))
        mid_freq_indices = list(range(1, i, 3))
        low_freq_indices = list(range(2, i, 3))
        x1 = x[:, high_freq_indices, :]  # high
        x2 = x[:, mid_freq_indices, :]  # mid
        x3 = x[:, low_freq_indices, :]  # low
        out = torch.cat([self.b1(x1), self.b2(x2), self.b3(x3)], dim=1)
        return out

class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, drop_out_rate, stride):
        super(BasicBlock1D, self).__init__()
        self.operation = nn.Sequential(
            BranchConv1D(in_channels, out_channels, stride),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_out_rate),
            BranchConv1D(out_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            SELayer1D(out_channels),
            SpatialAttention1D()  # 添加空间注意力模块
        )

        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut.add_module('MaxPool', nn.MaxPool1d(stride, ceil_mode=True))
        if in_channels != out_channels:
            self.shortcut.add_module('ShutConv', nn.Conv1d(in_channels, out_channels, 1))
            self.shortcut.add_module('ShutBN', nn.BatchNorm1d(out_channels))

    def forward(self, x):
        operation = self.operation(x)
        shortcut = self.shortcut(x)
        out = torch.relu(operation + shortcut)
        return out

class GSDNN_new(nn.Module):
    def __init__(self, num_classes=1, block_n=8, init_channels=18, growth_rate=12, base_channels=48,
                 stride=2, drop_out_rate=0.2):
        super(GSDNN_new, self).__init__()
        self.num_channels = init_channels
        block_n = 8
        block_c = [base_channels + i * growth_rate for i in range(block_n)]

        self.blocks = nn.Sequential()
        for i, C in enumerate(block_c):
            module = BasicBlock1D(self.num_channels, C, drop_out_rate, stride)
            self.blocks.add_module("block{}".format(i), module)
            self.num_channels = C

        module = nn.AdaptiveAvgPool1d(1)
        self.blocks.add_module("GlobalAvgPool", module)

        self.fc = nn.Linear(self.num_channels, num_classes)
        
        # 增加模型初始化
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.blocks(x)
        out = torch.squeeze(out)
        out = self.fc(out)
        return out

if __name__ == "__main__":
    gaits = torch.randn([64, 18, 101])  # 64个样本，18个通道，101个时间点
    net = GSDNN_new(block_n=8, init_channels=18, growth_rate=12, base_channels=48, stride=2, drop_out_rate=0.2)
    y = net(gaits)
    print(y.shape)  # 输出形状
    paras = sum([p.data.nelement() for p in net.parameters()])
    print('Number of params: {:.2f} M.\n'.format(paras / (1024 ** 2)))