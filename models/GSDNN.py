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
            nn.Sigmoid())

    def forward(self, x):
        alpha = torch.squeeze(self.globalavgpool(x))
        alpha = self.se_block(alpha)
        alpha = torch.unsqueeze(alpha, -1)
        out = torch.mul(x, alpha)
        return out



class BranchConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BranchConv1D, self).__init__()
        OC = out_channels//3
        OD = in_channels//3
        self.b1 = nn.Conv1d(OD, OC, k1, stride, p1, bias=False)
        self.b2 = nn.Conv1d(OD, OC, k2, stride, p2, bias=False)
        self.b3 = nn.Conv1d(OD, OC, k3, stride, p3, bias=False)

    def forward(self, x):
        i = x.size(1)
        high_freq_indices = list(range(0, i, 3))
        mid_freq_indices = list(range(1, i, 3))
        low_freq_indices = list(range(2, i, 3))
        x1 = x[:, high_freq_indices, :]#high
        x2 = x[:, mid_freq_indices, :]#mid
        x3 = x[:, low_freq_indices, :]#low
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
                SELayer1D(out_channels))

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

class GSDNN(nn.Module):

    def __init__(self, num_classes=1, init_channels=18, growth_rate=12, base_channels=48,
                 stride=2, drop_out_rate=0.2):
        super(GSDNN, self).__init__()
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

    def forward(self, x):
        out = self.blocks(x)
        out = torch.squeeze(out)
        out = self.fc(out)
        return out

if __name__ == "__main__":

    gaits = torch.randn([64, 18, 101])
    net = GSDNN()
    y = net(gaits)
    print(y.shape)
    paras = sum([p.data.nelement() for p in net.parameters()])
    print('Number of params: {:.2f} M.\n'.format(paras / (1024 ** 2)))
