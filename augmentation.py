import random
import numpy as np
import scipy

# 随机丢掉部分频率成分//弱增强
class gaitrequencyDropOut(object):
    def __init__(self, rate=0.3, default_len=101):
        self.rate = rate
        self.default_len = default_len
        self.num_zeros = int(self.rate * self.default_len)

    def __call__(self, data):
        num_zeros = random.randint(0, self.num_zeros)
        zero_idxs = sorted(np.random.choice(np.arange(self.default_len), num_zeros, replace=False))
        data_dct = scipy.fft.dct(data.copy())
        data_dct[:, zero_idxs] = 0
        data_idct = scipy.fft.idct(data_dct)

        return data_idct

# 随机裁剪10%并调整大小//弱增强
class gaitCropResize(object):
    def __init__(self, n=2, default_len=101, fs=5):
        self.min_len = n * fs
        self.default_len = default_len

    def __call__(self, data):
        crop_len = random.randint(self.min_len, self.default_len)
        crop_start = random.randint(0, self.default_len - crop_len)
        data_crop = data[:, crop_start:crop_start + crop_len]
        data_resize = np.empty_like(data)
        x = np.linspace(0, crop_len-1, crop_len)
        xnew = np.linspace(0, crop_len-1, self.default_len)
        for i in range(data.shape[0]):
            f = scipy.interpolate.interp1d(x, data_crop[i], kind='cubic')
            data_resize[i] = f(xnew)

        return data_resize

# 随机拉伸信号幅值到1-2倍//弱增强
class gaitChannelStretch(object):
    def __init__(self, default_channels=18):
        self.default_channels = default_channels

    def __call__(self, data):

        if data.shape[0] != self.default_channels:
            raise ValueError("数据的通道数与初始化时指定的通道数不匹配。")

        # 生成一个随机拉伸倍数，范围从1到3
        stretch_factor = random.uniform(1, 2)

        # 将拉伸倍数应用到所有通道
        stretched_data = data * stretch_factor

        return stretched_data

# 随机使一些通道变成0，置0通道从1到9//强增强
class gaitChannelMask(object):
    def __init__(self, masks=9, default_channels=18):
        self.masks = masks
        self.default_channels = default_channels

    def __call__(self, data):
        if not isinstance(data, torch.Tensor):
            raise TypeError("Input should be a torch.Tensor")

        # 确保数据是二维的，并且通道数符合预期
        if data.dim() != 2 or data.shape[0] != self.default_channels:
            raise ValueError("数据的通道数与初始化时指定的通道数不匹配。")

        # 随机选择要置零的通道数
        masks = random.randint(1, self.masks)
        channels_mask = np.random.choice(np.arange(self.default_channels), masks, replace=False)

        # 创建数据的副本以避免修改原始数据
        data_ = data.clone()

        # 将选定的通道置零
        data_[np.arange(self.default_channels)[channels_mask], :] = 0

        return data_

# 打乱通道//强增强
class gaitChannelShuffle(object):
    def __init__(self, default_channels=18):

        self.default_channels = default_channels

    def __call__(self, data):
        if data.shape[0] != self.default_channels:
            raise ValueError("数据的通道数与初始化时指定的通道数不匹配。")

        # 创建行索引的列表
        rows = list(range(data.shape[0]))

        # 随机打乱行索引
        random.shuffle(rows)

        # 根据打乱后的索引重新排列行
        shuffled_data = np.empty_like(data)
        for i, row in enumerate(rows):
            shuffled_data[i, :] = data[row, :]

        return shuffled_data

# 将每个通道的信号沿着x轴翻转//强增强
class gaitChannelXturn(object):
    def __init__(self, default_channels=18):

        self.default_channels = default_channels

    def __call__(self, data):

        if data.shape[0] != self.default_channels:
            raise ValueError("数据的通道数与初始化时指定的通道数不匹配。")

        # 沿x轴翻转所有通道的信号值
        flipped_data = -data

        return flipped_data