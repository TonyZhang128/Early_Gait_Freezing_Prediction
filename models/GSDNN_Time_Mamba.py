import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import sqrt

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
        if len(alpha.shape) == 1:  # handle batch size 1
            alpha = alpha.unsqueeze(0)
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
        att = torch.cat([max_out, avg_out], dim=1)
        att = self.conv1(att)
        att = self.sigmoid(att)
        return x * att


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


class LightweightTimesBlock(nn.Module):
    """
    轻量级TimesNet模块，专为步态GRF设计
    核心思想：将1D时序reshape为2D（周期内×周期间），用深度可分离卷积建模
    """

    def __init__(self, in_channels, out_channels, seq_len=101, top_k=1, d_model=32):
        super(LightweightTimesBlock, self).__init__()
        self.seq_len = seq_len
        self.top_k = top_k
        self.d_model = d_model

        # 投影到d_model维度进行周期分析（降维以减少计算量）
        self.projection = nn.Conv1d(in_channels, d_model, 1)

        # 频率分析层
        self.frequency_layer = nn.Linear(d_model, top_k)

        # 2D深度可分离卷积（轻量化设计）
        # 使用3x3 kernel捕捉周期内和周期间关系
        self.conv2d = nn.Sequential(
            # Depthwise
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
            # Pointwise
            nn.Conv2d(d_model, d_model, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True)
        )

        # 投影回out_channels
        self.out_projection = nn.Conv1d(d_model, out_channels, 1)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        B, C, L = x.shape

        # 降维到d_model
        x_proj = self.projection(x)  # [B, d_model, L]

        # FFT分析主导频率
        xf = torch.fft.rfft(x_proj, dim=-1)  # [B, d_model, L//2+1]
        frequency_list = abs(xf).mean(dim=-1)  # [B, d_model]

        # 找top_k主导频率
        frequency_list = frequency_list.mean(dim=0)  # [d_model]
        _, index = torch.topk(frequency_list, self.top_k, dim=-1, largest=True, sorted=True)
        period = (L / (index + 1)).int()  # 计算周期长度

        # 多周期融合
        res = []
        for i in range(self.top_k):
            period_len = period[i].item()
            # 如果周期太长或太短，跳过（适配步态数据特性）
            if period_len > L // 2 or period_len < 2:
                continue

            # 1D -> 2D reshape: [B, d_model, L] -> [B, d_model, period_len, num_periods]
            num_periods = L // period_len
            if num_periods == 0:
                continue

            x_2d = x_proj[:, :, :num_periods * period_len]
            x_2d = x_2d.reshape(B, self.d_model, num_periods, period_len)
            x_2d = x_2d.permute(0, 1, 3, 2)  # [B, d_model, period_len, num_periods]

            # 2D卷积捕捉周期内(intra-period)和周期间(inter-period)模式
            x_2d = self.conv2d(x_2d)

            # 2D -> 1D reshape back
            x_1d = x_2d.permute(0, 1, 3, 2)  # [B, d_model, num_periods, period_len]
            x_1d = x_1d.reshape(B, self.d_model, num_periods * period_len)

            # 补齐长度（如有截断）
            if x_1d.shape[-1] < L:
                x_1d = F.pad(x_1d, (0, L - x_1d.shape[-1]))

            res.append(x_1d)

        # 如果没有有效周期（fallback）
        if len(res) == 0:
            res = x_proj
        else:
            # 平均融合多个周期表示
            res = torch.stack(res, dim=-1).mean(dim=-1)

        # 投影回原维度
        out = self.out_projection(res)
        out = self.bn(out)
        return out


class _MambaCore(nn.Module):
    """
    纯PyTorch实现的Mamba核心模块（简化版）。
    输入输出格式与官方mamba_ssm.Mamba一致: [B, L, D]
    适用于短序列(L~100)，在Windows等无法安装mamba-ssm的环境自动fallback。
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2,
                 dt_scale=1.0, dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        super(_MambaCore, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(d_model * expand)
        self.d_conv = d_conv

        # 官方Mamba中，dt_rank = ceil(d_model / 16)，减少delta投影参数量
        dt_rank = math.ceil(d_model / 16)
        self.dt_rank = dt_rank

        # 输入投影：一半给SSM主干，一半给门控
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 因果卷积（分组卷积，逐通道）
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=0,
            groups=self.d_inner,
            bias=True,
        )

        # x_proj: 输出 (delta_raw, B, C)，其中delta_raw维度为dt_rank
        self.x_proj = nn.Linear(self.d_inner, dt_rank + d_state * 2, bias=False)
        # dt_proj: dt_rank -> d_inner
        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)

        # dt_proj 初始化：使softplus(dt_bias)落在[dt_min, dt_max]
        dt_init_std = dt_rank ** -0.5 * dt_scale
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # A: [d_inner, d_state]，使用log参数化确保离散化稳定
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))

        # D: skip connection参数
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """
        Args:
            x: [B, L, d_model]
        Returns:
            out: [B, L, d_model]
        """
        B, L, _ = x.shape

        # 输入投影并分割
        x_and_gate = self.in_proj(x)                          # [B, L, 2*d_inner]
        x_inner, gate = x_and_gate.chunk(2, dim=-1)           # [B, L, d_inner]

        # 因果卷积: 左侧pad d_conv-1个，保证因果性
        x_conv = F.pad(x_inner.permute(0, 2, 1), (self.d_conv - 1, 0))
        x_conv = self.conv1d(x_conv)                          # [B, d_inner, L]
        x_conv = x_conv.permute(0, 2, 1).contiguous()         # [B, L, d_inner]
        x_conv = F.silu(x_conv)

        # SSM参数投影: delta_raw, B, C
        x_proj_out = self.x_proj(x_conv)                      # [B, L, dt_rank + 2*d_state]
        delta_raw, B_param, C_param = x_proj_out.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta_raw))           # [B, L, d_inner]

        # 离散化A
        A = -torch.exp(self.A_log)                            # [d_inner, d_state]

        # 递归扫描（序列短，循环完全可行）
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            # 离散化系数 dA = exp(delta_t * A), dB = delta_t * B_t
            dA = torch.exp(delta[:, t, :].unsqueeze(-1) * A.unsqueeze(0))  # [B, d_inner, d_state]
            dB = delta[:, t, :].unsqueeze(-1) * B_param[:, t, :].unsqueeze(1)  # [B, d_inner, d_state]

            # 状态更新: h_t = dA * h_{t-1} + dB * x_t
            h = dA * h + dB * x_conv[:, t, :].unsqueeze(-1)   # [B, d_inner, d_state]

            # 输出: y_t = sum(C_t * h_t, dim=-1)
            y_t = (h * C_param[:, t, :].unsqueeze(1)).sum(dim=-1)  # [B, d_inner]
            ys.append(y_t)

        y = torch.stack(ys, dim=1)                            # [B, L, d_inner]

        # 跳跃连接 + 门控
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_conv     # [B, L, d_inner]
        y = y * F.silu(gate)                                  # [B, L, d_inner]

        # 输出投影
        y = self.out_proj(y)                                  # [B, L, d_model]
        return y


class MambaBlock(nn.Module):
    """
    Mamba (Selective State Space Model) Block for 1D time-series.
    自动检测mamba_ssm是否安装，若无则回退到纯PyTorch实现。
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super(MambaBlock, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        try:
            from mamba_ssm import Mamba
            self.core = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self._backend = "official"
        except ImportError:
            self.core = _MambaCore(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self._backend = "pytorch"
            print(f"[MambaBlock] mamba_ssm未安装，使用纯PyTorch回退实现 (d_model={d_model})")

    def forward(self, x):
        """
        Args:
            x: [B, C, L]  (channels = d_model)
        Returns:
            out: [B, C, L]
        """
        # Mamba expects [B, L, D]
        x = x.permute(0, 2, 1).contiguous()  # [B, L, C]
        x = self.norm(x)
        x = self.core(x)                     # [B, L, C]
        x = x.permute(0, 2, 1).contiguous()  # [B, C, L]
        return x


class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, drop_out_rate, stride, seq_len=101,
                 use_timesblock=False, use_mamba=False, mamba_d_state=16):
        super(BasicBlock1D, self).__init__()
        self.use_timesblock = use_timesblock
        self.use_mamba = use_mamba

        # 原有的多尺度分支
        self.branch_conv = BranchConv1D(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(drop_out_rate)

        # 第二段卷积
        self.branch_conv2 = BranchConv1D(out_channels, out_channels, 1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # 注意力模块
        self.se = SELayer1D(out_channels)
        self.spatial_att = SpatialAttention1D()

        # 可选：轻量级TimesBlock
        if use_timesblock:
            self.times_block = LightweightTimesBlock(
                out_channels, out_channels, seq_len=seq_len
            )

        # 可选：MambaBlock（SSM全局时序建模）
        if use_mamba:
            self.mamba_block = MambaBlock(
                d_model=out_channels, d_state=mamba_d_state
            )

        # 可学习融合权重（CNN主干 + Times + Mamba）
        num_branches = 1 + int(use_timesblock) + int(use_mamba)
        if num_branches > 1:
            self.fusion_weight = nn.Parameter(torch.ones(num_branches))

        # 残差连接
        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut.add_module('MaxPool', nn.MaxPool1d(stride, ceil_mode=True))
        if in_channels != out_channels:
            self.shortcut.add_module('ShutConv', nn.Conv1d(in_channels, out_channels, 1))
            self.shortcut.add_module('ShutBN', nn.BatchNorm1d(out_channels))

    def forward(self, x):
        # 主干路径（局部CNN特征）
        out = self.branch_conv(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.branch_conv2(out)
        out = self.bn2(out)

        # 收集所有分支特征
        branches = [out]

        if self.use_timesblock:
            times_feat = self.times_block(out)
            branches.append(times_feat)

        if self.use_mamba:
            mamba_feat = self.mamba_block(out)
            branches.append(mamba_feat)

        # 可学习的软融合
        if len(branches) > 1:
            weight = F.softmax(self.fusion_weight, dim=0)
            out = sum(w * b for w, b in zip(weight, branches))

        out = self.se(out)
        out = self.spatial_att(out)

        # 残差连接
        shortcut = self.shortcut(x)
        out = torch.relu(out + shortcut)
        return out


class GSDNN_Time_Mamba(nn.Module):
    def __init__(self, num_classes=27, init_channels=18, growth_rate=12, base_channels=48,
                 stride=2, drop_out_rate=0.2, seq_len=101,
                 use_times_blocks=[False, False, False, True, True, True, True, True],
                 use_mamba_blocks=[True, True, True, True, False, False, False, False],
                 mamba_d_state=16):
        """
        GSDNN + TimesBlock + Mamba 融合模型（性能优化版）

        Args:
            use_times_blocks: 指定哪些block启用TimesBlock（推荐后几层，捕捉高级周期特征）
            use_mamba_blocks: 指定哪些block启用MambaBlock（推荐前几层，序列长度足够时建模全局依赖）
            mamba_d_state: Mamba的SSM状态维度，默认16
        """
        super(GSDNN_Time_Mamba, self).__init__()
        self.num_channels = init_channels
        block_n = 8
        block_c = [base_channels + i * growth_rate for i in range(block_n)]

        # 动态计算每层的序列长度（考虑stride）
        current_len = seq_len
        self.blocks = nn.Sequential()

        for i, C in enumerate(block_c):
            # 更新序列长度（用于TimesBlock的周期分析）
            if i > 0:
                current_len = (current_len + stride - 1) // stride  # 考虑MaxPool的ceil_mode

            use_times = use_times_blocks[i] if i < len(use_times_blocks) else False
            use_mamba = use_mamba_blocks[i] if i < len(use_mamba_blocks) else False

            module = BasicBlock1D(
                self.num_channels, C, drop_out_rate, stride,
                seq_len=current_len,
                use_timesblock=use_times,
                use_mamba=use_mamba,
                mamba_d_state=mamba_d_state
            )
            self.blocks.add_module("block{}".format(i), module)
            self.num_channels = C

        module = nn.AdaptiveAvgPool1d(1)
        self.blocks.add_module("GlobalAvgPool", module)

        self.fc = nn.Linear(self.num_channels, num_classes)

        # 初始化
        self._initialize_weights()

    def forward(self, x):
        out = self.blocks(x)
        out = torch.squeeze(out, dim=-1)  # 明确指定dim避免batch=1时问题
        out = self.fc(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    # 测试代码
    batch_size = 64
    gaits = torch.randn([batch_size, 18, 101])  # 64个样本，18个通道，101个时间点

    # 配置1：推荐配置（前4层Mamba，后4层TimesBlock）。第四层都是ture做一下敏感性分析发现没有显著差异，这个第四层都打开ture没有意义
    net_recommended = GSDNN_Time_Mamba(
        num_classes=27,
        use_times_blocks=[False, False, False, False, True, True, True, True],
        use_mamba_blocks=[True, True, True, True, False, False, False, False]
    )

    # 配置2：全部开启（对比实验用）
    net_all = GSDNN_Time_Mamba(
        num_classes=27,
        use_times_blocks=[True] * 8,
        use_mamba_blocks=[True] * 8
    )

    # 配置3：纯Mamba（前6层）
    net_mamba_only = GSDNN_Time_Mamba(
        num_classes=27,
        use_times_blocks=[False] * 8,
        use_mamba_blocks=[True, True, True, True, True, True, False, False]
    )

    y1 = net_recommended(gaits)
    print(f"推荐配置输出形状: {y1.shape}")

    paras_recommended = sum([p.data.nelement() for p in net_recommended.parameters()])
    paras_all = sum([p.data.nelement() for p in net_all.parameters()])
    paras_mamba_only = sum([p.data.nelement() for p in net_mamba_only.parameters()])

    print(f'推荐配置参数量: {paras_recommended / (1024 ** 2):.3f} M')
    print(f'全开启参数量: {paras_all / (1024 ** 2):.3f} M')
    print(f'纯Mamba参数量: {paras_mamba_only / (1024 ** 2):.3f} M')

    # 测试梯度流动
    loss = y1.mean()
    loss.backward()
    print("梯度回传测试通过！")

    # 打印每层配置
    print("\n推荐配置各层模块启用情况：")
    for name, module in net_recommended.named_modules():
        if isinstance(module, BasicBlock1D):
            print(f"  {name}: TimesBlock={module.use_timesblock}, Mamba={module.use_mamba}")
