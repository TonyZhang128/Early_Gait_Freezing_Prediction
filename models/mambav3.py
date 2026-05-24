"""Mamba-v3: Selective State Space Model for gait signal classification.

Pure PyTorch implementation of the Mamba selective SSM (S6) architecture,
adapted for 1D gait sensor signals [B, C, T].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def selective_scan(
    x: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
) -> torch.Tensor:
    """Selective scan (S6) -- the core recurrence of Mamba.

    Args:
        x:    (B, D_inner, L)  -- projected input
        delta: (B, D_inner, L) -- discretisation step (softplus'd before call)
        A:    (D_inner, N)     -- state transition (negative)
        B:    (B, N, L)        -- input-dependent input matrix
        C:    (B, N, L)        -- input-dependent output matrix
        D:    (D_inner,)       -- skip / feedthrough

    Returns:
        y: (B, D_inner, L)
    """
    batch, d_inner, seq_len = x.shape
    n = A.shape[1]

    # Discretise A, B:  A_bar = exp(delta * A),  B_bar = delta * B
    # A: (D, N) -> (1, D, 1, N) for broadcasting with delta: (B, D, L, 1)
    A_expanded = A.unsqueeze(0).unsqueeze(2)              # (1, D, 1, N)
    deltaA = torch.exp(delta.unsqueeze(-1) * A_expanded)  # (B, D, L, N)
    # B: (B, N, L) -> (B, 1, L, N) for broadcasting with delta and x
    B_expanded = B.unsqueeze(1).transpose(-2, -1)         # (B, 1, L, N)
    deltaB_x = delta.unsqueeze(-1) * B_expanded * x.unsqueeze(-1)  # (B, D, L, N)

    # Sequential scan
    h = torch.zeros(batch, d_inner, n, device=x.device, dtype=x.dtype)
    ys: list[torch.Tensor] = []
    for t in range(seq_len):
        h = deltaA[:, :, t, :] * h + deltaB_x[:, :, t, :]
        y_t = (h * C[:, :, t].unsqueeze(1)).sum(dim=-1)   # (B, D)
        ys.append(y_t)
    y = torch.stack(ys, dim=-1)                            # (B, D, L)

    y = y + x * D.unsqueeze(-1)
    return y


class MambaBlock(nn.Module):
    """Single Mamba block: SSM path + gated feedforward."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        d_inner = d_model * expand

        # Input-dependent parameter projections (the "selective" part)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=d_conv,
                                padding=d_conv - 1, groups=d_inner)
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1, bias=False)  # B, C, delta
        self.dt_proj = nn.Linear(1, d_inner, bias=True)

        # SSM parameters (learnable, shared across time)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) -> (B, L, D)"""
        residual = x
        x = self.norm(x)

        # Project and split into SSM path (x_ssm) and gate (z)
        xz = self.in_proj(x)                         # (B, L, 2*d_inner)
        x_ssm, z = xz.chunk(2, dim=-1)               # each (B, L, d_inner)

        # Depthwise conv + SiLU
        x_ssm = x_ssm.transpose(1, 2)                # (B, d_inner, L)
        x_ssm = self.conv1d(x_ssm)[:, :, :residual.size(1)]
        x_ssm = F.silu(x_ssm)

        # Compute input-dependent SSM parameters
        x_proj_out = self.x_proj(x_ssm.transpose(1, 2))  # (B, L, 2*N+1)
        B_param = x_proj_out[:, :, :self.A_log.shape[1]]  # (B, L, N)
        C_param = x_proj_out[:, :, self.A_log.shape[1]:2*self.A_log.shape[1]]
        delta = x_proj_out[:, :, -1:]                     # (B, L, 1)

        # Project delta and apply softplus
        delta = F.softplus(self.dt_proj(delta))            # (B, L, d_inner)
        delta = delta.transpose(1, 2)                      # (B, d_inner, L)

        A = -torch.exp(self.A_log)                         # (d_inner, N)
        B_param = B_param.transpose(1, 2)                  # (B, N, L)
        C_param = C_param.transpose(1, 2)                  # (B, N, L)

        # Selective scan
        y = selective_scan(x_ssm, delta, A, B_param, C_param, self.D)

        # Gate and project
        y = y.transpose(1, 2)                              # (B, L, d_inner)
        y = y * F.silu(z)
        y = self.out_proj(y)

        return y + residual


class MambaEncoder(nn.Module):
    """Feature extractor: input projection + Mamba layers + pooling.

    Output: (B, d_model) -- ready for a classifier head.
    """

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=1),
            nn.BatchNorm1d(d_model),
        )
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand) for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) -> (B, d_model)"""
        x = self.input_proj(x)           # (B, d_model, T)
        x = x.transpose(1, 2)           # (B, T, d_model)
        for layer in self.layers:
            x = layer(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)           # (B, d_model, T)
        return self.pool(x).squeeze(-1)  # (B, d_model)


class MambaV3(nn.Module):
    """Mamba-v3 model for 1D gait signal classification.

    Architecture:  MambaEncoder -> FC
    Input:  (B, C, T) where C=18 channels, T=101 time steps
    Output: (B, num_classes)

    children() = [MambaEncoder, Linear] -- removing the last child
    yields the encoder for SimCLR / fine-tuning pipelines.
    """

    def __init__(
        self,
        num_classes: int = 27,
        in_channels: int = 18,
        d_model: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = MambaEncoder(
            in_channels, d_model, d_state, d_conv, expand, n_layers, dropout
        )
        self.fc = nn.Linear(d_model, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) -> (B, num_classes)"""
        return self.fc(self.encoder(x))


if __name__ == '__main__':
    x = torch.randn(64, 18, 101)
    model = MambaV3(num_classes=27)
    y = model(x)
    print(f'Output: {y.shape}')
    params = sum(p.numel() for p in model.parameters())
    print(f'Params: {params / 1e6:.2f}M')

    # Verify encoder extraction (SimCLR pattern)
    encoder = nn.Sequential(*list(model.children())[:-1])
    h = encoder(x)
    h = torch.flatten(h, start_dim=1)
    print(f'Encoder output: {h.shape}')
