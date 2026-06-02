import torch
from thop import profile  # 安装：pip install thop


def FLOPs_calculat(model, device, data_shape):
    print("=" * 70)
    print("模型参数量与FLOPS")
    print("=" * 70)
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"模型参数量：总参数量={total_params / 1e6:.2f}M，可训练参数量={trainable_params / 1e6:.2f}M"
    )
    # 计算FLOPs（需传入与实际输入一致的张量）
    dummy_input = torch.randn(*data_shape).to(device)  # 适配步态数据的输入尺寸
    flops, _ = profile(model, inputs=(dummy_input,))
    print(f"模型FLOPs：{flops / 1e9:.5f}G")
    print("=" * 70)
