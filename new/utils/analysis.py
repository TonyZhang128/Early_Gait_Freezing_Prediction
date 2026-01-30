import os
import random
import numpy as np

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

def set_seed(seed):
    """设置随机种子，确保实验可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
def save_checkpoint(model, save_dir, filename, epoch, loss, is_best=False):
    """保存模型检查点

    Args:
        model: 模型
        save_dir: 保存目录
        filename: 文件名
        epoch: 当前epoch
        loss: 当前损失
        is_best: 是否为最佳模型
    """
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
    }

    # 保存检查点
    save_path = os.path.join(save_dir, filename)
    torch.save(checkpoint, save_path)

    # 如果是最佳模型，额外保存
    if is_best:
        best_path = os.path.join(save_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f'Epoch {epoch + 1}, New minimum loss: {loss:.6f}, Best model saved.')


def plot_loss_curve(loss_values, save_dir):
    """绘制损失曲线

    Args:
        loss_values: 损失值列表
        save_dir: 保存目录
    """
    plt.figure(figsize=(10, 5))
    plt.plot(loss_values, label='Training Loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 保存图像
    save_path = os.path.join(save_dir, 'loss_curve.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Loss curve saved to {save_path}')
    plt.close()