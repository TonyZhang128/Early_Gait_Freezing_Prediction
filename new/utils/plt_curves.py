import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# ================================ 可视化模块 ================================

def plot_training_curves(losses, train_accs, test_accs, save_path=None):
    """
    绘制训练曲线

    Args:
        losses: 损失列表
        train_accs: 训练准确率列表
        test_accs: 测试准确率列表
        save_path: 保存路径（可选）
    """
    plt.figure(figsize=(15, 5))

    # 绘制损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(losses, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 绘制准确率曲线
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(test_accs, label='Testing Accuracy')
    plt.title('Training and Testing Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Training curves saved to {save_path}")
    else:
        plt.show()


def plot_confusion_matrix(y_labels, y_preds, save_path=None):
    """
    绘制混淆矩阵

    Args:
        y_labels: 真实标签
        y_preds: 预测标签
        save_path: 保存路径（可选）
    """
    conf_matrix = confusion_matrix(y_labels, y_preds)

    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    num_classes = conf_matrix.shape[0]
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    thresh = conf_matrix.max() / 2.0
    for i, j in np.ndindex(conf_matrix.shape):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                horizontalalignment="center",
                color="white" if conf_matrix[i, j] > thresh else "black")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
