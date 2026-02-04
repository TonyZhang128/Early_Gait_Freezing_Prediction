from datasets.Gait_dataset_old import create_dataloaders
from datasets.Gait_dataset_new import GaitDataModule
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='步态识别分类训练')
    parser.add_argument('--exp_name', type=str, default='Gait_finetune_training')
    parser.add_argument('--mode', type=str, default='debug', help='normal, debug')
    parser.add_argument('--print_params', type=bool, default=True, help='打印参数')

    # 数据相关
    parser.add_argument('--data_path', type=str, default='./datasets/data_10000/', help='数据文件路径')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='训练集比例')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载线程数')
    
    parser.add_argument('--augmentation_prob', type=float, default=0.5, help='数据增强概率')
    parser.add_argument('--freq_keep_ratio', type=float, default=0.6, help='频率成分保留比例')
    
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()
    return args

def count_testset_classes(test_loader):
    """
    统计测试集每个类别的样本数量（纯统计，无打印）
    :param test_loader: PyTorch DataLoader 测试集加载器
    :return: 有序类别统计字典 {类别标签: 样本数量}（按类别升序）
    """
    class_count = defaultdict(int)
    for batch in test_loader:
        labels = batch[1]  # 常规(data, label)格式，若特殊可修改此行
        for label in labels:
            label_int = int(label)  # 兼容tensor/int类型标签
            class_count[label_int] += 1
    # 按类别标签升序排序，返回有序字典
    return dict(sorted(class_count.items()))

def print_class_distribution(class_count, loader_name):
    """打印类别分布文本结果"""
    print(f"\n========== {loader_name} 测试集类别分布 ==========")
    for cls in class_count:
        print(f"类别 {cls:>2d} : {class_count[cls]:>4d} 个样本")
    total_samples = sum(class_count.values())
    print(f"------------------------------------------")
    print(f"测试集总样本数       : {total_samples:>4d} 个")
    return total_samples

def plot_class_distribution(old_count, new_count):
    """
    绘制新旧数据加载方式的类别分布并列柱状图
    :param old_count: 旧方式统计字典 {类别: 数量}
    :param new_count: 新方式统计字典 {类别: 数量}
    """
    # 统一类别标签（确保新旧方式类别一致，按升序排列）
    classes = sorted(list(set(old_count.keys()).union(set(new_count.keys()))))
    # 提取对应类别数量（若某方式无该类别，数量为0）
    old_nums = [old_count.get(cls, 0) for cls in classes]
    new_nums = [new_count.get(cls, 0) for cls in classes]

    # 设置柱状图参数：并列显示的柱宽、间距
    bar_width = 0.35
    x = range(len(classes))  # x轴位置

    # 绘制并列柱状图
    plt.bar([i - bar_width/2 for i in x], old_nums, bar_width, 
            label='旧数据加载方式', color='#1f77b4', alpha=0.8)
    plt.bar([i + bar_width/2 for i in x], new_nums, bar_width, 
            label='新数据加载方式', color='#ff7f0e', alpha=0.8)

    # 添加数值标签（在每个柱子顶部显示样本数）
    for i, (o, n) in enumerate(zip(old_nums, new_nums)):
        plt.text(i - bar_width/2, o + 1, str(o), ha='center', va='bottom', fontsize=9)
        plt.text(i + bar_width/2, n + 1, str(n), ha='center', va='bottom', fontsize=9)

    # 设置图表标签和标题
    plt.xlabel('类别标签', fontsize=12, fontweight='bold')
    plt.ylabel('样本数量', fontsize=12, fontweight='bold')
    plt.title('新旧数据加载方式 - 测试集类别分布对比', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(x, classes)  # x轴刻度替换为类别标签
    plt.legend(loc='upper right', frameon=True, shadow=True)  # 显示图例
    plt.grid(axis='y', linestyle='--', alpha=0.3)  # 添加水平网格线，增强可读性
    plt.tight_layout()  # 自动调整布局，避免标签截断

    # 显示图表 + 可选保存图表（取消注释即可保存为高清图片）
    # plt.savefig('testset_class_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

# 主执行逻辑
if __name__ == "__main__":
    args = parse_args()
    # 加载新旧方式的测试集DataLoader
    _, test_loader_old = create_dataloaders(args)
    data_module = GaitDataModule(args)
    _, test_loader_new = data_module.setup()
    
    # 1. 统计类别数量
    old_class_count = count_testset_classes(test_loader_old)
    new_class_count = count_testset_classes(test_loader_new)
    
    # 2. 打印文本统计结果
    old_total = print_class_distribution(old_class_count, "旧数据加载方式")
    new_total = print_class_distribution(new_class_count, "新数据加载方式")
    print(f"\n========== 对比总结 ==========")
    print(f"旧方式总样本数: {old_total}, 新方式总样本数: {new_total}")
    print(f"样本数是否一致: {'✅ 是' if old_total == new_total else '❌ 否'}")
    
    # 3. 绘制并列柱状图对比
    plot_class_distribution(old_class_count, new_class_count)