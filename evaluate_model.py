import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import UCF101DynamicDataset, custom_collate_fn
from model import HybridModel
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import pandas as pd
import os
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
DEFAULT_CONFIG = {
    'dataset_path': r"E:\PyCharm\YOLO\YOLO\毕设\data\UCF101",
    'model_path': None,
    'label_path': r"E:\PyCharm\YOLO\YOLO\毕设\ucf101_labels_chinese.txt",
    'batch_size': 8,
    'num_workers': 4,
    'clip_len': 16,
    'frame_step': 2,
    'num_classes': 101
}


def load_chinese_labels(label_path):
    """加载中文标签"""
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"中文标签文件不存在于: {label_path}")

    with open(label_path, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f if line.strip()]

    if len(labels) != 101:
        raise ValueError(f"标签数量错误，应为101个，实际找到{len(labels)}个")
    return labels


def evaluate_model(config=None, model_path=None, save_dir=None, return_dict=False):
    """
    增强版评估函数
    :param config: 自定义配置字典（可选）
    :param model_path: 模型文件路径（覆盖配置）
    :param save_dir: 结果保存目录
    :param return_dict: 是否返回结构化数据
    :return: 评估结果字典（当return_dict=True时）
    """
    # 合并配置参数
    final_config = DEFAULT_CONFIG.copy()
    if config:
        final_config.update(config)
    if model_path:
        final_config['model_path'] = model_path

    # 参数验证
    if not final_config['model_path'] or not os.path.exists(final_config['model_path']):
        raise FileNotFoundError(f"模型文件不存在: {final_config['model_path']}")

    # 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载中文标签
    chinese_labels = load_chinese_labels(final_config['label_path'])

    # 初始化数据集
    test_dataset = UCF101DynamicDataset(
        root_dir=final_config['dataset_path'],
        split='test',
        clip_len=final_config['clip_len'],
        frame_step=final_config['frame_step']
    )

    # 验证标签一致性
    assert len(chinese_labels) == len(test_dataset.classes), "标签数量不匹配"

    # 预处理设置
    test_dataset.transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=final_config['batch_size'],
        shuffle=False,
        num_workers=final_config['num_workers'],
        collate_fn=custom_collate_fn
    )

    # 加载模型
    model = HybridModel(num_classes=final_config['num_classes']).to(device)
    checkpoint = torch.load(final_config['model_path'], map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # 推理过程
    all_preds, all_labels = run_inference(model, test_loader, device)

    # 计算指标
    metrics = calculate_metrics(all_labels, all_preds, chinese_labels)

    # 保存结果
    if save_dir:
        save_results(metrics, chinese_labels, all_labels, all_preds, save_dir)

    return metrics if return_dict else None


def run_inference(model, test_loader, device):
    """执行模型推理"""
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for clips, labels in tqdm(test_loader, desc="评估进度"):
            if clips is None or clips.dim() != 5:
                continue

            try:
                clips = clips.permute(0, 2, 1, 3, 4).to(device)
                labels = labels.to(device)

                outputs = model(clips)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            except Exception as e:
                print(f"处理批次时出错: {str(e)}")
                continue

    return all_preds, all_labels


def calculate_metrics(true_labels, pred_labels, chinese_labels):
    """计算评估指标"""
    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, pred_labels, zero_division=0
    )

    # 找到最佳和最差类别
    f1_scores = dict(zip(chinese_labels, f1))
    best_class = max(f1_scores, key=f1_scores.get)
    worst_class = min(f1_scores, key=f1_scores.get)

    return {
        'accuracy': accuracy,
        'macro_precision': np.mean(precision),
        'macro_recall': np.mean(recall),
        'macro_f1': np.mean(f1),
        'best_class': best_class,
        'best_f1': f1_scores[best_class],
        'worst_class': worst_class,
        'worst_f1': f1_scores[worst_class],
        'class_metrics': {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        },
        'total_samples': len(true_labels)
    }


def save_results(metrics, chinese_labels, true_labels, pred_labels, save_dir):
    """保存评估结果"""
    os.makedirs(save_dir, exist_ok=True)

    # 保存分类报告
    report_df = pd.DataFrame({
        '精确率': metrics['class_metrics']['precision'].round(2),
        '召回率': metrics['class_metrics']['recall'].round(2),
        'F1值': metrics['class_metrics']['f1'].round(2),
        '支持数': metrics['class_metrics']['support']
    }, index=chinese_labels)

    # 添加整体指标
    accuracy_row = pd.DataFrame({
        '精确率': metrics['accuracy'],
        '召回率': np.nan,
        'F1值': np.nan,
        '支持数': metrics['total_samples']
    }, index=['整体指标'])

    report_df = pd.concat([report_df, accuracy_row])
    report_path = os.path.join(save_dir, 'classification_report.csv')
    report_df.to_csv(report_path, encoding='utf-8-sig')

    # 保存混淆矩阵
    plt.figure(figsize=(20, 20))
    cm = confusion_matrix(true_labels, pred_labels)
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                xticklabels=chinese_labels,
                yticklabels=chinese_labels)
    plt.title("混淆矩阵")
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()


if __name__ == "__main__":
    # 独立运行示例
    result = evaluate_model(
        model_path=r"E:\PyCharm\YOLO\YOLO\毕设\data\models\final_model.pth",
        save_dir=r"E:\PyCharm\YOLO\YOLO\毕设\eval_results",
        return_dict=True
    )

    print(f"\n评估结果：")
    print(f"整体准确率: {result['accuracy']:.2%}")
    print(f"最佳类别: {result['best_class']} (F1: {result['best_f1']:.2f})")
    print(f"最差类别: {result['worst_class']} (F1: {result['worst_f1']:.2f})")