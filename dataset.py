import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import default_collate

class UCF101DynamicDataset(Dataset):
    def __init__(self, root_dir, split='train', clip_len=16, frame_step=2, test_size=0.2):
        # === 1. 先初始化所有参数 ===
        self.root_dir = os.path.normpath(root_dir)
        self.split = split
        self.clip_len = clip_len  # 关键修正：优先初始化
        self.frame_step = frame_step  # 关键修正：优先初始化
        self.test_size = test_size
        self.min_frames = clip_len * frame_step

        # === 2. 获取有效样本 ===
        self._init_classes()
        self.valid_samples = self._filter_valid_samples()
        self.samples = self._split_dataset()  # 重新组织后的样本列表

        # === 3. 数据处理转换 ===
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _init_classes(self):
        """初始化类别标签映射"""
        self.classes = sorted([d for d in os.listdir(self.root_dir)
                               if os.path.isdir(os.path.join(self.root_dir, d))])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def _filter_valid_samples(self):
        """过滤符合要求的样本"""
        valid_samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            for video_file in os.listdir(class_dir):
                if not video_file.lower().endswith(('.avi', '.mp4')):
                    continue
                video_path = os.path.join(class_dir, video_file)
                if self._is_video_valid(video_path):
                    valid_samples.append((video_path, self.class_to_idx[class_name]))
        return valid_samples

    def _is_video_valid(self, video_path):
        """验证视频有效性"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total_frames >= self.min_frames  # 现在可以正确使用self.min_frames

    def _split_dataset(self):
        """划分训练测试集"""
        if not self.valid_samples:
            raise ValueError("没有找到有效样本，请检查数据集路径和过滤条件")

        train_samples, test_samples = train_test_split(
            self.valid_samples,
            test_size=self.test_size,
            random_state=42,
            stratify=[s[1] for s in self.valid_samples]
        )
        return train_samples if self.split == 'train' else test_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        max_retry = 3
        for _ in range(max_retry):
            try:
                video_path, label = self.samples[idx]
                cap = cv2.VideoCapture(video_path)

                # === 定义关键帧范围和采样间隔 ===
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                start_frame = np.random.randint(0, total_frames - (self.clip_len * self.frame_step) + 1)

                # === 抽取视频帧 ===
                frames = []
                for i in range(0, self.clip_len):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + i * self.frame_step)
                    ret, frame = cap.read()
                    if not ret:
                        raise RuntimeError(f"读取失败: {video_path} 第{start_frame + i * self.frame_step}帧")

                    # === 转换为张量 ===
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                    frames.append(self.transform(frame))

                cap.release()
                return torch.stack(frames), label
            except Exception as e:
                print(f"\033[91m加载失败: {video_path}\n错误细节: {str(e)}\033[0m")
                print(f"加载失败重试中({_ + 1}/{max_retry}): {video_path}")
                idx = np.random.randint(0, len(self))  # 随机重试其他样本
        return None, -1  # 标记无效样本

def custom_collate_fn(batch):
    """强化过滤逻辑"""
    valid_batch = []
    for item in batch:
        if item[0] is None or item[1] == -1:
            continue
        if item[0].shape != (16, 3, 224, 224):  # 添加形状验证
            continue
        valid_batch.append(item)
    if not valid_batch:
        return torch.Tensor(), torch.LongTensor()
    return default_collate(valid_batch)

def visualize_distribution(dataset, title):
    """可视化类别分布"""
    plt.figure(figsize=(14, 6))

    # === 统计类别分布 ===
    class_counts = Counter([s[1] for s in dataset.samples])
    labels = [dataset.classes[i] for i in class_counts.keys()]
    counts = list(class_counts.values())

    # === 中级图 ===
    ax1 = plt.subplot(1, 2, 1)
    bars = ax1.barh(labels, counts, color=sns.color_palette("hls", len(labels)))
    plt.title(f'{title}类别分布')
    plt.xlabel('样本数量')
    plt.xticks(rotation=45, ha='right')

    # === 统计摘要 ===
    ax2 = plt.subplot(1, 2, 2)
    summary_stats = {
        '总样本量': len(dataset),
        '最大样本量': max(counts),
        '最小样本量': min(counts),
        '平均样本量': sum(counts) / len(counts)
    }
    ax2.axis('off')
    ax2.table(cellText=[[str(round(v, 2))] for v in summary_stats.values()],
              rowLabels=summary_stats.keys(),
              loc='center')
    plt.tight_layout()
    plt.show()


# === 测试运行 ===
if __name__ == '__main__':
    dataset = UCF101DynamicDataset(
        root_dir=r"E:\PyCharm\YOLO\YOLO\毕设\data\UCF101",
        split='train',
        clip_len=16,
        frame_step=2,
        test_size=0.2
    )

    print(f"成功加载样本数量: {len(dataset)}")
    visualize_distribution(dataset, "训练集")