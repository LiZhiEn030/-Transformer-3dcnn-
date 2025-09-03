from torch.utils.data.dataloader import default_collate
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from dataset import UCF101DynamicDataset
from model import HybridModel
import torch.nn as nn
import os
from datetime import datetime

# 定义检查点保存路径
CHECKPOINT_DIR = "E:/PyCharm/YOLO/YOLO/毕设/data/models"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def custom_collate_fn(batch):
    valid_batch = [item for item in batch if item is not None and item[1] != -1]
    return default_collate(valid_batch) if valid_batch else (None, None)


class Trainer:
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config


        # ----------------------- 数据集初始化（新增验证集）-----------------------
        self.train_dataset = UCF101DynamicDataset(
            root_dir=config['dataset_path'],
            split='train',  # 训练集
            clip_len=config['clip_len'],
            frame_step=config['frame_step']
        )

        # 初始化数据加载器
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=min(2, os.cpu_count()),
            pin_memory=False,
            collate_fn=custom_collate_fn
        )

        self.val_dataset = UCF101DynamicDataset(
            root_dir=config['dataset_path'],
            split='val',  # 验证集（假设数据集类支持split='val'）
            clip_len=config['clip_len'],
            frame_step=config['frame_step']
        )


        # ----------------------- 数据加载器初始化（新增验证集加载器）-----------------------
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=min(config['num_workers'], os.cpu_count()),  # 使用配置中的num_workers
            pin_memory=torch.cuda.is_available(),  # GPU优化
            collate_fn=custom_collate_fn
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config['batch_size'],  # 验证集batch_size通常与训练集一致
            shuffle=False,
            num_workers=min(config['num_workers'], os.cpu_count()),
            pin_memory=torch.cuda.is_available(),
            collate_fn=custom_collate_fn
        )


        # ----------------------- 模型与优化器初始化 -----------------------
        self.model = HybridModel(num_classes=config['num_classes']).to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
        self.scaler = GradScaler(enabled=config['use_amp'])

        # ----------------------- 检查点加载-----------------------
        self.start_epoch = 0
        self.best_val_loss = float('inf')  # 修复：正确初始化最佳验证损失
        self._load_latest_checkpoint()


    def _save_checkpoint(self, epoch, train_loss, val_loss):
        """保存检查点（包含验证损失和早停状态）"""
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,  # 保存最佳验证损失
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all(),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }, checkpoint_path)
        print(f"\n检查点已保存至: {checkpoint_path}")

    def _load_latest_checkpoint(self):
        """加载检查点（增强兼容性）"""
        checkpoints = [f for f in os.listdir(CHECKPOINT_DIR)
                       if f.startswith("epoch_") and f.endswith(".pth")]
        if not checkpoints:
            return  # 无检查点时初始化新训练

        latest_checkpoint = max(
            [os.path.join(CHECKPOINT_DIR, f) for f in checkpoints],
            key=os.path.getctime
        )
        checkpoint = torch.load(latest_checkpoint)

        # 加载模型和优化器状态
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # 加载混合精度状态
        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # 加载训练状态
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))  # 兼容旧检查点

        # 修复验证Loss打印逻辑
        val_loss = checkpoint.get('val_loss')
        val_loss_str = f"{val_loss:.4f}" if val_loss is not None else "未知"

        print(f"\n成功恢复检查点：从 Epoch {checkpoint['epoch']} 继续训练")
        print(f"上次验证Loss：{val_loss_str}")

    def validate(self):
        """验证集评估（修复版）"""
        self.model.eval()
        total_loss = 0.0
        num_samples = 0

        with torch.no_grad():
            for clips, labels in self.val_loader:
                if clips is None or labels is None:
                    continue

                clips = clips.permute(0, 2, 1, 3, 4).to(self.device)
                labels = labels.to(self.device)

                with autocast(enabled=self.config['use_amp']):
                    outputs = self.model(clips)
                    loss = nn.CrossEntropyLoss()(outputs, labels)

                total_loss += loss.item() * len(labels)
                num_samples += len(labels)

        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        self.model.train()
        return avg_loss

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        valid_batches = 0
        num_samples = 0

        for batch_idx, (clips, labels) in enumerate(self.train_loader):
            if clips is None or labels is None:
                continue

            # 数据预处理
            clips = clips.permute(0, 2, 1, 3, 4).to(self.device)
            labels = labels.to(self.device)

            # 混合精度训练
            with autocast(enabled=self.config['use_amp']):
                outputs = self.model(clips)
                loss = nn.CrossEntropyLoss()(outputs, labels) / self.config['grad_accum_steps']

            # 反向传播和梯度累积
            self.scaler.scale(loss).backward()
            if (batch_idx + 1) % self.config['grad_accum_steps'] == 0:
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            # 打印训练信息
            total_loss += loss.item() * len(labels)
            num_samples += len(labels)
            valid_batches += 1

            if batch_idx % 50 == 0:
                mem = torch.cuda.max_memory_allocated() / 1024 ** 3
                print(f"Epoch {epoch} | Batch {batch_idx} | 显存峰值: {mem:.2f}GB | Loss: {loss.item():.4f}")

        # 计算平均训练损失
        avg_loss = total_loss / valid_batches if valid_batches > 0 else 0

        # 保存检查点，传入默认的val_loss
        val_loss = float('inf')  # 默认值，表示当前epoch没有有效的验证损失
        self._save_checkpoint(epoch, avg_loss, val_loss)

        return avg_loss

    def memory_profile(self):
        """显存分析工具"""
        if torch.cuda.is_available():
            print("\n显存使用报告:")
            print(f"分配峰值: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f}GB")
            print(f"缓存峰值: {torch.cuda.max_memory_reserved() / 1024 ** 3:.2f}GB")