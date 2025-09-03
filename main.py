from train import Trainer
import os
import torch
import glob
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 配置基目录
BASE_DIR = "E:\\PyCharm\\YOLO\\YOLO\\毕设\\data"
CHECKPOINT_DIR = os.path.join(BASE_DIR, "models")


def find_latest_checkpoint(checkpoint_dir):
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "epoch_*.pth"))
    if not checkpoint_files:
        return None
    checkpoints = sorted(checkpoint_files,
                         key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]))
    return checkpoints[-1]


if __name__ == "__main__":
    # 环境变量设置（保持不变）
    os.environ.update({
        'TORCH_HOME': os.path.join(BASE_DIR, "cache/torch"),
        'HF_HOME': os.path.join(BASE_DIR, "cache/huggingface"),
        'TEMP': os.path.join(BASE_DIR, "temp"),
        'TMPDIR': os.path.join(BASE_DIR, "temp")
    })
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # 训练配置 - 增加batch size并添加早停和学习率调度器参数
    config = {
        'dataset_path': os.path.join(BASE_DIR, "UCF101"),
        'num_classes': 101,
        'clip_len': 16,
        'frame_step': 2,
        'batch_size': 16,  # 增加batch size从8到16
        'num_workers': 4,  # 增加数据加载工作线程
        'lr': 3e-5,
        'weight_decay': 1e-4,
        'epochs': 50,
        'grad_accum_steps': 1,  # 由于batch size增大，减少梯度累积步数
        'use_amp': True,
        'save_path': CHECKPOINT_DIR,
        'early_stopping_patience': 5,  # 早停耐心值
        'min_delta': 0.001,  # 早停最小改善量
        'lr_patience': 2,  # 学习率调度器耐心值
        'lr_factor': 0.5  # 学习率衰减因子
    }

    # 初始化训练器
    trainer = Trainer(config)
    start_epoch = 0
    best_val_loss = float('inf')
    early_stop_counter = 0

    # 尝试加载检查点（新增兼容逻辑）
    latest_checkpoint = find_latest_checkpoint(config['save_path'])
    if latest_checkpoint:
        choice = input(f"发现最新检查点 {latest_checkpoint}，是否继续训练？(y/n): ")
        if choice.lower() == 'y':
            checkpoint = torch.load(latest_checkpoint)

            # 兼容新旧键名
            model_key = 'model_state' if 'model_state' in checkpoint else 'model_state_dict'
            optimizer_key = 'optimizer_state' if 'optimizer_state' in checkpoint else 'optimizer_state_dict'

            trainer.model.load_state_dict(checkpoint[model_key])
            trainer.optimizer.load_state_dict(checkpoint[optimizer_key])
            start_epoch = checkpoint['epoch']

            # 加载早停和学习率调度器状态
            if 'best_val_loss' in checkpoint:
                best_val_loss = checkpoint['best_val_loss']
            if 'early_stop_counter' in checkpoint:
                early_stop_counter = checkpoint['early_stop_counter']

            print(f"成功恢复训练状态，将从 Epoch {start_epoch} 继续训练")
            if 'train_loss' in checkpoint:
                print(f"上次训练Loss值: {checkpoint['train_loss']:.4f}")
            if 'val_loss' in checkpoint:
                print(f"上次验证Loss值: {checkpoint['val_loss']:.4f}")

    # 创建学习率调度器
    scheduler = ReduceLROnPlateau(
        trainer.optimizer,
        mode='min',
        factor=config['lr_factor'],
        patience=config['lr_patience'],
        verbose=True,
        min_lr=1e-7
    )

    print("开始训练...")

    try:
        # 训练循环（从start_epoch开始）
        for epoch in range(start_epoch, config['epochs']):
            train_loss = trainer.train_epoch(epoch + 1)
            val_loss = trainer.validate()  # 获取真实验证损失

            # 保存检查点
            trainer._save_checkpoint(epoch + 1, train_loss, val_loss)
            # 更新学习率
            scheduler.step(val_loss)

            # 早停检查
            if val_loss < best_val_loss - config['min_delta']:
                best_val_loss = val_loss
                early_stop_counter = 0

                # 保存最佳模型
                best_path = os.path.join(config['save_path'], "best_model.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': trainer.model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss,
                    'early_stop_counter': early_stop_counter,
                    'lr': trainer.optimizer.param_groups[0]['lr']
                }, best_path)
                print(f"Epoch {epoch + 1} | 最佳模型已保存: {best_path} (val_loss: {val_loss:.4f})")
            else:
                early_stop_counter += 1
                print(
                    f"Epoch {epoch + 1} | 验证损失未改善，计数器: {early_stop_counter}/{config['early_stopping_patience']}")

                # 检查是否应该早停
                if early_stop_counter >= config['early_stopping_patience']:
                    print(f"早停触发：验证损失在{config['early_stopping_patience']}轮内没有改善")
                    break

            # 保存当前epoch的检查点
            checkpoint_path = os.path.join(
                config['save_path'],
                f"epoch_{epoch + 1}.pth"
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'early_stop_counter': early_stop_counter,
                'lr': trainer.optimizer.param_groups[0]['lr']
            }, checkpoint_path)
            print(
                f"Epoch {epoch + 1} | 检查点已保存: {checkpoint_path} (train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f})")

    except KeyboardInterrupt:
        # 意外中断时保存应急模型
        interrupt_path = os.path.join(config['save_path'], "interrupted_model.pth")
        torch.save(trainer.model.state_dict(), interrupt_path)
        print(f"\n训练中断！应急模型已保存至: {interrupt_path}")
        exit(1)

    # 最终模型保存
    final_path = os.path.join(config['save_path'], "final_model.pth")
    torch.save(trainer.model.state_dict(), final_path)
    print(f"\n训练完成！最终模型已保存至: {final_path}")
    print(f"最佳验证损失: {best_val_loss:.4f}")
