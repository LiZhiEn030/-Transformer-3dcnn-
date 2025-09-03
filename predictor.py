import torch
import cv2
import numpy as np
from model import HybridModel
from collections import deque
import os

class ActionPredictor:
    def __init__(self, model_path, clip_len=16, frame_step=3):
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_len = clip_len
        self.frame_step = frame_step

        # 预处理参数（正确版本）
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device)

        # 模型加载（修正键名问题）
        self.model = HybridModel(num_classes=101).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)

        if "model_state_dict" in state_dict:  # 兼容检查点文件
            self.model.load_state_dict(state_dict["model_state_dict"])  # ✅ 正确键名
        else:  # 直接加载权重
            self.model.load_state_dict(state_dict)
        self.model.eval()

        # 缓存系统（正确初始化）
        self.frame_buffer = deque(maxlen=clip_len * frame_step * 3)
        self.required_frames = clip_len * frame_step

        # 类别名称（示例需补全）
        self.class_names = self._load_chinese_labels()
        assert len(self.class_names) == 101, "中文标签数量错误"

    def _load_chinese_labels(self):
        """加载UTF-8编码的中文标签"""
        label_path = os.path.join(os.path.dirname(__file__), "ucf101_labels_chinese.txt")
        try:
            with open(label_path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f.readlines() if line.strip()]
        except Exception as e:
            raise RuntimeError(f"无法加载中文标签: {str(e)}")


    def preprocess(self, frame):
        """改进的预处理流程"""
        if frame is None or frame.size < 100:
            return None

        # 调整尺寸和颜色空间
        resized = cv2.resize(frame, (224, 224))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # 转换为Tensor并归一化（正确方式）
        tensor = torch.from_numpy(rgb).float().to(self.device) / 255.0
        return (tensor - self.mean) / self.std  # [H, W, C]

    def predict_frame(self, frame):
        try:
            # 输入验证
            if frame is None or frame.size < 100:
                return "无效输入"

            # 预处理
            processed = self.preprocess(frame)
            if processed is None:
                return "预处理失败"

            # 更新缓存（自动维护长度）
            self.frame_buffer.append(processed.permute(2, 0, 1))  # 调整为[C, H, W]

            # 检查缓冲是否充足
            if len(self.frame_buffer) < self.required_frames:
                return f"缓冲中 ({len(self.frame_buffer)}/{self.required_frames})"

            # 安全获取索引
            start = max(0, len(self.frame_buffer) - self.required_frames)  # ✅ 确保非负
            indices = range(start, len(self.frame_buffer), self.frame_step)
            valid_indices = list(indices)[:self.clip_len]

            # 提取有效帧
            clip_frames = []
            for idx in valid_indices:
                if 0 <= idx < len(self.frame_buffer):
                    clip_frames.append(self.frame_buffer[idx])
                else:
                    print(f"警告：跳过无效索引 {idx}")

            # 检查帧数
            if len(clip_frames) < self.clip_len:
                return "帧不足"

            # 构建输入张量
            clip = torch.stack(clip_frames)  # [T, C, H, W]
            clip = clip.unsqueeze(0)  # [B, T, C, H, W]
            clip = clip.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]

            # 预测
            with torch.no_grad():
                outputs = self.model(clip)

            # 解析结果
            pred_idx = torch.argmax(outputs).item()
            if 0 <= pred_idx < len(self.class_names):
                return self.class_names[pred_idx]
            else:
                return "未知行为"

        except Exception as e:
            print(f"预测异常: {str(e)}")
            return "预测失败"

    def clear_buffer(self):
        self.frame_buffer.clear()