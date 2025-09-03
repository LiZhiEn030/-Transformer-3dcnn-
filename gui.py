import sys
import cv2
import os
import numpy as np
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QMutex
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QSlider, QSizePolicy  # 添加缺失的导入
)
from predictor import ActionPredictor

import os
import sys

# 自动判断打包环境
if getattr(sys, 'frozen', False):
    # 打包后的基准目录（exe所在目录）
    BASE_DIR = os.path.dirname(sys.executable)
else:
    # 开发环境的基准目录（当前脚本所在目录）
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 构建模型路径（使用相对路径）
MODEL_PATH = os.path.join(BASE_DIR, "data", "models", "epoch_41.pth")



class VideoThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    prediction_ready = pyqtSignal(str)
    position_changed = pyqtSignal(int, int)

    def __init__(self, predictor, parent=None):
        super().__init__(parent)
        self.predictor = predictor
        self.cap = None
        self.is_running = False
        self.pause = False
        self.lock = QMutex()
        self.current_frame = 0
        self.total_frames = 0
        self.use_camera = False

    def load_video(self, path):
        self.lock.lock()
        try:
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(path)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.current_frame = 0
            self.use_camera = False
        finally:
            self.lock.unlock()

    def start_camera(self):
        self.lock.lock()
        try:
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("无法打开摄像头")
            self.use_camera = True
            self.is_running = True
            self.total_frames = 0
        finally:
            self.lock.unlock()
        self.start()

    def seek(self, frame_num):
        self.lock.lock()
        try:
            if self.cap and not self.use_camera:
                frame_num = max(0, min(frame_num, self.total_frames-1))
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                self.current_frame = frame_num
        except Exception as e:
            print(f"定位错误: {str(e)}")
        finally:
            self.lock.unlock()

    def run(self):
        while True:
            self.lock.lock()
            if not self.is_running:
                self.lock.unlock()
                break

            if self.pause or not self.cap:
                self.lock.unlock()
                QThread.msleep(50)
                continue

            try:
                ret, frame = self.cap.read()
                if not ret:
                    self.is_running = False
                    self.lock.unlock()
                    break

                # 更新进度（仅视频文件）
                if not self.use_camera:
                    frame_num = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    self.position_changed.emit(frame_num, self.total_frames)

                # 执行预测
                label = self.predictor.predict_frame(frame)
                self.prediction_ready.emit(label)

                # 转换颜色空间
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame_ready.emit(rgb_frame)

            except Exception as e:
                print(f"视频处理异常: {str(e)}")
                self.is_running = False

            self.lock.unlock()
            QThread.msleep(30)

        try:
            self.cap.release()
        except:
            pass

    def stop(self):
        self.lock.lock()
        try:
            self.is_running = False
            if self.cap:
                self.cap.release()
        finally:
            self.lock.unlock()
        self.wait(2000)

    def start_video(self, path):
        self.load_video(path)
        self.lock.lock()
        try:
            self.is_running = True
        finally:
            self.lock.unlock()
        self.start()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setFont(QFont("Microsoft YaHei", 12))
        if not os.path.isfile(MODEL_PATH):
            raise FileNotFoundError(f"模型文件不存在于: {MODEL_PATH}")

        self.is_seeking = False
        self.predictor = ActionPredictor(MODEL_PATH)
        self.init_ui()
        self.video_thread = None

    def toggle_camera(self):
        # 控制摄像头开启/关闭的核心逻辑
        if self.video_thread and self.video_thread.is_running:
            # 停止线程
            self.video_thread.stop()
            self.video_thread = None
            # 更新界面状态
            self.camera_btn.setText("📷 开启摄像头")
            self.pred_label.setText("当前行为: 未知")
        else:
            try:
                # 创建新线程并连接信号
                self.video_thread = VideoThread(self.predictor, self)
                self.video_thread.frame_ready.connect(self.update_frame)
                self.video_thread.prediction_ready.connect(self.update_prediction)
                # 启动摄像头
                self.video_thread.start_camera()
                # 更新按钮文本
                self.camera_btn.setText("⏹️ 关闭摄像头")
            except Exception as e:
                self.show_error_dialog(str(e))

    def toggle_pause(self):
        if self.video_thread:
            self.video_thread.pause = not self.video_thread.pause
            self.pause_btn.setText("▶️ 继续" if self.video_thread.pause else "⏸️ 暂停")

    def init_ui(self):
        self.setWindowTitle("行为识别系统")
        self.setGeometry(100, 100, 1024, 768)

        # 主布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()

        # 视频显示区域
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 600)
        layout.addWidget(self.video_label)

        # 进度条控件
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.progress_slider.sliderPressed.connect(self.start_seek)
        self.progress_slider.sliderMoved.connect(self.do_seek)
        self.progress_slider.sliderReleased.connect(self.end_seek)
        layout.addWidget(self.progress_slider)

        # 控制按钮
        btn_layout = QHBoxLayout()
        self.camera_btn = QPushButton("📷 开启摄像头")
        self.file_btn = QPushButton("📂 选择文件")
        self.pause_btn = QPushButton("⏸️ 暂停/继续")

        for btn in [self.camera_btn, self.file_btn, self.pause_btn]:
            btn.setStyleSheet("font-size:14px; padding:8px;")

        self.camera_btn.clicked.connect(self.toggle_camera)
        self.file_btn.clicked.connect(self.open_file)
        self.pause_btn.clicked.connect(self.toggle_pause)

        btn_layout.addWidget(self.camera_btn)
        btn_layout.addWidget(self.file_btn)
        btn_layout.addWidget(self.pause_btn)
        layout.addLayout(btn_layout)

        # 预测结果
        self.pred_label = QLabel("当前行为: 未知")
        self.pred_label.setStyleSheet("""
            font-size:24px; color:#FF0000; font-weight:bold;
            background-color:rgba(255,255,255,0.7); padding:10px; border-radius:5px;
        """)
        self.pred_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.pred_label)

        main_widget.setLayout(layout)

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频",
            os.path.join(BASE_DIR, "data/videos"),
            "视频文件 (*.mp4 *.avi)"
        )
        if path:
            if self.video_thread:
                self.video_thread.stop()

            self.video_thread = VideoThread(self.predictor, self)
            self.video_thread.frame_ready.connect(self.update_frame)
            self.video_thread.prediction_ready.connect(self.update_prediction)
            self.video_thread.position_changed.connect(self.update_progress)

            try:
                self.video_thread.start_video(path)
                self.progress_slider.setEnabled(True)
            except Exception as e:
                self.show_error_dialog(str(e))

    def update_frame(self, frame):
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(
            QPixmap.fromImage(q_img).scaled(
                800, 600,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

    def update_prediction(self, label):
        self.current_prediction = label
        red_shade = min(255, 100 + len(label) * 10)
        self.pred_label.setStyleSheet(f"""
            font-size: 24px; 
            color: rgb({red_shade},0,0);
            font-weight: bold;
            background-color: rgba(255,255,255,0.7);
            padding: 10px;
            border-radius: 5px;
        """)
        self.pred_label.setText(f"当前行为: {label}")

    def show_error_dialog(self, message):
        error_dialog = QLabel(self)
        error_dialog.setText(f"❌ 错误: {message}")
        error_dialog.setStyleSheet("""
            background-color: #FF0000;
            color: white;
            padding: 20px;
            border-radius: 10px;
        """)
        error_dialog.setWindowModality(Qt.WindowModal)
        error_dialog.show()
        QTimer.singleShot(5000, error_dialog.close)

    def closeEvent(self, event):
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread.wait(2000)  # 等待2秒确保线程退出
        event.accept()

    def start_seek(self):
        self.is_seeking = True

    def do_seek(self, value):
        if self.video_thread and self.progress_slider.maximum() > 0:
            frame_num = int(value * self.video_thread.total_frames / 100)
            self.video_thread.seek(frame_num)

    def end_seek(self):
        self.is_seeking = False

    def update_progress(self, current, total):
        if not self.is_seeking and total > 0:
            progress = int(current * 100 / total)
            self.progress_slider.setValue(progress)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 10))
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())