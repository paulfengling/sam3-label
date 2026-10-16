"""A separate, cancellable video workflow without changing image annotations."""
from pathlib import Path
import threading

import torch  # Keep the Python 3.10 Qt/typing import order compatible.
import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QRectF, QThread, Signal, QUrl
from PySide6.QtGui import QColor, QPen, QPixmap, QImage, QDesktopServices
from PySide6.QtWidgets import (
    QApplication, QDialog, QFileDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QSlider, QCheckBox, QProgressBar, QGraphicsView, QGraphicsScene,
)

from labelpaw.data.video_io import decode_video, VideoCancelledError
from labelpaw.video_pipeline import track_sequence


class VideoSeedView(QGraphicsView):
    box_changed = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setMinimumSize(560, 300)
        self.setAlignment(Qt.AlignCenter)
        self._anchor = None
        self._roi = None
        self._mask_item = None
        self.box = None
        self.image_size = None

    def set_frame(self, path):
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            raise ValueError("无法显示视频帧")
        self.scene().clear()
        self._anchor = self._roi = self._mask_item = None
        self.box = None
        self.image_size = (pixmap.width(), pixmap.height())
        self.scene().addPixmap(pixmap)
        self.setSceneRect(QRectF(0, 0, *self.image_size))
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
        self.box_changed.emit(None)

    def show_mask(self, path):
        if self._mask_item is not None:
            self.scene().removeItem(self._mask_item)
            self._mask_item = None
        if not path or not Path(path).is_file():
            return
        with Image.open(path) as source:
            mask = np.asarray(source.convert("L")) > 0
        rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
        rgba[mask] = (40, 230, 100, 110)
        image = QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0], QImage.Format_RGBA8888).copy()
        self._mask_item = self.scene().addPixmap(QPixmap.fromImage(image))
        self._mask_item.setZValue(1)

    def _position(self, point):
        p = self.mapToScene(point)
        p.setX(min(max(0, p.x()), self.image_size[0]))
        p.setY(min(max(0, p.y()), self.image_size[1]))
        return p

    def clear_box(self):
        if self._roi is not None:
            self.scene().removeItem(self._roi)
        self._roi = self._anchor = None
        self.box = None
        self.box_changed.emit(None)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.image_size is not None:
            if not self.sceneRect().contains(self.mapToScene(event.position().toPoint())):
                return
            self.clear_box()
            self._anchor = self._position(event.position().toPoint())
            pen = QPen(QColor("#ffb700"), 2, Qt.DashLine)
            pen.setCosmetic(True)
            self._roi = self.scene().addRect(QRectF(self._anchor, self._anchor), pen)
            self._roi.setZValue(2)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._anchor is not None and event.buttons() & Qt.LeftButton:
            self._roi.setRect(QRectF(self._anchor, self._position(event.position().toPoint())).normalized())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._anchor is not None and event.button() == Qt.LeftButton:
            rect = QRectF(self._anchor, self._position(event.position().toPoint())).normalized()
            self._anchor = None
            if rect.width() < 2 or rect.height() < 2:
                self.clear_box()
            else:
                self._roi.setRect(rect)
                self.box = [rect.left(), rect.top(), rect.right(), rect.bottom()]
                self.box_changed.emit(self.box)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.image_size:
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        if event.modifiers() & (Qt.ControlModifier | Qt.MetaModifier):
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.scale(factor, factor)
            event.accept()
        else:
            super().wheelEvent(event)


class VideoJob(QThread):
    result_ready = Signal(object)
    failed = Signal(str)
    progress = Signal(int, int, str)
    cancelled = Signal()

    def __init__(self, operation, parent=None):
        super().__init__(parent)
        self.operation = operation
        self.cancel_event = threading.Event()

    def run(self):
        try:
            result = self.operation(self.progress.emit, self.cancel_event.is_set)
            self.result_ready.emit(result)
        except VideoCancelledError:
            self.cancelled.emit()
        except Exception as error:
            self.failed.emit(str(error))

    def cancel(self):
        self.cancel_event.set()


class VideoTrackingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("视频目标跟踪")
        self.resize(1040, 790)
        self.sequence = None
        self.tracking_result = None
        self.job = None
        self._closing = False
        QApplication.instance().aboutToQuit.connect(self.shutdown)
        self.video_path = None
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        self.open_button = QPushButton("打开视频")
        self.source_label = QLabel("先选择视频，再用左键拖框指定同一个目标")
        self.source_label.setWordWrap(True)
        top.addWidget(self.open_button)
        top.addWidget(self.source_label, 1)
        layout.addLayout(top)
        destination = QHBoxLayout()
        destination.addWidget(QLabel("输出到"))
        self.output_path = QLineEdit(str(Path(__file__).resolve().parents[1] / "outputs/video_tracking"))
        self.output_path.setReadOnly(True)
        self.output_button = QPushButton("选择目录")
        destination.addWidget(self.output_path, 1)
        destination.addWidget(self.output_button)
        layout.addLayout(destination)
        self.view = VideoSeedView()
        layout.addWidget(self.view, 1)
        frame_row = QHBoxLayout()
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_label = QLabel("第 0 帧")
        self.clear_button = QPushButton("清除选框")
        frame_row.addWidget(self.frame_slider, 1)
        frame_row.addWidget(self.frame_label)
        frame_row.addWidget(self.clear_button)
        layout.addLayout(frame_row)
        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("目标描述／类别"))
        self.prompt = QLineEdit("白色胶带")
        self.prompt.setPlaceholderText("文字找到多个目标时，请拖框指定其中一个")
        self.yolo = QCheckBox("同时导出 YOLO 分割")
        self.yolo.setChecked(True)
        target_row.addWidget(self.prompt, 1)
        target_row.addWidget(self.yolo)
        layout.addLayout(target_row)
        self.selection_label = QLabel("未拖框时按文字寻找一个目标；拖框时按选框定位，文字作为类别名称。")
        self.selection_label.setWordWrap(True)
        layout.addWidget(self.selection_label)
        note = QLabel("输出轮廓视频（无音轨）、每帧 JSON 和掩码。当前使用 SAM3 确定目标，SAM2.1 连续跟踪。遮挡后可换参考帧重新框选并生成新一轮结果。")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.status = QLabel("等待打开视频")
        self.status.setWordWrap(True)
        self.progress_bar = QProgressBar()
        layout.addWidget(self.status)
        layout.addWidget(self.progress_bar)
        actions = QHBoxLayout()
        self.start_button = QPushButton("开始跟踪并导出")
        self.stop_button = QPushButton("停止")
        self.video_button = QPushButton("查看结果视频")
        self.folder_button = QPushButton("打开结果目录")
        for button in (self.start_button, self.stop_button, self.video_button, self.folder_button):
            actions.addWidget(button)
        layout.addLayout(actions)
        self.open_button.clicked.connect(self.choose_video)
        self.output_button.clicked.connect(self.choose_output)
        self.start_button.clicked.connect(self.start_tracking)
        self.stop_button.clicked.connect(self.stop)
        self.clear_button.clicked.connect(self.view.clear_box)
        self.frame_slider.valueChanged.connect(self.show_frame)
        self.view.box_changed.connect(self.box_changed)
        self.video_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.tracking_result.video_path))))
        self.folder_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.tracking_result.output_dir))))
        self._update_controls()

    def _update_controls(self):
        busy = self.job is not None
        for widget in (self.open_button, self.output_button, self.prompt, self.yolo, self.view):
            widget.setEnabled(not busy)
        self.frame_slider.setEnabled(not busy and self.sequence is not None)
        self.clear_button.setEnabled(not busy and self.sequence is not None)
        self.start_button.setEnabled(not busy and self.sequence is not None)
        self.stop_button.setEnabled(busy)
        self.video_button.setEnabled(not busy and self.tracking_result is not None)
        self.folder_button.setEnabled(not busy and self.tracking_result is not None)

    def choose_output(self):
        path = QFileDialog.getExistingDirectory(self, "选择视频跟踪输出目录", self.output_path.text())
        if path:
            self.output_path.setText(path)
            if self.video_path:
                self.prepare_video(self.video_path)

    def choose_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开视频", "", "视频 (*.mp4 *.mov *.avi *.mkv *.m4v);;所有文件 (*)")
        if path:
            self.prepare_video(path)

    def _run_job(self, operation, on_result):
        self.job = VideoJob(operation, self)
        job = self.job
        job.progress.connect(self.update_progress)
        job.result_ready.connect(on_result)
        job.failed.connect(lambda message: self.status.setText("未完成：" + message))
        job.cancelled.connect(lambda: self.status.setText("已停止。原视频未改动；已解码画面可继续使用。"))
        job.finished.connect(lambda: self._job_finished(job))
        self._update_controls()
        job.start()

    def _job_finished(self, job):
        self.job = None
        job.deleteLater()
        self._update_controls()
        if self._closing:
            self.reject()

    def prepare_video(self, path):
        if self.job is not None:
            return
        self.video_path = str(path)
        self.sequence = self.tracking_result = None
        self.source_label.setText(Path(path).name)
        self.status.setText("正在读取视频，按原帧率解码全部画面")
        output_dir = self.output_path.text()
        self._run_job(
            lambda progress, cancelled: decode_video(path, output_dir, progress=progress, cancelled=cancelled),
            self.video_prepared,
        )

    def video_prepared(self, sequence):
        self.sequence = sequence
        self.frame_slider.setRange(0, sequence.frame_count - 1)
        self.frame_slider.setValue(0)
        self.show_frame(0)
        self.status.setText(f"已读取 {sequence.frame_count} 帧，{sequence.width} × {sequence.height}，{sequence.fps:.3f} 帧/秒。选择清晰画面，左键拖框指定目标。")

    def show_frame(self, index):
        if self.sequence is None:
            return
        self.view.set_frame(self.sequence.frames[index].path)
        self.frame_label.setText(f"第 {index + 1}/{self.sequence.frame_count} 帧  {index / self.sequence.fps:.2f} 秒")
        if self.tracking_result is not None:
            self.view.show_mask(self.tracking_result.masks_dir / f"{index:05d}.png")

    def box_changed(self, box):
        if box is None:
            self.selection_label.setText("未拖框时按文字寻找一个目标；文字找到多个目标时请拖框指定。")
        else:
            values = ", ".join(str(round(value)) for value in box)
            self.selection_label.setText(f"已指定这一帧的目标范围：{values}。将向前、向后跟踪同一个目标。")

    def start_tracking(self):
        if self.sequence is None or self.job is not None:
            return
        label = self.prompt.text().strip()
        if not label:
            self.status.setText("请填写目标描述或类别名称")
            return
        sequence, seed_index, box = self.sequence, self.frame_slider.value(), self.view.box
        export_yolo = self.yolo.isChecked()
        self.tracking_result = None
        self.status.setText("正在确定目标。CPU 加载模型和逐帧跟踪需要等待，可随时停止。")
        self._run_job(
            lambda progress, cancelled: track_sequence(
                sequence, label, seed_frame_index=seed_index, box_xyxy=box,
                export_yolo=export_yolo, progress=progress, cancelled=cancelled,
            ), self.tracking_finished,
        )

    def tracking_finished(self, result):
        self.tracking_result = result
        self.show_frame(self.frame_slider.value())
        self.status.setText(f"已导出 {result.frame_count} 帧，其中 {result.frames_with_target} 帧有目标轮廓。可拖动时间轴检查遮挡前后结果；结果保存在 {result.output_dir}")

    def update_progress(self, done, total, message):
        self.status.setText(message)
        self.progress_bar.setRange(0, max(1, total) if total else 0)
        self.progress_bar.setValue(done)

    def stop(self):
        if self.job is not None:
            self.job.cancel()
            self.status.setText("正在停止，等待当前帧或模型加载结束")
            self.stop_button.setEnabled(False)

    def shutdown(self):
        # Application quit bypasses the dialog's ordinary close/reject path.
        # Keep the owning dialog alive until the cancellable worker finishes.
        if self.job is not None:
            self.job.cancel()
            self.job.wait()

    def reject(self):
        if self.job is not None:
            self._closing = True
            self.stop()
            return
        super().reject()

    def closeEvent(self, event):
        if self.job is not None:
            self._closing = True
            self.stop()
            event.ignore()
        else:
            super().closeEvent(event)
