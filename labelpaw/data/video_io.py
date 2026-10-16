"""Video/frame export for a single tracked target, without sampling or interpolation.

``progress(done, total, message)`` is shared by decoding and export. ``cancelled``
is a no-argument callback. Video files produced by OpenCV contain no audio track.
Decoded frames retain their source resolution; JPEG storage is lossy.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import uuid

import cv2
import numpy as np

from labelpaw.models.sam_geometry import mask_to_annotation


class VideoIOError(RuntimeError):
    """A video could not be read, aligned, or exported safely."""


class VideoCancelledError(VideoIOError):
    """The user cancelled at a frame boundary."""


@dataclass(frozen=True)
class VideoInfo:
    source_path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float


@dataclass(frozen=True)
class VideoFrame:
    frame_index: int
    timestamp: float
    path: Path
    timestamp_source: str = "frame_index_over_fps"


@dataclass(frozen=True)
class VideoSequence:
    root_dir: Path
    frames_dir: Path
    metadata_path: Path
    source_video: Path
    width: int
    height: int
    fps: float
    frame_count: int
    frames: tuple


@dataclass(frozen=True)
class TrackingExport:
    output_dir: Path
    video_path: Path
    labels_dir: Path
    masks_dir: Path
    index_path: Path
    yolo_dir: Path | None
    codec: str
    frame_count: int
    frames_with_target: int
    audio_included: bool = False


def _check_cancelled(cancelled):
    if cancelled is not None and cancelled():
        raise VideoCancelledError("视频处理已取消")


def _progress(callback, done, total, message):
    if callback is not None:
        callback(int(done), int(total), message)


def _write_json(path, data):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)


def _write_image(path, image, parameters=None):
    if not cv2.imwrite(str(path), image, parameters or []):
        raise VideoIOError(f"无法写入图片：{path}")


def _open_capture(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise VideoIOError(f"无法打开视频：{path}")
    return capture


def probe_video(video_path):
    """Read video properties; frame_count/duration are container estimates."""
    source = Path(video_path).expanduser().resolve()
    if not source.is_file():
        raise VideoIOError(f"视频文件不存在：{source}")
    capture = _open_capture(source)
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        success, frame = capture.read()
        if not success or frame is None:
            raise VideoIOError("视频没有可读取的画面")
        if not math.isfinite(fps) or fps <= 0:
            raise VideoIOError("无法读取有效帧率，不能确定视频时间或输出播放速度")
        height, width = frame.shape[:2]
        frame_count = int(round(count)) if math.isfinite(count) and count > 0 else 0
        return VideoInfo(source, width, height, fps, frame_count, frame_count / fps)
    finally:
        capture.release()


def decode_video(video_path, output_parent, *, progress=None, cancelled=None):
    """Decode every frame into a new private job directory, never resizing.

    On failure/cancellation only this newly created job directory is removed.
    Source presentation timestamps are recorded when OpenCV supplies a valid,
    increasing value; otherwise the recorded timestamp explicitly uses FPS.
    """
    _check_cancelled(cancelled)
    info = probe_video(video_path)
    parent = Path(output_parent).expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="labelpaw_video_", dir=parent))
    frames_dir = root / "frames"
    frames_dir.mkdir()
    capture = None
    try:
        capture = _open_capture(info.source_path)
        frames = []
        _progress(progress, 0, info.frame_count, "正在解码视频原始帧")
        while True:
            _check_cancelled(cancelled)
            success, image = capture.read()
            if not success:
                break
            if image is None or image.shape[:2] != (info.height, info.width):
                raise VideoIOError("视频帧尺寸发生变化，不能与原分辨率跟踪结果对齐")
            index = len(frames)
            milliseconds = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            timestamp = milliseconds / 1000
            source = "opencv_pos_msec"
            if (not math.isfinite(timestamp) or timestamp < 0
                    or (frames and timestamp <= frames[-1].timestamp)):
                timestamp = index / info.fps
                if frames and timestamp <= frames[-1].timestamp:
                    timestamp = frames[-1].timestamp + 1 / info.fps
                source = "frame_index_over_fps"
            path = frames_dir / f"{index:05d}.jpg"
            _write_image(path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            frames.append(VideoFrame(index, timestamp, path, source))
            _progress(progress, len(frames), info.frame_count, "正在解码视频原始帧")
        if not frames:
            raise VideoIOError("视频没有可读取的画面")
        _check_cancelled(cancelled)
        metadata_path = root / "metadata.json"
        _write_json(metadata_path, {
            "source_video": str(info.source_path), "width": info.width,
            "height": info.height, "fps": info.fps, "frame_count": len(frames),
            "reported_frame_count": info.frame_count,
            "duration_seconds": frames[-1].timestamp + 1 / info.fps,
            "frame_sampling": False, "resized": False,
            "frame_encoding": "JPEG quality 95; original decoded resolution",
            "timestamp_units": "seconds",
            "frames": [
                {"frame_index": frame.frame_index, "timestamp": frame.timestamp,
                 "timestamp_source": frame.timestamp_source,
                 "path": frame.path.relative_to(root).as_posix()}
                for frame in frames
            ],
        })
        _progress(progress, len(frames), len(frames), "视频解码完成，全部帧已按原顺序保存")
        return VideoSequence(root, frames_dir, metadata_path, info.source_path,
                             info.width, info.height, info.fps, len(frames), tuple(frames))
    except BaseException:
        if capture is not None:
            capture.release()
        shutil.rmtree(root)
        raise
    finally:
        if capture is not None:
            capture.release()


def _frame_index(value, frame_count):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise VideoIOError("跟踪帧编号必须是从 0 开始的整数")
    index = int(value)
    if not 0 <= index < frame_count:
        raise VideoIOError(f"跟踪帧编号越界：{index}")
    return index


def _mask_record(record, frame_count):
    if hasattr(record, "frame_index") and hasattr(record, "mask"):
        index, mask = record.frame_index, record.mask
        if getattr(record, "track_id", 1) != 1:
            raise VideoIOError("只允许导出选定目标 track_id=1")
    else:
        try:
            index, mask = record
        except (TypeError, ValueError) as error:
            raise VideoIOError("跟踪结果应为 (frame_index, mask) 或 TrackingFrame") from error
    return _frame_index(index, frame_count), mask


def _binary_mask(mask, width, height):
    if mask is None:
        return np.zeros((height, width), dtype=np.uint8), "no_mask", False
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    array = np.asarray(mask)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.shape != (height, width):
        raise VideoIOError(f"模型 mask 尺寸 {array.shape} 与视频原尺寸 {(height, width)} 不一致")
    if not np.issubdtype(array.dtype, np.number) and array.dtype != np.bool_:
        raise VideoIOError("模型 mask 必须是二值或数值数组")
    if not np.isfinite(array).all():
        raise VideoIOError("模型 mask 含非有限数值")
    binary = (array > 0.5).astype(np.uint8)
    return binary, "visible" if binary.any() else "empty", True


def _save_frame_result(sequence, final_dir, stage, frame, mask, label, status, available, export_yolo):
    name = f"{frame.frame_index:05d}"
    _write_image(stage / "masks" / f"{name}.png", mask * 255)
    annotation = mask_to_annotation(mask)
    points = annotation["poly_pts"] if annotation is not None else []
    bbox = annotation["rect"] if annotation is not None else None
    image_path = Path(os.path.relpath(frame.path, final_dir / "labels")).as_posix()
    data = {
        "version": "LabelPaw.VideoTracking.1", "imagePath": image_path,
        "imageWidth": sequence.width, "imageHeight": sequence.height,
        "width": sequence.width, "height": sequence.height,
        "frame_index": frame.frame_index, "timestamp": frame.timestamp,
        "timestamp_units": "seconds", "track_id": 1, "label": label,
        "status": status, "prediction_available": available,
        "visible": annotation is not None,
        "polygon": points, "bbox": bbox, "bbox_format": "xywh",
        "maskPath": f"../masks/{name}.png", "flags": {},
        "shapes": ([{"label": label, "points": points, "group_id": 1,
                     "shape_type": "polygon", "flags": {"track_id": 1}}]
                   if annotation is not None else []),
    }
    _write_json(stage / "labels" / f"{name}.json", data)
    if export_yolo:
        # Same normalized polygon columns as Exporter.save_yolo; no Qt import.
        coordinates = " ".join(f"{x / sequence.width:.6f} {y / sequence.height:.6f}" for x, y in points)
        (stage / "yolo" / f"{name}.txt").write_text("0 " + coordinates + "\n" if points else "", encoding="utf-8")
    return {
        "frame_index": frame.frame_index, "timestamp": frame.timestamp,
        "timestamp_source": frame.timestamp_source,
        "frame_path": Path(os.path.relpath(frame.path, final_dir)).as_posix(),
        "mask_path": f"masks/{name}.png", "label_path": f"labels/{name}.json",
        "yolo_path": f"yolo/{name}.txt" if export_yolo else None,
        "status": status, "prediction_available": available,
        "visible": annotation is not None, "track_id": 1, "bbox": bbox,
    }


def _open_video_writer(path, fps, size, codecs):
    for codec in codecs:
        if not isinstance(codec, str) or len(codec) != 4:
            raise VideoIOError("视频编码器名称必须是四个字符")
        try:
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
            if writer.isOpened():
                return writer, codec
            writer.release()
        except cv2.error:
            # Some backends throw instead of returning an unopened writer.
            pass
        if path.exists():
            path.unlink()
    raise VideoIOError("当前 OpenCV 无可用的 MP4 编码器（已尝试 avc1/mp4v）")


def _verify_video(path, sequence, progress, cancelled):
    capture = _open_capture(path)
    count = 0
    try:
        while True:
            _check_cancelled(cancelled)
            success, image = capture.read()
            if not success:
                break
            if image is None or image.shape[:2] != (sequence.height, sequence.width):
                raise VideoIOError("编码器改变了画面尺寸，不能导出原分辨率视频")
            count += 1
            _progress(progress, count, sequence.frame_count, "正在校验视频与逐帧标注对齐")
        if count != sequence.frame_count:
            raise VideoIOError(f"输出视频帧数不匹配：{count}/{sequence.frame_count}")
    finally:
        capture.release()


def export_tracking(sequence, masks, label, *, export_yolo=False, track_id=1,
                    tracking_metadata=None, progress=None, cancelled=None,
                    codecs=("avc1", "mp4v")):
    """Save every model result, then assemble a chronological video-only MP4.

    ``masks`` is a mapping or an iterable of (frame_index, mask)/TrackingFrame.
    Any order is accepted, allowing seed-to-end then backward propagation.
    Each mask is immediately written to disk, avoiding an in-memory mask movie.
    Missing records, None, and empty masks remain explicitly distinguishable;
    all produce a blank PNG and no invented polygon/bounding box. Predictions
    are never interpolated. Failure removes only this call's private staging
    directory; decoded frames and earlier exports remain intact.
    """
    if not isinstance(label, str) or not label.strip() or "\n" in label or "\r" in label:
        raise VideoIOError("请输入单行目标类别名称")
    label = label.strip()
    if track_id != 1:
        raise VideoIOError("当前单目标导出固定使用 track_id=1")
    if sequence.frame_count <= 0 or len(sequence.frames) != sequence.frame_count:
        raise VideoIOError("视频帧索引不完整")
    if any(frame.frame_index != index for index, frame in enumerate(sequence.frames)):
        raise VideoIOError("原视频帧必须按从 0 开始的连续编号排列")
    metadata = dict(tracking_metadata or {})
    json.dumps(metadata, allow_nan=False)
    _check_cancelled(cancelled)
    stage = Path(tempfile.mkdtemp(prefix=".tracking-export-", dir=sequence.root_dir))
    final_dir = sequence.root_dir / f"tracking_{uuid.uuid4().hex[:12]}"
    writer, iterator = None, None
    try:
        (stage / "masks").mkdir()
        (stage / "labels").mkdir()
        if export_yolo:
            (stage / "yolo").mkdir()
            (stage / "yolo" / "classes.txt").write_text(label + "\n", encoding="utf-8")
        iterator = iter(masks.items() if isinstance(masks, Mapping) else masks)
        records = {}
        _progress(progress, 0, sequence.frame_count, "正在保存模型逐帧预测")
        while True:
            _check_cancelled(cancelled)
            try:
                item = next(iterator)
            except StopIteration:
                break
            index, mask = _mask_record(item, sequence.frame_count)
            if index in records:
                raise VideoIOError(f"模型重复返回帧编号：{index}")
            binary, status, available = _binary_mask(mask, sequence.width, sequence.height)
            records[index] = _save_frame_result(sequence, final_dir, stage,
                                                sequence.frames[index], binary, label,
                                                status, available, export_yolo)
            _progress(progress, len(records), sequence.frame_count, "正在保存模型逐帧预测")
            del item, mask, binary

        predictions_received = len(records)
        for frame in sequence.frames:
            _check_cancelled(cancelled)
            if frame.frame_index not in records:
                blank = np.zeros((sequence.height, sequence.width), dtype=np.uint8)
                records[frame.frame_index] = _save_frame_result(
                    sequence, final_dir, stage, frame, blank, label,
                    "not_provided", False, export_yolo,
                )

        video_name = "overlay.mp4"
        writer, codec = _open_video_writer(stage / video_name, sequence.fps,
                                           (sequence.width, sequence.height), codecs)
        for frame in sequence.frames:
            _check_cancelled(cancelled)
            image = cv2.imread(str(frame.path), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(stage / "masks" / f"{frame.frame_index:05d}.png"), cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape[:2] != (sequence.height, sequence.width) or mask is None:
                raise VideoIOError(f"无法读取对齐的视频帧或 mask：{frame.frame_index}")
            foreground = mask > 0
            if foreground.any():
                color = np.array([50, 220, 50], dtype=np.float32)
                image[foreground] = np.rint(image[foreground] * 0.6 + color * 0.4).astype(np.uint8)
                contours, _ = cv2.findContours(foreground.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image, contours, -1, (50, 220, 50), 2)
            writer.write(image)
            _progress(progress, frame.frame_index + 1, sequence.frame_count, "正在合成轮廓视频（不含音轨）")
        writer.release()
        writer = None
        _verify_video(stage / video_name, sequence, progress, cancelled)
        ordered = [records[index] for index in range(sequence.frame_count)]
        visible = sum(record["visible"] for record in ordered)
        _write_json(stage / "index.json", {
            "source_video": str(sequence.source_video), "width": sequence.width,
            "height": sequence.height, "fps": sequence.fps,
            "frame_count": sequence.frame_count, "track_id": 1, "label": label,
            "timestamp_units": "seconds", "output_timing": "constant_fps",
            "frame_sampling": False, "prediction_interpolation": False,
            "predictions_received": predictions_received, "frames_with_target": visible,
            "video_path": video_name, "codec": codec, "audio_included": False,
            "audio_note": "OpenCV 导出仅包含画面，不保留原视频音轨",
            "raw_mask_values": [0, 255], "polygon_is_approximation": True,
            "tracking_metadata": metadata, "frames": ordered,
        })
        _check_cancelled(cancelled)
        if final_dir.exists() or final_dir.is_symlink():
            raise VideoIOError("新导出目录意外已存在，已停止以避免覆盖")
        os.rename(stage, final_dir)
        _progress(progress, sequence.frame_count, sequence.frame_count, "轮廓视频和逐帧标注已导出（不含音轨）")
        return TrackingExport(final_dir, final_dir / video_name, final_dir / "labels",
                              final_dir / "masks", final_dir / "index.json",
                              final_dir / "yolo" if export_yolo else None,
                              codec, sequence.frame_count, visible)
    except BaseException:
        if writer is not None:
            writer.release()
        if stage.exists():
            shutil.rmtree(stage)
        raise
    finally:
        if iterator is not None and hasattr(iterator, "close"):
            iterator.close()
