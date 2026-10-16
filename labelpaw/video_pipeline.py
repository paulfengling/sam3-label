"""Connect decoded videos, one-object propagation, and persistent exports."""
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

from labelpaw.data.video_io import export_tracking, VideoCancelledError
from labelpaw.models.video_tracking import VideoTrackingBackend, VideoTrackingRequest


def track_sequence(sequence, label, *, seed_frame_index=0, box_xyxy=None,
                   text_prompt=None, export_yolo=True, progress=None,
                   cancelled=None, backend_factory=VideoTrackingBackend):
    """Use disk-backed masks so a long video need not remain in RAM."""
    label = label.strip()
    if not label:
        raise ValueError("请填写目标描述或类别名称")
    cancelled = cancelled or (lambda: False)
    progress = progress or (lambda done, total, message: None)
    if cancelled():
        raise VideoCancelledError("已停止视频跟踪")
    request = VideoTrackingRequest(
        frames_dir=str(sequence.frames_dir), seed_frame_index=seed_frame_index,
        box_xyxy=box_xyxy, text_prompt=text_prompt if text_prompt is not None else label,
        label=label, direction="both",
    )
    backend = backend_factory()
    seen = set()
    unavailable = set()
    progress(0, sequence.frame_count, "正在确定初始目标并加载跟踪模型")
    with tempfile.TemporaryDirectory(prefix=".tracking-", dir=str(sequence.root_dir)) as temporary:
        cache = Path(temporary)

        def save_frame(frame):
            index = int(frame.frame_index)
            if index in seen or not 0 <= index < sequence.frame_count:
                raise ValueError("跟踪返回了重复或越界的帧编号")
            if frame.mask is None:
                unavailable.add(index)
                seen.add(index)
                return
            mask = np.asarray(frame.mask, dtype=np.bool_)
            if mask.shape != (sequence.height, sequence.width):
                raise ValueError("跟踪轮廓与原视频尺寸不一致")
            Image.fromarray(mask.astype(np.uint8) * 255).save(cache / f"{index:05d}.png")
            seen.add(index)

        metadata = backend.track(request, on_frame=save_frame,
                                 on_progress=progress, is_cancelled=cancelled)
        if metadata.get("status") == "cancelled" or cancelled():
            raise VideoCancelledError("已停止视频跟踪，未导出未完成的结果")
        if len(seen) != sequence.frame_count:
            raise ValueError(f"跟踪结果不完整：收到 {len(seen)}/{sequence.frame_count} 帧")

        def masks():
            for index in range(sequence.frame_count):
                if cancelled():
                    raise VideoCancelledError("已停止视频导出")
                if index in unavailable:
                    yield index, None
                    continue
                with Image.open(cache / f"{index:05d}.png") as image:
                    yield index, np.asarray(image) > 0

        return export_tracking(
            sequence, masks(), label, export_yolo=export_yolo,
            tracking_metadata=metadata, progress=progress, cancelled=cancelled,
        )
