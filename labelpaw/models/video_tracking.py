"""Single-object video propagation using local SAM2 weights on CPU or CUDA.

SAM3 may produce the seed mask; propagation is SAM2.1, not SAM3 video inference.
The caller must pause image inference/model loading before starting this backend.
Pass the same ``model_lock`` to every user of a shared SAM3 model. Frame callbacks
run synchronously in the tracking thread, so exporters can persist each mask
without retaining a full video's masks in memory.
"""

from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import tempfile
import threading
import os
import sys

import torch  # Import before Qt on the project's Python 3.10 runtime.
import numpy as np
from PIL import Image
from PySide6.QtCore import QThread, Signal

from labelpaw.models.sam_geometry import mask_to_annotation
from labelpaw.models.sam_prompts import normalize_sam_prompt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "weights/sam_weights/sam2.1_hiera_large.pt"
DEFAULT_SAM3_CHECKPOINT = PROJECT_ROOT / "weights/sam_weights/sam3.pt"
DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
_MODEL_LOCK = threading.RLock()


@dataclass
class VideoTrackingRequest:
    frames_dir: str
    seed_frame_index: int = 0
    seed_mask: object = None
    box_xyxy: object = None
    text_prompt: str = ""
    label: str = "object"
    direction: str = "both"
    seed_source: str = "provided_mask"


@dataclass
class TrackingFrame:
    frame_index: int
    mask: object
    annotation: object
    label: str
    track_id: int = 1


class _Cancelled(Exception):
    pass


class _LazyFrames:
    """Match SAM2's PIL preprocessing while keeping at most two input tensors."""

    def __init__(self, paths, image_size, cache_size=2):
        self.paths = paths
        self.image_size = image_size
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.mean = torch.tensor((0.485, 0.456, 0.406))[:, None, None]
        self.std = torch.tensor((0.229, 0.224, 0.225))[:, None, None]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        with Image.open(self.paths[index]) as image:
            array = np.asarray(image.convert("RGB").resize((self.image_size, self.image_size)), dtype=np.float32) / 255.0
        tensor = (torch.from_numpy(array).permute(2, 0, 1) - self.mean) / self.std
        self.cache[index] = tensor
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return tensor


def _frame_paths(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError("请先将视频解码为逐帧 JPEG 图片目录")
    paths = [path for path in directory.iterdir() if path.suffix.lower() in (".jpg", ".jpeg")]
    if not paths:
        raise ValueError("视频帧目录中没有 JPEG 图片")
    if any(not path.stem.isdecimal() for path in paths):
        raise ValueError("视频帧文件名须为连续的数字，例如 00000.jpg")
    paths.sort(key=lambda path: int(path.stem))
    if [int(path.stem) for path in paths] != list(range(len(paths))):
        raise ValueError("视频帧须从 0 开始连续编号，不能有重复或缺帧")
    return paths


def _box(box, width, height):
    try:
        values = np.asarray(box, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as error:
        raise ValueError("初始选框须包含四个有限数字") from error
    if values.size != 4 or not np.isfinite(values).all():
        raise ValueError("初始选框须包含四个有限数字")
    x1, x2 = np.clip(sorted(values[[0, 2]]), 0, width)
    y1, y2 = np.clip(sorted(values[[1, 3]]), 0, height)
    if x2 - x1 < 2 or y2 - y1 < 2:
        raise ValueError("初始选框须至少为 2 × 2 像素")
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def _numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


class VideoTrackingBackend:
    """Track one seed through a decoded sequence, yielding original-size masks."""

    def __init__(self, sam3_model=None, sam3_processor=None, model_lock=None,
                 checkpoint_path=None, device=None, predictor_factory=None,
                 sam3_checkpoint_path=None, seed_model_factory=None):
        self.sam3_model = sam3_model
        self.sam3_processor = sam3_processor
        self.model_lock = model_lock if model_lock is not None else _MODEL_LOCK
        self.checkpoint_path = Path(checkpoint_path or DEFAULT_CHECKPOINT)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("视频跟踪目前支持 CPU 或 CUDA")
        self.predictor_factory = predictor_factory
        self.sam3_checkpoint_path = Path(sam3_checkpoint_path or DEFAULT_SAM3_CHECKPOINT)
        self.seed_model_factory = seed_model_factory

    def _load_seed_model(self):
        """Load only local SAM3 weights; no Hub access or dependency installation."""
        if self.seed_model_factory is not None:
            return self.seed_model_factory()
        if not self.sam3_checkpoint_path.is_file():
            raise FileNotFoundError(f"找不到本地 SAM3 初始分割权重：{self.sam3_checkpoint_path}")
        repository = str(PROJECT_ROOT / "sam3")
        if repository not in sys.path:
            sys.path.insert(0, repository)
        from sam3 import model_builder
        from sam3.model.sam3_image_processor import Sam3Processor
        bpe = Path(model_builder.__file__).parent / "assets/bpe_simple_vocab_16e6.txt.gz"
        model = model_builder.build_sam3_image_model(
            checkpoint_path=str(self.sam3_checkpoint_path), bpe_path=str(bpe),
            device="cpu", load_from_HF=False, enable_inst_interactivity=True,
        )
        if self.device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
            for parameter in model.parameters():
                if parameter.is_floating_point():
                    parameter.data = parameter.data.to(dtype=dtype)
            model.transformer.decoder.float()
            model = model.to(device=self.device)
        if sys.platform == "win32":
            model.inst_interactive_predictor._transforms.max_hole_area = 0
        return model, Sam3Processor(model, device=self.device)

    def _autocast(self):
        if self.device != "cuda":
            return nullcontext()
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
        return torch.autocast("cuda", dtype=dtype)

    def _load_predictor(self):
        if self.predictor_factory is not None:
            return self.predictor_factory()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"找不到本地视频跟踪权重：{self.checkpoint_path}")
        repository = str(PROJECT_ROOT / "sam2")
        if repository not in sys.path:
            sys.path.insert(0, repository)
        from sam2.build_sam import build_sam2_video_predictor
        # Avoid the optional CUDA connected-components extension on CPU/Windows.
        predictor = build_sam2_video_predictor(
            DEFAULT_CONFIG, str(self.checkpoint_path), device=self.device,
        )
        if self.device == "cpu" or sys.platform == "win32":
            predictor.fill_hole_area = 0
        return predictor

    def _init_state(self, predictor, paths):
        # init_state warms frame 0. Give it one frame rather than letting the
        # upstream loader allocate N*3*1024*1024 FP32 input tensors.
        with tempfile.TemporaryDirectory(prefix="labelpaw-video-init-") as temporary:
            destination = Path(temporary) / "00000.jpg"
            try:
                os.link(paths[0], destination)
            except OSError:
                import shutil
                shutil.copyfile(paths[0], destination)
            state = predictor.init_state(
                video_path=temporary, offload_video_to_cpu=True,
                offload_state_to_cpu=True, async_loading_frames=False,
            )
        state["images"] = _LazyFrames(paths, predictor.image_size)
        state["num_frames"] = len(paths)
        return state

    def _seed(self, request, paths, width, height):
        if request.seed_mask is not None:
            mask = _numpy(request.seed_mask)
            if mask.shape != (height, width):
                raise ValueError("初始轮廓尺寸与视频原始帧尺寸不一致")
            if not np.isfinite(mask).all():
                raise ValueError("初始轮廓包含无效数值")
            mask = mask > 0.5
            if not mask.any():
                raise ValueError("初始轮廓为空，请先选中一个物体")
            return mask, request.seed_source or "provided_mask"
        if self.sam3_model is None or self.sam3_processor is None:
            raise ValueError("选框或文字初始化需要已加载的 SAM3 图像模型；也可传入已有轮廓")
        if request.box_xyxy is None and not request.text_prompt.strip():
            raise ValueError("请提供初始物体轮廓、选框或文字提示")
        box = _box(request.box_xyxy, width, height) if request.box_xyxy is not None else None
        with Image.open(paths[request.seed_frame_index]) as image:
            state = self.sam3_processor.set_image(image.convert("RGB"))
        if box is None:
            output = self.sam3_processor.set_text_prompt(
                prompt=normalize_sam_prompt(request.text_prompt), state=state,
            )
            masks = output.get("masks", [])
            if len(masks) != 1:
                raise ValueError(f"文字提示找到 {len(masks)} 个物体；连续跟踪需要唯一目标，请在初始帧框选一个物体")
            mask = _numpy(masks[0]).reshape(height, width) > 0.5
            if not mask.any():
                raise ValueError("文字提示返回空轮廓，请在初始帧框选一个物体")
            return mask, "sam3_image_text"
        masks, scores, _ = self.sam3_model.predict_inst(
            inference_state=state, box=box, multimask_output=False,
        )
        if len(masks) == 0:
            raise ValueError("初始选框内未找到物体，请调整选框")
        index = int(np.argmax(_numpy(scores)))
        mask = _numpy(masks[index]).reshape(height, width) > 0.5
        if not mask.any():
            raise ValueError("初始选框内未找到物体，请调整选框")
        return mask, "sam3_image_box"

    @staticmethod
    def _prune_memory(predictor, state, frame_index):
        # SAM2 reads a finite window of non-conditioning memories and pointers.
        # Preserve that full window and every conditioning frame (the seed).
        stride = max(1, getattr(predictor, "memory_temporal_stride_for_eval", 1))
        window = max(getattr(predictor, "num_maskmem", 7) * stride + 2,
                     getattr(predictor, "max_obj_ptrs_in_encoder", 16) + 2)
        for outputs in state.get("output_dict_per_obj", {}).values():
            non_cond = outputs.get("non_cond_frame_outputs", {})
            for index in list(non_cond):
                if abs(index - frame_index) > window:
                    del non_cond[index]

    def track(self, request, on_frame=None, on_progress=None, is_cancelled=None):
        """Block until complete/cancelled; callbacks execute in this thread.

        ``on_frame(TrackingFrame)`` runs once for each requested frame, including
        empty masks. Both-direction order is seed..end followed by seed-1..0.
        ``on_progress(done, total, message)`` reports completed unique frames.
        Cancellation is cooperative between model calls, never mid-tensor-op.
        """
        on_frame = on_frame or (lambda frame: None)
        on_progress = on_progress or (lambda done, total, message: None)
        is_cancelled = is_cancelled or (lambda: False)
        paths = _frame_paths(request.frames_dir)
        if isinstance(request.seed_frame_index, bool) or not isinstance(request.seed_frame_index, (int, np.integer)):
            raise ValueError("初始帧索引须为整数")
        seed = int(request.seed_frame_index)
        if not 0 <= seed < len(paths):
            raise ValueError("初始帧索引超出视频范围")
        if request.direction not in ("both", "forward", "backward"):
            raise ValueError("跟踪方向须为 both、forward 或 backward")
        if request.seed_mask is None and request.box_xyxy is None and not request.text_prompt.strip():
            raise ValueError("请提供初始物体轮廓、选框或文字提示")
        with Image.open(paths[0]) as image:
            width, height = image.size
        for path in paths:
            with Image.open(path) as image:
                if image.size != (width, height):
                    raise ValueError("视频各帧尺寸不一致，请重新解码视频")
        if request.seed_mask is None and request.box_xyxy is not None:
            _box(request.box_xyxy, width, height)
        total = len(paths) if request.direction == "both" else (len(paths) - seed if request.direction == "forward" else seed + 1)
        metadata = {"algorithm": "mask_seed_sam2_video", "tracking_model": "SAM2.1 Hiera Large",
                    "seed_source": request.seed_source, "track_id": 1, "label": request.label,
                    "text_prompt": request.text_prompt,
                    "normalized_prompt": normalize_sam_prompt(request.text_prompt) if request.text_prompt.strip() and request.box_xyxy is None and request.seed_mask is None else "",
                    "device": self.device, "checkpoint": str(self.checkpoint_path),
                    "seed_frame_index": seed, "direction": request.direction,
                    "frame_count": len(paths), "width": width, "height": height,
                    "processed_frames": 0, "status": "completed"}
        predictor, state = None, None

        def check_cancel():
            if is_cancelled():
                raise _Cancelled()

        try:
            check_cancel()
            # Timed acquisition makes cancellation possible while another
            # authorised user of the shared model finishes its current call.
            while not self.model_lock.acquire(timeout=0.1):
                check_cancel()
            try:
                check_cancel()
                with torch.inference_mode(), self._autocast():
                    on_progress(0, total, "正在准备初始轮廓")
                    owns_seed_model = False
                    if request.seed_mask is None and (self.sam3_model is None or self.sam3_processor is None):
                        if (self.sam3_model is None) != (self.sam3_processor is None):
                            raise ValueError("复用 SAM3 时须同时提供 model 与 processor")
                        check_cancel()
                        on_progress(0, total, "正在加载本地 SAM3 初始分割模型")
                        self.sam3_model, self.sam3_processor = self._load_seed_model()
                        owns_seed_model = True
                    try:
                        check_cancel()
                        seed_mask, source = self._seed(request, paths, width, height)
                    finally:
                        if owns_seed_model:
                            # The NumPy mask owns its CPU result independently of
                            # the model. Release SAM3 before loading SAM2 on Macs.
                            self.sam3_processor = None
                            self.sam3_model = None
                            import gc
                            gc.collect()
                            if self.device == "cuda":
                                torch.cuda.empty_cache()
                    metadata["seed_source"] = source
                    metadata["algorithm"] = "sam3_image_seed_sam2_video" if source.startswith("sam3") else "mask_seed_sam2_video"
                    check_cancel()
                    on_progress(0, total, "正在加载本地 SAM2.1 视频跟踪模型")
                    predictor = self._load_predictor()
                    check_cancel()
                    state = self._init_state(predictor, paths)
                    check_cancel()
                    predictor.add_new_mask(state, frame_idx=seed, obj_id=1, mask=seed_mask)
                    # Preserve exactly the accepted initial mask at the seed.
                    on_frame(TrackingFrame(seed, seed_mask, mask_to_annotation(seed_mask), request.label))
                    metadata["processed_frames"] = 1
                    on_progress(1, total, f"已跟踪 1 / {total} 帧")
                    for reverse in ([False, True] if request.direction == "both" else [request.direction == "backward"]):
                        check_cancel()
                        if reverse:
                            for outputs in state.get("output_dict_per_obj", {}).values():
                                outputs.get("non_cond_frame_outputs", {}).clear()
                        for frame_index, object_ids, logits in predictor.propagate_in_video(
                                state, start_frame_idx=seed, reverse=reverse):
                            check_cancel()
                            if int(frame_index) == seed:
                                continue
                            ids = list(object_ids)
                            if ids != [1]:
                                raise RuntimeError("视频跟踪返回了非预期的物体编号")
                            mask = _numpy(logits[0]).reshape(height, width) > 0
                            on_frame(TrackingFrame(int(frame_index), mask, mask_to_annotation(mask), request.label))
                            metadata["processed_frames"] += 1
                            on_progress(metadata["processed_frames"], total,
                                        f"已跟踪 {metadata['processed_frames']} / {total} 帧")
                            self._prune_memory(predictor, state, int(frame_index))
                    check_cancel()
            finally:
                try:
                    if predictor is not None and state is not None:
                        predictor.reset_state(state)
                        state.clear()
                finally:
                    self.model_lock.release()
        except _Cancelled:
            metadata["status"] = "cancelled"
        return metadata


class VideoTrackingWorker(QThread):
    frame_ready = Signal(object)
    progress = Signal(int, int, str)
    completed = Signal(dict)
    cancelled = Signal(dict)
    failed = Signal(str)

    def __init__(self, backend, request, on_frame=None, parent=None):
        super().__init__(parent)
        self.backend = backend
        self.request = request
        self.on_frame = on_frame
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()
        self.requestInterruption()

    def run(self):
        def accept_frame(frame):
            if self.on_frame is not None:
                self.on_frame(frame)
            self.frame_ready.emit(frame)
        try:
            result = self.backend.track(self.request, on_frame=accept_frame,
                                        on_progress=self.progress.emit,
                                        is_cancelled=self._cancel.is_set)
            if result["status"] == "cancelled":
                self.cancelled.emit(result)
            else:
                self.completed.emit(result)
        except Exception as error:
            self.failed.emit(str(error))
