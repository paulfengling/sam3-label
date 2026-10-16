# -*- coding: utf-8 -*-
import torch
import numpy as np
import queue
import threading
from PySide6.QtCore import QObject, QThread, Signal
from PIL import Image

import os
import sys
from contextlib import nullcontext
from labelpaw.models.sam_geometry import mask_to_annotation
from labelpaw.models.sam_prompts import normalize_sam_prompt, split_sam_prompts


def _sam_numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _sam_instance_mask(value):
    mask = _sam_numpy(value)
    if mask.ndim < 2:
        raise ValueError("SAM instance mask must have two spatial dimensions")
    return mask.reshape(mask.shape[-2:])


def _sam_box(box_xyxy, width, height):
    """Keep box prompts in source-image pixels, with either drag direction."""
    try:
        box = np.asarray(box_xyxy, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as error:
        raise ValueError("框选坐标应为四个数字") from error
    if box.size != 4 or not np.isfinite(box).all():
        raise ValueError("框选坐标应为四个有限数字")
    if width is None or height is None or width <= 0 or height <= 0:
        raise ValueError("请先打开并载入图片，再框选物体")
    x1, x2 = np.clip(sorted([box[0], box[2]]), 0, width)
    y1, y2 = np.clip(sorted([box[1], box[3]]), 0, height)
    if x2 - x1 < 2 or y2 - y1 < 2:
        raise ValueError("框选范围太小，请拖出至少 2 × 2 像素的区域")
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _queue_sam_task(task_queue, task_lock, task):
    """Coalesce previews while keeping queued confirmation/text work intact."""
    with task_lock:
        try:
            previous = task_queue.get_nowait()
        except queue.Empty:
            previous = None
        is_preview = task[0] == "point" and not task[2]
        if is_preview and previous is not None and (previous[0] == "text" or previous[2]):
            task_queue.put_nowait(previous)
            return False
        task_queue.put_nowait(task)
        return True

# Editable SAM repositories inside the project must resolve to their inner
# Python packages, rather than the outer repository namespace directories.
if not getattr(sys, "frozen", False):
    _source_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for _package_name in ("sam2", "sam3"):
        _repository = os.path.join(_source_root, _package_name)
        _package_dir = os.path.join(_repository, _package_name)
        if os.path.isfile(os.path.join(_package_dir, "__init__.py")):
            if _repository not in sys.path:
                sys.path.insert(0, _repository)


def sam_autocast():
    """Use the precision supported by the current GPU (including RTX 20xx)."""
    if not torch.cuda.is_available():
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)

def sam_device():
    """Keep the CUDA path and support CPU-only/Apple Silicon installations."""
    return "cuda" if torch.cuda.is_available() else "cpu"


if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ============== SAM Model Map ==============
import sys

# 设置一个更加通用的模型权重基准目录
# 优先从当前项目根目录下的 weights 文件夹寻找，如果没有则回退到硬编码路径
if getattr(sys, 'frozen', False):
    PROJECT_ROOT = os.path.dirname(sys.executable)
else:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LOCAL_WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")
HARDCODED_DEV_DIR = r"C:\Users\Paul\Desktop\LabelPaw\weights"

if os.path.exists(LOCAL_WEIGHTS_DIR):
    MODEL_BASE_DIR = LOCAL_WEIGHTS_DIR
elif os.path.exists(HARDCODED_DEV_DIR):
    MODEL_BASE_DIR = HARDCODED_DEV_DIR
else:
    MODEL_BASE_DIR = LOCAL_WEIGHTS_DIR  # 默认指向项目下的 weights

def get_sam_model_map():
    sam_weights_dir = os.path.join(MODEL_BASE_DIR, "sam_weights")
    model_map = {}
    
    if not os.path.exists(sam_weights_dir):
        return model_map
        
    for f in os.listdir(sam_weights_dir):
        if not f.endswith(".pt"):
            continue
            
        path = os.path.join(sam_weights_dir, f)
        key = f.replace(".pt", "")
        size_mb = os.path.getsize(path) // (1024 * 1024)
        size_label = f"{size_mb} MB" if size_mb < 1024 else f"{size_mb/1024:.1f} GB"
        
        # Determine type and config based on filename
        if "sam3" in f.lower():
            model_map[key] = {
                "display_name": "SAM 3" if f.lower() == "sam3.pt" else f"SAM 3 ({key})",
                "type": "sam3",
                "weight": path,
                "config": None,
                "supports_text": True,
                "size_label": size_label,
            }
        else:
            # SAM 2 logic
            config = None
            f_lower = f.lower()
            if "tiny" in f_lower or "_t.pt" in f_lower:
                config = "configs/sam2.1/sam2.1_hiera_t.yaml"
            elif "small" in f_lower or "_s.pt" in f_lower:
                config = "configs/sam2.1/sam2.1_hiera_s.yaml"
            elif "base_plus" in f_lower or "_b+.pt" in f_lower or "base" in f_lower:
                config = "configs/sam2.1/sam2.1_hiera_b+.yaml"
            elif "large" in f_lower or "_l.pt" in f_lower:
                config = "configs/sam2.1/sam2.1_hiera_l.yaml"
                
            model_map[key] = {
                "display_name": key.replace("_", " ").title(),
                "type": "sam2",
                "weight": path,
                "config": config,
                "supports_text": False,
                "size_label": size_label,
            }
    return model_map

SAM_MODEL_MAP = get_sam_model_map()

# ============== SAM 3 Workers ==============
class Sam3ModelLoadWorker(QThread):
    loaded = Signal(object, object, bool, str)

    def __init__(self, checkpoint_path):
        super().__init__()
        self.checkpoint_path = checkpoint_path

    def run(self):
        try:
            from sam3 import model_builder
            from sam3.model.sam3_image_processor import Sam3Processor

            device = sam_device()
            bpe_path = os.path.join(os.path.dirname(model_builder.__file__), "assets", "bpe_simple_vocab_16e6.txt.gz")
            model = model_builder.build_sam3_image_model(
                checkpoint_path=self.checkpoint_path, bpe_path=bpe_path,
                device="cpu", load_from_HF=False, enable_inst_interactivity=True,
            )
            if device == "cuda":
                # Keep weights compact to avoid spilling inference into shared RAM
                # on 6 GB GPUs; autocast handles mixed-precision operations.
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
                for parameter in model.parameters():
                    if parameter.is_floating_point():
                        parameter.data = parameter.data.to(dtype=dtype)
                # Decoder deliberately disables autocast for its coordinate heads.
                model.transformer.decoder.float()
                model = model.to(device=device)
            if sys.platform == "win32":
                # Optional Triton connected-components kernel is unavailable on Windows.
                model.inst_interactive_predictor._transforms.max_hole_area = 0
            processor = Sam3Processor(model, device=device)
            self.loaded.emit(model, processor, True, "模型加载成功")
        except Exception as e:
            self.loaded.emit(None, None, False, str(e))


class Sam3InferenceWorker(QThread):
    result_ready = Signal(list, list, list, float, bool)
    result_ready_labeled = Signal(list, list, list, float, bool, str)
    text_result_ready = Signal(list, str)
    error = Signal(str)

    def __init__(self):
        super().__init__()
        self.model = None
        self.processor = None
        self._inference_state = None
        self._state_generation = 0
        self._semantic_cache = None
        self.task_queue = queue.Queue(maxsize=1)
        self._task_lock = threading.Lock()
        self.running = True

    @property
    def inference_state(self):
        return self._inference_state

    @inference_state.setter
    def inference_state(self, state):
        if state is not self._inference_state:
            self._semantic_cache = None
            self._state_generation += 1
        self._inference_state = state

    def _semantic_instances(self, prompt, state):
        normalized = normalize_sam_prompt(prompt)
        cache = self._semantic_cache
        if cache is not None and cache['state'] is state and cache['prompt'] == normalized:
            return cache['instances']
        with torch.inference_mode(), sam_autocast():
            output = self.processor.set_text_prompt(prompt=normalized, state=state)
        masks = output.get('masks', [])
        scores = output.get('scores', [])
        boxes = output.get('boxes', [])
        instances = [
            {
                'mask': _sam_instance_mask(mask),
                'score': float(_sam_numpy(scores[index])),
                'box': _sam_numpy(boxes[index]).reshape(-1) if index < len(boxes) else None,
            }
            for index, mask in enumerate(masks)
        ]
        # LabelPaw consumes binary masks, not the full-resolution FP32
        # probability maps (several GB on a large group photo).
        output.pop('masks_logits', None)
        if self.inference_state is state:
            self._semantic_cache = {'state': state, 'prompt': normalized, 'instances': instances}
        return instances

    def _point_annotation(self, x, y, prompt, state):
        if prompt and self.processor is not None:
            instances = self._semantic_instances(prompt, state)
            px, py = int(np.floor(x)), int(np.floor(y))
            hits = [instance for instance in instances
                    if 0 <= py < instance['mask'].shape[0]
                    and 0 <= px < instance['mask'].shape[1]
                    and instance['mask'][py, px] > 0.5]
            if hits:
                instance = max(hits, key=lambda item: item['score'])
                annotation = mask_to_annotation(instance['mask'], box=instance['box'])
                if annotation is not None:
                    annotation['score'] = instance['score']
                    return annotation

        # Single-point interaction is ambiguous. Use the same candidate set
        # for preview and confirmation so a click does not silently change it.
        with torch.inference_mode(), sam_autocast():
            masks, scores, _ = self.model.predict_inst(
                inference_state=state, point_coords=np.array([[x, y]]),
                point_labels=np.array([1]), multimask_output=True,
            )
        scores = _sam_numpy(scores)
        if len(scores) == 0:
            return None
        best_index = int(np.argmax(scores))
        annotation = mask_to_annotation(masks[best_index])
        if annotation is None:
            annotation = {'poly_pts': [], 'rect': [], 'obb': []}
        annotation['score'] = float(scores[best_index])
        return annotation

    def _box_annotation(self, box, state):
        # The drawn region is the visual prompt. A class label must not
        # redirect this request to a different text-detected object.
        with torch.inference_mode(), sam_autocast():
            masks, scores, _ = self.model.predict_inst(
                inference_state=state, box=box, multimask_output=False,
            )
        scores = _sam_numpy(scores)
        if len(scores) == 0:
            raise ValueError("框选区域内没有找到可分割的物体，请调整选框")
        best_index = int(np.argmax(scores))
        annotation = mask_to_annotation(masks[best_index])
        if annotation is None:
            raise ValueError("框选区域内没有找到可分割的物体，请调整选框")
        annotation['score'] = float(scores[best_index])
        return annotation

    def run(self):
        while self.running:
            try:
                task_type, data, is_click = self.task_queue.get(timeout=0.05)
                state = self.inference_state
                if self.model is None or state is None:
                    continue

                if task_type in ('point', 'box'):
                    if task_type == 'point':
                        x, y, prompt = data
                        annotation = self._point_annotation(x, y, prompt, state)
                    else:
                        box, prompt, generation = data
                        if generation != self._state_generation:
                            continue
                        annotation = self._box_annotation(box, state)
                    if self.inference_state is not state or annotation is None:
                        continue
                    args = (annotation['poly_pts'], annotation['rect'], annotation['obb'],
                            annotation['score'], is_click)
                    self.result_ready.emit(*args)
                    self.result_ready_labeled.emit(*args, prompt or '')
                elif task_type == 'text':
                    if self.processor is None:
                        continue
                    prompts = split_sam_prompts(data) if isinstance(data, str) else [prompt.strip() for prompt in data if prompt.strip()]
                    results = []
                    for raw_prompt in prompts:
                        raw_prompt = raw_prompt.strip()
                        if not raw_prompt:
                            continue
                        for instance in self._semantic_instances(raw_prompt, state):
                            annotation = mask_to_annotation(instance['mask'], box=instance['box'])
                            if annotation is not None:
                                annotation.update(score=instance['score'], label=raw_prompt)
                                results.append(annotation)
                        if self.inference_state is not state:
                            break
                    if self.inference_state is state:
                        self.text_result_ready.emit(results, ', '.join(prompts))
            except queue.Empty:
                continue
            except Exception as error:
                print(f'SAM3 推理错误: {error}')
                self.error.emit(str(error))

    def request_inference(self, x, y, is_click=False, prompt=None):
        hint = prompt.strip() if prompt else None
        return _queue_sam_task(self.task_queue, self._task_lock, ('point', (x, y, hint), is_click))

    def request_text_inference(self, prompt_text):
        return _queue_sam_task(self.task_queue, self._task_lock, ('text', prompt_text, True))

    def request_box_inference(self, box_xyxy, prompt=None):
        state = self.inference_state
        generation = self._state_generation
        if self.model is None or state is None:
            self.error.emit("请先载入 SAM 模型和图片，再框选物体")
            return False
        try:
            box = _sam_box(box_xyxy, state.get('original_width'), state.get('original_height'))
        except ValueError as error:
            self.error.emit(str(error))
            return False
        label = prompt.strip() if prompt else None
        if self.inference_state is not state or generation != self._state_generation:
            self.error.emit("图片已切换，请重新框选物体")
            return False
        return _queue_sam_task(self.task_queue, self._task_lock,
                               ('box', (box, label, generation), True))

    def stop(self):
        self.running = False
        self.inference_state = None
        self.wait()


# ============== SAM 2.1 Workers ==============
class Sam2ModelLoadWorker(QThread):
    loaded = Signal(object, bool, str)

    def __init__(self, config_file, checkpoint_path):
        super().__init__()
        self.config_file = config_file
        self.checkpoint_path = checkpoint_path

    def run(self):
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            if not self.config_file:
                raise ValueError("无法识别 SAM2 配置，请检查权重文件名")
            model = build_sam2(self.config_file, ckpt_path=self.checkpoint_path, device=sam_device())
            predictor = SAM2ImagePredictor(model)
            self.loaded.emit(predictor, True, "SAM 2.1 模型加载成功")
        except Exception as e:
            self.loaded.emit(None, False, str(e))


class Sam2InferenceWorker(QThread):
    """SAM 2.1 point interaction; semantic prompts are only available in SAM3."""
    result_ready = Signal(list, list, list, float, bool)
    result_ready_labeled = Signal(list, list, list, float, bool, str)
    error = Signal(str)

    def __init__(self):
        super().__init__()
        self.predictor = None
        self._image_set = False
        self._image_generation = 0
        self.task_queue = queue.Queue(maxsize=1)
        self._task_lock = threading.Lock()
        self.running = True

    @property
    def image_set(self):
        return self._image_set

    @image_set.setter
    def image_set(self, ready):
        self._image_set = bool(ready)
        self._image_generation += 1

    def run(self):
        while self.running:
            try:
                task_type, data, is_click = self.task_queue.get(timeout=0.05)
                if self.predictor is None or not self.image_set:
                    continue
                predictor = self.predictor
                generation = self._image_generation
                prompt = None
                if task_type in ('point', 'box'):
                    if task_type == 'point':
                        x, y = data
                        kwargs = {'point_coords': np.array([[x, y]]), 'point_labels': np.array([1]),
                                  'multimask_output': True}
                    else:
                        box, prompt, queued_generation = data
                        if queued_generation != generation:
                            continue
                        kwargs = {'box': box, 'multimask_output': False}
                    with torch.inference_mode(), sam_autocast():
                        masks, scores, _ = predictor.predict(**kwargs)
                    if generation != self._image_generation or self.predictor is not predictor or not self.image_set:
                        continue
                    if len(scores) > 0:
                        best_index = int(np.argmax(scores))
                        annotation = mask_to_annotation(masks[best_index])
                        if annotation is None:
                            if task_type == 'box':
                                raise ValueError("框选区域内没有找到可分割的物体，请调整选框")
                            annotation = {'poly_pts': [], 'rect': [], 'obb': []}
                        args = (annotation['poly_pts'], annotation['rect'], annotation['obb'],
                                float(scores[best_index]), is_click)
                        self.result_ready.emit(*args)
                        self.result_ready_labeled.emit(*args, prompt or '')
                    elif task_type == 'box':
                        raise ValueError("框选区域内没有找到可分割的物体，请调整选框")
            except queue.Empty:
                continue
            except Exception as error:
                print(f'SAM2 推理错误: {error}')
                self.error.emit(str(error))

    def request_inference(self, x, y, is_click=False, prompt=None):
        return _queue_sam_task(self.task_queue, self._task_lock, ('point', (x, y), is_click))

    def request_box_inference(self, box_xyxy, prompt=None):
        predictor = self.predictor
        generation = self._image_generation
        original_sizes = getattr(predictor, '_orig_hw', None)
        if not self.image_set or not original_sizes:
            self.error.emit("请先载入 SAM 模型和图片，再框选物体")
            return False
        height, width = original_sizes[0]
        try:
            box = _sam_box(box_xyxy, width, height)
        except ValueError as error:
            self.error.emit(str(error))
            return False
        label = prompt.strip() if prompt else None
        if self.predictor is not predictor or generation != self._image_generation:
            self.error.emit("图片已切换，请重新框选物体")
            return False
        return _queue_sam_task(self.task_queue, self._task_lock,
                               ('box', (box, label, generation), True))

    def stop(self):
        self.running = False
        self.image_set = False
        self.wait()


# ============== Unified SAM Client ==============
class SAMClient(QObject):
    model_status_changed = Signal(bool, str)
    inference_result = Signal(list, list, list, float, bool)
    inference_result_labeled = Signal(list, list, list, float, bool, str)
    text_result_ready = Signal(list, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = None
        self.processor = None
        self.current_model_key = None   # 当前模型的 key
        self.current_model_type = None  # "sam3" / "sam2"

        # SAM3 worker
        self._sam3_worker = None
        # SAM2 worker
        self._sam2_worker = None
        # 当前活跃的推理 worker
        self._active_worker = None

        self.load_worker = None
        self._load_completed = True

    def load_model_by_key(self, model_key):
        """通过 key 从 SAM_MODEL_MAP 中选择并加载模型"""
        if model_key not in SAM_MODEL_MAP:
            self.model_status_changed.emit(False, f"未知模型: {model_key}")
            return

        # Retain the running loader until it finishes; replacing a live
        # QThread can otherwise terminate the application during a switch.
        if self.load_worker and self.load_worker.isRunning():
            self.model_status_changed.emit(False, "模型仍在加载，请稍候再切换")
            return
        
        # 如果已经是当前模型，不重复加载
        if model_key == self.current_model_key and self._active_worker is not None:
            self.model_status_changed.emit(True, f"{SAM_MODEL_MAP[model_key]['display_name']} 已就绪")
            return

        info = SAM_MODEL_MAP[model_key]
        model_type = info.get("type", "sam2")
        weight_path = info.get("weight")

        # 如果权重文件不存在，提示用户并不继续加载
        if not weight_path or not os.path.exists(weight_path):
            self.model_status_changed.emit(False, f"模型权重文件不存在或未下载")
            return

        # 先停止旧的推理 worker
        self._stop_active_worker()

        self.current_model_key = model_key
        self.current_model_type = model_type
        self.model_status_changed.emit(False, f"正在加载 {info.get('display_name', model_key)}，请稍候...")

        if model_type == "sam3":
            self.load_worker = Sam3ModelLoadWorker(weight_path)
            self._load_completed = False
            self.load_worker.loaded.connect(self._on_sam3_loaded)
            self.load_worker.start()
        elif model_type == "sam2":
            config = info.get("config")
            self.load_worker = Sam2ModelLoadWorker(config, weight_path)
            self._load_completed = False
            self.load_worker.loaded.connect(self._on_sam2_loaded)
            self.load_worker.start()

    # 兼容旧接口
    def load_model_async(self, checkpoint_path):
        """向后兼容：直接加载 SAM3 模型"""
        if self.load_worker and self.load_worker.isRunning():
            self.model_status_changed.emit(False, "模型仍在加载，请稍候再切换")
            return
        self.current_model_key = "sam3"
        self.current_model_type = "sam3"
        self._stop_active_worker()
        self.model_status_changed.emit(False, "正在后台加载模型，请稍候...")
        self.load_worker = Sam3ModelLoadWorker(checkpoint_path)
        self._load_completed = False
        self.load_worker.loaded.connect(self._on_sam3_loaded)
        self.load_worker.start()

    def _stop_active_worker(self):
        """停止当前的推理 worker"""
        if self._sam3_worker:
            self._sam3_worker.stop()
            self._sam3_worker = None
        if self._sam2_worker:
            self._sam2_worker.stop()
            self._sam2_worker = None
        self._active_worker = None
        self.model = None
        self.processor = None

    def _accept_load_result(self):
        # Qt can still deliver an already-queued signal after disconnect().
        # Only the current loader may install a model, once per load request.
        sender = self.sender()
        if sender is None or sender is not self.load_worker or self._load_completed:
            return False
        self._load_completed = True
        return True

    def _on_sam3_loaded(self, model, processor, success, msg):
        if not self._accept_load_result():
            return
        if success:
            self.model = model
            self.processor = processor
            # 启动 SAM3 推理线程
            self._sam3_worker = Sam3InferenceWorker()
            self._sam3_worker.model = model
            self._sam3_worker.processor = processor
            self._sam3_worker.result_ready.connect(self.inference_result)
            self._sam3_worker.result_ready_labeled.connect(self.inference_result_labeled)
            self._sam3_worker.text_result_ready.connect(self.text_result_ready)
            self._sam3_worker.error.connect(lambda msg: self.model_status_changed.emit(False, f"SAM3 推理失败: {msg}"))
            self._sam3_worker.start()
            self._active_worker = self._sam3_worker
        self.model_status_changed.emit(success, msg)

    def _on_sam2_loaded(self, predictor, success, msg):
        if not self._accept_load_result():
            return
        if success:
            self.model = predictor
            self.processor = None  # SAM2 没有文本处理器
            # 启动 SAM2 推理线程
            self._sam2_worker = Sam2InferenceWorker()
            self._sam2_worker.predictor = predictor
            self._sam2_worker.result_ready.connect(self.inference_result)
            self._sam2_worker.result_ready_labeled.connect(self.inference_result_labeled)
            self._sam2_worker.error.connect(lambda msg: self.model_status_changed.emit(False, f"SAM2 推理失败: {msg}"))
            self._sam2_worker.start()
            self._active_worker = self._sam2_worker
        self.model_status_changed.emit(success, msg)

    def set_image(self, image_path):
        """设置当前图像（自动适配 SAM3 / SAM2）"""
        if self.current_model_type == "sam3":
            if not self.processor:
                return
            if self._sam3_worker:
                self._sam3_worker.inference_state = None
            try:
                pil_img = Image.open(image_path).convert("RGB")
                with torch.inference_mode(), sam_autocast():
                    state = self.processor.set_image(pil_img)
                    if self._sam3_worker:
                        self._sam3_worker.inference_state = state
            except Exception as e:
                print(f"SAM3 图像特征提取失败: {e}")
                self.model_status_changed.emit(False, f"SAM3 图像特征提取失败: {e}")

        elif self.current_model_type == "sam2":
            if not self.model:
                return
            if self._sam2_worker:
                self._sam2_worker.image_set = False
            try:
                pil_img = Image.open(image_path).convert("RGB")
                img_np = np.array(pil_img)
                with torch.inference_mode(), sam_autocast():
                    self.model.set_image(img_np)
                    if self._sam2_worker:
                        self._sam2_worker.image_set = True
            except Exception as e:
                print(f"SAM2 图像特征提取失败: {e}")
                self.model_status_changed.emit(False, f"SAM2 图像特征提取失败: {e}")

    def request_inference(self, x, y, is_click, prompt=None):
        if self._active_worker:
            if prompt is None:
                self._active_worker.request_inference(x, y, is_click)
            else:
                self._active_worker.request_inference(x, y, is_click, prompt=prompt)

    def request_text_inference(self, prompt_text):
        """文本提示词推理（仅 SAM3 支持）"""
        if self.current_model_type == "sam3" and self._sam3_worker:
            prompts = split_sam_prompts(prompt_text)
            if prompts:
                self._sam3_worker.request_text_inference(prompts)

    def request_box_inference(self, box_xyxy, prompt=None):
        if self._active_worker is None:
            self.model_status_changed.emit(False, "请先加载 SAM 模型并打开图片，再使用框选")
            return False
        return self._active_worker.request_box_inference(box_xyxy, prompt=prompt)

    def supports_text_prompt(self):
        """当前模型是否支持文本提示词"""
        if self.current_model_key and self.current_model_key in SAM_MODEL_MAP:
            return SAM_MODEL_MAP[self.current_model_key]["supports_text"]
        return False

    def cleanup(self):
        loader = self.load_worker
        self.load_worker = None
        self._load_completed = True
        if loader:
            loader.loaded.disconnect()
            if loader.isRunning():
                loader.wait()
        self._stop_active_worker()
        
        # 卸载模型以释放显存
        self.model = None
        self.processor = None
        self.predictor = None
        
        # 强制垃圾回收和显存清理
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
