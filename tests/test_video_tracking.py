"""Checkpoint-free contracts for single-object video propagation."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock

import torch
import numpy as np
from PIL import Image
from PySide6.QtCore import QCoreApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from labelpaw.models.video_tracking import (
    VideoTrackingBackend, VideoTrackingRequest, VideoTrackingWorker, _LazyFrames,
)


class FakePredictor:
    image_size = 8

    def __init__(self, masks):
        self.masks = masks
        self.calls = []
        self.reset = False

    def init_state(self, **kwargs):
        self.calls.append(("init", kwargs))
        return {"output_dict_per_obj": {0: {"non_cond_frame_outputs": {}}}}

    def add_new_mask(self, state, **kwargs):
        self.calls.append(("seed", kwargs))

    def propagate_in_video(self, state, start_frame_idx, reverse):
        self.calls.append(("propagate", start_frame_idx, reverse))
        frames = range(start_frame_idx, -1, -1) if reverse else range(start_frame_idx, len(self.masks))
        for index in frames:
            yield index, [1], self.masks[index][None, None].astype(np.float32) * 2 - 1

    def reset_state(self, state):
        self.reset = True


class VideoTrackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QCoreApplication.instance() or QCoreApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.paths = []
        self.masks = []
        for index in range(4):
            path = Path(self.temporary.name) / f"{index:05}.jpg"
            Image.new("RGB", (20, 12), (index * 30, 60, 90)).save(path)
            self.paths.append(path)
            mask = np.zeros((12, 20), dtype=bool)
            mask[2:8, 3 + index:9 + index] = True
            self.masks.append(mask)
        self.predictor = FakePredictor(self.masks)
        self.backend = VideoTrackingBackend(device="cpu", predictor_factory=lambda: self.predictor)

    def request(self, **kwargs):
        values = dict(frames_dir=self.temporary.name, seed_frame_index=2,
                      seed_mask=self.masks[2], label="胶带")
        values.update(kwargs)
        return VideoTrackingRequest(**values)

    def test_both_directions_once_per_frame_and_single_track_identity(self):
        frames, progress = [], []
        result = self.backend.track(self.request(), on_frame=frames.append,
                                    on_progress=lambda *args: progress.append(args))
        self.assertEqual([frame.frame_index for frame in frames], [2, 3, 1, 0])
        self.assertEqual({frame.track_id for frame in frames}, {1})
        self.assertEqual({frame.label for frame in frames}, {"胶带"})
        self.assertEqual(result["processed_frames"], 4)
        self.assertEqual(progress[-1][:2], (4, 4))
        self.assertTrue(self.predictor.reset)
        np.testing.assert_array_equal(frames[0].mask, self.masks[2])
        self.assertEqual(self.predictor.calls[1][1]["obj_id"], 1)

    def test_forward_and_backward_ranges(self):
        for direction, expected in (("forward", [2, 3]), ("backward", [2, 1, 0])):
            frames = []
            result = self.backend.track(self.request(direction=direction), on_frame=frames.append)
            self.assertEqual([frame.frame_index for frame in frames], expected)
            self.assertEqual(result["processed_frames"], len(expected))

    def test_empty_propagated_mask_is_reported_without_inventing_geometry(self):
        self.masks[3][:] = False
        frames = []
        self.backend.track(self.request(direction="forward"), on_frame=frames.append)
        self.assertFalse(frames[1].mask.any())
        self.assertIsNone(frames[1].annotation)

    def test_cancel_between_frames_releases_state(self):
        frames = []
        result = self.backend.track(self.request(), on_frame=frames.append,
                                    is_cancelled=lambda: len(frames) >= 2)
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(result["processed_frames"], 2)
        self.assertTrue(self.predictor.reset)

    def test_cancel_before_loading_does_not_load_predictor(self):
        factory = Mock(side_effect=AssertionError("must not load"))
        self.backend.predictor_factory = factory
        result = self.backend.track(self.request(), is_cancelled=lambda: True)
        self.assertEqual(result["processed_frames"], 0)
        self.assertEqual(result["status"], "cancelled")
        factory.assert_not_called()

    def test_can_cancel_while_waiting_for_shared_model_lock(self):
        lock = threading.Lock()
        lock.acquire()
        self.addCleanup(lock.release)
        self.backend.model_lock = lock
        checks = []
        result = self.backend.track(self.request(), is_cancelled=lambda: (checks.append(True) or len(checks) >= 3))
        self.assertEqual(result["status"], "cancelled")

    def test_sam3_visual_box_is_clipped_and_never_uses_text(self):
        processor = Mock()
        processor.set_image.return_value = {"image": 1}
        model = Mock()
        model.predict_inst.return_value = (self.masks[2][None], np.array([0.8]), None)
        self.backend.sam3_model, self.backend.sam3_processor = model, processor
        result = self.backend.track(self.request(seed_mask=None, box_xyxy=[25, 15, 1, -4], text_prompt="白色胶带"))
        np.testing.assert_array_equal(model.predict_inst.call_args.kwargs["box"], [1, 0, 20, 12])
        self.assertFalse(model.predict_inst.call_args.kwargs["multimask_output"])
        processor.set_text_prompt.assert_not_called()
        self.assertEqual(result["algorithm"], "sam3_image_seed_sam2_video")
        self.assertEqual(result["seed_source"], "sam3_image_box")
        self.assertEqual(result["label"], "胶带")
        self.assertEqual(result["normalized_prompt"], "")
        self.assertIs(self.backend.sam3_model, model)
        self.assertIs(self.backend.sam3_processor, processor)

    def test_owned_sam3_model_is_released_before_sam2_loading(self):
        processor, model = Mock(), Mock()
        processor.set_text_prompt.return_value = {"masks": self.masks[2][None, None]}
        factory = Mock(return_value=(model, processor))
        self.backend.seed_model_factory = factory
        def load_tracker():
            self.assertIsNone(self.backend.sam3_model)
            self.assertIsNone(self.backend.sam3_processor)
            return self.predictor
        self.backend.predictor_factory = load_tracker
        result = self.backend.track(self.request(seed_mask=None, text_prompt="胶带"))
        factory.assert_called_once_with()
        self.assertEqual(result["seed_source"], "sam3_image_text")

    def test_cancel_after_owned_sam3_load_releases_it_without_loading_tracker(self):
        self.backend.seed_model_factory = Mock(return_value=(Mock(), Mock()))
        self.backend.predictor_factory = Mock(side_effect=AssertionError("must not load"))
        result = self.backend.track(self.request(seed_mask=None, text_prompt="胶带"),
                                    is_cancelled=lambda: self.backend.sam3_model is not None)
        self.assertEqual(result["status"], "cancelled")
        self.assertIsNone(self.backend.sam3_model)
        self.backend.predictor_factory.assert_not_called()

    def test_text_seed_normalizes_prompt_and_preserves_label(self):
        processor, model = Mock(), Mock()
        processor.set_text_prompt.return_value = {"masks": self.masks[2][None, None]}
        self.backend.sam3_model, self.backend.sam3_processor = model, processor
        result = self.backend.track(self.request(seed_mask=None, text_prompt="蓝色胶带", label="蓝色胶带"))
        self.assertEqual(processor.set_text_prompt.call_args.kwargs["prompt"], "blue tape")
        self.assertEqual(result["normalized_prompt"], "blue tape")
        self.assertEqual(result["label"], "蓝色胶带")
        self.assertEqual(result["seed_source"], "sam3_image_text")
        model.predict_inst.assert_not_called()

    def test_ambiguous_or_missing_text_target_requests_visual_selection(self):
        for count in (0, 2):
            processor = Mock()
            processor.set_text_prompt.return_value = {"masks": np.zeros((count, 1, 12, 20))}
            self.backend.sam3_model, self.backend.sam3_processor = Mock(), processor
            with self.assertRaisesRegex(ValueError, f"{count} 个物体.*框选"):
                self.backend.track(self.request(seed_mask=None, text_prompt="胶带"))
            self.assertFalse(self.predictor.calls)

    def test_invalid_seed_mask_is_rejected_before_loading(self):
        for mask in (np.zeros((12, 20)), np.ones((11, 20)), np.full((12, 20), np.nan)):
            with self.assertRaises(ValueError):
                self.backend.track(self.request(seed_mask=mask))
        self.assertFalse(self.predictor.calls)

    def test_bad_seed_indices_directions_and_small_boxes_are_rejected(self):
        for index in (-1, 4, 1.5, True):
            with self.assertRaises(ValueError):
                self.backend.track(self.request(seed_frame_index=index))
        with self.assertRaises(ValueError):
            self.backend.track(self.request(direction="random"))
        self.backend.sam3_model, self.backend.sam3_processor = Mock(), Mock()
        with self.assertRaisesRegex(ValueError, "2 × 2"):
            self.backend.track(self.request(seed_mask=None, box_xyxy=[1, 1, 2, 2]))

    def test_missing_or_non_contiguous_frames_are_rejected(self):
        self.paths[1].unlink()
        with self.assertRaisesRegex(ValueError, "缺帧"):
            self.backend.track(self.request())

    def test_lazy_input_cache_stays_bounded_and_uses_sam2_normalization(self):
        images = _LazyFrames(self.paths, image_size=8)
        for index in range(len(self.paths)):
            tensor = images[index]
            self.assertEqual(tuple(tensor.shape), (3, 8, 8))
            self.assertEqual(tensor.dtype, torch.float32)
            self.assertLessEqual(len(images.cache), 2)
        with Image.open(self.paths[-1]) as image:
            pixels = np.asarray(image.convert("RGB").resize((8, 8))) / 255.0
        expected = (torch.tensor(pixels).permute(2, 0, 1).float() - images.mean) / images.std
        torch.testing.assert_close(tensor, expected)

    def test_finite_memory_pruning_keeps_conditioning_and_required_history(self):
        seed_output = object()
        state = {"output_dict_per_obj": {0: {
            "cond_frame_outputs": {2: seed_output},
            "non_cond_frame_outputs": {index: object() for index in range(40)},
        }}}
        self.backend._prune_memory(self.predictor, state, 39)
        outputs = state["output_dict_per_obj"][0]
        self.assertEqual(outputs["cond_frame_outputs"], {2: seed_output})
        self.assertEqual(min(outputs["non_cond_frame_outputs"]), 21)

    def test_worker_stream_callback_and_signals_and_cancellation(self):
        frames, completed, signalled = [], [], []
        worker = VideoTrackingWorker(self.backend, self.request(), on_frame=frames.append)
        worker.completed.connect(completed.append)
        worker.frame_ready.connect(signalled.append)
        worker.run()
        self.assertEqual(len(frames), 4)
        self.assertEqual(len(signalled), 4)
        self.assertEqual(completed[0]["status"], "completed")
        worker = VideoTrackingWorker(self.backend, self.request())
        cancelled = []
        worker.cancelled.connect(cancelled.append)
        worker.cancel()
        worker.run()
        self.assertEqual(cancelled[0]["status"], "cancelled")

    def test_worker_reports_error_and_model_lock_is_released(self):
        backend = self.backend
        backend._load_predictor = Mock(side_effect=RuntimeError("failed to load"))
        failures = []
        worker = VideoTrackingWorker(backend, self.request())
        worker.failed.connect(failures.append)
        worker.run()
        self.assertEqual(failures, ["failed to load"])
        self.assertTrue(backend.model_lock.acquire(blocking=False))
        backend.model_lock.release()


if __name__ == "__main__":
    unittest.main()
