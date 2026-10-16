"""Behavioral regression checks for SAM annotation requests without checkpoints."""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np
from PySide6.QtCore import QCoreApplication
from labelpaw.models.sam_client import SAMClient, Sam2InferenceWorker, Sam3InferenceWorker
from labelpaw.models.sam_prompts import normalize_sam_prompt, split_sam_prompts


def instance_output(masks, scores=None, boxes=None):
    masks = np.asarray(masks, dtype=np.bool_)
    return {
        'masks': masks[:, None],
        'scores': np.asarray(scores if scores is not None else [0.96] * len(masks)),
        'boxes': np.asarray(boxes) if boxes is not None else [],
        'masks_logits': np.zeros_like(masks, dtype=np.float32),
    }


class FakeProcessor:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []

    def set_text_prompt(self, prompt, state):
        self.calls.append((prompt, state))
        state.update(self.outputs[prompt])
        return state


class FakeModel:
    def __init__(self, masks, scores):
        self.masks = np.asarray(masks)
        self.scores = np.asarray(scores)
        self.calls = []

    def predict_inst(self, **kwargs):
        self.calls.append(kwargs)
        return self.masks, self.scores, None


class FakePredictor(FakeModel):
    def __init__(self, masks, scores):
        super().__init__(masks, scores)
        self._orig_hw = [(40, 30)]

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return self.masks, self.scores, None


class SamAnnotationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QCoreApplication.instance() or QCoreApplication([])

    def setUp(self):
        self.body = np.zeros((40, 30), dtype=np.bool_)
        self.body[2:38, 8:22] = True
        self.shirt = np.zeros_like(self.body)
        self.shirt[12:25, 9:21] = True

    def worker(self, outputs=None):
        worker = Sam3InferenceWorker()
        worker.model = FakeModel([self.shirt, self.body], [0.99, 0.56])
        worker.processor = FakeProcessor(outputs or {'person': instance_output([self.body])})
        worker.inference_state = {'backbone_out': object(), 'original_width': 30, 'original_height': 40}
        return worker

    def collect_point(self, worker):
        results, labeled, errors = [], [], []
        worker.result_ready.connect(lambda *args: (results.append(args), setattr(worker, 'running', False)))
        worker.result_ready_labeled.connect(lambda *args: labeled.append(args))
        worker.error.connect(lambda error: (errors.append(error), setattr(worker, 'running', False)))
        worker.run()
        self.assertFalse(errors, errors)
        self.assertEqual(len(results), 1)
        return results[0], labeled[0]

    def test_chinese_hint_selects_semantic_body_instead_of_higher_score_shirt(self):
        worker = self.worker()
        worker.request_inference(14, 16, True, prompt='人')
        result, labeled = self.collect_point(worker)
        self.assertEqual(result[1], [8.0, 2.0, 14.0, 36.0])
        self.assertEqual(worker.processor.calls[0][0], 'person')
        self.assertFalse(worker.model.calls)
        self.assertEqual(labeled[-1], '人')
        self.assertEqual(len(result), 5)

    def test_hand_prompt_and_confirmed_typo_use_hand_and_preserve_labels(self):
        hand = np.zeros_like(self.body)
        hand[16:22, 5:11] = True
        for label in ('手掌', '手章'):
            with self.subTest(label=label):
                worker = self.worker({'hand': instance_output([hand])})
                text_results = []
                worker.text_result_ready.connect(
                    lambda *args: (text_results.append(args), setattr(worker, 'running', False))
                )
                worker.request_text_inference(label)
                worker.run()
                self.assertEqual(worker.processor.calls[0][0], 'hand')
                self.assertEqual(text_results[0][0][0]['label'], label)
                worker.running = True
                worker.request_inference(7, 19, True, prompt=label)
                result, labeled = self.collect_point(worker)
                self.assertEqual(result[1], [5.0, 16.0, 6.0, 6.0])
                self.assertEqual(labeled[-1], label)
                self.assertFalse(worker.model.calls)
                self.assertEqual(len(worker.processor.calls), 1)

    def test_bounded_chinese_tape_phrases_preserve_colors_and_hand_relation(self):
        examples = {
            '胶带': 'tape', '胶布': 'tape',
            '白色胶带': 'white tape', '蓝色胶布': 'blue tape',
            '白色的胶带': 'white tape', '蓝胶带': 'blue tape',
            '红色胶带': 'red tape', '黑色胶布': 'black tape',
            '透明胶带': 'transparent tape',
            '人手中拿的胶带': 'tape held in a hand',
            '手中拿着的白色胶带': 'white tape held in a hand',
            '手里拿的蓝色胶带': 'blue tape held in a hand',
            '拿在手里的蓝色的胶布': 'blue tape held in a hand',
            '人手里拿着的透明胶带': 'transparent tape held in a hand',
            '白色胶带卷': 'roll of white tape',
            '蓝色胶带卷': 'roll of blue tape',
            '一卷胶带': 'a roll of tape',
            '手中拿着的一卷蓝色胶带': 'a roll of blue tape held in a hand',
            '人手中拿的 胶带': 'tape held in a hand',
            '白色 胶带': 'white tape',
            '手里拿的 蓝色 胶带': 'blue tape held in a hand',
            '手里拿的 一卷 蓝色 的 胶带': 'a roll of blue tape held in a hand',
        }
        for raw_prompt, expected in examples.items():
            with self.subTest(raw_prompt=raw_prompt):
                self.assertEqual(normalize_sam_prompt(raw_prompt), expected)

    def test_unknown_tape_qualifiers_and_arbitrary_english_are_not_discarded(self):
        unchanged = [
            '深蓝色胶带', '深蓝色 胶带', '手里拿的 深蓝色 胶带',
            '蓝白相间的胶带', '不是白色的胶带', '不要分割蓝色胶带',
            '桌子上的白色胶带', '张三左手中拿的白色胶带', '右手中拿着的胶带',
            '拿在手里的破损白色胶带', '白色胶带盒', '白色胶带和蓝色胶带',
            '一卷胶带卷', '粘在墙上的透明胶带',
            'white tape', 'blue tape', 'roll of white tape',
            'blue tape held by the person on the left', 'WHITE TAPE',
        ]
        for raw_prompt in unchanged:
            with self.subTest(raw_prompt=raw_prompt):
                self.assertEqual(normalize_sam_prompt(raw_prompt), raw_prompt)
        self.assertEqual(normalize_sam_prompt('人体'), 'person')
        self.assertEqual(normalize_sam_prompt('手章'), 'hand')

    def test_worker_translates_tape_model_prompts_and_keeps_original_labels(self):
        raw_prompts = ['人手中拿的 白色 胶带', '蓝色 胶带', '桌子上的 破损 白色 胶带']
        model_prompts = ['white tape held in a hand', 'blue tape', raw_prompts[2]]
        worker = self.worker({prompt: instance_output([self.shirt]) for prompt in model_prompts})
        results, errors = [], []
        worker.text_result_ready.connect(lambda *args: (results.append(args), setattr(worker, 'running', False)))
        worker.error.connect(lambda error: (errors.append(error), setattr(worker, 'running', False)))
        worker.request_text_inference('，'.join(raw_prompts))
        worker.run()
        self.assertFalse(errors, errors)
        self.assertEqual([prompt for prompt, _ in worker.processor.calls], model_prompts)
        self.assertEqual([item['label'] for item in results[0][0]], raw_prompts)
        self.assertEqual(results[0][1], ', '.join(raw_prompts))

    def test_tape_color_and_relation_short_phrases_remain_whole_prompts(self):
        self.assertEqual(
            split_sam_prompts('white tape held in a hand, blue tape，手里拿的白色胶带'),
            ['white tape held in a hand', 'blue tape', '手里拿的白色胶带'],
        )

    def test_semantic_cache_reuses_alias_and_invalidates_for_new_image(self):
        worker = self.worker()
        old_state = worker.inference_state
        worker._point_annotation(14, 16, '人', old_state)
        worker._point_annotation(14, 16, '人体', old_state)
        self.assertEqual(len(worker.processor.calls), 1)
        worker.inference_state = {'backbone_out': object()}
        self.assertIsNone(worker._semantic_cache)
        worker._point_annotation(14, 16, '人', worker.inference_state)
        self.assertEqual(len(worker.processor.calls), 2)
        worker.inference_state = None
        self.assertIsNone(worker._semantic_cache)

    def test_semantic_cache_keeps_only_the_latest_prompt(self):
        worker = self.worker({'person': instance_output([self.body]), 'dog': instance_output([self.body])})
        for prompt in ['人', 'dog', '人']:
            worker._point_annotation(14, 16, prompt, worker.inference_state)
        self.assertEqual([prompt for prompt, _ in worker.processor.calls], ['person', 'dog', 'person'])

    def test_cache_discards_unused_probability_maps_without_copying_masks(self):
        output = instance_output([self.body])
        worker = self.worker({'person': output})
        backbone = worker.inference_state['backbone_out']
        cached = worker._semantic_instances('人', worker.inference_state)
        self.assertNotIn('masks_logits', worker.inference_state)
        self.assertIs(worker.inference_state['backbone_out'], backbone)
        self.assertTrue(np.shares_memory(cached[0]['mask'], output['masks']))

    def test_point_outside_instance_mask_falls_back_even_inside_detector_box(self):
        worker = self.worker({'person': instance_output([self.body], boxes=[[0, 0, 30, 40]])})
        fallback = np.zeros_like(self.body)
        fallback[12:18, 1:5] = True
        worker.model = FakeModel([fallback], [0.88])
        worker.request_inference(2, 15, True, prompt='人')
        result, _ = self.collect_point(worker)
        self.assertEqual(len(worker.model.calls), 1)
        self.assertEqual(result[1], [1.0, 12.0, 4.0, 6.0])

    def test_point_chooses_highest_score_among_instances_covering_the_click(self):
        worker = self.worker({'person': instance_output([self.body, self.shirt], scores=[0.8, 0.95])})
        annotation = worker._point_annotation(14, 16, '人', worker.inference_state)
        self.assertEqual(annotation['score'], 0.95)
        self.assertFalse(worker.model.calls)

    def test_generic_point_keeps_both_disconnected_components(self):
        worker = self.worker()
        mask = np.zeros_like(self.body)
        mask[3:20, 2:8] = True
        mask[25:35, 20:27] = True
        worker.model = FakeModel([mask], [0.9])
        worker.request_inference(4, 10, True)
        result, _ = self.collect_point(worker)
        self.assertEqual(result[1], [2.0, 3.0, 25.0, 32.0])
        self.assertGreater(max(point[0] for point in result[0]), 19)
        self.assertGreater(max(point[1] for point in result[0]), 24)

    def test_semantic_mask_preserves_single_row_spatial_dimension(self):
        mask = np.zeros((1, 30), dtype=np.bool_)
        mask[:, 2:8] = True
        worker = self.worker({'person': instance_output([mask])})
        worker.request_inference(3, 0, True, prompt='人')
        result, _ = self.collect_point(worker)
        self.assertEqual(result[1], [2.0, 0.0, 6.0, 1.0])
        self.assertFalse(worker.model.calls)

    def test_preview_and_confirmation_use_the_same_multimask_candidate_selection(self):
        worker = self.worker()
        results = []
        worker.result_ready.connect(lambda *args: (results.append(args), setattr(worker, 'running', False)))
        for click in [False, True]:
            worker.running = True
            worker.request_inference(14, 16, click)
            worker.run()
        self.assertEqual([call['multimask_output'] for call in worker.model.calls], [True, True])
        self.assertEqual(results[0][:4], results[1][:4])

    def test_hover_cannot_replace_pending_confirmed_click(self):
        worker = self.worker()
        worker.request_inference(14, 16, True, prompt='人')
        self.assertFalse(worker.request_inference(2, 3, False))
        kind, data, click = worker.task_queue.get_nowait()
        self.assertEqual((kind, data, click), ('point', (14, 16, '人'), True))

    def test_hover_cannot_replace_pending_text_request(self):
        worker = self.worker()
        worker.request_text_inference(['人', 'dog'])
        self.assertFalse(worker.request_inference(2, 3, False))
        self.assertEqual(worker.task_queue.get_nowait(), ('text', ['人', 'dog'], True))

    def test_multiple_text_prompts_emit_once_with_original_labels(self):
        dog = np.zeros_like(self.body)
        dog[28:38, 1:6] = True
        worker = self.worker({'person': instance_output([self.body]), 'dog': instance_output([dog])})
        results, errors = [], []
        worker.text_result_ready.connect(lambda *args: (results.append(args), setattr(worker, 'running', False)))
        worker.error.connect(lambda error: (errors.append(error), setattr(worker, 'running', False)))
        worker.request_text_inference('人， dog')
        worker.run()
        self.assertFalse(errors, errors)
        self.assertEqual(len(results), 1)
        annotations, summary = results[0]
        self.assertEqual([annotation['label'] for annotation in annotations], ['人', 'dog'])
        self.assertEqual(summary, '人, dog')
        self.assertEqual([prompt for prompt, _ in worker.processor.calls], ['person', 'dog'])

    def test_empty_last_prompt_does_not_erase_previous_prompt_results(self):
        worker = self.worker({'person': instance_output([self.body]), 'dog': instance_output(np.empty((0, 40, 30)))})
        results = []
        worker.text_result_ready.connect(lambda *args: (results.append(args), setattr(worker, 'running', False)))
        worker.request_text_inference(['人', 'dog'])
        worker.run()
        self.assertEqual(len(results), 1)
        self.assertEqual([annotation['label'] for annotation in results[0][0]], ['人'])

    def test_sam2_confirmation_is_also_protected_from_hover(self):
        worker = Sam2InferenceWorker()
        worker.request_inference(4, 10, True)
        self.assertFalse(worker.request_inference(8, 20, False))
        self.assertEqual(worker.task_queue.get_nowait(), ('point', (4, 10), True))

    def test_sam3_box_sorts_clips_pixels_and_never_queries_text(self):
        worker = self.worker()
        worker.model = FakeModel([self.body], [0.88])
        self.assertTrue(worker.request_box_inference([39, 42, -4, -5], prompt='手掌'))
        result, labeled = self.collect_point(worker)
        call = worker.model.calls[0]
        np.testing.assert_array_equal(call['box'], [0, 0, 30, 40])
        self.assertFalse(call['multimask_output'])
        self.assertNotIn('point_coords', call)
        self.assertFalse(worker.processor.calls)
        self.assertTrue(result[-1])
        self.assertEqual(labeled[-1], '手掌')

    def test_box_result_retains_disconnected_visible_parts(self):
        worker = self.worker()
        mask = np.zeros_like(self.body)
        mask[3:20, 2:8] = True
        mask[25:35, 20:27] = True
        worker.model = FakeModel([mask], [0.91])
        worker.request_box_inference([0, 0, 29, 39])
        result, _ = self.collect_point(worker)
        self.assertEqual(result[1], [2.0, 3.0, 25.0, 32.0])
        self.assertGreater(max(point[0] for point in result[0]), 19)

    def test_box_label_is_captured_before_later_requests(self):
        worker = self.worker()
        worker.model = FakeModel([self.body], [0.9])
        worker.request_box_inference([1, 1, 25, 39], prompt='未知物体')
        self.assertFalse(worker.request_inference(5, 5, False, prompt='人'))
        _, labeled = self.collect_point(worker)
        self.assertEqual(labeled[-1], '未知物体')
        self.assertFalse(worker.processor.calls)

    def test_bad_or_too_small_box_is_rejected_without_model_call(self):
        bad_boxes = [[], [0, 0, 2], [0, 0, float('nan'), 5], [0, 0, float('inf'), 5],
                     [1, 1, 1, 8], [1, 1, 2, 8], [-10, -10, -2, -2]]
        for box in bad_boxes:
            with self.subTest(box=box):
                worker = self.worker()
                errors = []
                worker.error.connect(errors.append)
                self.assertFalse(worker.request_box_inference(box))
                self.assertEqual(len(errors), 1)
                self.assertIn('框选', errors[0])
                self.assertTrue(worker.task_queue.empty())
                self.assertFalse(worker.model.calls)

    def test_box_requires_loaded_image(self):
        worker = self.worker()
        worker.inference_state = None
        errors = []
        worker.error.connect(errors.append)
        self.assertFalse(worker.request_box_inference([0, 0, 20, 20]))
        self.assertIn('载入', errors[0])

    def test_empty_box_mask_reports_adjust_selection(self):
        worker = self.worker()
        worker.model = FakeModel([np.zeros_like(self.body)], [0.7])
        worker.request_box_inference([0, 0, 20, 30])
        results, errors = [], []
        worker.result_ready.connect(lambda *args: results.append(args))
        worker.error.connect(lambda error: (errors.append(error), setattr(worker, 'running', False)))
        worker.run()
        self.assertFalse(results)
        self.assertIn('调整选框', errors[0])

    def test_finished_box_result_from_previous_image_is_not_emitted(self):
        worker = self.worker()
        worker.request_box_inference([0, 0, 20, 30])
        original_predict = worker.model.predict_inst

        def change_image(**kwargs):
            output = original_predict(**kwargs)
            worker.inference_state = {'backbone_out': object(), 'original_width': 30, 'original_height': 40}
            worker.running = False
            return output

        worker.model.predict_inst = change_image
        results = []
        worker.result_ready.connect(lambda *args: results.append(args))
        worker.result_ready_labeled.connect(lambda *args: results.append(args))
        worker.run()
        self.assertFalse(results)

    def test_queued_box_for_previous_image_is_discarded_before_inference(self):
        worker = self.worker()
        worker.request_box_inference([0, 0, 20, 30])
        worker.inference_state = {'backbone_out': object(), 'original_width': 30, 'original_height': 40}
        read_task = worker.task_queue.get

        def one_task(**kwargs):
            task = read_task(**kwargs)
            worker.running = False
            return task

        results = []
        worker.result_ready.connect(lambda *args: results.append(args))
        with patch.object(worker.task_queue, 'get', side_effect=one_task):
            worker.run()
        self.assertFalse(worker.model.calls)
        self.assertFalse(results)

    def test_sam2_box_passes_visual_prompt_and_keeps_task_label(self):
        worker = Sam2InferenceWorker()
        worker.predictor = FakePredictor([self.body], [0.9])
        worker.image_set = True
        worker.request_box_inference([35, 44, -2, -3], prompt='未知物体')
        self.assertFalse(worker.request_inference(14, 16, False))
        result, labeled = self.collect_point(worker)
        call = worker.predictor.calls[0]
        np.testing.assert_array_equal(call['box'], [0, 0, 30, 40])
        self.assertFalse(call['multimask_output'])
        self.assertNotIn('point_coords', call)
        self.assertEqual(labeled[-1], '未知物体')
        self.assertTrue(result[-1])

    def test_sam2_box_result_after_image_change_is_discarded(self):
        worker = Sam2InferenceWorker()
        worker.predictor = FakePredictor([self.body], [0.9])
        worker.image_set = True
        worker.request_box_inference([0, 0, 20, 30])
        original_predict = worker.predictor.predict

        def change_image(**kwargs):
            output = original_predict(**kwargs)
            worker.image_set = False
            worker.image_set = True
            worker.running = False
            return output

        worker.predictor.predict = change_image
        results = []
        worker.result_ready.connect(lambda *args: results.append(args))
        worker.result_ready_labeled.connect(lambda *args: results.append(args))
        worker.run()
        self.assertFalse(results)

    def test_client_forwards_box_and_reports_no_loaded_model(self):
        client = SAMClient()
        statuses = []
        client.model_status_changed.connect(lambda *args: statuses.append(args))
        self.assertFalse(client.request_box_inference([0, 0, 20, 30]))
        self.assertIn('加载', statuses[0][1])
        worker = Mock()
        worker.request_box_inference.return_value = True
        client._active_worker = worker
        self.assertTrue(client.request_box_inference([2, 3, 20, 30], prompt='物体'))
        worker.request_box_inference.assert_called_once_with([2, 3, 20, 30], prompt='物体')


if __name__ == '__main__':
    unittest.main()
