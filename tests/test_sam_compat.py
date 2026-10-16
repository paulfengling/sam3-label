"""Small CPU compatibility checks; real checkpoints are covered by smoke_sam*.py."""
import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from PySide6.QtCore import QCoreApplication, Qt
from labelpaw.models.sam_client import SAMClient, Sam2ModelLoadWorker, Sam3ModelLoadWorker
from sam3.model.decoder import TransformerDecoder, TransformerDecoderLayer
from sam3.model.geometry_encoders import SequenceGeometryEncoder
from sam3.model.position_encoding import PositionEmbeddingSine
from sam3.model.sam3_tracker_utils import sample_one_point_from_error_center
from sam3.model.vitdet import Mlp
from sam3.perflib.connected_components import connected_components


class SamCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QCoreApplication.instance() or QCoreApplication([])

    def test_sam3_import_without_triton(self):
        script = """
import builtins, sys
sys.path.insert(0, 'sam3')
original_import = builtins.__import__
def no_triton(name, *args, **kwargs):
    if name == 'triton' or name.startswith('triton.'):
        raise ModuleNotFoundError('Triton intentionally unavailable')
    return original_import(name, *args, **kwargs)
builtins.__import__ = no_triton
from sam3.model_builder import build_sam3_image_model
"""
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=PROJECT_ROOT,
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_sam2_loader_selects_cpu_without_cuda(self):
        calls = []
        fake_model = object()
        fake_predictor = object()
        modules = {
            "sam2.build_sam": SimpleNamespace(
                build_sam2=lambda config, **kwargs: (calls.append(kwargs), fake_model)[1]
            ),
            "sam2.sam2_image_predictor": SimpleNamespace(
                SAM2ImagePredictor=lambda model: fake_predictor
            ),
        }
        loaded = []
        worker = Sam2ModelLoadWorker("config.yaml", "checkpoint.pt")
        worker.loaded.connect(lambda *args: loaded.append(args))
        with patch.dict(sys.modules, modules), patch("torch.cuda.is_available", return_value=False):
            worker.run()
        self.assertEqual(calls[0]["device"], "cpu")
        self.assertIs(loaded[0][0], fake_predictor)
        self.assertTrue(loaded[0][1], loaded[0][2])

    def test_sam3_missing_dependency_is_reported(self):
        import builtins

        original_import = builtins.__import__

        def missing_dependency(name, *args, **kwargs):
            if name == "sam3":
                raise ModuleNotFoundError("No module named 'missing_sam3_dependency'")
            return original_import(name, *args, **kwargs)

        loaded = []
        worker = Sam3ModelLoadWorker("checkpoint.pt")
        worker.loaded.connect(lambda *args: loaded.append(args))
        with patch("builtins.__import__", side_effect=missing_dependency):
            worker.run()
        self.assertFalse(loaded[0][2])
        self.assertIn("missing_sam3_dependency", loaded[0][3])

    def test_position_cache_builds_on_cpu_and_follows_input_device(self):
        position = PositionEmbeddingSine(num_pos_feats=16, precompute_resolution=112)
        encoded = position(torch.zeros(2, 1, 8, 8))
        self.assertEqual(encoded.shape, (2, 16, 8, 8))
        self.assertTrue(torch.isfinite(encoded).all())
        # A meta tensor exercises cache migration without requiring a GPU.
        encoded = position(torch.empty(2, 1, 8, 8, device="meta"))
        self.assertEqual(encoded.device.type, "meta")

    def test_decoder_coordinate_cache_follows_model_device(self):
        layer = TransformerDecoderLayer(
            activation="relu", d_model=16, dim_feedforward=32, dropout=0,
            cross_attention=torch.nn.MultiheadAttention(16, 4), n_heads=4,
        )
        decoder = TransformerDecoder(
            d_model=16, frozen=False, interaction_layer=None, layer=layer,
            num_layers=1, num_queries=2, return_intermediate=True,
            box_refine=True, boxRPB="log", resolution=16, stride=4,
        ).eval()
        boxes = torch.full((2, 1, 4), 0.5)
        bias = decoder._get_rpb_matrix(boxes, (4, 4))
        self.assertTrue(torch.isfinite(bias).all())
        decoder.to("meta")
        bias = decoder._get_rpb_matrix(boxes.to("meta"), (4, 4))
        self.assertEqual(bias.device.type, "meta")

    def test_cpu_components_use_eight_connectivity_and_zero_background(self):
        mask = torch.tensor([[[[1, 0, 0], [0, 1, 0], [0, 0, 0]]]], dtype=torch.uint8)
        labels, counts = connected_components(mask)
        self.assertEqual(labels.shape, mask.shape)
        self.assertEqual(labels[0, 0, 0, 0], labels[0, 0, 1, 1])
        np.testing.assert_array_equal(counts.numpy(), [[[[2, 0, 0], [0, 2, 0], [0, 0, 0]]]])

    def test_cpu_error_point_targets_region_center(self):
        mask = torch.zeros(1, 1, 9, 9, dtype=torch.bool)
        mask[:, :, 2:7, 2:7] = True
        point, label = sample_one_point_from_error_center(mask, None)
        self.assertEqual(point.tolist(), [[[4.0, 4.0]]])
        self.assertEqual(label.tolist(), [[1]])
        point, label = sample_one_point_from_error_center(torch.zeros_like(mask), mask)
        self.assertEqual(point.tolist(), [[[4.0, 4.0]]])
        self.assertEqual(label.tolist(), [[0]])

    def test_cpu_vision_mlp_preserves_float32(self):
        mlp = Mlp(in_features=16, hidden_features=32).eval()
        image_features = torch.randn(2, 4, 16)
        with torch.inference_mode():
            output = mlp(image_features)
        self.assertEqual(output.dtype, torch.float32)
        self.assertEqual(output.shape, image_features.shape)
        self.assertTrue(torch.isfinite(output).all())

    def test_cpu_geometry_pooling_does_not_use_accelerator_pinned_memory(self):
        encoder = SequenceGeometryEncoder(
            encode_boxes_as_points=False, points_direct_project=True,
            points_pool=False, points_pos_enc=False, boxes_direct_project=False,
            boxes_pool=True, boxes_pos_enc=False, d_model=8, pos_enc=None,
            num_layers=0, layer=None, roi_size=2,
        ).eval()
        boxes = torch.full((1, 1, 4), 0.5)
        mask = torch.zeros(1, 1, dtype=torch.bool)
        labels = torch.ones(1, 1, dtype=torch.long)
        image_features = torch.randn(1, 8, 8, 8)
        with torch.inference_mode(), patch.object(
            torch.Tensor, "pin_memory", side_effect=AssertionError("CPU inference must not probe an accelerator")
        ):
            encoded, _ = encoder._encode_boxes(boxes, mask, labels, image_features)
        self.assertEqual(encoded.device.type, "cpu")
        self.assertEqual(encoded.shape, (1, 1, 8))
        self.assertTrue(torch.isfinite(encoded).all())

    def test_cleanup_ignores_already_queued_load_result(self):
        client = SAMClient()
        loader = Sam2ModelLoadWorker("config.yaml", "checkpoint.pt")
        client.load_worker = loader
        client._load_completed = False
        loader.loaded.connect(client._on_sam2_loaded, Qt.ConnectionType.QueuedConnection)
        loader.loaded.emit(object(), True, "loaded")
        try:
            client.cleanup()
            QCoreApplication.processEvents()
            self.assertIsNone(client.model)
            self.assertIsNone(client._active_worker)
        finally:
            client.cleanup()

    def test_replaced_loader_ignores_old_queued_result(self):
        client = SAMClient()
        old_loader = Sam2ModelLoadWorker("old.yaml", "old.pt")
        new_loader = Sam2ModelLoadWorker("new.yaml", "new.pt")
        old_loader.loaded.connect(client._on_sam2_loaded, Qt.ConnectionType.QueuedConnection)
        new_loader.loaded.connect(client._on_sam2_loaded, Qt.ConnectionType.QueuedConnection)
        client.load_worker = old_loader
        client._load_completed = False
        old_loader.loaded.emit(object(), True, "old model loaded")
        client.load_worker = new_loader
        try:
            QCoreApplication.processEvents()
            self.assertIsNone(client.model)
            self.assertIsNone(client._active_worker)
        finally:
            client.cleanup()


if __name__ == "__main__":
    unittest.main()
