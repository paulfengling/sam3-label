"""Pipeline alignment, cancellation, and missing-prediction contracts."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import numpy as np
from labelpaw.data.video_io import VideoCancelledError
from labelpaw.video_pipeline import track_sequence


class VideoPipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.sequence = SimpleNamespace(root_dir=self.root, frames_dir=self.root,
                                        width=20, height=12, frame_count=3)
        self.mask = np.zeros((12, 20), dtype=bool)
        self.mask[3:9, 4:15] = True

    def backend(self, items, status="completed"):
        def track(request, on_frame, on_progress, is_cancelled):
            self.request = request
            for index, mask in items:
                on_frame(SimpleNamespace(frame_index=index, mask=mask))
            return {"status": status, "algorithm": "contract-test"}
        return lambda: SimpleNamespace(track=track)

    def test_out_of_order_predictions_and_unavailable_masks_export_in_order(self):
        def export(sequence, masks, label, **kwargs):
            records = list(masks)
            self.assertEqual([index for index, _ in records], [0, 1, 2])
            np.testing.assert_array_equal(records[0][1], self.mask)
            self.assertIsNone(records[1][1])
            self.assertFalse(records[2][1].any())
            self.assertEqual(kwargs["tracking_metadata"]["algorithm"], "contract-test")
            self.assertEqual(label, "胶带")
            return "exported"
        with patch("labelpaw.video_pipeline.export_tracking", side_effect=export):
            result = track_sequence(
                self.sequence, "胶带", seed_frame_index=2, box_xyxy=[1, 2, 19, 10],
                backend_factory=self.backend([(2, np.zeros_like(self.mask)), (1, None), (0, self.mask)]),
            )
        self.assertEqual(result, "exported")
        self.assertEqual(self.request.seed_frame_index, 2)
        self.assertEqual(self.request.direction, "both")
        self.assertEqual(list(self.root.iterdir()), [])

    def test_missing_or_duplicate_frames_do_not_produce_successful_export(self):
        cases = [[(0, self.mask)], [(0, self.mask), (0, self.mask)]]
        for items in cases:
            with self.subTest(items=len(items)), patch("labelpaw.video_pipeline.export_tracking") as export:
                with self.assertRaises(ValueError):
                    track_sequence(self.sequence, "胶带", backend_factory=self.backend(items))
                export.assert_not_called()
                self.assertEqual(list(self.root.iterdir()), [])

    def test_cancelled_backend_cleans_cache_without_publishing_partial_video(self):
        with patch("labelpaw.video_pipeline.export_tracking") as export:
            with self.assertRaises(VideoCancelledError):
                track_sequence(self.sequence, "胶带", backend_factory=self.backend([(0, self.mask)], "cancelled"))
            export.assert_not_called()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_wrong_mask_size_rejected_before_export(self):
        with self.assertRaisesRegex(ValueError, "尺寸"):
            track_sequence(self.sequence, "胶带", backend_factory=self.backend([(0, self.mask[:2])]))
        self.assertEqual(list(self.root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
