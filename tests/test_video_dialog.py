"""Exercise real video preparation and reference-frame selection through Qt."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import time

import torch
import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from labelpaw.video_dialog import VideoTrackingDialog, VideoSeedView, VideoJob


class VideoDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.image = self.root / "frame.png"
        Image.new("RGB", (200, 120), "gray").save(self.image)

    def wait_for_job(self, dialog):
        deadline = time.monotonic() + 5
        while dialog.job is not None and time.monotonic() < deadline:
            self.app.processEvents()
            QTest.qWait(5)
        self.assertIsNone(dialog.job, dialog.status.text())

    def test_drag_box_uses_original_image_coordinates_at_view_scale(self):
        view = VideoSeedView()
        self.addCleanup(view.close)
        view.resize(700, 420)
        view.show()
        view.set_frame(self.image)
        self.app.processEvents()
        start = view.mapFromScene(150, 90)
        end = view.mapFromScene(30, 20)
        QTest.mousePress(view.viewport(), Qt.LeftButton, pos=start)
        QTest.mouseMove(view.viewport(), end)
        QTest.mouseRelease(view.viewport(), Qt.LeftButton, pos=end)
        np.testing.assert_allclose(view.box, [30, 20, 150, 90], atol=1)
        view.set_frame(self.image)
        self.assertIsNone(view.box)

    def test_prepare_choose_middle_frame_track_and_review_mask(self):
        video = self.root / "source.mp4"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5, (200, 120))
        self.assertTrue(writer.isOpened())
        for index in range(3):
            writer.write(np.full((120, 200, 3), index * 40, dtype=np.uint8))
        writer.release()
        dialog = VideoTrackingDialog()
        self.addCleanup(dialog.close)
        dialog.output_path.setText(str(self.root / "output"))
        dialog.show()
        dialog.prepare_video(video)
        self.wait_for_job(dialog)
        self.assertEqual(dialog.sequence.frame_count, 3)
        self.assertTrue(dialog.start_button.isEnabled())
        dialog.frame_slider.setValue(1)
        dialog.view.box = [20, 15, 90, 80]
        mask_dir = self.root / "test-masks"
        mask_dir.mkdir()
        Image.new("L", (200, 120), 255).save(mask_dir / "00001.png")
        result = SimpleNamespace(frame_count=3, frames_with_target=3,
                                 output_dir=self.root, video_path=video, masks_dir=mask_dir)
        with patch("labelpaw.video_dialog.track_sequence", return_value=result) as track:
            dialog.start_tracking()
            self.wait_for_job(dialog)
        self.assertEqual(track.call_args.kwargs["seed_frame_index"], 1)
        self.assertEqual(track.call_args.kwargs["box_xyxy"], [20, 15, 90, 80])
        self.assertTrue(dialog.video_button.isEnabled())
        self.assertIsNotNone(dialog.view._mask_item)
        dialog.frame_slider.setValue(2)
        self.assertIsNone(dialog.view.box)
        self.assertIsNone(dialog.view._mask_item)

    def test_application_shutdown_cancels_and_joins_active_worker(self):
        import threading
        entered = threading.Event()

        def operation(progress, cancelled):
            entered.set()
            while not cancelled():
                time.sleep(0.005)
            return None

        dialog = VideoTrackingDialog()
        self.addCleanup(dialog.close)
        job = VideoJob(operation, dialog)
        dialog.job = job
        job.start()
        self.assertTrue(entered.wait(2))
        dialog.shutdown()
        self.assertTrue(job.cancel_event.is_set())
        self.assertFalse(job.isRunning())
        dialog.job = None


if __name__ == "__main__":
    unittest.main()
