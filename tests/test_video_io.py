"""Actual short-video I/O checks, independent of SAM weights and Qt."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labelpaw.data.video_io import (
    VideoCancelledError, VideoIOError, decode_video, export_tracking, probe_video,
)


@dataclass
class TrackingFrame:
    frame_index: int
    mask: object
    track_id: int = 1


class VideoIOTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.source = self.directory / "moving_ring.mp4"
        self.output = self.directory / "user_outputs"
        self.output.mkdir()
        self.user_file = self.output / "user_photo.jpg"
        self.user_file.write_bytes(b"existing user file must remain unchanged")
        self.width, self.height, self.count, self.fps = 192, 128, 8, 12.5
        writer = cv2.VideoWriter(str(self.source), cv2.VideoWriter_fourcc(*"mp4v"), self.fps,
                                 (self.width, self.height))
        if not writer.isOpened():
            writer.release()
            self.skipTest("The environment has no MP4 writer")
        self.masks = {}
        for index in range(self.count):
            mask = np.zeros((self.height, self.width), dtype=np.uint8)
            center = (28 + index * 16, 65)
            cv2.circle(mask, center, 16, 1, -1)
            cv2.circle(mask, center, 7, 0, -1)
            self.masks[index] = mask.astype(bool)
            image = np.full((self.height, self.width, 3), (15, 20, 25), dtype=np.uint8)
            image[mask != 0] = (240, 240, 240)
            writer.write(image)
        writer.release()
        self.source_hash = hashlib.sha256(self.source.read_bytes()).hexdigest()

    def decode(self, **kwargs):
        return decode_video(self.source, self.output, **kwargs)

    def assert_originals_untouched(self):
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), self.source_hash)
        self.assertEqual(self.user_file.read_bytes(), b"existing user file must remain unchanged")

    @staticmethod
    def read_frames(path):
        capture = cv2.VideoCapture(str(path))
        frames = []
        while True:
            success, frame = capture.read()
            if not success:
                break
            frames.append(frame)
        capture.release()
        return frames

    def test_probe_and_decode_keep_every_frame_order_resolution_and_timestamps(self):
        info = probe_video(self.source)
        self.assertEqual((info.width, info.height, info.frame_count), (192, 128, 8))
        self.assertAlmostEqual(info.fps, self.fps)
        self.assertAlmostEqual(info.duration_seconds, self.count / self.fps)
        progress = []
        sequence = self.decode(progress=lambda *args: progress.append(args))
        self.assertIsInstance(sequence.frames_dir, Path)
        self.assertEqual([frame.path.name for frame in sequence.frames], [f"{i:05d}.jpg" for i in range(self.count)])
        self.assertEqual(sequence.frame_count, self.count)
        metadata = json.loads(sequence.metadata_path.read_text())
        self.assertFalse(metadata["frame_sampling"])
        self.assertFalse(metadata["resized"])
        for index, frame in enumerate(sequence.frames):
            self.assertEqual(frame.frame_index, index)
            self.assertAlmostEqual(frame.timestamp, index / self.fps, places=5)
            decoded = cv2.imread(str(frame.path))
            self.assertEqual(decoded.shape[:2], (self.height, self.width))
            foreground = decoded.mean(axis=2) > 150
            self.assertAlmostEqual(np.nonzero(foreground)[1].mean(), 28 + index * 16, delta=1)
            self.assertEqual(metadata["frames"][index]["frame_index"], index)
        self.assertEqual(progress[-1][:2], (self.count, self.count))
        self.assertTrue(sequence.root_dir.is_relative_to(self.output.resolve()))
        self.assert_originals_untouched()

    def test_export_unordered_model_masks_keeps_all_files_and_video_aligned(self):
        sequence = self.decode()
        records = {5: self.masks[5], 0: self.masks[0], 3: None,
                   4: np.zeros((self.height, self.width), dtype=bool), 7: self.masks[7]}
        result = export_tracking(sequence, records, "白色胶带", export_yolo=True,
                                 tracking_metadata={"backend": "test_propagator", "seed_frame_index": 5},
                                 codecs=("mp4v",))
        self.assertFalse(result.audio_included)
        self.assertEqual(result.frame_count, self.count)
        self.assertEqual(result.frames_with_target, 3)
        self.assertEqual(result.codec, "mp4v")
        self.assertEqual(len(list(result.labels_dir.glob("*.json"))), self.count)
        self.assertEqual(len(list(result.masks_dir.glob("*.png"))), self.count)
        self.assertEqual(len(list(result.yolo_dir.glob("[0-9]*.txt"))), self.count)
        self.assertEqual((result.yolo_dir / "classes.txt").read_text(), "白色胶带\n")
        index = json.loads(result.index_path.read_text())
        self.assertEqual(index["tracking_metadata"]["seed_frame_index"], 5)
        self.assertEqual(index["predictions_received"], len(records))
        self.assertFalse(index["frame_sampling"])
        self.assertFalse(index["prediction_interpolation"])
        self.assertFalse(index["audio_included"])
        self.assertEqual([frame["frame_index"] for frame in index["frames"]], list(range(self.count)))
        for frame_index in range(self.count):
            name = f"{frame_index:05d}"
            data = json.loads((result.labels_dir / f"{name}.json").read_text())
            raw = cv2.imread(str(result.masks_dir / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
            self.assertEqual(data["track_id"], 1)
            self.assertEqual(data["label"], "白色胶带")
            self.assertEqual(data["frame_index"], frame_index)
            self.assertAlmostEqual(data["timestamp"], frame_index / self.fps)
            self.assertEqual((result.labels_dir / data["imagePath"]).resolve(), sequence.frames[frame_index].path)
            self.assertTrue(set(np.unique(raw)).issubset({0, 255}))
            if frame_index in (0, 5, 7):
                np.testing.assert_array_equal(raw != 0, self.masks[frame_index])
                self.assertEqual(data["status"], "visible")
                self.assertIsNotNone(data["bbox"])
                self.assertTrue(data["polygon"])
                values = (result.yolo_dir / f"{name}.txt").read_text().split()
                self.assertEqual(values[0], "0")
                polygon = np.array(values[1:], dtype=float).reshape(-1, 2) * [self.width, self.height]
                raster = np.zeros_like(raw)
                cv2.fillPoly(raster, [np.rint(polygon).astype(np.int32)], 1)
                self.assertEqual(raster[65, 28 + frame_index * 16], 0)
            else:
                self.assertFalse(raw.any())
                self.assertIsNone(data["bbox"])
                self.assertEqual(data["polygon"], [])
                self.assertEqual(data["shapes"], [])
                self.assertEqual((result.yolo_dir / f"{name}.txt").read_text(), "")
        self.assertEqual(json.loads((result.labels_dir / "00003.json").read_text())["status"], "no_mask")
        self.assertEqual(json.loads((result.labels_dir / "00004.json").read_text())["status"], "empty")
        self.assertEqual(json.loads((result.labels_dir / "00001.json").read_text())["status"], "not_provided")
        video = self.read_frames(result.video_path)
        self.assertEqual(len(video), self.count)
        for frame in video:
            self.assertEqual(frame.shape[:2], (self.height, self.width))
        blue, green, red = map(int, video[5][65, 28 + 5 * 16 + 12])
        self.assertGreater(green - red, 30)
        blue, green, red = map(int, video[1][65, 28 + 16 + 12])
        self.assertLess(abs(green - red), 20)  # No interpolated mask on this missing frame.
        self.assert_originals_untouched()

    def test_streaming_trackingframes_are_written_before_requesting_the_next_mask(self):
        sequence = self.decode()
        def model_results():
            yield TrackingFrame(5, self.masks[5])
            stages = list(sequence.root_dir.glob(".tracking-export-*"))
            self.assertEqual(len(stages), 1)
            self.assertTrue((stages[0] / "masks" / "00005.png").exists())
            self.assertTrue((stages[0] / "labels" / "00005.json").exists())
            yield TrackingFrame(2, self.masks[2])
            yield TrackingFrame(0, self.masks[0])
        result = export_tracking(sequence, model_results(), "胶带", codecs=("mp4v",))
        self.assertEqual(result.frames_with_target, 3)

    def test_all_missing_frames_have_no_invented_geometry(self):
        sequence = self.decode()
        result = export_tracking(sequence, {}, "胶带", codecs=("mp4v",))
        self.assertEqual(result.frames_with_target, 0)
        index = json.loads(result.index_path.read_text())
        self.assertEqual(index["predictions_received"], 0)
        self.assertTrue(all(not frame["visible"] and frame["bbox"] is None for frame in index["frames"]))
        self.assertEqual(len(self.read_frames(result.video_path)), self.count)

    def test_cancel_decode_removes_only_its_new_job(self):
        cancel = False
        def progress(done, total, message):
            nonlocal cancel
            cancel = done >= 2
        with self.assertRaises(VideoCancelledError):
            self.decode(progress=progress, cancelled=lambda: cancel)
        self.assertEqual(list(self.output.iterdir()), [self.user_file])
        self.assert_originals_untouched()

    def test_cancel_export_keeps_frames_prior_export_and_closes_generator(self):
        sequence = self.decode()
        original = export_tracking(sequence, {0: self.masks[0]}, "胶带", codecs=("mp4v",))
        cancel, closed = False, False
        def model_results():
            nonlocal closed
            try:
                for index in range(self.count):
                    yield TrackingFrame(index, self.masks[index])
            finally:
                closed = True
        def progress(done, total, message):
            nonlocal cancel
            cancel = message == "正在保存模型逐帧预测" and done >= 2
        with self.assertRaises(VideoCancelledError):
            export_tracking(sequence, model_results(), "胶带", progress=progress,
                            cancelled=lambda: cancel, codecs=("mp4v",))
        self.assertTrue(closed)
        self.assertTrue(original.video_path.exists())
        self.assertEqual(len(list(sequence.frames_dir.glob("*.jpg"))), self.count)
        self.assertEqual(list(sequence.root_dir.glob(".tracking-export-*")), [])
        self.assertEqual(list(sequence.root_dir.glob("tracking_*")), [original.output_dir])
        self.assert_originals_untouched()

    def test_invalid_mask_duplicate_and_out_of_range_frames_leave_no_partial_export(self):
        sequence = self.decode()
        for records in (
            [(0, np.ones((10, 10)))],
            [(0, self.masks[0]), (0, self.masks[0])],
            [(self.count, self.masks[0])],
            [TrackingFrame(0, self.masks[0], track_id=2)],
        ):
            with self.subTest(records_type=type(records[0]).__name__), self.assertRaises(VideoIOError):
                export_tracking(sequence, records, "胶带", codecs=("mp4v",))
            self.assertEqual(list(sequence.root_dir.glob(".tracking-export-*")), [])
            self.assertEqual(list(sequence.root_dir.glob("tracking_*")), [])
        self.assert_originals_untouched()

    def test_codec_fallback_uses_mp4v_and_records_the_actual_codec(self):
        sequence = self.decode()
        actual_writer = cv2.VideoWriter
        for throws in (False, True):
            with self.subTest(throws=throws):
                calls = []
                def writer(*args):
                    calls.append(args)
                    if len(calls) == 1:
                        if throws:
                            raise cv2.error("Encoder unavailable")
                        return SimpleNamespace(isOpened=lambda: False, release=lambda: None)
                    return actual_writer(*args)
                with patch("labelpaw.data.video_io.cv2.VideoWriter", side_effect=writer):
                    result = export_tracking(sequence, {0: self.masks[0]}, "胶带")
                self.assertEqual(len(calls), 2)
                self.assertEqual(result.codec, "mp4v")
                self.assertEqual(len(self.read_frames(result.video_path)), self.count)

    def test_unreadable_video_does_not_create_or_remove_user_files(self):
        with self.assertRaises(VideoIOError):
            decode_video(self.directory / "missing.mp4", self.output)
        self.assertEqual(list(self.output.iterdir()), [self.user_file])
        self.assert_originals_untouched()


if __name__ == "__main__":
    unittest.main()
