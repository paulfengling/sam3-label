"""CPU-only checks for complete SAM instance geometry and prompt aliases."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labelpaw.models.sam_geometry import mask_to_annotation
from labelpaw.models.sam_prompts import normalize_sam_prompt, split_sam_prompts


class SamGeometryTests(unittest.TestCase):
    @staticmethod
    def rasterize(annotation, shape):
        raster = np.zeros(shape, dtype=np.uint8)
        points = np.rint(annotation["poly_pts"]).astype(np.int32)
        cv2.fillPoly(raster, [points], 1)
        return raster

    def assert_mask_iou(self, actual, expected, minimum=0.98):
        union = np.count_nonzero((actual != 0) | (expected != 0))
        intersection = np.count_nonzero((actual != 0) & (expected != 0))
        self.assertGreaterEqual(intersection / union, minimum)

    @staticmethod
    def nested_holes_mask():
        mask = np.zeros((280, 400), dtype=np.uint8)
        mask[20:250, 20:250] = 1
        mask[50:220, 50:220] = 0
        mask[85:185, 85:185] = 1  # An island in the first hole.
        mask[110:160, 110:160] = 0  # A hole in that island.
        mask[35:105, 300:370] = 1  # Another disconnected part of this instance.
        return mask

    @staticmethod
    def disconnected_mask():
        mask = np.zeros((100, 120), dtype=np.uint8)
        # One instance has two visible body parts separated by an occluder.
        mask[10:31, 20:41] = 1
        mask[65:81, 65:81] = 1
        return mask

    def assert_rectangle_contains_foreground(self, annotation, mask):
        y, x = np.nonzero(mask)
        left, top, width, height = annotation["rect"]
        self.assertTrue(np.all(x >= left))
        self.assertTrue(np.all(x < left + width))
        self.assertTrue(np.all(y >= top))
        self.assertTrue(np.all(y < top + height))

    def test_rectangle_encloses_every_disconnected_body_part(self):
        mask = self.disconnected_mask()
        annotation = mask_to_annotation(mask)
        self.assertEqual(annotation["rect"], [20.0, 10.0, 61.0, 71.0])
        self.assert_rectangle_contains_foreground(annotation, mask)

    def test_rectangle_retains_full_detector_box_and_mask(self):
        mask = self.disconnected_mask()
        # Detector extends above and left, while the mask extends below and right.
        annotation = mask_to_annotation(mask, [5, 3, 70, 72])
        self.assertEqual(annotation["rect"], [5.0, 3.0, 76.0, 78.0])
        self.assert_rectangle_contains_foreground(annotation, mask)

    def test_rectangle_clips_detector_box_to_image(self):
        mask = self.disconnected_mask()
        annotation = mask_to_annotation(mask, [-50, -10, 300, 200])
        self.assertEqual(annotation["rect"], [0.0, 0.0, 120.0, 100.0])

    def test_invalid_detector_boxes_do_not_discard_mask(self):
        mask = self.disconnected_mask()
        for box in ([40, 30, 20, 10], [0, 0, np.nan, 90], [1, 2, 3]):
            with self.subTest(box=box):
                annotation = mask_to_annotation(mask, box)
                self.assertEqual(annotation["rect"], [20.0, 10.0, 61.0, 71.0])

    def test_empty_mask_does_not_invent_annotation_from_detector_box(self):
        self.assertIsNone(mask_to_annotation(np.zeros((50, 60)), [2, 3, 20, 30]))

    def test_polygon_fills_both_parts_without_filling_large_gap(self):
        mask = self.disconnected_mask()
        annotation = mask_to_annotation(mask)
        points = np.rint(annotation["poly_pts"]).astype(np.int32)
        raster = np.zeros_like(mask)
        cv2.fillPoly(raster, [points], 1)
        self.assertTrue(np.all(raster[mask != 0] == 1))
        # A doubled bridge may draw a single pixel line, but cannot fill the
        # background area that a convex hull would introduce between parts.
        self.assertEqual(raster[50, 60], 0)
        self.assertLess(np.count_nonzero(raster[31:65, 20:81]), 100)
        self.assertLess(np.count_nonzero(raster), np.count_nonzero(mask) + 100)
        # The bridge endpoint occurs twice because each bridge is traversed
        # in both directions, preserving each disconnected component ring.
        self.assertLess(len(np.unique(points, axis=0)), len(points))

    def test_oriented_box_contains_every_contour(self):
        mask = self.disconnected_mask()
        annotation = mask_to_annotation(mask)
        center_x, center_y, width, height, angle = annotation["obb"]
        oriented = cv2.boxPoints(((center_x, center_y), (width, height), angle))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            for x, y in contour.reshape(-1, 2):
                distance = cv2.pointPolygonTest(oriented, (float(x), float(y)), True)
                self.assertGreaterEqual(distance, -1e-4)

    def test_thin_and_single_pixel_components_are_preserved(self):
        mask = np.zeros((60, 70), dtype=np.uint8)
        mask[5:21, 8:19] = 1
        mask[40, 36:51] = 1
        mask[54, 61] = 1
        annotation = mask_to_annotation(mask)
        self.assert_rectangle_contains_foreground(annotation, mask)
        self.assertEqual(annotation["rect"], [8.0, 5.0, 54.0, 50.0])
        raster = np.zeros_like(mask)
        cv2.fillPoly(raster, [np.rint(annotation["poly_pts"]).astype(np.int32)], 1)
        self.assertTrue(np.all(raster[mask != 0] == 1))
        self.assertTrue(np.isfinite(annotation["obb"]).all())

    def test_single_row_and_single_column_masks_keep_spatial_dimensions(self):
        for shape in ((1, 12), (12, 1), (1, 1)):
            with self.subTest(shape=shape):
                mask = np.ones(shape, dtype=np.uint8)
                annotation = mask_to_annotation(mask)
                self.assertEqual(annotation["rect"], [0.0, 0.0, float(shape[1]), float(shape[0])])

    def test_leading_batch_dimensions_are_accepted(self):
        mask = self.disconnected_mask()
        self.assertEqual(mask_to_annotation(mask[None, None])["rect"], [20.0, 10.0, 61.0, 71.0])

    def test_multiple_instances_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "two spatial dimensions"):
            mask_to_annotation(np.zeros((2, 30, 40)))

    def test_tape_ring_polygon_keeps_its_center_empty(self):
        mask = np.zeros((260, 260), dtype=np.uint8)
        cv2.circle(mask, (130, 130), 100, 1, -1)
        cv2.circle(mask, (130, 130), 54, 0, -1)
        annotation = mask_to_annotation(mask)
        raster = self.rasterize(annotation, mask.shape)
        self.assertEqual(raster[130, 130], 0)
        self.assertEqual(raster[130, 210], 1)
        self.assert_mask_iou(raster, mask)
        self.assert_rectangle_contains_foreground(annotation, mask)

    def test_multiple_holes_and_disconnected_part_are_preserved(self):
        mask = np.zeros((280, 400), dtype=np.uint8)
        mask[20:250, 20:250] = 1
        cv2.circle(mask, (85, 85), 30, 0, -1)
        cv2.circle(mask, (175, 170), 38, 0, -1)
        mask[35:105, 300:370] = 1
        annotation = mask_to_annotation(mask)
        raster = self.rasterize(annotation, mask.shape)
        self.assertEqual(raster[85, 85], 0)
        self.assertEqual(raster[170, 175], 0)
        self.assertEqual(raster[65, 335], 1)
        self.assertEqual(raster[150, 280], 0)
        self.assert_mask_iou(raster, mask)
        self.assertEqual(annotation["rect"], [20.0, 20.0, 350.0, 230.0])
        self.assert_rectangle_contains_foreground(annotation, mask)

    def test_island_inside_hole_and_nested_hole_keep_odd_even_fill(self):
        mask = self.nested_holes_mask()
        annotation = mask_to_annotation(mask)
        raster = self.rasterize(annotation, mask.shape)
        for x, y, expected in ((30, 130, 1), (65, 130, 0), (95, 130, 1),
                               (135, 135, 0), (335, 65, 1), (280, 150, 0)):
            with self.subTest(point=(x, y)):
                self.assertEqual(raster[y, x], expected)
        self.assert_mask_iou(raster, mask)
        self.assert_rectangle_contains_foreground(annotation, mask)

    def test_holes_survive_qt_fill_and_json_yolo_round_trip(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtCore import QPointF, QRectF, Qt
            from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPolygonF
            from PySide6.QtWidgets import QApplication, QGraphicsScene
        except ImportError:
            self.skipTest("Qt is only needed for the native polygon/export roundtrip")
        from labelpaw.data.exporter import Exporter
        from labelpaw.graphics.shapes import PolyShape

        app = QApplication.instance() or QApplication([])
        mask = self.nested_holes_mask()
        annotation = mask_to_annotation(mask)
        polygon = QPolygonF([QPointF(*point) for point in annotation["poly_pts"]])
        shape = PolyShape(polygon, "胶带")
        self.assertEqual(shape.fillRule(), Qt.OddEvenFill)
        scene = QGraphicsScene()
        scene.setSceneRect(QRectF(0, 0, mask.shape[1], mask.shape[0]))
        scene.addItem(shape)
        for child in shape.childItems():
            child.hide()
        shape.setPen(QPen(Qt.NoPen))
        shape.setBrush(QColor("red"))
        image = QImage(mask.shape[1], mask.shape[0], QImage.Format_RGB32)
        image.fill(QColor("white"))
        painter = QPainter(image)
        scene.render(painter, scene.sceneRect(), scene.sceneRect())
        painter.end()
        self.assertEqual(image.pixelColor(95, 130), QColor("red"))
        self.assertEqual(image.pixelColor(65, 130), QColor("white"))
        self.assertEqual(image.pixelColor(135, 135), QColor("white"))
        shapes = Exporter.extract_shapes(scene)
        with tempfile.TemporaryDirectory() as directory:
            target_json = Path(directory) / "tape.json"
            target_yolo = Path(directory) / "tape.txt"
            Exporter.save_json(str(target_json), "tape.png", mask.shape[1], mask.shape[0], shapes)
            Exporter.save_yolo(str(target_yolo), mask.shape[1], mask.shape[0], shapes, ["胶带"])
            restored_json = json.loads(target_json.read_text(encoding="utf-8"))["shapes"][0]
            self.assertEqual(restored_json["label"], "胶带")
            values = target_yolo.read_text().split()
            restored_yolo = np.array(values[1:], dtype=float).reshape(-1, 2) * [mask.shape[1], mask.shape[0]]
            for points in (restored_json["points"], restored_yolo):
                restored = {"poly_pts": points}
                raster = self.rasterize(restored, mask.shape)
                self.assert_mask_iou(raster, mask)
                self.assertEqual(raster[130, 65], 0)
                self.assertEqual(raster[135, 135], 0)
                restored_polygon = QPolygonF([QPointF(*point) for point in points])
                self.assertTrue(restored_polygon.containsPoint(QPointF(95, 130), Qt.OddEvenFill))
                self.assertFalse(restored_polygon.containsPoint(QPointF(65, 130), Qt.OddEvenFill))
                self.assertFalse(restored_polygon.containsPoint(QPointF(135, 135), Qt.OddEvenFill))
        app.processEvents()


class SamPromptTests(unittest.TestCase):
    def test_exact_person_aliases_use_english_model_prompt(self):
        for prompt in ("人", "人体", "人物", "人员", "行人", "人群", "人类", "完整的人", " Person ", "PEOPLE", "Human", "humans"):
            with self.subTest(prompt=prompt):
                self.assertEqual(normalize_sam_prompt(prompt), "person")

    def test_custom_phrases_are_retained(self):
        for prompt in ("穿红色衣服的人", "背包的人", "snow person", "person wearing a hat", "个人物品"):
            with self.subTest(prompt=prompt):
                self.assertEqual(normalize_sam_prompt(prompt), prompt)
        self.assertEqual(normalize_sam_prompt("  穿红色衣服的人  "), "穿红色衣服的人")

    def test_chinese_and_english_separators_split_prompts(self):
        self.assertEqual(split_sam_prompts(" 人， 背包、树, ,石头，"), ["人", "背包", "树", "石头"])


if __name__ == "__main__":
    unittest.main()
