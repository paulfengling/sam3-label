"""Real Qt mouse drags keep annotation geometry, history, and exports in sync."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main
from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPolygonF
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QGraphicsItem, QGraphicsScene, QGraphicsView

from labelpaw.data.exporter import Exporter
from labelpaw.graphics.shapes import PolyShape, RectShape, RotatedRectShape


class EditingScene(QGraphicsScene):
    state_changed = Signal()


class ShapeEditingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.scene = EditingScene()
        self.scene.setSceneRect(0, 0, 500, 400)
        self.view = QGraphicsView(self.scene)
        self.view.resize(620, 520)
        self.view.show()
        self.app.processEvents()
        self.addCleanup(self.view.close)
        self.commits = []
        self.scene.state_changed.connect(lambda: self.commits.append(Exporter.extract_shapes(self.scene)))

    def drag(self, start, end):
        viewport = self.view.viewport()
        begin, finish = self.view.mapFromScene(start), self.view.mapFromScene(end)
        QTest.mouseMove(viewport, begin)
        QTest.mousePress(viewport, Qt.LeftButton, Qt.NoModifier, begin)
        # Explicit pressed buttons make the drag portable to offscreen Qt.
        move = QMouseEvent(
            QEvent.MouseMove, QPointF(finish),
            QPointF(viewport.mapToGlobal(finish)),
            Qt.NoButton, Qt.LeftButton, Qt.NoModifier,
        )
        QApplication.sendEvent(viewport, move)
        QTest.mouseRelease(viewport, Qt.LeftButton, Qt.NoModifier, finish)
        self.app.processEvents()

    def click(self, point):
        QTest.mouseClick(self.view.viewport(), Qt.LeftButton, Qt.NoModifier, self.view.mapFromScene(point))
        self.app.processEvents()

    def polygon(self):
        return PolyShape(QPolygonF([
            QPointF(60, 60), QPointF(160, 60),
            QPointF(160, 160), QPointF(60, 160),
        ]), "人")

    def test_existing_rectangle_polygon_and_rotated_box_move_and_commit(self):
        for shape in (
            RectShape(QRectF(60, 60, 100, 100), "人"),
            self.polygon(),
            RotatedRectShape(110, 110, 100, 80, 25, "人"),
        ):
            with self.subTest(shape=type(shape).__name__):
                self.scene.addItem(shape)
                before = Exporter.extract_shapes(self.scene)[0]
                self.commits.clear()
                self.drag(QPointF(110, 110), QPointF(130, 140))
                after = Exporter.extract_shapes(self.scene)[0]
                self.assertTrue(shape.isSelected())
                self.assertEqual(len(self.commits), 1)
                for old, new in zip(before["points"], after["points"]):
                    self.assertAlmostEqual(new[0] - old[0], 20)
                    self.assertAlmostEqual(new[1] - old[1], 30)
                self.assertTrue(shape.flags() & QGraphicsItem.ItemIsMovable)
                self.scene.removeItem(shape)

    def test_rectangle_corner_and_edge_resizing_export_scene_coordinates(self):
        shape = RectShape(QRectF(60, 60, 100, 100), "人")
        self.scene.addItem(shape)
        shape.setPos(30, 20)
        shape.setSelected(True)
        self.drag(shape.rb_handle.scenePos(), shape.rb_handle.scenePos() + QPointF(25, 20))
        self.assertEqual(Exporter.extract_shapes(self.scene)[0]["points"], [[90.0, 80.0], [215.0, 200.0]])
        self.drag(shape.t_handle.scenePos(), shape.t_handle.scenePos() + QPointF(0, -15))
        self.assertEqual(Exporter.extract_shapes(self.scene)[0]["points"], [[90.0, 65.0], [215.0, 200.0]])
        self.assertEqual(len(self.commits), 2)
        self.assertTrue(shape.flags() & QGraphicsItem.ItemIsMovable)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "sample.txt"
            Exporter.save_yolo(str(target), 500, 400, Exporter.extract_shapes(self.scene), ["人"])
            self.assertEqual(target.read_text(), "0 0.305000 0.331250 0.250000 0.337500")

    def test_small_polygon_vertex_drag_does_not_delete_vertex(self):
        shape = self.polygon()
        self.scene.addItem(shape)
        shape.setSelected(True)
        self.drag(shape.handles[0].scenePos(), shape.handles[0].scenePos() + QPointF(1, 1))
        self.assertEqual(shape.polygon().count(), 4)
        self.assertEqual(Exporter.extract_shapes(self.scene)[0]["points"][0], [61.0, 61.0])
        self.assertEqual(len(self.commits), 1)
        self.assertTrue(shape.flags() & QGraphicsItem.ItemIsMovable)

    def test_polygon_edge_inserts_adjustable_vertex_and_commits(self):
        shape = self.polygon()
        self.scene.addItem(shape)
        shape.setSelected(True)
        self.drag(QPointF(110, 60), QPointF(110, 40))
        self.assertEqual(shape.polygon().count(), 5)
        self.assertIn([110.0, 40.0], Exporter.extract_shapes(self.scene)[0]["points"])
        self.assertEqual(len(self.commits), 1)
        self.assertEqual(shape._dragging_edge_idx, -1)
        self.assertTrue(shape.flags() & QGraphicsItem.ItemIsMovable)

    def test_polygon_vertex_click_deletes_and_commits(self):
        shape = self.polygon()
        self.scene.addItem(shape)
        shape.setSelected(True)
        self.click(shape.handles[0].scenePos())
        self.assertEqual(shape.polygon().count(), 3)
        self.assertEqual(len(self.commits), 1)
        self.assertEqual(len(Exporter.extract_shapes(self.scene)[0]["points"]), 3)
        self.assertTrue(shape.flags() & QGraphicsItem.ItemIsMovable)

    def test_rotated_box_resize_keeps_opposite_edge_and_commits(self):
        shape = RotatedRectShape(180, 170, 100, 80, 25, "人")
        self.scene.addItem(shape)
        shape.setSelected(True)
        opposite = shape.h_left.scenePos()
        destination = shape.mapToScene(QPointF(75, 0))
        self.drag(shape.h_right.scenePos(), destination)
        self.assertAlmostEqual(shape.box_w, 125, delta=1.2)
        self.assertAlmostEqual(shape.h_left.scenePos().x(), opposite.x(), delta=1e-6)
        self.assertAlmostEqual(shape.h_left.scenePos().y(), opposite.y(), delta=1e-6)
        exported = Exporter.extract_shapes(self.scene)[0]
        self.assertAlmostEqual(exported["rect"][2], shape.box_w)
        self.assertAlmostEqual(exported["angle"], 25)
        self.assertEqual(len(self.commits), 1)
        self.assertIsNone(shape._dragging_handle)
        self.assertTrue(shape.flags() & QGraphicsItem.ItemIsMovable)

    def test_rotated_box_rotation_exports_angle_and_unlocked_item(self):
        shape = RotatedRectShape(180, 170, 100, 80, 0, "人")
        self.scene.addItem(shape)
        shape.setSelected(True)
        self.drag(shape.h_rotate.scenePos(), QPointF(250, 170))
        exported = Exporter.extract_shapes(self.scene)[0]
        self.assertAlmostEqual(exported["angle"], 90)
        self.assertEqual(len(self.commits), 1)
        self.assertIsNone(shape._dragging_handle)
        self.assertTrue(shape.flags() & QGraphicsItem.ItemIsMovable)

    def test_selection_without_geometry_change_does_not_commit(self):
        for shape in (RectShape(QRectF(60, 60, 100, 100), "人"), RotatedRectShape(110, 110, 100, 80, 0, "人")):
            with self.subTest(shape=type(shape).__name__):
                self.scene.addItem(shape)
                self.commits.clear()
                self.click(QPointF(110, 110))
                self.assertTrue(shape.isSelected())
                self.assertEqual(self.commits, [])
                self.scene.removeItem(shape)

    def test_rotated_box_handle_click_does_not_commit_or_lock_item(self):
        shape = RotatedRectShape(180, 170, 100, 80, 0, "人")
        self.scene.addItem(shape)
        shape.setSelected(True)
        self.click(shape.h_right.scenePos())
        self.assertEqual(self.commits, [])
        self.assertIsNone(shape._dragging_handle)
        self.assertTrue(shape.flags() & QGraphicsItem.ItemIsMovable)

    def test_bridged_polygon_duplicate_vertices_move_together(self):
        # Doubled bridge endpoints must remain identical after a vertex drag.
        points = [
            [60, 60], [220, 160], [260, 160], [260, 200], [220, 200],
            [220, 160], [60, 60], [120, 60], [120, 120], [60, 120],
        ]
        shape = PolyShape(QPolygonF([QPointF(*point) for point in points]), "人")
        self.scene.addItem(shape)
        shape.setSelected(True)
        self.drag(QPointF(220, 160), QPointF(230, 150))
        exported = Exporter.extract_shapes(self.scene)[0]["points"]
        self.assertEqual(exported[1], [230.0, 150.0])
        self.assertEqual(exported[5], exported[1])
        self.assertFalse(shape.polygon().containsPoint(QPointF(150, 90), Qt.OddEvenFill))
        self.assertEqual(len(self.commits), 1)

    def test_bridged_polygon_shared_anchors_are_not_deleted(self):
        points = [
            [60, 60], [220, 160], [260, 160], [260, 200], [220, 200],
            [220, 160], [60, 60], [120, 60], [120, 120], [60, 120],
        ]
        shape = PolyShape(QPolygonF([QPointF(*point) for point in points]), "人")
        self.scene.addItem(shape)
        shape.setSelected(True)
        self.click(QPointF(220, 160))
        self.assertEqual(Exporter.extract_shapes(self.scene)[0]["points"], points)
        self.assertEqual(self.commits, [])
        self.assertTrue(shape.flags() & QGraphicsItem.ItemIsMovable)
        # Hovering over the bridge itself must not create a new boundary node.
        QTest.mouseMove(self.view.viewport(), self.view.mapFromScene(QPointF(140, 110)))
        self.app.processEvents()
        self.assertEqual(shape.ghost_idx, -1)

    def test_sam_generated_rectangle_resize_saves_and_can_undo_redo(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = temporary.name
        # Window shutdown saves before the temporary image folder is removed.
        image_path = Path(directory) / "sample.png"
        image = QImage(500, 400, QImage.Format_RGB32)
        image.fill(QColor("white"))
        self.assertTrue(image.save(str(image_path)))
        for target in ("main.MainWindow.restore_last_state", "main.QSettings", "main.DialogOver"):
            patcher = patch(target)
            patcher.start()
            self.addCleanup(patcher.stop)
        window = main.MainWindow(auto_load_model=False)
        self.addCleanup(window.close)
        window.current_dir = directory
        window.current_image_path = str(image_path)
        window.add_class_to_list("人")
        window.scene.load_image(str(image_path))
        window.sam_client.current_model_key = "sam3"
        window.sam_client.current_model_type = "sam3"
        window.scene.set_sam_enabled(True)
        shape = RectShape(QRectF(60, 60, 100, 100), "人")
        window.scene.addItem(shape)
        window.push_state()
        original = Exporter.extract_shapes(window.scene)
        self.assertTrue(window.auto_save_annotation())
        self.view.setScene(window.scene)
        shape.setSelected(True)
        self.drag(shape.rb_handle.scenePos(), shape.rb_handle.scenePos() + QPointF(25, 20))
        edited = Exporter.extract_shapes(window.scene)
        self.assertNotEqual(edited, original)
        saved = json.loads(image_path.with_suffix(".json").read_text(encoding="utf-8"))
        self.assertEqual(saved["shapes"][0]["points"], edited[0]["points"])
        self.assertEqual(len(window.undo_stack), 2)
        window.undo()
        self.assertEqual(Exporter.extract_shapes(window.scene), original)
        saved = json.loads(image_path.with_suffix(".json").read_text(encoding="utf-8"))
        self.assertEqual(saved["shapes"][0]["points"], original[0]["points"])
        window.redo()
        self.assertEqual(Exporter.extract_shapes(window.scene), edited)
        saved = json.loads(image_path.with_suffix(".json").read_text(encoding="utf-8"))
        self.assertEqual(saved["shapes"][0]["points"], edited[0]["points"])


if __name__ == "__main__":
    unittest.main()
