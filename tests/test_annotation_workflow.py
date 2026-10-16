"""Desktop annotation regressions that do not require model weights.

Run with: QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests
"""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import xml.etree.ElementTree as ET

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Exercise the application's import order before importing Qt directly.
import main
import numpy as np

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QKeyEvent, QPolygonF
from PySide6.QtWidgets import QApplication, QGraphicsSceneMouseEvent, QGraphicsView, QListWidgetItem

from labelpaw.data.exporter import Exporter
from labelpaw.graphics.canvas import Canvas, CanvasMode
from labelpaw.graphics.shapes import PointShape, PolyShape, PoseShape, RectShape, RotatedRectShape


class AnnotationWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.directory = Path(self.temp_dir.name)
        self.image_path = self.directory / "sample.png"
        image = QImage(200, 160, QImage.Format_RGB32)
        image.fill(QColor("white"))
        self.assertTrue(image.save(str(self.image_path)))
        # Tests must not restore or overwrite the user's desktop session.
        for target in ("main.MainWindow.restore_last_state", "main.QSettings", "main.DialogOver"):
            patcher = patch(target)
            patcher.start()
            self.addCleanup(patcher.stop)

    def new_window(self):
        window = main.MainWindow(auto_load_model=False)
        self.addCleanup(window.close)
        window.current_dir = str(self.directory)
        window.load_classes(str(self.directory))
        window.add_class_to_list("dog")
        window.current_image_path = str(self.image_path)
        window.scene.load_image(str(self.image_path))
        window.push_state()
        return window

    def smart_canvas(self, mode=CanvasMode.RECT):
        scene = Canvas()
        scene.load_image(str(self.image_path))
        scene.set_mode(mode)
        scene.sam_client = SimpleNamespace(
            current_model_key="sam3", current_model_type="sam3",
            request_inference=Mock(), request_box_inference=Mock(),
        )
        scene.set_sam_enabled(True)
        return scene

    def scene_mouse(self, scene, event_type, point, modifiers=Qt.NoModifier):
        position = QPointF(*point)
        event = QGraphicsSceneMouseEvent(event_type)
        event.setScenePos(position)
        event.setLastScenePos(getattr(scene, "_test_mouse_position", position))
        event.setModifiers(modifiers)
        if event_type == QEvent.GraphicsSceneMousePress:
            scene._test_mouse_anchor = position
        event.setButtonDownScenePos(Qt.LeftButton, getattr(scene, "_test_mouse_anchor", position))
        event.setButton(Qt.NoButton if event_type == QEvent.GraphicsSceneMouseMove else Qt.LeftButton)
        event.setButtons(Qt.NoButton if event_type == QEvent.GraphicsSceneMouseRelease else Qt.LeftButton)
        handler = {
            QEvent.GraphicsSceneMousePress: scene.mousePressEvent,
            QEvent.GraphicsSceneMouseMove: scene.mouseMoveEvent,
            QEvent.GraphicsSceneMouseRelease: scene.mouseReleaseEvent,
        }[event_type]
        handler(event)
        scene._test_mouse_position = position

    def test_default_startup_selects_a_model_without_requiring_weights_in_test(self):
        with patch.object(main.SAMClient, "load_model_by_key") as load_model:
            window = main.MainWindow()
            self.addCleanup(window.close)
        load_model.assert_called_once_with("sam3")
        self.assertIsNone(window.current_image_path)
        self.assertIsNone(window.sam_client.model)

    def test_moved_point_exports_scene_coordinates(self):
        scene = Canvas()
        scene.load_image(str(self.image_path))
        point = PointShape(QPointF(20, 30), "dog")
        scene.addItem(point)
        point.setPos(50, 40)
        self.assertEqual(Exporter.extract_shapes(scene)[0]["points"], [[70.0, 70.0]])

    def test_manual_rectangle_draws_without_a_loaded_model(self):
        window = self.new_window()
        window.sam_client.current_model_key = None
        events = (
            (QEvent.GraphicsSceneMousePress, window.scene.mousePressEvent, QPointF(20, 30), Qt.LeftButton),
            (QEvent.GraphicsSceneMouseMove, window.scene.mouseMoveEvent, QPointF(100, 90), Qt.NoButton),
            (QEvent.GraphicsSceneMouseRelease, window.scene.mouseReleaseEvent, QPointF(100, 90), Qt.LeftButton),
        )
        for event_type, handler, position, button in events:
            event = QGraphicsSceneMouseEvent(event_type)
            event.setScenePos(position)
            event.setButton(button)
            event.setButtons(Qt.NoButton if event_type == QEvent.GraphicsSceneMouseRelease else Qt.LeftButton)
            handler(event)
        shapes = Exporter.extract_shapes(window.scene)
        self.assertEqual(len(shapes), 1)
        self.assertEqual(shapes[0]["type"], "rectangle")
        self.assertEqual(shapes[0]["points"], [[20.0, 30.0], [100.0, 90.0]])

    def test_selected_person_class_is_sent_to_hover_and_click(self):
        window = self.new_window()
        window.sam_client.current_model_key = "sam3"
        window.sam_client.current_model_type = "sam3"
        window.scene.set_sam_enabled(True)
        window.samPromptInput.setText("dog")
        window.add_class_to_list("人")
        window.classListWidget._class_items["人"].setSelected(True)
        with patch.object(window.sam_client, "request_inference") as request:
            hover = QGraphicsSceneMouseEvent(QEvent.GraphicsSceneMouseMove)
            hover.setScenePos(QPointF(40, 50))
            window.scene.mouseMoveEvent(hover)
            request.assert_called_once_with(40.0, 50.0, is_click=False, prompt="人")
            click = QGraphicsSceneMouseEvent(QEvent.GraphicsSceneMousePress)
            click.setScenePos(QPointF(40, 50))
            click.setButton(Qt.LeftButton)
            window.scene.mousePressEvent(click)
            self.assertEqual(request.call_count, 1)
            self.scene_mouse(window.scene, QEvent.GraphicsSceneMouseRelease, (40, 50))
            request.assert_called_with(40.0, 50.0, is_click=True, prompt="人")

    def test_text_person_hint_and_result_keep_the_original_chinese_label(self):
        window = self.new_window()
        window.sam_client.current_model_key = "sam3"
        window.sam_client.current_model_type = "sam3"
        window.scene.set_sam_enabled(True)
        window.samPromptInput.setText("人")
        with patch.object(window.sam_client, "request_inference") as request:
            click = QGraphicsSceneMouseEvent(QEvent.GraphicsSceneMousePress)
            click.setScenePos(QPointF(40, 50))
            click.setButton(Qt.LeftButton)
            window.scene.mousePressEvent(click)
            request.assert_not_called()
            self.scene_mouse(window.scene, QEvent.GraphicsSceneMouseRelease, (40, 50))
            request.assert_called_once_with(40.0, 50.0, is_click=True, prompt="人")
        window.samPromptInput.setText("dog")
        polygon = [[30, 10], [90, 10], [90, 155], [30, 155]]
        args = (polygon, [30, 10, 60, 145], [60, 82.5, 60, 145, 0], 0.99, True)
        window.sam_client.inference_result.emit(*args)
        self.assertEqual(Exporter.extract_shapes(window.scene), [])
        window.sam_client.inference_result_labeled.emit(*args, "人")
        shapes = Exporter.extract_shapes(window.scene)
        self.assertEqual(len(shapes), 1)
        self.assertEqual(shapes[0]["label"], "人")
        self.assertEqual(shapes[0]["points"], [[30.0, 10.0], [90.0, 155.0]])

    def test_ambiguous_text_and_sam2_keep_the_original_point_request_interface(self):
        window = self.new_window()
        window.sam_client.current_model_key = "sam3"
        window.sam_client.current_model_type = "sam3"
        window.scene.set_sam_enabled(True)
        for text in ("", "人, dog"):
            with self.subTest(text=text), patch.object(window.sam_client, "request_inference") as request:
                window.samPromptInput.setText(text)
                click = QGraphicsSceneMouseEvent(QEvent.GraphicsSceneMousePress)
                click.setScenePos(QPointF(40, 50))
                click.setButton(Qt.LeftButton)
                window.scene.mousePressEvent(click)
                request.assert_not_called()
                self.scene_mouse(window.scene, QEvent.GraphicsSceneMouseRelease, (40, 50))
                request.assert_called_once_with(40.0, 50.0, is_click=True)
        window.sam_client.current_model_key = "sam2.1_hiera_large"
        window.sam_client.current_model_type = "sam2"
        window.add_class_to_list("人")
        window.classListWidget._class_items["人"].setSelected(True)
        with patch.object(window.sam_client, "request_inference") as request:
            window.scene._request_sam_inference(QPointF(40, 50), is_click=True)
            request.assert_called_once_with(40.0, 50.0, is_click=True)

    def test_smart_press_waits_and_short_release_confirms_point(self):
        scene = self.smart_canvas()
        self.scene_mouse(scene, QEvent.GraphicsSceneMousePress, (40, 50))
        scene.sam_client.request_inference.assert_not_called()
        scene.sam_client.request_box_inference.assert_not_called()
        self.scene_mouse(scene, QEvent.GraphicsSceneMouseRelease, (42, 52))
        scene.sam_client.request_inference.assert_called_once_with(40.0, 50.0, is_click=True)
        scene.sam_client.request_box_inference.assert_not_called()
        self.assertIsNone(scene._sam_press_pt)

    def test_smart_drag_normalizes_and_clips_box_without_hover_requests(self):
        for mode in (CanvasMode.RECT, CanvasMode.POLY, CanvasMode.RBOX):
            with self.subTest(mode=mode):
                scene = self.smart_canvas(mode)
                polygon = [[30, 10], [90, 10], [90, 155], [30, 155]]
                scene.handle_sam_result(polygon, [30, 10, 60, 145], [60, 82.5, 60, 145, 0], 0.99, False)
                self.assertIsNotNone(scene.sam_hover_item)
                self.scene_mouse(scene, QEvent.GraphicsSceneMousePress, (160, 130))
                self.assertIsNone(scene.sam_hover_item)
                self.scene_mouse(scene, QEvent.GraphicsSceneMouseMove, (-30, -50))
                self.assertIsNotNone(scene._sam_roi_item)
                self.assertEqual(scene._sam_roi_item.rect(), QRectF(0, 0, 160, 130))
                self.assertEqual(Exporter.extract_shapes(scene), [])
                scene.handle_sam_result(polygon, [30, 10, 60, 145], [60, 82.5, 60, 145, 0], 0.99, False)
                self.assertIsNone(scene.sam_hover_item)
                self.scene_mouse(scene, QEvent.GraphicsSceneMouseRelease, (-30, -50))
                scene.sam_client.request_box_inference.assert_called_once_with([0.0, 0.0, 160.0, 130.0])
                scene.sam_client.request_inference.assert_not_called()
                self.assertIsNone(scene._sam_roi_item)
                self.assertIsNone(scene._sam_press_pt)

    def test_ctrl_smart_drag_inside_existing_shape_creates_only_a_box_request(self):
        for modifiers in (Qt.ControlModifier, Qt.MetaModifier):
            with self.subTest(modifiers=modifiers):
                scene = self.smart_canvas()
                person = RectShape(QRectF(20, 10, 170, 140), "人")
                scene.addItem(person)
                person.setSelected(True)
                scene.sam_prompt_provider = lambda: "人"
                commits = []
                scene.state_changed.connect(lambda: commits.append(True))
                original = Exporter.extract_shapes(scene)
                self.scene_mouse(scene, QEvent.GraphicsSceneMousePress, (55, 60), modifiers)
                self.scene_mouse(scene, QEvent.GraphicsSceneMouseMove, (105, 120), modifiers)
                self.scene_mouse(scene, QEvent.GraphicsSceneMouseRelease, (105, 120), modifiers)
                scene.sam_client.request_box_inference.assert_called_once_with([55.0, 60.0, 105.0, 120.0], prompt="人")
                scene.sam_client.request_inference.assert_not_called()
                self.assertEqual(Exporter.extract_shapes(scene), original)
                self.assertEqual(commits, [])

    def test_modified_rotation_shortcuts_do_not_rotate_an_obb(self):
        scene = Canvas()
        scene.load_image(str(self.image_path))
        shape = RotatedRectShape(100, 80, 100, 60, 15, "dog")
        scene.addItem(shape)
        shape.setSelected(True)
        commits = []
        scene.state_changed.connect(lambda: commits.append(True))
        for modifiers in (Qt.ControlModifier | Qt.ShiftModifier, Qt.ControlModifier,
                          Qt.MetaModifier, Qt.MetaModifier | Qt.ShiftModifier, Qt.ShiftModifier):
            for key in (Qt.Key_Z, Qt.Key_X, Qt.Key_C, Qt.Key_V):
                with self.subTest(modifiers=modifiers, key=key):
                    scene.keyPressEvent(QKeyEvent(QEvent.KeyPress, key, modifiers))
                    self.assertEqual(shape.rotation(), 15)
        self.assertEqual(commits, [])
        scene.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Z, Qt.NoModifier))
        self.assertEqual(shape.rotation(), 10)
        self.assertEqual(commits, [True])

    def test_existing_shape_editing_does_not_start_smart_inference(self):
        scene = self.smart_canvas()
        person = RectShape(QRectF(20, 10, 170, 140), "人")
        scene.addItem(person)
        person.setSelected(True)
        commits = []
        scene.state_changed.connect(lambda: commits.append(True))
        self.scene_mouse(scene, QEvent.GraphicsSceneMousePress, (55, 60))
        self.assertTrue(scene._sam_editing)
        self.assertIsNone(scene._sam_press_pt)
        self.scene_mouse(scene, QEvent.GraphicsSceneMouseMove, (65, 70))
        self.scene_mouse(scene, QEvent.GraphicsSceneMouseRelease, (65, 70))
        scene.sam_client.request_box_inference.assert_not_called()
        scene.sam_client.request_inference.assert_not_called()
        self.assertIsNone(scene._sam_roi_item)
        self.assertFalse(scene._sam_editing)
        self.assertEqual(commits, [True])

    def test_smart_box_results_keep_the_current_shape_mode_and_remain_editable(self):
        for mode, expected_type in ((CanvasMode.RECT, "rectangle"), (CanvasMode.POLY, "polygon"), (CanvasMode.RBOX, "obb")):
            with self.subTest(mode=mode):
                window = self.new_window()
                scene = window.scene
                scene.set_mode(mode)
                scene.set_sam_enabled(True)
                window.sam_client.current_model_key = "sam3"
                window.sam_client.current_model_type = "sam3"
                geometry = ([[30, 20], [90, 20], [90, 150], [30, 150]],
                            [30, 20, 60, 130], [60, 85, 60, 130, 0], 0.99, True, "")
                with patch.object(window.sam_client, "request_inference") as point, patch.object(
                    window.sam_client, "request_box_inference",
                    side_effect=lambda box: window.sam_client.inference_result_labeled.emit(*geometry),
                ) as box:
                    self.scene_mouse(scene, QEvent.GraphicsSceneMousePress, (30, 20))
                    self.scene_mouse(scene, QEvent.GraphicsSceneMouseMove, (90, 150))
                    self.scene_mouse(scene, QEvent.GraphicsSceneMouseRelease, (90, 150))
                    shapes = Exporter.extract_shapes(scene)
                    self.assertEqual(len(shapes), 1)
                    self.assertEqual(shapes[0]["type"], expected_type)
                    self.scene_mouse(scene, QEvent.GraphicsSceneMousePress, (60, 80))
                    self.assertTrue(scene._sam_editing)
                    self.scene_mouse(scene, QEvent.GraphicsSceneMouseMove, (65, 85))
                    self.scene_mouse(scene, QEvent.GraphicsSceneMouseRelease, (65, 85))
                    self.assertEqual(box.call_count, 1)
                    point.assert_not_called()
                    self.assertTrue(scene.selectedItems())

    def test_ctrl_manual_drag_can_create_a_nested_rectangle_or_obb(self):
        for mode in (CanvasMode.RECT, CanvasMode.RBOX):
            with self.subTest(mode=mode):
                scene = Canvas()
                scene.load_image(str(self.image_path))
                scene.set_mode(mode)
                scene.shape_drawn.connect(scene.addItem)
                person = RectShape(QRectF(20, 10, 170, 140), "人")
                scene.addItem(person)
                person.setSelected(True)
                self.scene_mouse(scene, QEvent.GraphicsSceneMousePress, (55, 60), Qt.ControlModifier)
                self.scene_mouse(scene, QEvent.GraphicsSceneMouseMove, (105, 120), Qt.ControlModifier)
                self.scene_mouse(scene, QEvent.GraphicsSceneMouseRelease, (105, 120), Qt.ControlModifier)
                shapes = Exporter.extract_shapes(scene)
                self.assertEqual(len(shapes), 2)
                self.assertEqual(shapes[0]["points"], [[20.0, 10.0], [190.0, 150.0]])
                self.assertEqual(shapes[1]["type"], "rectangle" if mode == CanvasMode.RECT else "obb")
                self.assertEqual(shapes[1]["points"], [[55.0, 60.0], [105.0, 120.0]] if mode == CanvasMode.RECT
                                 else [[55.0, 60.0], [105.0, 60.0], [105.0, 120.0], [55.0, 120.0]])

    def test_smart_drag_threshold_uses_view_pixels_and_rejects_thin_boxes(self):
        for zoom, end, expects_box in ((4.0, (42, 52), True), (0.25, (60, 60), False)):
            with self.subTest(zoom=zoom):
                scene = self.smart_canvas()
                view = QGraphicsView(scene)
                view.scale(zoom, zoom)
                self.scene_mouse(scene, QEvent.GraphicsSceneMousePress, (40, 50))
                self.scene_mouse(scene, QEvent.GraphicsSceneMouseMove, end)
                self.scene_mouse(scene, QEvent.GraphicsSceneMouseRelease, end)
                if expects_box:
                    scene.sam_client.request_box_inference.assert_called_once_with([40.0, 50.0, 42.0, 52.0])
                    scene.sam_client.request_inference.assert_not_called()
                else:
                    scene.sam_client.request_inference.assert_called_once_with(40.0, 50.0, is_click=True)
                    scene.sam_client.request_box_inference.assert_not_called()
                view.close()
        scene = self.smart_canvas()
        self.scene_mouse(scene, QEvent.GraphicsSceneMousePress, (40, 50))
        self.scene_mouse(scene, QEvent.GraphicsSceneMouseMove, (40, 130))
        self.scene_mouse(scene, QEvent.GraphicsSceneMouseRelease, (40, 130))
        scene.sam_client.request_box_inference.assert_not_called()
        scene.sam_client.request_inference.assert_not_called()
        self.assertIsNone(scene._sam_roi_item)
        scene = self.smart_canvas()
        view = QGraphicsView(scene)
        view.scale(4, 4)
        self.scene_mouse(scene, QEvent.GraphicsSceneMousePress, (40, 50))
        self.scene_mouse(scene, QEvent.GraphicsSceneMouseMove, (41.5, 60))
        self.scene_mouse(scene, QEvent.GraphicsSceneMouseRelease, (41.5, 60))
        scene.sam_client.request_box_inference.assert_not_called()
        scene.sam_client.request_inference.assert_not_called()
        view.close()

    def test_smart_roi_is_cleared_on_escape_mode_image_and_smart_changes(self):
        for change in ("escape", "mode", "image", "smart"):
            with self.subTest(change=change):
                scene = self.smart_canvas()
                self.scene_mouse(scene, QEvent.GraphicsSceneMousePress, (40, 50))
                self.scene_mouse(scene, QEvent.GraphicsSceneMouseMove, (100, 120))
                self.assertIsNotNone(scene._sam_roi_item)
                if change == "escape":
                    scene.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier))
                elif change == "mode":
                    scene.set_mode(CanvasMode.POLY)
                elif change == "image":
                    scene.load_image(str(self.image_path))
                else:
                    scene.set_sam_enabled(False)
                self.assertIsNone(scene._sam_roi_item)
                self.assertIsNone(scene._sam_press_pt)
                self.scene_mouse(scene, QEvent.GraphicsSceneMouseRelease, (100, 120))
                scene.sam_client.request_box_inference.assert_not_called()
                scene.sam_client.request_inference.assert_not_called()

    def test_multi_prompt_results_overwrite_once_and_keep_all_labels(self):
        window = self.new_window()
        window.scene.addItem(RectShape(QRectF(10, 10, 30, 30), "old"))
        window.btnOverwrite.setChecked(True)
        results = [
            {"label": "人", "rect": [30, 10, 40, 145]},
            {"label": "人", "rect": [90, 10, 40, 145]},
            {"label": "dog", "rect": [140, 80, 40, 50]},
        ]
        with patch.object(window.scene, "clear_shapes", wraps=window.scene.clear_shapes) as clear:
            window.handle_text_results(results, "人, dog")
            clear.assert_called_once()
        shapes = Exporter.extract_shapes(window.scene)
        self.assertEqual([shape["label"] for shape in shapes], ["人", "人", "dog"])
        self.assertEqual(window.class_list, ["dog", "人"])
        saved = json.loads(self.image_path.with_suffix(".json").read_text(encoding="utf-8"))
        self.assertEqual([shape["label"] for shape in saved["shapes"]], ["人", "人", "dog"])

    def test_batch_chinese_prompt_uses_person_and_preserves_complete_yolo_box(self):
        class Processor:
            def __init__(self):
                self.prompts = []

            def set_image(self, image):
                return {}

            def set_text_prompt(self, prompt, state):
                self.prompts.append(prompt)
                mask = np.zeros((160, 200), dtype=np.float32)
                mask[20:106, 30:91] = 1
                mask[120:151, 60:76] = 1  # A detached visible body part.
                return {"masks": [mask], "scores": np.array([0.99]), "boxes": [np.array([30, 20, 91, 106])]}

        original_classes = ["dog"]
        for format_type in ("json", "yolo"):
            with self.subTest(format_type=format_type):
                processor = Processor()
                worker = main.SamBatchWorker(
                    processor, object(), [str(self.image_path)], ["人"],
                    format_type, original_classes, CanvasMode.RECT, overwrite=True,
                )
                errors = []
                worker.error.connect(errors.append)
                worker.run()
                self.assertEqual(errors, [])
                self.assertEqual(processor.prompts, ["person"])
                self.assertEqual(worker.class_list, ["dog", "人"])
                self.assertEqual(original_classes, ["dog"])
                if format_type == "json":
                    data = json.loads(self.image_path.with_suffix(".json").read_text(encoding="utf-8"))
                    self.assertEqual(data["shapes"][0]["label"], "人")
                    self.assertEqual(data["shapes"][0]["points"], [[30.0, 20.0], [91.0, 151.0]])
                else:
                    self.assertEqual(
                        self.image_path.with_suffix(".txt").read_text(encoding="utf-8"),
                        "1 0.302500 0.534375 0.305000 0.818750",
                    )

    def test_batch_tape_descriptions_keep_chinese_classes_in_json_and_yolo(self):
        labels = ["白色胶带", "蓝色胶带", "人手中拿的胶带"]
        expected = ["white tape", "blue tape", "tape held in a hand"]

        class Processor:
            def __init__(self):
                self.prompts = []

            def set_image(self, image):
                return {}

            def set_text_prompt(self, prompt, state):
                self.prompts.append(prompt)
                index = expected.index(prompt)
                mask = np.zeros((160, 200), dtype=np.bool_)
                mask[30:60, 10 + index * 50:40 + index * 50] = True
                return {"masks": [mask], "scores": np.array([0.99])}

        for format_type in ("json", "yolo"):
            with self.subTest(format_type=format_type):
                processor = Processor()
                worker = main.SamBatchWorker(
                    processor, object(), [str(self.image_path)], labels,
                    format_type, ["dog"], CanvasMode.POLY, overwrite=True,
                )
                errors, finished = [], []
                worker.error.connect(errors.append)
                worker.finished.connect(lambda *args: finished.append(args))
                worker.run()
                self.assertEqual(errors, [])
                self.assertEqual(finished, [(1, 1)])
                self.assertEqual(processor.prompts, expected)
                self.assertEqual(worker.class_list, ["dog"] + labels)
                if format_type == "json":
                    data = json.loads(self.image_path.with_suffix(".json").read_text(encoding="utf-8"))
                    self.assertEqual([shape["label"] for shape in data["shapes"]], labels)
                    self.assertEqual([shape["shape_type"] for shape in data["shapes"]], ["polygon"] * 3)
                else:
                    lines = self.image_path.with_suffix(".txt").read_text(encoding="utf-8").splitlines()
                    self.assertEqual([line.split()[0] for line in lines], ["1", "2", "3"])
                    self.assertTrue(all(len(line.split()) >= 7 for line in lines))

    def test_pose_rotation_survives_json_round_trip(self):
        window = self.new_window()
        template = {
            "name": "Regression pose",
            "keypoints": [
                {"name": "a", "default_pos": [0.2, 0.2]},
                {"name": "b", "default_pos": [0.8, 0.2]},
                {"name": "c", "default_pos": [0.5, 0.8]},
            ],
            "connections": [[0, 1], [1, 2]],
        }
        pose = PoseShape(QRectF(50, 30, 100, 100), template, "dog")
        window.scene.addItem(pose)
        pose.setRotation(32)
        self.assertTrue(window.auto_save_annotation())
        json_path = self.image_path.with_suffix(".json")
        data = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(data["shapes"][0]["angle"], 32)
        window.scene.clear_shapes()
        window._load_json(str(json_path))
        self.assertEqual(Exporter.extract_shapes(window.scene)[0]["angle"], 32)

    def test_loading_another_image_cancels_unfinished_drawing(self):
        scene = Canvas()
        scene.load_image(str(self.image_path))
        scene.set_mode(CanvasMode.POLY)
        scene.poly_pts = [QPointF(10, 10), QPointF(20, 30)]
        scene.update_temp_poly()
        scene.drawing = True
        scene.load_image(str(self.image_path))
        self.assertEqual(scene.poly_pts, [])
        self.assertIsNone(scene.temp_item)
        self.assertFalse(scene.drawing)
        self.assertEqual(Exporter.extract_shapes(scene), [])

    def test_switching_images_clears_polygon_vertex_redo(self):
        window = self.new_window()
        window.scene.set_mode(CanvasMode.POLY)
        window.scene.poly_pts = [QPointF(10, 10), QPointF(20, 30), QPointF(40, 20)]
        window.undo()
        self.assertTrue(window._poly_redo_pts)
        window.on_file_selected(QListWidgetItem(str(self.image_path)), None)
        window.redo()
        self.assertEqual(window.scene.poly_pts, [])
        self.assertEqual(window._poly_redo_pts, [])

    def test_switching_to_new_format_preserves_and_saves_annotations(self):
        window = self.new_window()
        window.scene.addItem(RectShape(QRectF(20, 30, 80, 60), "dog"))
        original = Exporter.extract_shapes(window.scene)
        window.set_current_format("yolo")
        self.assertEqual(Exporter.extract_shapes(window.scene), original)
        self.assertEqual(len(window.undo_stack), 1)
        self.assertTrue(self.image_path.with_suffix(".json").exists())
        self.assertTrue(window.auto_save_annotation())
        self.assertEqual(
            self.image_path.with_suffix(".txt").read_text(encoding="utf-8"),
            "0 0.300000 0.375000 0.400000 0.375000",
        )
        self.assertTrue(self.image_path.with_suffix(".json").exists())

    def test_switching_to_existing_format_loads_its_annotations(self):
        window = self.new_window()
        window.scene.addItem(RectShape(QRectF(20, 30, 80, 60), "dog"))
        self.image_path.with_suffix(".txt").write_text("0 0.5 0.5 0.2 0.25", encoding="utf-8")
        window.set_current_format("yolo")
        shapes = Exporter.extract_shapes(window.scene)
        self.assertEqual(shapes[0]["points"], [[80.0, 60.0], [120.0, 100.0]])
        saved_json = json.loads(self.image_path.with_suffix(".json").read_text(encoding="utf-8"))
        self.assertEqual(saved_json["shapes"][0]["points"], [[20.0, 30.0], [100.0, 90.0]])

    def test_failed_save_keeps_original_format_and_canvas(self):
        window = self.new_window()
        window.scene.addItem(RectShape(QRectF(20, 30, 80, 60), "dog"))
        original = Exporter.extract_shapes(window.scene)
        self.image_path.with_suffix(".txt").write_text("0 0.5 0.5 0.2 0.25", encoding="utf-8")
        with patch.object(Exporter, "save_json", side_effect=OSError("read-only directory")):
            window.set_current_format("yolo")
        self.assertEqual(window.current_format, "json")
        self.assertEqual(Exporter.extract_shapes(window.scene), original)

    def test_empty_save_only_removes_the_selected_format(self):
        window = self.new_window()
        for extension in (".json", ".txt", ".xml"):
            self.image_path.with_suffix(extension).write_text("independent annotations", encoding="utf-8")
        self.assertTrue(window.auto_save_annotation())
        self.assertFalse(self.image_path.with_suffix(".json").exists())
        self.assertTrue(self.image_path.with_suffix(".txt").exists())
        self.assertTrue(self.image_path.with_suffix(".xml").exists())
        window.save_annotation("yolo")
        self.assertFalse(self.image_path.with_suffix(".txt").exists())
        self.assertTrue(self.image_path.with_suffix(".xml").exists())

    def test_four_vertex_yolo_uses_polygon_or_obb_mode(self):
        window = self.new_window()
        txt_path = self.image_path.with_suffix(".txt")
        txt_path.write_text("0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5", encoding="utf-8")
        window.scene.set_mode(CanvasMode.POLY)
        window._load_yolo(str(txt_path), 200, 160)
        self.assertTrue(any(isinstance(item, PolyShape) for item in window.scene.items()))
        self.assertFalse(any(isinstance(item, RotatedRectShape) for item in window.scene.items()))
        window.scene.clear_shapes()
        window.scene.set_mode(CanvasMode.RBOX)
        window._load_yolo(str(txt_path), 200, 160)
        self.assertTrue(any(isinstance(item, RotatedRectShape) for item in window.scene.items()))

    def test_rectangle_polygon_and_obb_json_round_trip(self):
        window = self.new_window()
        polygon = QPolygonF([QPointF(20, 20), QPointF(90, 20), QPointF(100, 50), QPointF(60, 100), QPointF(20, 80)])
        for shape in (
            RectShape(QRectF(20, 30, 80, 60), "dog"),
            PolyShape(polygon, "dog"),
            RotatedRectShape(100, 80, 100, 40, 45, "dog"),
        ):
            window.scene.addItem(shape)
        original = {shape["type"]: shape for shape in Exporter.extract_shapes(window.scene)}
        self.assertTrue(window.auto_save_annotation())
        window.scene.clear_shapes()
        window.load_annotations(str(self.image_path))
        restored = {shape["type"]: shape for shape in Exporter.extract_shapes(window.scene)}
        self.assertEqual(restored, original)

    def test_rectangle_polygon_and_obb_yolo_round_trip(self):
        window = self.new_window()
        polygon = QPolygonF([QPointF(20, 20), QPointF(90, 20), QPointF(100, 50), QPointF(60, 100), QPointF(20, 80)])
        cases = (
            (CanvasMode.RECT, RectShape(QRectF(20, 30, 80, 60), "dog")),
            (CanvasMode.POLY, PolyShape(polygon, "dog")),
            (CanvasMode.RBOX, RotatedRectShape(100, 80, 100, 40, 45, "dog")),
        )
        for mode, shape in cases:
            with self.subTest(mode=mode):
                window.scene.clear_shapes()
                window.scene.set_mode(mode)
                window.scene.addItem(shape)
                original = Exporter.extract_shapes(window.scene)
                txt_path = self.image_path.with_suffix(".txt")
                Exporter.save_yolo(str(txt_path), 200, 160, original, ["dog"])
                window.scene.clear_shapes()
                window._load_yolo(str(txt_path), 200, 160)
                restored = Exporter.extract_shapes(window.scene)
                self.assertEqual(len(restored), 1)
                self.assertEqual(restored[0]["type"], original[0]["type"])
                for expected_point, actual_point in zip(original[0]["points"], restored[0]["points"]):
                    self.assertAlmostEqual(expected_point[0], actual_point[0], delta=0.001)
                    self.assertAlmostEqual(expected_point[1], actual_point[1], delta=0.001)

    def test_bridged_yolo_polygons_do_not_become_pose_annotations(self):
        window = self.new_window()
        for mode, predictor in ((CanvasMode.POLY, None), (CanvasMode.RECT, SimpleNamespace(task="segment"))):
            window.scene.set_mode(mode)
            window.current_yolo_predictor = predictor
            for count in (19, 28):
                with self.subTest(vertex_count=count, mode=mode):
                    angles = np.linspace(0, 2 * np.pi, count - 6, endpoint=False)
                    outer = [QPointF(60 + 30 * np.cos(angle), 70 + 30 * np.sin(angle)) for angle in angles]
                    component = [QPointF(140, 70), QPointF(165, 70), QPointF(150, 100)]
                    points = outer[:1] + component + [component[0], outer[0]] + outer[1:] + outer[:1]
                    self.assertEqual(len(points), count)
                    window.scene.clear_shapes()
                    window.scene.addItem(PolyShape(QPolygonF(points), "dog"))
                    original = Exporter.extract_shapes(window.scene)
                    txt_path = self.image_path.with_suffix(".txt")
                    Exporter.save_yolo(str(txt_path), 200, 160, original, ["dog"])
                    window.scene.clear_shapes()
                    window._load_yolo(str(txt_path), 200, 160)
                    restored = Exporter.extract_shapes(window.scene)
                    self.assertEqual(restored[0]["type"], "polygon")
                    self.assertEqual(len(restored[0]["points"]), count)
                    for expected_point, actual_point in zip(original[0]["points"], restored[0]["points"]):
                        self.assertAlmostEqual(expected_point[0], actual_point[0], delta=0.001)
                        self.assertAlmostEqual(expected_point[1], actual_point[1], delta=0.001)

    def test_rotated_box_xml_encloses_its_corners_and_reloads(self):
        window = self.new_window()
        window.scene.addItem(RotatedRectShape(100, 80, 100, 40, 45, "dog"))
        shapes = Exporter.extract_shapes(window.scene)
        xml_path = self.image_path.with_suffix(".xml")
        for points in (shapes[0]["points"], []):
            with self.subTest(has_explicit_corners=bool(points)):
                data = dict(shapes[0], points=points)
                Exporter.save_xml(str(xml_path), str(self.image_path), 200, 160, [data])
                box = ET.parse(xml_path).getroot().find("object/bndbox")
                self.assertEqual([int(box.findtext(key)) for key in ("xmin", "ymin", "xmax", "ymax")], [50, 30, 150, 130])
                window.scene.clear_shapes()
                window._load_xml(str(xml_path))
                restored = Exporter.extract_shapes(window.scene)
                self.assertEqual(restored[0]["type"], "rectangle")
                self.assertEqual(restored[0]["points"], [[50.0, 30.0], [150.0, 130.0]])


if __name__ == "__main__":
    unittest.main()
