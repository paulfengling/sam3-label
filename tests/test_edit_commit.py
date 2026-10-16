"""Completed annotation edits persist once and remain undoable."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import main
from PySide6.QtCore import QRectF
from PySide6.QtGui import QImage, QColor
from PySide6.QtWidgets import QApplication
from labelpaw.data.exporter import Exporter
from labelpaw.graphics.shapes import RectShape


class EditCommitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        for target in ("main.MainWindow.restore_last_state", "main.QSettings", "main.DialogOver"):
            patcher = patch(target)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.image = Path(self.directory.name) / "image.png"
        image = QImage(200, 160, QImage.Format_RGB32)
        image.fill(QColor("white"))
        self.assertTrue(image.save(str(self.image)))
        self.window = main.MainWindow(auto_load_model=False)
        self.addCleanup(self.window.close)
        self.window.current_dir = self.directory.name
        self.window.current_image_path = str(self.image)
        self.window.load_classes(self.directory.name)
        self.window.scene.load_image(str(self.image))
        self.window.push_state()
        self.shape = RectShape(QRectF(20, 30, 80, 60))
        self.window.handle_new_shape(self.shape)

    def saved_points(self):
        return json.loads(self.image.with_suffix(".json").read_text())["shapes"][0]["points"]

    def test_new_annotation_is_saved_without_waiting_for_image_switch(self):
        self.assertEqual(self.saved_points(), [[20.0, 30.0], [100.0, 90.0]])

    def test_smart_edit_saves_and_undo_redo_restore_saved_geometry(self):
        self.window.scene.sam_enabled = True
        self.window.sam_client.current_model_key = "sam3"
        self.shape.setPos(10, 15)
        self.window.scene.state_changed.emit()
        changed = [[30.0, 45.0], [110.0, 105.0]]
        self.assertEqual(self.saved_points(), changed)
        self.window.undo()
        self.assertEqual(self.saved_points(), [[20.0, 30.0], [100.0, 90.0]])
        self.window.redo()
        self.assertEqual(self.saved_points(), changed)
        self.assertEqual(Exporter.extract_shapes(self.window.scene)[0]["points"], changed)

    def test_duplicate_commit_does_not_write_or_add_undo_step(self):
        self.shape.setPos(10, 15)
        with patch.object(self.window, "auto_save_annotation", wraps=self.window.auto_save_annotation) as save:
            self.window.scene.state_changed.emit()
            history_length = len(self.window.undo_stack)
            self.window.scene.state_changed.emit()
            self.assertEqual(save.call_count, 1)
            self.assertEqual(len(self.window.undo_stack), history_length)


if __name__ == "__main__":
    unittest.main()
