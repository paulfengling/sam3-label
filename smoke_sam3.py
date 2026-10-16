"""Run real SAM3 text and point annotation using the local checkpoint."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
import argparse
import torch
import numpy as np
import json
import tempfile
from PIL import Image
from PySide6.QtWidgets import QApplication
from labelpaw.models.sam_client import (
    SAM_MODEL_MAP, Sam3ModelLoadWorker, Sam3InferenceWorker, sam_autocast,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=Path(__file__).parent / "sam3/assets/videos/0001/0.jpg")
    parser.add_argument("--prompt", default="person")
    parser.add_argument("--point", type=float, nargs=2, metavar=("X", "Y"))
    parser.add_argument("--box", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"))
    args = parser.parse_args()
    app = QApplication.instance() or QApplication([])
    loaded = []
    loader = Sam3ModelLoadWorker(SAM_MODEL_MAP["sam3"]["weight"])
    loader.loaded.connect(lambda *args: loaded.append(args))
    loader.run()
    model, processor, success, message = loaded[0]
    if not success:
        raise RuntimeError(message)
    print("Model loaded", flush=True)
    image_path = args.image
    image = Image.open(image_path).convert("RGB")
    worker = Sam3InferenceWorker()
    worker.model, worker.processor = model, processor
    with torch.inference_mode(), sam_autocast():
        worker.inference_state = processor.set_image(image)
    print("Image encoded", image.size, flush=True)
    points, texts = [], []
    errors = []
    worker.error.connect(lambda message: (errors.append(message), setattr(worker, "running", False)))
    worker.result_ready.connect(lambda *args: (points.append(args), setattr(worker, "running", False)))
    worker.text_result_ready.connect(lambda *args: (texts.append(args), setattr(worker, "running", False)))
    if args.box:
        # Prove rectangle selection works without text inference.
        worker.processor = None
        worker.request_box_inference(args.box, prompt=args.prompt)
    else:
        worker.request_text_inference(args.prompt)
        worker.run()
        assert not errors, errors
        assert texts and texts[0][0], "Text annotation returned no polygons"
        print("Text polygons:", len(texts[0][0]), flush=True)
        rect = texts[0][0][0]["rect"]
        worker.running = True
        x, y = args.point or (rect[0] + rect[2] / 2, rect[1] + rect[3] / 2)
        worker.request_inference(x, y, True, prompt=args.prompt)
    worker.run()
    assert not errors, errors
    assert points and len(points[0][0]) >= 3, "Point annotation returned no polygon"
    print("Box" if args.box else "Point", "polygon vertices:", len(points[0][0]), flush=True)
    print("Annotation box:", points[0][1], flush=True)
    if not args.box:
        matching = [result for result in texts[0][0] if np.allclose(result["rect"], points[0][1])]
        assert matching, "Semantic point selection should preserve the complete detected instance box"
    from labelpaw.data.exporter import Exporter
    shapes = ([{"label": args.prompt, "type": "polygon", "points": points[0][0]}] if args.box else
              [{"label": args.prompt, "type": "polygon", "points": result["poly_pts"]}
               for result in texts[0][0]])
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "annotation.json"
        Exporter.save_json(str(output), str(image_path), image.width, image.height, shapes)
        assert len(json.loads(output.read_text(encoding="utf-8"))["shapes"]) == len(shapes)
        output = Path(directory) / "annotation.txt"
        Exporter.save_yolo(str(output), image.width, image.height, shapes, [args.prompt])
        assert len(output.read_text().splitlines()) == len(shapes)
    print("JSON and YOLO export passed", flush=True)


if __name__ == "__main__":
    main()
