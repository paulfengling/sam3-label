"""Verify local SAM2 loading and point segmentation in .sam3_label."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from PySide6.QtWidgets import QApplication
from labelpaw.models.sam_client import SAM_MODEL_MAP, Sam2ModelLoadWorker, sam_autocast


def main():
    app = QApplication.instance() or QApplication([])
    info = SAM_MODEL_MAP["sam2.1_hiera_large"]
    loaded = []
    worker = Sam2ModelLoadWorker(info["config"], info["weight"])
    worker.loaded.connect(lambda *args: loaded.append(args))
    worker.run()
    predictor, success, message = loaded[0]
    assert success, message
    print("SAM2 loaded", flush=True)
    image = Image.open(Path(__file__).parent / "sam3/assets/videos/0001/0.jpg").convert("RGB")
    with torch.inference_mode(), sam_autocast():
        predictor.set_image(np.array(image))
        masks, scores, _ = predictor.predict(
            point_coords=np.array([[image.width / 2, image.height / 2]]),
            point_labels=np.array([1]), multimask_output=True,
        )
    assert masks.shape == (3, image.height, image.width), masks.shape
    assert np.isfinite(scores).all() and masks.any()
    print("SAM2 point segmentation passed:", masks.shape, scores, flush=True)


if __name__ == "__main__":
    main()
