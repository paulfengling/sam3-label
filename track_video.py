"""Track one object through a video and export annotated video and every frame."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import torch  # The local Python 3.10 environment requires torch before Qt.
from PySide6.QtCore import QCoreApplication

from labelpaw.data.video_io import decode_video
from labelpaw.video_pipeline import track_sequence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "outputs/video_tracking")
    parser.add_argument("--prompt", default="白色胶带")
    parser.add_argument("--label", help="Annotation label; defaults to the prompt")
    parser.add_argument("--seed-frame", type=int, default=0)
    parser.add_argument("--box", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--no-yolo", action="store_true")
    args = parser.parse_args()
    app = QCoreApplication.instance() or QCoreApplication([])

    def progress(done, total, message):
        print(f"[{done}/{total or '?'}] {message}", flush=True)

    sequence = decode_video(args.video, args.output, progress=progress)
    result = track_sequence(
        sequence, args.label or args.prompt, seed_frame_index=args.seed_frame,
        box_xyxy=args.box, text_prompt=args.prompt, export_yolo=not args.no_yolo,
        progress=progress,
    )
    print(json.dumps({
        "video": str(result.video_path), "output_dir": str(result.output_dir),
        "index": str(result.index_path), "frames": sequence.frame_count,
        "audio_included": result.audio_included,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
