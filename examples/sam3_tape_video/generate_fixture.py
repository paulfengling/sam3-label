"""Animate two generated stills into a disclosed synthetic tracking fixture.

This produces motion and partial occlusion for diagnosis. It is not a generated
natural video and makes no claim about performance on real recorded footage.
The foreground hand's original alpha is preserved for every composition.
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
WIDTH, HEIGHT, FPS, COUNT = 768, 512, 10, 30


def make_fixture():
    bg = Image.open(ROOT / "assets/source_hand_tape.png").convert("RGB")
    bg = bg.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    bg_np = np.asarray(bg)
    hand = Image.open(ROOT / "assets/source_foreground_hand.png").convert("RGBA")
    # Ignore near-transparent generation speckles only when finding the crop;
    # preserve the original alpha values inside the resulting foreground asset.
    crop_box = hand.getchannel("A").point(lambda p: 255 if p >= 8 else 0).getbbox()
    hand = hand.crop(crop_box)
    target_hand_height = 420
    hand = hand.resize(
        (round(hand.width * target_hand_height / hand.height), target_hand_height),
        Image.Resampling.LANCZOS,
    )
    hand_np = np.asarray(hand)
    (ROOT / "frames").mkdir(exist_ok=True)
    writer = cv2.VideoWriter(
        str(ROOT / "synthetic_tape_occlusion.mp4"),
        cv2.VideoWriter_fourcc(*"avc1"), FPS, (WIDTH, HEIGHT),
    )
    if not writer.isOpened():
        raise RuntimeError("The local video encoder could not create the MP4.")
    first_box = [154, 207, 641, 273]
    corners = np.float32([[154,207],[641,207],[641,273],[154,273]])
    records = []
    frames = []
    for index in range(COUNT):
        t = index / (COUNT - 1)
        angle = 1.8 * math.sin(2 * math.pi * t)
        tx = 12 * math.sin(math.pi * t)
        ty = 7 * math.sin(2 * math.pi * t)
        transform = cv2.getRotationMatrix2D((WIDTH / 2, HEIGHT / 2), angle, 1)
        transform[:, 2] += [tx, ty]
        frame = cv2.warpAffine(
            bg_np, transform, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        # The second hand moves in front of the same strip, then withdraws.
        phase = max(0.0, min(1.0, (index - 5) / 20.0))
        envelope = math.sin(math.pi * phase) ** 2
        hand_x = round(390 + 24 * math.sin(2 * math.pi * t))
        hand_y = round(540 - 480 * envelope)
        x0, y0 = max(0, hand_x), max(0, hand_y)
        x1, y1 = min(WIDTH, hand_x + hand.width), min(HEIGHT, hand_y + hand.height)
        if x1 > x0 and y1 > y0:
            layer = hand_np[y0-hand_y:y1-hand_y, x0-hand_x:x1-hand_x]
            alpha = layer[..., 3:4].astype(np.float32) / 255
            frame[y0:y1, x0:x1] = np.round(
                layer[..., :3] * alpha + frame[y0:y1, x0:x1] * (1-alpha)
            ).astype(np.uint8)
        out = Image.fromarray(frame)
        draw = ImageDraw.Draw(out)
        draw.rounded_rectangle((10, 10, 357, 49), radius=5, fill=(20, 24, 28))
        draw.text((18, 14), "SYNTHETIC TEST VIDEO", fill=(255,255,255))
        draw.text((18, 30), "AI stills + composited motion | not recorded footage", fill=(205,214,220))
        frame = np.asarray(out)
        out.save(ROOT / "frames" / f"{index:06d}.png")
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        frames.append(out)
        transformed = cv2.transform(corners[None], transform)[0]
        box = [float(transformed[:,0].min()), float(transformed[:,1].min()),
               float(transformed[:,0].max()), float(transformed[:,1].max())]
        records.append({"frame_index": index, "time_seconds": index / FPS,
                        "source_affine_2x3": transform.tolist(),
                        "target_search_box_xyxy": box,
                        "foreground_hand_xy": [hand_x,hand_y],
                        "occlusion_envelope": envelope})
    writer.release()
    thumb_w, thumb_h = 384, 256
    sheet = Image.new("RGB", (3 * thumb_w, 2 * (thumb_h + 27)), (239,242,245))
    painter = ImageDraw.Draw(sheet)
    for position, index in enumerate([0, 6, 11, 15, 20, 29]):
        x = (position % 3) * thumb_w
        y = (position // 3) * (thumb_h + 27)
        sheet.paste(frames[index].resize((thumb_w,thumb_h),Image.Resampling.LANCZOS), (x,y))
        painter.text((x+8,y+thumb_h+7), f"Frame {index:02d} | {index/FPS:.1f}s", fill=(22,28,35))
    sheet.save(ROOT / "contact_sheet.jpg", quality=93)
    metadata = {
        "fixture_type": "synthetic layered animation; not recorded footage",
        "target": "one long straight white strip of adhesive tape held at the left end",
        "width": WIDTH, "height": HEIGHT, "fps": FPS, "frame_count": COUNT,
        "duration_seconds": COUNT / FPS,
        "initial_box_xyxy": first_box, "initial_positive_point_xy": [450,240],
        "occlusion": "Foreground hand enters and partly covers the strip, strongest near frame 15; never covers the entire strip.",
        "source": "Two built-in image_gen stills; original files and prompts retained.",
        "annotation_warning": "Search boxes describe the source fixture and are not SAM predictions or exact masks.",
        "frames": records,
    }
    (ROOT / "fixture_metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    cap = cv2.VideoCapture(str(ROOT / "synthetic_tape_occlusion.mp4"))
    decoded = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        assert frame.shape[:2] == (HEIGHT, WIDTH)
        decoded += 1
    cap.release()
    assert decoded == COUNT, decoded
    print(json.dumps({"video":str(ROOT / "synthetic_tape_occlusion.mp4"),
                      "decoded_frames":decoded,"box":first_box,"size":[WIDTH,HEIGHT]},ensure_ascii=False))


if __name__ == "__main__":
    make_fixture()
