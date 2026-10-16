"""Convert one SAM instance mask, preserving disconnected parts and holes."""

import cv2
import numpy as np


def _join_polygon_components(components):
    """Join boundary rings with doubled bridges for odd-even polygon filling.

    A bridge is traversed in both directions, so it adds no filled area. Keeping
    every contour ring also preserves holes and islands nested inside holes.
    """
    merged = np.asarray(components[0], dtype=np.float64)
    for component in components[1:]:
        ring = np.asarray(component, dtype=np.float64)
        distances = ((merged[:, None, :] - ring[None, :, :]) ** 2).sum(axis=2)
        outer_index, inner_index = np.unravel_index(distances.argmin(), distances.shape)
        ring = np.concatenate((ring[inner_index:], ring[:inner_index], ring[inner_index:inner_index + 1]))
        merged = np.concatenate((merged[:outer_index + 1], ring,
                                 merged[outer_index:outer_index + 1], merged[outer_index + 1:]))
    return merged.tolist()


def mask_to_annotation(mask, box=None):
    """Enclose all visible components; optionally retain the detector's full box."""
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    while mask.ndim > 2 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim != 2:
        raise ValueError("SAM instance mask must have two spatial dimensions")
    foreground = (mask > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(foreground, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    all_points = np.concatenate(contours, axis=0)
    x, y, width, height = cv2.boundingRect(all_points)
    x1, y1, x2, y2 = float(x), float(y), float(x + width), float(y + height)
    if box is not None:
        if hasattr(box, "detach"):
            box = box.detach().cpu().numpy()
        detector_box = np.asarray(box, dtype=np.float64).reshape(-1)
        if detector_box.size == 4 and np.isfinite(detector_box).all():
            bx1, by1, bx2, by2 = detector_box
            if bx2 >= bx1 and by2 >= by1:
                x1, y1, x2, y2 = min(x1, bx1), min(y1, by1), max(x2, bx2), max(y2, by2)
    image_height, image_width = mask.shape
    x1, x2 = np.clip([x1, x2], 0, image_width)
    y1, y2 = np.clip([y1, y2], 0, image_height)

    components = []
    for contour in contours:
        epsilon = 0.002 * cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(polygon) < 3:
            px, py, pw, ph = cv2.boundingRect(contour)
            polygon = np.array([[px, py], [px + pw, py], [px + pw, py + ph], [px, py + ph]])
        components.append(polygon)
    if not components:
        components = [np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])]

    center, size, angle = cv2.minAreaRect(all_points)
    return {
        "poly_pts": _join_polygon_components(components),
        "rect": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
        "obb": [float(center[0]), float(center[1]), float(size[0]), float(size[1]), float(angle)],
    }
