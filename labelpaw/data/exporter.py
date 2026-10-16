import json
import math
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from labelpaw.graphics.shapes import RectShape, PolyShape, PointShape, RotatedRectShape, PoseShape


class Exporter:
    @staticmethod
    def extract_shapes(scene):
        shapes_data = []
        for item in scene.items():
            if isinstance(item, RectShape) and not getattr(item, 'is_temp', False):
                rect = item.rect()
                p1 = item.mapToScene(rect.topLeft())
                p2 = item.mapToScene(rect.bottomRight())
                shapes_data.append({
                    "label": item.label,
                    "type": "rectangle",
                    "points": [[p1.x(), p1.y()], [p2.x(), p2.y()]]
                })
            elif isinstance(item, PoseShape) and not getattr(item, 'is_temp', False):
                # Export pose
                keypoints = []
                for kp in item.kps:
                    pos = kp.scenePos()
                    keypoints.append([pos.x(), pos.y(), kp.visible_state])
                shapes_data.append({
                    "label": item.label,
                    "type": "pose",
                    "points": [],
                    "rect": [item.pos().x(), item.pos().y(), item.box_w, item.box_h],
                    "angle": item.rotation(),
                    "keypoints": keypoints,
                    "kpt_shape": item.template.get("kpt_shape", [len(item.kps), 3]),
                    "template_name": item.template.get("name", "Unknown")
                })
            elif isinstance(item, PolyShape) and not getattr(item, 'is_temp', False):
                poly = item.polygon()
                mapped_poly = item.mapToScene(poly)
                points = [[pt.x(), pt.y()] for pt in mapped_poly]
                shapes_data.append({
                    "label": item.label,
                    "type": "polygon",
                    "points": points
                })
            elif isinstance(item, PointShape) and not getattr(item, 'is_temp', False):
                center = item.mapToScene(item.rect().center())
                shapes_data.append({
                    "label": item.label,
                    "type": "point",
                    "points": [[center.x(), center.y()]]
                })
            elif isinstance(item, RotatedRectShape) and not getattr(item, 'is_temp', False):
                poly = item.polygon()
                points = [[poly[i].x(), poly[i].y()] for i in range(4)]

                cx = item.pos().x()
                cy = item.pos().y()
                w = item.box_w
                h = item.box_h
                angle = item.rotation()

                shapes_data.append({
                    "label": item.label,
                    "type": "obb",
                    "points": points,
                    "rect": [cx, cy, w, h],  # JSON 中保存: [中心X, 中心Y, 宽, 高]
                    "angle": angle
                })
        shapes_data.reverse()
        return shapes_data

    @staticmethod
    def save_json(filepath, image_path, image_width, image_height, shapes):
        data = {
            "version": "5.2.1",
            "flags": {},
            "shapes": [],
            "imagePath": os.path.basename(image_path),
            "imageHeight": int(image_height),
            "imageWidth": int(image_width)
        }
        for s in shapes:
            shape_type = s["type"]
            shape_dict = {
                "label": s["label"],
                "points": s["points"],
                "group_id": None,
                "shape_type": shape_type,
                "flags": {}
            }
            if shape_type == "obb":
                shape_dict["angle"] = s.get("angle", 0)
                shape_dict["rect"] = s.get("rect", [0, 0, 0, 0])
            elif shape_type == "pose":
                shape_dict["rect"] = s.get("rect", [0, 0, 0, 0])
                shape_dict["angle"] = s.get("angle", 0)
                shape_dict["kpt_shape"] = s.get("kpt_shape", [])
                shape_dict["keypoints"] = s.get("keypoints", [])
                shape_dict["template_name"] = s.get("template_name", "")

            data["shapes"].append(shape_dict)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def save_yolo(filepath, image_width, image_height, shapes, classes_list):
        lines = []
        for s in shapes:
            if s["label"] not in classes_list:
                continue
            class_id = classes_list.index(s["label"])

            if s["type"] == "rectangle":
                x1, y1 = s["points"][0]
                x2, y2 = s["points"][1]
                cx = ((x1 + x2) / 2.0) / image_width
                cy = ((y1 + y2) / 2.0) / image_height
                w = abs(x2 - x1) / image_width
                h = abs(y2 - y1) / image_height
                lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            elif s["type"] == "pose":
                rect = s.get("rect", [0, 0, 0, 0])
                val0, val1, w_val, h_val = rect
                kps = s.get("keypoints", [])
                
                kpts_x = [kp[0] for kp in kps if len(kp) >= 2]
                if kpts_x:
                    min_kp_x = min(kpts_x)
                    max_kp_x = max(kpts_x)
                    mid_kp_x = (min_kp_x + max_kp_x) / 2
                    if val0 < mid_kp_x - (max_kp_x - min_kp_x) * 0.15:
                        # rect is [xmin, ymin, w, h] (top-left aligned)
                        cx = val0 + w_val / 2
                        cy = val1 + h_val / 2
                    else:
                        # rect is [cx, cy, w, h] (center aligned)
                        cx = val0
                        cy = val1
                else:
                    cx, cy = val0, val1

                cx = cx / image_width
                cy = cy / image_height
                w = w_val / image_width
                h = h_val / image_height
                
                # Ensure coordinates are properly normalized and within bounds [0, 1]
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                w = max(0.0, min(1.0, w))
                h = max(0.0, min(1.0, h))
                
                parts = [f"{class_id}", f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]
                for kp in s.get("keypoints", []):
                    kx = kp[0] / image_width
                    ky = kp[1] / image_height
                    kx = max(0.0, min(1.0, kx))
                    ky = max(0.0, min(1.0, ky))
                    vis = kp[2]
                    parts.extend([f"{kx:.6f}", f"{ky:.6f}", f"{vis}"])
                lines.append(" ".join(parts))

            elif s["type"] == "obb":
                flat_pts = []
                for pt in s["points"][:4]:
                    flat_pts.append(f"{pt[0] / image_width:.6f} {pt[1] / image_height:.6f}")
                lines.append(f"{class_id} " + " ".join(flat_pts))

            elif s["type"] == "polygon":
                flat_pts = []
                for pt in s["points"]:
                    flat_pts.append(f"{pt[0] / image_width:.6f} {pt[1] / image_height:.6f}")
                lines.append(f"{class_id} " + " ".join(flat_pts))

            elif s["type"] == "point":
                cx = s["points"][0][0] / image_width
                cy = s["points"][0][1] / image_height
                pw, ph = 0.02, 0.02  # 微小框宽高
                cx = max(pw / 2, min(1.0 - pw / 2, cx))
                cy = max(ph / 2, min(1.0 - ph / 2, cy))
                lines.append(f"{class_id} {cx:.6f} {cy:.6f} {pw:.6f} {ph:.6f}")

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))

    @staticmethod
    def save_xml(filepath, image_path, image_width, image_height, shapes):
        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = "images"
        ET.SubElement(root, "filename").text = os.path.basename(image_path)
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(int(image_width))
        ET.SubElement(size, "height").text = str(int(image_height))
        ET.SubElement(size, "depth").text = "3"

        for s in shapes:
            if s["type"] == "pose":
                rect = s.get("rect", [0, 0, 0, 0])
                val0, val1, w, h = rect
                kps = s.get("keypoints", [])
                
                kpts_x = [kp[0] for kp in kps if len(kp) >= 2]
                if kpts_x:
                    min_kp_x = min(kpts_x)
                    max_kp_x = max(kpts_x)
                    mid_kp_x = (min_kp_x + max_kp_x) / 2
                    if val0 < mid_kp_x - (max_kp_x - min_kp_x) * 0.15:
                        # rect is [xmin, ymin, w, h] (top-left aligned)
                        cx = val0 + w / 2
                        cy = val1 + h / 2
                    else:
                        # rect is [cx, cy, w, h] (center aligned)
                        cx = val0
                        cy = val1
                else:
                    cx, cy = val0, val1

                min_x = cx - w / 2
                max_x = cx + w / 2
                min_y = cy - h / 2
                max_y = cy + h / 2
                
            elif s["type"] == "obb":
                # VOC stores an axis-aligned box enclosing all rotated corners.
                points = s.get("points", [])
                if points:
                    min_x = min(p[0] for p in points)
                    max_x = max(p[0] for p in points)
                    min_y = min(p[1] for p in points)
                    max_y = max(p[1] for p in points)
                else:
                    cx, cy, w, h = s.get("rect", [0, 0, 0, 0])
                    angle = math.radians(s.get("angle", 0))
                    half_w = (abs(w * math.cos(angle)) + abs(h * math.sin(angle))) / 2
                    half_h = (abs(w * math.sin(angle)) + abs(h * math.cos(angle))) / 2
                    min_x, max_x = cx - half_w, cx + half_w
                    min_y, max_y = cy - half_h, cy + half_h
                min_x, min_y = math.floor(min_x), math.floor(min_y)
                max_x, max_y = math.ceil(max_x), math.ceil(max_y)
            elif s["points"]:
                min_x = min(p[0] for p in s["points"])
                max_x = max(p[0] for p in s["points"])
                min_y = min(p[1] for p in s["points"])
                max_y = max(p[1] for p in s["points"])
            else:
                continue

            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = s["label"]
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"

            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(int(min_x))
            ET.SubElement(bndbox, "ymin").text = str(int(min_y))
            ET.SubElement(bndbox, "xmax").text = str(int(max_x))
            ET.SubElement(bndbox, "ymax").text = str(int(max_y))

        xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(xmlstr)
