"""
Convert x-AnyLabeling linestrip annotations into YOLO segmentation labels.

This script walks the annotation directory, finds every x-AnyLabeling JSON
file, and for each linestrip shape it inflates the polyline into a closed
polygon of fixed half-width (perpendicular offset on each side). The result
is written as:
    - a YOLO segmentation .txt file (one line per instance: class_id followed
      by normalized polygon vertices) next to each source JSON;
    - a visualization folder containing a random subset of images together
      with a JSON in which the linestrips have been replaced by the inflated
      polygons, so the conversion can be inspected in x-AnyLabeling.

The linestrip-to-polygon conversion uses averaged segment normals at interior
vertices, with miter scaling, so the offset distance stays close to the
requested half-width on gentle bends without introducing self-intersections.
Shapes whose type is not "linestrip" are passed through unchanged in the
visualization JSON and ignored for the YOLO label file.

Inputs:
    - A YAML config file passed as the first command-line argument (without
      the .yaml extension), providing at least: annotation_dir, half_width_px,
      class_registry, viz_folder_name, viz_max, random_seed.
    - x-AnyLabeling JSON files under <annotation_dir> (searched recursively)
      with their matching .png images alongside.

Outputs:
    - A <stem>.txt file next to each processed <stem>.json, in YOLO
      segmentation format with normalized polygon coordinates.
    - <annotation_dir>/<viz_folder_name>/ containing up to viz_max sampled
      (image, JSON) pairs with polygons instead of linestrips, for visual QA.
    - A short summary printed to stdout.

Example: python json_to_yolo.py config

author: nicolas.gault@aviation-civile.gouv.fr
"""


import json
import shutil
import random
from pathlib import Path
import numpy as np
from PIL import Image
from utils import get_conf

def linestrip_to_polygon(points, half_width):
    """
    Convert a linestrip (list of [x, y]) into a closed polygon with given half-width
    on each side. Uses averaged normals at interior vertices to avoid self-intersection
    on gentle bends.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return None

    # Segment directions (unit vectors)
    seg_dirs = []
    for i in range(len(pts) - 1):
        d = pts[i + 1] - pts[i]
        n = np.linalg.norm(d)
        if n < 1e-9:
            seg_dirs.append(np.array([1.0, 0.0]))
        else:
            seg_dirs.append(d / n)
    seg_dirs = np.array(seg_dirs)

    # Normal of a 2D vector (dx, dy) -> (-dy, dx)
    seg_normals = np.stack([-seg_dirs[:, 1], seg_dirs[:, 0]], axis=1)

    # Per-vertex normal (averaged for interior vertices)
    vertex_normals = np.zeros_like(pts)
    vertex_normals[0] = seg_normals[0]
    vertex_normals[-1] = seg_normals[-1]
    for i in range(1, len(pts) - 1):
        n = seg_normals[i - 1] + seg_normals[i]
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            n = seg_normals[i]
        else:
            # Miter scaling so the offset stays at half_width perpendicular distance
            n = n / norm
            cos_half = np.dot(n, seg_normals[i])
            if abs(cos_half) > 1e-6:
                n = n / cos_half
        vertex_normals[i] = n

    left = pts + vertex_normals * half_width
    right = pts - vertex_normals * half_width

    # Closed polygon: left side forward, right side backward
    polygon = np.concatenate([left, right[::-1]], axis=0)
    return polygon


def process_json_file(json_path, conf):
    """Process a single x-AnyLabeling JSON: build YOLO txt and polygon-shapes list."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        # try:
        #     data = json.load(f)
        # except:
        #     print(f"Error reading json file: {json_path}")
        #     return None, None

    img_w = data.get("imageWidth")
    img_h = data.get("imageHeight")
    if img_w is None or img_h is None:
        # Try to read from the actual image
        img_path = json_path.with_suffix(".png")
        if not img_path.exists():
            return None, None
        with Image.open(img_path) as im:
            img_w, img_h = im.size

    yolo_lines = []
    polygon_shapes = []

    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "linestrip":
            # Pass through other shapes unchanged for the visualization JSON
            polygon_shapes.append(shape)
            continue

        label = shape.get("label", "object")
        points = shape.get("points", [])
        if len(points) < 2:
            continue

        polygon = linestrip_to_polygon(points, conf.half_width_px)
        if polygon is None:
            continue

        # Clip to image bounds
        polygon[:, 0] = np.clip(polygon[:, 0], 0, img_w - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, img_h - 1)

        # Class id
        class_id = conf.class_registry[label]

        # YOLO normalized coords
        norm = polygon.copy()
        norm[:, 0] /= img_w
        norm[:, 1] /= img_h
        flat = " ".join(f"{v:.6f}" for v in norm.flatten())
        yolo_lines.append(f"{class_id} {flat}")

        # Polygon shape entry for visualization JSON
        new_shape = dict(shape)
        new_shape["shape_type"] = "polygon"
        new_shape["points"] = polygon.tolist()
        polygon_shapes.append(new_shape)

    return yolo_lines, polygon_shapes


def write_visualization(json_path, polygon_shapes, viz_dir):
    """Copy image + write modified (polygon) JSON into the visualization folder."""
    img_path = json_path.with_suffix(".png")
    if not img_path.exists():
        return False

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["shapes"] = polygon_shapes

    # Make destination names unique by including the parent batch folder name
    stem = json_path.stem
    dst_img = viz_dir / f"{stem}.png"
    dst_json = viz_dir / f"{stem}.json"

    shutil.copy2(img_path, dst_img)
    data["imagePath"] = dst_img.name
    # imageData is large and not needed; drop it for the viz copy
    data["imageData"] = None
    with open(dst_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return True


def main():
    conf = get_conf()

    random.seed(conf.random_seed)
    processed_jsons = []  # list of (json_path, polygon_shapes)

    annotation_path = Path(conf.annotation_dir)
    for json_path in annotation_path.rglob("*.json"):
        yolo_lines, polygon_shapes = process_json_file(
            json_path, conf
        )
        if yolo_lines is None:
            print(f"[warn] could not process {json_path}")
            continue

        txt_path = json_path.with_suffix(".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))

        processed_jsons.append((json_path, polygon_shapes))

    # Visualization
    viz_dir = annotation_path / conf.viz_folder_name
    if viz_dir.exists():
        shutil.rmtree(viz_dir)
    viz_dir.mkdir(parents=True, exist_ok=True)

    candidates = [p for p in processed_jsons if p[1]]  # only those with shapes
    random.shuffle(candidates)
    sample = candidates[: conf.viz_max]

    n_written = 0
    for json_path, polygon_shapes in sample:
        if write_visualization(json_path, polygon_shapes, viz_dir):
            n_written += 1

    print(f"Processed {len(processed_jsons)} JSON files.")
    print(f"Classes: {conf.class_registry}")
    print(f"Visualization images written: {n_written} -> {viz_dir}")


if __name__ == "__main__":
    main()
