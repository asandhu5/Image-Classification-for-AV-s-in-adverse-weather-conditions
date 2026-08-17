"""
scripts/prepare_cityscapes.py
─────────────────────────────
Convert Cityscapes instance annotations to the generic rois.json format.

Cityscapes layout expected at data/cityscapes/:
    data/cityscapes/
    ├── leftImg8bit/
    │   └── val/
    │       └── <city>/
    │           └── <city>_<seq>_<frame>_leftImg8bit.png
    └── gtFine/
        └── val/
            └── <city>/
                └── <city>_<seq>_<frame>_gtFine_instanceIds.png  (or _polygons.json)

Cityscapes provides per-image JSON annotations (_gtFine_polygons.json)
with bounding-box info per object instance.

Usage
─────
    python scripts/prepare_cityscapes.py
    python scripts/prepare_cityscapes.py --cs-root /path/to/cityscapes --split val
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

# Cityscapes label name → our BDD100K class mapping
CS_TO_BDD: Dict[str, str] = {
    "car":          "car",
    "truck":        "truck",
    "bus":          "bus",
    "motorcycle":   "motor",
    "bicycle":      "bicycle",
    "person":       "person",
    "rider":        "rider",
    "train":        "train",
    "traffic light": "traffic light",
    "traffic sign":  "traffic sign",
    "caravan":      "truck",   # closest equivalent
    "trailer":      "truck",
}

CLASS2IDX = {
    "traffic light": 0, "traffic sign": 1, "car": 2, "bus": 3,
    "person": 4, "train": 5, "truck": 6, "rider": 7, "motor": 8, "bicycle": 9,
}

MIN_BOX_AREA = 400


def polygon_to_bbox(polygon: List[List[float]]) -> tuple:
    """Convert a list of (x, y) vertices to (x1, y1, x2, y2)."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def process_city(
    city_gt_dir: Path,
    city_img_dir: Path,
) -> List[Dict]:
    rois: List[Dict] = []
    for gt_file in sorted(city_gt_dir.glob("*_gtFine_polygons.json")):
        img_stem = gt_file.stem.replace("_gtFine_polygons", "_leftImg8bit")
        img_path = city_img_dir / (img_stem + ".png")
        if not img_path.exists():
            img_path = city_img_dir / (img_stem + ".jpg")
        if not img_path.exists():
            continue

        with open(gt_file) as f:
            data = json.load(f)

        for obj in data.get("objects", []):
            label = obj.get("label", "").lower()
            # Strip group suffix (e.g. "carGroup" → "car")
            label_clean = label.replace("group", "").strip()

            if label_clean not in CS_TO_BDD:
                continue
            bdd_label = CS_TO_BDD[label_clean]
            if bdd_label not in CLASS2IDX:
                continue

            polygon = obj.get("polygon", [])
            if len(polygon) < 3:
                continue

            x1, y1, x2, y2 = polygon_to_bbox(polygon)
            w, h = x2 - x1, y2 - y1
            if w * h < MIN_BOX_AREA or w <= 0 or h <= 0:
                continue

            rois.append({
                "img_path":  str(img_path),
                "x1":        float(x1),
                "y1":        float(y1),
                "x2":        float(x2),
                "y2":        float(y2),
                "label_idx": CLASS2IDX[bdd_label],
            })
    return rois


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cs-root", default="data/cityscapes")
    p.add_argument("--split",   default="val", choices=["train", "val", "test"])
    args = p.parse_args()

    cs_root  = Path(args.cs_root)
    gt_root  = cs_root / "gtFine"     / args.split
    img_root = cs_root / "leftImg8bit" / args.split
    out_path = cs_root / "rois.json"

    all_rois: List[Dict] = []
    for city_dir in sorted(gt_root.iterdir()):
        if not city_dir.is_dir():
            continue
        city_img_dir = img_root / city_dir.name
        rois = process_city(city_dir, city_img_dir)
        all_rois.extend(rois)
        print(f"  {city_dir.name:<20}: {len(rois):,} ROIs")

    with open(out_path, "w") as f:
        json.dump(all_rois, f)
    print(f"\nCityscapes ({args.split}): {len(all_rois):,} total ROIs → {out_path}")


if __name__ == "__main__":
    main()
