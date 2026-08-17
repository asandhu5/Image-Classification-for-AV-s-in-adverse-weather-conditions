"""
scripts/prepare_acdc.py
───────────────────────
Convert ACDC (Adverse Conditions Dataset with Correspondences) object
detection annotations to the generic rois.json format consumed by
CrossDatasetROI.

ACDC dataset layout expected at data/acdc/:
    data/acdc/
    ├── rgb_anon/               ← images
    │   ├── fog/
    │   ├── night/
    │   ├── rain/
    │   └── snow/
    └── gt/                     ← annotations (COCO-format JSON files)
        ├── instancesonly_fog_gt_val.json
        ├── instancesonly_night_gt_val.json
        ├── instancesonly_rain_gt_val.json
        └── instancesonly_snow_gt_val.json

Output
──────
    data/acdc/rois.json   — list of ROI dicts for CrossDatasetROI

Usage
─────
    python scripts/prepare_acdc.py
    python scripts/prepare_acdc.py --acdc-root /path/to/acdc
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

# ACDC (Cityscapes-derived) category names → our 10-class mapping
ACDC_TO_BDD: Dict[str, str] = {
    "car":            "car",
    "truck":          "truck",
    "bus":            "bus",
    "motorcycle":     "motor",
    "bicycle":        "bicycle",
    "person":         "person",
    "rider":          "rider",
    "train":          "train",
    "traffic light":  "traffic light",
    "traffic sign":   "traffic sign",
    # Cityscapes also has "caravan", "trailer" — not in our taxonomy; skip
}

# Our 10-class → index mapping (must match Config.BDD_CLASSES order)
CLASS2IDX = {
    "traffic light": 0, "traffic sign": 1, "car": 2, "bus": 3,
    "person": 4, "train": 5, "truck": 6, "rider": 7, "motor": 8, "bicycle": 9,
}

MIN_BOX_AREA = 400


def parse_coco_json(json_path: Path, img_root: Path) -> List[Dict]:
    """Parse a COCO-format detection JSON and return ROI dicts."""
    with open(json_path) as f:
        data = json.load(f)

    # Build image id → file path map
    id2path: Dict[int, Path] = {}
    for img in data.get("images", []):
        img_path = img_root / img["file_name"]
        id2path[img["id"]] = img_path

    # Build category id → our label index
    cat_map: Dict[int, int] = {}
    for cat in data.get("categories", []):
        name = cat["name"].lower()
        if name in ACDC_TO_BDD and ACDC_TO_BDD[name] in CLASS2IDX:
            cat_map[cat["id"]] = CLASS2IDX[ACDC_TO_BDD[name]]

    rois: List[Dict] = []
    for ann in data.get("annotations", []):
        if ann["category_id"] not in cat_map:
            continue
        img_path = id2path.get(ann["image_id"])
        if img_path is None or not img_path.exists():
            continue

        x, y, w, h = ann["bbox"]   # COCO format: [x, y, width, height]
        if w * h < MIN_BOX_AREA:
            continue
        if ann.get("iscrowd", 0):
            continue

        rois.append({
            "img_path":  str(img_path),
            "x1":        float(x),
            "y1":        float(y),
            "x2":        float(x + w),
            "y2":        float(y + h),
            "label_idx": cat_map[ann["category_id"]],
        })
    return rois


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--acdc-root", default="data/acdc")
    args = p.parse_args()

    acdc_root = Path(args.acdc_root)
    gt_root   = acdc_root / "gt"
    img_root  = acdc_root / "rgb_anon"
    out_path  = acdc_root / "rois.json"

    all_rois: List[Dict] = []
    conditions = ["fog", "night", "rain", "snow"]

    for cond in conditions:
        json_path = gt_root / f"instancesonly_{cond}_gt_val.json"
        if not json_path.exists():
            print(f"  [skip] {json_path} not found")
            continue
        rois = parse_coco_json(json_path, img_root / cond)
        all_rois.extend(rois)
        print(f"  {cond:<8}: {len(rois):,} ROIs")

    with open(out_path, "w") as f:
        json.dump(all_rois, f)
    print(f"\nACDC: {len(all_rois):,} total ROIs saved → {out_path}")


if __name__ == "__main__":
    main()
