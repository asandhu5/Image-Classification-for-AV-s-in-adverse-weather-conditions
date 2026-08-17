"""
scripts/prepare_cadc.py
───────────────────────
Convert CADC (Canadian Adverse Driving Conditions) 2D annotations
to the generic rois.json format.

CADC layout expected at data/cadc/:
    data/cadc/
    ├── 2018_03_06/                 ← date-based sequences
    │   └── <sequence>/
    │       ├── labeled/
    │       │   └── image_00/
    │       │       └── data/       ← PNG images
    │       └── 3d_ann.json         ← 3D + 2D annotations per frame
    └── (other date directories)

CADC provides 3D bounding box annotations with 2D projections.
This script extracts the 2D projected boxes from the annotation JSON.

CADC annotation format per frame (in 3d_ann.json):
{
  "frames": [
    {
      "file_id": "0000000000",
      "annotations": [
        {
          "label": "Pedestrian",
          "camera_used": 0,
          "2d_bbox": {"xmin": 123, "ymin": 456, "xmax": 200, "ymax": 600}
        }
      ]
    }
  ]
}

Usage
─────
    python scripts/prepare_cadc.py
    python scripts/prepare_cadc.py --cadc-root /path/to/cadc
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

# CADC label → our BDD100K class mapping
CADC_TO_BDD: Dict[str, str] = {
    "pedestrian": "person",
    "car":        "car",
    "pickup_truck": "truck",
    "truck":      "truck",
    "bus":        "bus",
    "motorbike":  "motor",
    "bicycle":    "bicycle",
    "cyclist":    "rider",
    "animal":     None,       # not in our taxonomy
    "garbage_bins": None,
}

CLASS2IDX = {
    "traffic light": 0, "traffic sign": 1, "car": 2, "bus": 3,
    "person": 4, "train": 5, "truck": 6, "rider": 7, "motor": 8, "bicycle": 9,
}

MIN_BOX_AREA = 400
CAMERA_ID    = 0   # use front camera (camera_00 / image_00)


def process_sequence(seq_dir: Path) -> List[Dict]:
    """Process one CADC sequence directory."""
    ann_path = seq_dir / "3d_ann.json"
    img_dir  = seq_dir / "labeled" / "image_00" / "data"

    if not ann_path.exists() or not img_dir.exists():
        return []

    with open(ann_path) as f:
        data = json.load(f)

    rois: List[Dict] = []
    for frame in data.get("frames", []):
        fid      = frame.get("file_id", "0000000000")
        img_path = img_dir / f"{fid}.png"
        if not img_path.exists():
            continue

        for ann in frame.get("annotations", []):
            # Only use annotations visible from the front camera
            if ann.get("camera_used", CAMERA_ID) != CAMERA_ID:
                continue

            label = ann.get("label", "").lower()
            bdd_label = CADC_TO_BDD.get(label)
            if bdd_label is None or bdd_label not in CLASS2IDX:
                continue

            bbox = ann.get("2d_bbox", {})
            x1   = float(bbox.get("xmin", 0))
            y1   = float(bbox.get("ymin", 0))
            x2   = float(bbox.get("xmax", 0))
            y2   = float(bbox.get("ymax", 0))

            if (x2 - x1) * (y2 - y1) < MIN_BOX_AREA:
                continue

            rois.append({
                "img_path":  str(img_path),
                "x1":        x1, "y1": y1, "x2": x2, "y2": y2,
                "label_idx": CLASS2IDX[bdd_label],
            })
    return rois


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cadc-root", default="data/cadc")
    args = p.parse_args()

    cadc_root = Path(args.cadc_root)
    out_path  = cadc_root / "rois.json"

    all_rois: List[Dict] = []
    for date_dir in sorted(cadc_root.iterdir()):
        if not date_dir.is_dir():
            continue
        for seq_dir in sorted(date_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            rois = process_sequence(seq_dir)
            all_rois.extend(rois)
            print(f"  {date_dir.name}/{seq_dir.name}: {len(rois):,} ROIs")

    with open(out_path, "w") as f:
        json.dump(all_rois, f)
    print(f"\nCADC: {len(all_rois):,} total ROIs → {out_path}")


if __name__ == "__main__":
    main()
