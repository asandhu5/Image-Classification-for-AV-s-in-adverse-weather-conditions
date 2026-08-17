"""
scripts/prepare_once.py
───────────────────────
Convert ONCE (One Million Scenes) dataset 2D annotations
to the generic rois.json format.

ONCE layout expected at data/once/:
    data/once/
    ├── data/
    │   └── <sequence_id>/
    │       ├── cam01/              ← front-camera images (JPEG)
    │       │   └── <timestamp>.jpg
    │       └── <sequence_id>.json  ← annotation JSON
    └── ImageSets/
        └── val.txt

ONCE annotation JSON structure (per sequence):
{
  "sequence_id": "000001",
  "frames": [
    {
      "frame_id": "1616003433400",
      "annos": {
        "names": ["Car", "Pedestrian"],
        "boxes_2d": {
          "cam01": [[x1, y1, x2, y2], [x1, y1, x2, y2]]
        }
      }
    }
  ]
}

Usage
─────
    python scripts/prepare_once.py
    python scripts/prepare_once.py --once-root /path/to/once --split val
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

# ONCE class names → our BDD100K mapping
ONCE_TO_BDD: Dict[str, str] = {
    "car":          "car",
    "truck":        "truck",
    "bus":          "bus",
    "pedestrian":   "person",
    "cyclist":      "rider",
    "motorcycle":   "motor",
    "bicycle":      "bicycle",
    "barricade":    None,
    "trafficcone":  None,
}

CLASS2IDX = {
    "traffic light": 0, "traffic sign": 1, "car": 2, "bus": 3,
    "person": 4, "train": 5, "truck": 6, "rider": 7, "motor": 8, "bicycle": 9,
}

MIN_BOX_AREA = 400
CAMERA       = "cam01"   # front camera


def process_sequence(seq_dir: Path) -> List[Dict]:
    """Process one ONCE sequence directory."""
    ann_file = seq_dir / f"{seq_dir.name}.json"
    img_dir  = seq_dir / CAMERA

    if not ann_file.exists() or not img_dir.exists():
        return []

    with open(ann_file) as f:
        data = json.load(f)

    rois: List[Dict] = []
    for frame in data.get("frames", []):
        frame_id = frame.get("frame_id", "")
        annos    = frame.get("annos", {})
        names    = annos.get("names", [])
        boxes_2d = annos.get("boxes_2d", {}).get(CAMERA, [])

        img_path = img_dir / f"{frame_id}.jpg"
        if not img_path.exists():
            img_path = img_dir / f"{frame_id}.png"
        if not img_path.exists():
            continue

        for name, box in zip(names, boxes_2d):
            bdd_label = ONCE_TO_BDD.get(name.lower())
            if bdd_label is None or bdd_label not in CLASS2IDX:
                continue

            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            if (x2 - x1) * (y2 - y1) < MIN_BOX_AREA:
                continue

            rois.append({
                "img_path":  str(img_path),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "label_idx": CLASS2IDX[bdd_label],
            })
    return rois


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--once-root", default="data/once")
    p.add_argument("--split",     default="val", choices=["train", "val", "test"])
    args = p.parse_args()

    once_root = Path(args.once_root)
    split_file = once_root / "ImageSets" / f"{args.split}.txt"
    data_root  = once_root / "data"
    out_path   = once_root / "rois.json"

    # Collect sequence IDs from split file (if it exists)
    if split_file.exists():
        with open(split_file) as f:
            seq_ids = [l.strip() for l in f if l.strip()]
    else:
        # Fall back: use all available sequence directories
        seq_ids = [d.name for d in sorted(data_root.iterdir()) if d.is_dir()]

    all_rois: List[Dict] = []
    for seq_id in seq_ids:
        seq_dir = data_root / seq_id
        if not seq_dir.exists():
            continue
        rois = process_sequence(seq_dir)
        all_rois.extend(rois)
        print(f"  {seq_id}: {len(rois):,} ROIs")

    with open(out_path, "w") as f:
        json.dump(all_rois, f)
    print(f"\nONCE ({args.split}): {len(all_rois):,} total ROIs → {out_path}")


if __name__ == "__main__":
    main()
