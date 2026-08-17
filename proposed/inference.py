"""
inference.py
────────────
End-to-end inference using the full two-stage pipeline
(YOLOv8m detection → CustomViT classification).

Usage
─────
    # Single image — print detections JSON
    python inference.py --source image.jpg

    # Single image — draw boxes and save
    python inference.py --source image.jpg --draw --save annotated.jpg

    # Video file — save annotated video
    python inference.py --source video.mp4 --output out.mp4

    # Live webcam (device index)
    python inference.py --source 0
"""
import argparse
import json
import logging
from pathlib import Path

from config import Config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visual Object Classification — Inference")
    p.add_argument("--source",  required=True,
                   help="Image path | video path | webcam index (0, 1, …)")
    p.add_argument("--ckpt",    default=str(Config.CHECKPOINT_DIR / "best.pth"),
                   help="Path to CustomViT checkpoint")
    p.add_argument("--draw",    action="store_true",
                   help="Draw bounding boxes and labels on the output image")
    p.add_argument("--save",    default=None,
                   help="Save annotated image to this path (image mode only)")
    p.add_argument("--output",  default="output.mp4",
                   help="Output video path (video mode only)")
    p.add_argument("--show",    action="store_true",
                   help="Display output in a window (video / webcam only)")
    p.add_argument("--device",  default=Config.DEVICE,
                   help="Torch device: cuda | cpu")
    return p.parse_args()


def is_image(source: str) -> bool:
    return Path(source).suffix.lower() in {
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"
    }


def is_video(source: str) -> bool:
    return Path(source).suffix.lower() in {
        ".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"
    }


# ── Image mode ───────────────────────────────────────────────────────────────

def run_image(pipeline, source: str, args) -> None:
    """Run pipeline on a single image file and print / save results."""
    result = pipeline.run_on_image(source, draw=args.draw or args.save is not None)

    print(json.dumps(result["detections"], indent=2))
    print(f"\nDetected objects : {len(result['detections'])}")
    print(f"Inference time   : {result['inference_s']}s")

    if "annotated" in result and args.save:
        result["annotated"].save(args.save)
        logging.info(f"Annotated image saved → {args.save}")


# ── Video mode ───────────────────────────────────────────────────────────────

def run_video(pipeline, source: str, args) -> None:
    """Run pipeline frame-by-frame on a video file."""
    pipeline.run_on_video(source, output_path=args.output, show=args.show)


# ── Webcam mode ───────────────────────────────────────────────────────────────

def run_webcam(pipeline, device_idx: int, args) -> None:
    """Live inference from a webcam feed."""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(device_idx)
    if not cap.isOpened():
        raise IOError(f"Cannot open webcam {device_idx}")

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Webcam read failed — exiting.")
            break

        result    = pipeline.run_on_image(frame, draw=True)
        display   = np.array(result["annotated"])[:, :, ::-1]   # RGB → BGR
        cv2.imshow("Visual Object Classification", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s [%(levelname)s] %(message)s",
    )
    args = parse_args()

    # Load the pipeline from checkpoint
    logging.info(f"Loading pipeline from: {args.ckpt}")
    from models.pipeline import ObjectClassificationPipeline
    pipeline = ObjectClassificationPipeline.from_checkpoint(
        args.ckpt, device=args.device
    )
    logging.info("Pipeline ready.")

    # Route to the right mode
    source = args.source
    try:
        webcam_idx = int(source)
        logging.info(f"Mode: Webcam (device {webcam_idx})")
        run_webcam(pipeline, webcam_idx, args)
    except ValueError:
        if is_video(source):
            logging.info(f"Mode: Video → {source}")
            run_video(pipeline, source, args)
        else:
            logging.info(f"Mode: Image → {source}")
            run_image(pipeline, source, args)


if __name__ == "__main__":
    main()
