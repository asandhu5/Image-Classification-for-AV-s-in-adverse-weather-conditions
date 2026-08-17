"""
models/pipeline.py
──────────────────
Full two-stage inference pipeline:

    Stage 1  →  YOLOv8m       → bounding boxes (localisation)
    Stage 2  →  CustomViT     → class label + confidence (classification)

Usage
─────
    pipe    = ObjectClassificationPipeline.from_checkpoint("checkpoints/best.pth")
    results = pipe.run_on_image("frame.jpg", draw=True)
    results["annotated"].save("out.jpg")
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from config import Config
from models.detector import YOLOv8Detector
from models.vit import CustomViT


class ObjectClassificationPipeline:
    """
    Combines YOLOv8m (detector) and CustomViT (classifier) into a
    single callable object.

    The detector is stateless (no training involved).
    The classifier is loaded from a trained checkpoint.

    Parameters
    ----------
    detector   : YOLOv8Detector instance
    classifier : CustomViT instance (already moved to device, eval mode)
    transform  : val/test transform applied to each ROI crop
    device     : torch device string
    """

    def __init__(
        self,
        detector:   YOLOv8Detector,
        classifier: CustomViT,
        transform:  T.Compose,
        device:     str = Config.DEVICE,
    ) -> None:
        self.detector   = detector
        self.classifier = classifier.to(device).eval()
        self.transform  = transform
        self.device     = device

    # ── Factory ───────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str,
        device:    str = Config.DEVICE,
    ) -> "ObjectClassificationPipeline":
        """Load a pipeline from a saved .pth checkpoint."""
        from dataset.transforms import get_val_transforms
        from utils.checkpoint import load_checkpoint

        classifier = CustomViT()
        load_checkpoint(ckpt_path, classifier, device=device)
        classifier.to(device).eval()

        detector  = YOLOv8Detector(device=device)
        detector.warmup()

        return cls(detector, classifier, get_val_transforms(), device)

    # ── Core inference ────────────────────────────────────────────────

    @torch.inference_mode()
    def run_on_image(
        self,
        image: "str | Path | np.ndarray | Image.Image",
        draw:  bool = False,
    ) -> Dict[str, Any]:
        """
        Run the full two-stage pipeline on a single image.

        Parameters
        ----------
        image : file path | BGR numpy array (cv2) | PIL Image
        draw  : if True, annotate image with boxes and labels

        Returns
        -------
        dict:
            "detections"  – list of per-object dicts (see below)
            "inference_s" – wall-clock seconds for the full pipeline
            "annotated"   – PIL Image (only present when draw=True)

        Each detection dict:
            "bbox"       : [x1, y1, x2, y2]
            "det_conf"   : YOLOv8m confidence score
            "class_idx"  : predicted class index (0–9)
            "class_name" : human-readable class label
            "cls_conf"   : ViT softmax confidence for predicted class
            "probs"      : full 10-class probability distribution
        """
        t0  = time.perf_counter()
        pil = self._to_pil(image)

        # Stage 1: detection
        detections = self.detector.detect(pil)

        # Stage 2: classification of each ROI
        predictions: List[Dict] = []
        if detections:
            crops   = self.detector.crop_rois(pil, detections)
            tensors = torch.stack(
                [self.transform(c) for c in crops]
            ).to(self.device)

            logits = self.classifier(tensors)       # (N, 10)
            probs  = F.softmax(logits, dim=-1)      # (N, 10)
            conf, pred_idx = probs.max(dim=-1)

            for i, (x1, y1, x2, y2, det_conf) in enumerate(detections):
                predictions.append({
                    "bbox":       [x1, y1, x2, y2],
                    "det_conf":   round(det_conf, 4),
                    "class_idx":  int(pred_idx[i]),
                    "class_name": Config.IDX2CLASS[int(pred_idx[i])],
                    "cls_conf":   round(float(conf[i]), 4),
                    "probs":      probs[i].cpu().tolist(),
                })

        result = {
            "detections":  predictions,
            "inference_s": round(time.perf_counter() - t0, 4),
        }
        if draw:
            result["annotated"] = self._draw(pil, predictions)
        return result

    def run_on_video(
        self,
        video_path:  str,
        output_path: str,
        show:        bool = False,
    ) -> None:
        """
        Process a video file frame-by-frame and write annotated output.

        Parameters
        ----------
        video_path  : input video (mp4, avi, …)
        output_path : output annotated video path
        show        : display frames in real time (press Q to stop)
        """
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        n_frames, total_time = 0, 0.0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result       = self.run_on_image(frame, draw=True)
            total_time  += result["inference_s"]
            n_frames    += 1

            bgr_frame = np.array(result["annotated"])[:, :, ::-1]
            writer.write(bgr_frame)

            if show:
                cv2.imshow("Pipeline", bgr_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        writer.release()
        if show:
            cv2.destroyAllWindows()

        avg_ms = (total_time / n_frames * 1000) if n_frames else 0
        print(
            f"Video complete: {n_frames} frames → {output_path} "
            f"(avg {avg_ms:.1f} ms/frame)"
        )

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _to_pil(image) -> Image.Image:
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        if isinstance(image, np.ndarray):
            return Image.fromarray(image[:, :, ::-1])   # BGR → RGB
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image)}")

    @staticmethod
    def _draw(image: Image.Image, predictions: List[Dict]) -> Image.Image:
        """Annotate image with bounding boxes and class labels."""
        from PIL import ImageDraw, ImageFont

        canvas = image.copy()
        draw   = ImageDraw.Draw(canvas)

        palette = [
            "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
            "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
        ]
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14
            )
        except Exception:
            font = ImageFont.load_default()

        for pred in predictions:
            x1, y1, x2, y2 = pred["bbox"]
            color = palette[pred["class_idx"] % len(palette)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            label = f"{pred['class_name']} {pred['cls_conf']:.2f}"
            # Background chip for text readability
            text_w = len(label) * 8
            draw.rectangle([x1, y1 - 18, x1 + text_w, y1], fill=color)
            draw.text((x1 + 2, y1 - 17), label, fill="white", font=font)

        return canvas
