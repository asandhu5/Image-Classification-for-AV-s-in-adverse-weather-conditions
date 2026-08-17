"""
models/detector.py
──────────────────
Thin wrapper around Ultralytics YOLOv8m for Stage 1 detection.

Role in the pipeline
────────────────────
YOLOv8m is responsible ONLY for localisation — predicting bounding boxes
and confidence scores.  Class labels are ignored; the CustomViT handles
all classification.  This design lets the ViT override YOLO's class head
with our 10-class taxonomy and adapt to adverse weather features that
YOLO's general COCO backbone may miss.

YOLOv8m is not trained during the thesis — its pre-trained weights are
used directly as a robust, real-time box proposer.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from config import Config


# Detection result type: (x1, y1, x2, y2, confidence_score)
Detection = Tuple[int, int, int, int, float]


class YOLOv8Detector:
    """
    YOLOv8m bounding-box detector.

    Parameters
    ----------
    weights : str
        Path to a .pt weights file, or "yolov8m.pt" to auto-download.
    conf    : float
        Minimum detection confidence threshold (0–1).
    iou     : float
        Non-maximum suppression IoU threshold (0–1).
    device  : str
        "cuda", "cpu", or a specific CUDA index like "cuda:0".
    """

    def __init__(
        self,
        weights: str   = Config.YOLO_WEIGHTS,
        conf:    float = Config.YOLO_CONF,
        iou:     float = Config.YOLO_IOU,
        device:  str   = Config.DEVICE,
    ) -> None:
        # Lazy import — ultralytics is only needed during inference,
        # not at training time.
        from ultralytics import YOLO

        self.conf   = conf
        self.iou    = iou
        self.device = device
        self.model  = YOLO(weights)

    # ── Core methods ──────────────────────────────────────────────────

    def detect(
        self,
        image: "str | Path | np.ndarray | Image.Image",
    ) -> List[Detection]:
        """
        Run YOLOv8m on a single image and return bounding boxes.

        Parameters
        ----------
        image : file path | BGR numpy array (cv2) | PIL Image

        Returns
        -------
        list of (x1, y1, x2, y2, confidence)  in pixel coordinates.
        Empty list if no detections pass the confidence threshold.
        """
        results = self.model.predict(
            source  = image,
            conf    = self.conf,
            iou     = self.iou,
            imgsz   = Config.YOLO_IMG_SIZE,
            device  = self.device,
            verbose = False,
        )

        detections: List[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                confidence      = float(box.conf[0])
                detections.append((x1, y1, x2, y2, confidence))

        return detections

    def crop_rois(
        self,
        image:      Image.Image,
        detections: List[Detection],
        padding:    float = Config.BOX_PADDING,
    ) -> List[Image.Image]:
        """
        Crop each detected region from a PIL image.

        A small fractional padding is added around each box to give
        the ViT edge context (same padding used in the dataset loader).

        Parameters
        ----------
        image      : PIL RGB image (source frame)
        detections : list of (x1, y1, x2, y2, conf) from self.detect()
        padding    : fractional padding (0.05 = 5 % of box side)

        Returns
        -------
        List of PIL crops — one per valid detection.
        """
        w_img, h_img = image.size
        crops: List[Image.Image] = []

        for (x1, y1, x2, y2, _) in detections:
            bw = x2 - x1
            bh = y2 - y1
            px = bw * padding
            py = bh * padding

            cx1 = max(0,     int(x1 - px))
            cy1 = max(0,     int(y1 - py))
            cx2 = min(w_img, int(x2 + px))
            cy2 = min(h_img, int(y2 + py))

            if cx2 > cx1 and cy2 > cy1:
                crops.append(image.crop((cx1, cy1, cx2, cy2)))

        return crops

    def detect_and_crop(
        self,
        image: "str | Path | np.ndarray | Image.Image",
    ) -> Tuple[List[Detection], List[Image.Image]]:
        """Convenience: detect + crop in one call."""
        pil  = self._to_pil(image)
        dets = self.detect(pil)
        return dets, self.crop_rois(pil, dets)

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _to_pil(image) -> Image.Image:
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        if isinstance(image, np.ndarray):
            # Assume BGR (OpenCV convention) → convert to RGB
            return Image.fromarray(image[:, :, ::-1])
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image)}")

    def warmup(self, img_size: int = Config.YOLO_IMG_SIZE) -> None:
        """
        Run one dummy forward pass so the first real detection isn't
        slow due to CUDA kernel compilation.
        """
        dummy = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        self.detect(dummy)
