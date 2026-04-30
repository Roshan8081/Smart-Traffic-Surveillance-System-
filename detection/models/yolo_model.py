"""
YOLOv8 vehicle detector.

Detects only: car, motorcycle, bus, truck
Returns per detection: bbox (x1,y1,x2,y2), class_name, confidence
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from ultralytics import YOLO
import cv2


BBoxXYXY = Tuple[int, int, int, int]


@dataclass(frozen=True)
class Detection:
    bbox: BBoxXYXY
    class_name: str
    confidence: float


class YOLODetector:
    """
    Wrapper around Ultralytics YOLOv8 for vehicle-only detection.
    """

    # COCO class names we care about
    _ALLOWED_NAMES = {"car", "motorcycle", "bus", "truck"}

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf: float = 0.4,
        max_det: int = 50,
        min_box_area: int = 900,
        infer_width: int = 640,
        detect_every_n_frames: int = 3,
    ) -> None:
        self.model = YOLO(model_path)
        self.conf = float(conf)
        self.max_det = int(max_det)
        self.min_box_area = int(min_box_area)
        self.infer_width = int(infer_width)
        self.detect_every_n_frames = max(1, int(detect_every_n_frames))
        self._frame_idx = 0
        self._last_detections: List[Detection] = []

        # Select device once and keep a safe fallback.
        import torch
        if torch.cuda.is_available():
            self.device = "cuda:0"
            try:
                self.model.to(self.device)
            except Exception:
                self.device = "cpu"
        else:
            self.device = "cpu"
        print(f"[YOLO] Inference device: {self.device}", flush=True)

        # Ultralytics exposes class names as a dict: {id: name}
        self.class_names: Dict[int, str] = dict(self.model.names)
        self.allowed_class_ids = {
            cls_id for cls_id, name in self.class_names.items() if name in self._ALLOWED_NAMES
        }

    def _prepare_infer_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Resize frame for faster inference and return scale factor
        (scaled -> original coordinate multiplier).
        """
        h, w = frame.shape[:2]
        if w <= 0 or h <= 0 or self.infer_width <= 0 or w <= self.infer_width:
            return frame, 1.0
        ratio = float(self.infer_width) / float(w)
        nh = max(2, int(h * ratio))
        resized = cv2.resize(frame, (self.infer_width, nh), interpolation=cv2.INTER_LINEAR)
        inv_scale = float(w) / float(self.infer_width)
        return resized, inv_scale

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection on a single BGR frame (numpy array).
        """
        self._frame_idx += 1
        # Reuse previous detections on skipped frames to improve FPS.
        if self.detect_every_n_frames > 1 and (self._frame_idx % self.detect_every_n_frames) != 1:
            return list(self._last_detections)

        infer_frame, inv_scale = self._prepare_infer_frame(frame)
        results = self.model.predict(
            source=infer_frame,
            conf=self.conf,
            max_det=self.max_det,
            verbose=False,
            device=self.device,
        )
        if not results:
            self._last_detections = []
            return []

        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            self._last_detections = []
            return []

        # xyxy: (N,4), cls: (N,), conf: (N,)
        xyxy = r0.boxes.xyxy
        cls = r0.boxes.cls
        conf = r0.boxes.conf

        dets: List[Detection] = []
        for i in range(len(r0.boxes)):
            cls_id = int(cls[i].item())
            if cls_id not in self.allowed_class_ids:
                continue

            x1, y1, x2, y2 = xyxy[i].tolist()
            x1i = int(x1 * inv_scale)
            y1i = int(y1 * inv_scale)
            x2i = int(x2 * inv_scale)
            y2i = int(y2 * inv_scale)
            area = max(0, x2i - x1i) * max(0, y2i - y1i)
            if area < self.min_box_area:
                continue

            dets.append(
                Detection(
                    bbox=(x1i, y1i, x2i, y2i),
                    class_name=self.class_names.get(cls_id, str(cls_id)),
                    confidence=float(conf[i].item()),
                )
            )

        self._last_detections = dets
        return dets


# Backward-compatible alias for `main.py` (imports `YOLOModel`)
YOLOModel = YOLODetector

