"""
ANPR (Automatic Number Plate Recognition) using EasyOCR.

Core API required by prompt:
- class ANPR
- extract_plate(image) -> (plate_text, confidence)

This is a demo-friendly implementation:
- If you pass a cropped vehicle ROI, it will optionally try a heuristic crop for the plate region
  (lower-middle band) and run OCR on it.
- Text is cleaned to uppercase alphanumeric only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import easyocr
except Exception as e:  # pragma: no cover
    easyocr = None


PlateResult = Tuple[str, float]


def _clean_plate_text(text: str) -> str:
    """
    Keep only A-Z0-9 and uppercase the result.
    """
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


def _preprocess_for_ocr(bgr: np.ndarray) -> np.ndarray:
    """
    Simple preprocessing to improve OCR reliability.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    # Adaptive threshold helps under varying lighting
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
    )
    return thr


def _upscale_if_small(bgr: np.ndarray, target_width: int = 520) -> np.ndarray:
    """
    Upscale small ROIs to improve OCR accuracy.
    """
    try:
        h, w = bgr.shape[:2]
        if w <= 0 or h <= 0:
            return bgr
        if w >= int(target_width):
            return bgr
        scale = float(target_width) / float(w)
        nw = int(w * scale)
        nh = int(h * scale)
        if nw < 2 or nh < 2:
            return bgr
        return cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_CUBIC)
    except Exception:
        return bgr


def _enhance_gray_for_ocr(bgr: np.ndarray) -> np.ndarray:
    """
    Alternate variant: grayscale + histogram equalization (contrast normalization).
    Useful when adaptive thresholding fails (motion blur / shadows).
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    try:
        gray = cv2.equalizeHist(gray)
    except Exception:
        pass
    return gray


def _heuristic_plate_crop(vehicle_roi: np.ndarray) -> np.ndarray:
    """
    Heuristic: number plates often appear in the lower-middle area of the vehicle.
    Returns a cropped ROI likely containing the plate.
    """
    h, w = vehicle_roi.shape[:2]
    if h <= 0 or w <= 0:
        return vehicle_roi

    # Lower band (roughly bottom 45% to 80%) and middle width (15% to 85%)
    y1 = int(h * 0.45)
    y2 = int(h * 0.90)
    x1 = int(w * 0.10)
    x2 = int(w * 0.90)

    y1 = max(0, min(h - 1, y1))
    y2 = max(y1 + 1, min(h, y2))
    x1 = max(0, min(w - 1, x1))
    x2 = max(x1 + 1, min(w, x2))

    return vehicle_roi[y1:y2, x1:x2]


@dataclass
class OCRCandidate:
    text: str
    conf: float


@dataclass
class _PlateCache:
    plate_text: str
    confidence: float
    last_seen_s: float
    last_attempt_s: float


class ANPR:
    """
    EasyOCR-based plate reader.
    """

    def __init__(
        self,
        languages: Optional[List[str]] = None,
        use_gpu: Optional[bool] = None,
        min_length: int = 5,
        max_length: int = 12,
        heuristic_crop: bool = True,
        refresh_interval_s: float = 2.0,
        max_ocr_per_frame: int = 2,
    ) -> None:
        self.languages = languages or ["en"]
        # Auto-select OCR device unless explicitly overridden.
        if use_gpu is None:
            try:
                import torch
                self.use_gpu = bool(torch.cuda.is_available())
            except Exception:
                self.use_gpu = False
        else:
            self.use_gpu = bool(use_gpu)
        self.min_length = int(min_length)
        self.max_length = int(max_length)
        self.heuristic_crop = bool(heuristic_crop)
        self.refresh_interval_s = float(refresh_interval_s)
        self.max_ocr_per_frame = int(max_ocr_per_frame)

        self._reader = None
        # object_id -> cached plate result
        self._cache: Dict[int, _PlateCache] = {}

    def _get_reader(self):
        if easyocr is None:
            raise ImportError(
                "easyocr is not installed. Add it to requirements.txt (easyocr) and install dependencies."
            )
        if self._reader is None:
            self._reader = easyocr.Reader(self.languages, gpu=self.use_gpu)
        return self._reader

    def extract_plate(self, image: np.ndarray) -> PlateResult:
        """
        Args:
            image: BGR image. Can be a full frame, a vehicle crop, or a plate crop.

        Returns:
            (plate_text, confidence). If nothing reliable is found: ("", 0.0)
        """
        if image is None or image.size == 0:
            return "", 0.0

        roi = image
        if self.heuristic_crop:
            roi = _heuristic_plate_crop(image)

        # OCR performs much better when the plate occupies more pixels.
        roi = _upscale_if_small(roi, target_width=520)

        # If EasyOCR isn't installed yet, keep the pipeline running.
        if easyocr is None:
            return "", 0.0

        reader = self._get_reader()
        allow = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

        # Try multiple variants; keep the best-confidence candidate.
        best_text = ""
        best_conf = 0.0

        thr = _preprocess_for_ocr(roi)
        variants: List[np.ndarray] = [thr]
        try:
            variants.append(cv2.bitwise_not(thr))
        except Exception:
            pass
        variants.append(_enhance_gray_for_ocr(roi))

        for v in variants:
            results = reader.readtext(v, detail=1, paragraph=False, allowlist=allow)
            if not results:
                continue
            for item in results:
                if not isinstance(item, (list, tuple)) or len(item) < 3:
                    continue
                raw_text = str(item[1])
                conf = float(item[2])
                cleaned = _clean_plate_text(raw_text)
                if not (self.min_length <= len(cleaned) <= self.max_length):
                    continue
                if conf > best_conf:
                    best_text = cleaned
                    best_conf = conf

        return str(best_text), float(best_conf)

    # Optional helpers for future integration with `main.py` (safe-called there)
    def recognize(self, frame: np.ndarray, tracks: Any) -> dict:
        """
        Best-effort plate extraction for each track (throttled + cached).

        Why: doing OCR for every object on every frame will make playback extremely slow.
        Instead, we:
        - Keep a cache per object_id
        - Refresh at most N tracks per frame
        - Reuse cached plate text for drawing on every bbox

        Expects tracks with `.object_id` and `.bbox` (x1,y1,x2,y2).
        Returns dict[object_id] = {"plate_text": str, "confidence": float}
        """
        out: Dict[int, Dict[str, Any]] = {}
        if easyocr is None:
            return out
        if frame is None:
            return out

        now = time.time()

        # First: return cached results (so we can draw plates for every bbox)
        for t in tracks or []:
            oid = int(getattr(t, "object_id"))
            cached = self._cache.get(oid)
            if cached and cached.plate_text:
                out[oid] = {"plate_text": cached.plate_text, "confidence": float(cached.confidence)}

        # Second: decide which tracks should be OCR'd this frame.
        # Performance mode: only OCR new tracks here.
        # Vehicles with violations are handled by forced OCR in main.py.
        candidates: List[Tuple[int, Tuple[int, int, int, int]]] = []
        for t in tracks or []:
            try:
                oid = int(getattr(t, "object_id"))
            except Exception:
                continue

            # Only attempt OCR for new tracks in background mode.
            if oid in self._cache:
                continue

            bbox = getattr(t, "bbox", None)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = max(0, int(x2)), max(0, int(y2))
            if x2 <= x1 or y2 <= y1:
                continue

            candidates.append((oid, (x1, y1, x2, y2)))

        # OCR only a small subset per frame
        budget = max(0, int(self.max_ocr_per_frame))
        for oid, (x1, y1, x2, y2) in candidates[:budget]:
            h, w = frame.shape[:2]
            x1, y1 = max(0, min(w - 1, x1)), max(0, min(h - 1, y1))
            x2, y2 = max(0, min(w, x2)), max(0, min(h, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            vehicle_roi = frame[y1:y2, x1:x2]
            plate_text, conf = self.extract_plate(vehicle_roi)
            if plate_text:
                self._cache[oid] = _PlateCache(
                    plate_text=str(plate_text),
                    confidence=float(conf),
                    last_seen_s=now,
                    last_attempt_s=now,
                )
                out[oid] = {"plate_text": str(plate_text), "confidence": float(conf)}
            else:
                # Record attempt time even if no plate found
                cached = self._cache.get(oid)
                if cached:
                    cached.last_attempt_s = now
                    cached.last_seen_s = now
                else:
                    self._cache[oid] = _PlateCache(
                        plate_text="",
                        confidence=0.0,
                        last_seen_s=now,
                        last_attempt_s=now,
                    )

        return out

    def draw(self, frame: np.ndarray, tracks: Any, plates: Any) -> None:
        """
        Draw recognized plates near each track bbox if provided.
        """
        if frame is None or not plates:
            return

        for t in tracks or []:
            oid = int(getattr(t, "object_id"))
            bbox = getattr(t, "bbox", None)
            if bbox is None:
                continue

            info = plates.get(oid) if isinstance(plates, dict) else None
            if not info:
                continue

            text = info.get("plate_text", "")
            if not text:
                continue

            x1, y1, x2, y2 = bbox
            label = f"{text}"

            # Clean overlay: draw a small dark background behind the plate text
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            tx, ty = int(x1), max(0, int(y1) - 10)
            x2b = min(frame.shape[1] - 1, tx + tw + 10)
            y1b = max(0, ty - th - 10)
            cv2.rectangle(frame, (tx, y1b), (x2b, ty + 4), (10, 10, 10), -1)
            cv2.putText(
                frame,
                label,
                (tx + 5, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 255, 200),
                2,
                cv2.LINE_AA,
            )

