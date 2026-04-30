"""
Violation manager:
- Keeps a small in-memory store of recent violations (dedupe)
- Saves evidence snapshots to detection/violations/
- Sends violations to backend via utils/api_client.py
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
from collections import deque

from utils.api_client import APIClient


@dataclass(frozen=True)
class ViolationRecord:
    violation_id: str
    violation_type: str
    object_id: int
    timestamp: float
    image_path: str
    meta: Dict[str, Any]


@dataclass(frozen=True)
class ViolationOverlayItem:
    violation_type: str
    object_id: int
    timestamp: float
    plate_text: str
    image: Optional[np.ndarray]  # small BGR thumbnail


class ViolationManager:
    """
    Central place to:
    - prevent duplicates (per object_id + violation_type within a time window)
    - persist evidence image
    - send data to backend
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        dedupe_window_s: float = 3.0,
        once_per_object_types: Optional[List[str]] = None,
        api_client: Optional[APIClient] = None,
    ) -> None:
        self.output_dir = output_dir or (Path(__file__).resolve().parents[1] / "violations")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.dedupe_window_s = float(dedupe_window_s)
        self.once_per_object_types = {
            str(x).strip().lower()
            for x in (once_per_object_types or ["overspeed", "red_light"])
            if str(x).strip()
        }
        self.api = api_client or APIClient()

        # key -> last_timestamp
        self._last_seen: Dict[Tuple[int, str], float] = {}
        # key -> True (types we only want to emit once per object for the whole run)
        self._seen_once: Dict[Tuple[int, str], bool] = {}

        # violation_id -> record (in-memory log)
        self.records: Dict[str, ViolationRecord] = {}

        # Recent items for on-screen overlay (newest first)
        self._recent_overlay: Deque[ViolationOverlayItem] = deque(maxlen=6)

    def add_violation(
        self,
        frame,
        violation_type: str,
        object_id: int,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        meta: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> Optional[ViolationRecord]:
        """
        Add a violation if it's not a recent duplicate.

        Returns the ViolationRecord if created, else None if deduped/skipped.
        """
        ts = float(timestamp if timestamp is not None else time.time())
        oid = int(object_id)
        vtype = str(violation_type).strip().lower()
        key = (oid, vtype)

        # For noisy, continuously-true conditions (e.g. overspeed), only emit once per object.
        if vtype in self.once_per_object_types and self._seen_once.get(key, False):
            return None

        last = self._last_seen.get(key)
        if last is not None and (ts - last) < self.dedupe_window_s:
            return None

        self._last_seen[key] = ts
        if vtype in self.once_per_object_types:
            self._seen_once[key] = True

        meta_dict = dict(meta or {})
        plate_text = str(meta_dict.get("plate_text", "")).strip()

        violation_id = str(uuid.uuid4())

        overlay_lines: List[str] = [vtype.upper()]
        if plate_text:
            overlay_lines.append(f"PLATE: {plate_text}")
        if "speed_kmh" in meta_dict:
            try:
                overlay_lines.append(f"SPEED: {float(meta_dict['speed_kmh']):.1f} km/h")
            except Exception:
                pass

        image_path, thumb = self.save_image(
            frame,
            violation_id=violation_id,
            violation_type=vtype,
            bbox=bbox,
            return_thumbnail=True,
            overlay_lines=overlay_lines,
        )

        # Save a dedicated "plate crop" image for this violation (debugging + OCR tuning).
        # This is best-effort; we don't require a plate string to exist.
        if bbox is not None and frame is not None and frame is not None:
            try:
                plate_crop_path = self.save_plate_crop(
                    frame=frame,
                    violation_id=violation_id,
                    violation_type=vtype,
                    bbox=bbox,
                )
                if plate_crop_path:
                    meta_dict.setdefault("plate_image_path", plate_crop_path)
            except Exception:
                pass

        record = ViolationRecord(
            violation_id=violation_id,
            violation_type=vtype,
            object_id=oid,
            timestamp=ts,
            image_path=image_path,
            meta=meta_dict,
        )
        self.records[violation_id] = record

        # Push overlay item (most recent first)
        self._recent_overlay.appendleft(
            ViolationOverlayItem(
                violation_type=vtype,
                object_id=oid,
                timestamp=ts,
                plate_text=plate_text,
                image=thumb,
            )
        )

        # Fire-and-forget network call (API client should handle failures gracefully)
        try:
            self.api.send_violation(
                {
                    "violationId": record.violation_id,
                    "type": record.violation_type,
                    "objectId": record.object_id,
                    "timestamp": record.timestamp,
                    "imagePath": record.image_path,
                    "meta": record.meta,
                }
            )
        except Exception:
            # Keep pipeline running even if backend is down.
            pass

        return record

    def save_image(
        self,
        frame,
        violation_id: str,
        violation_type: str,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        return_thumbnail: bool = False,
        thumbnail_size: Tuple[int, int] = (180, 110),
        overlay_lines: Optional[List[str]] = None,
    ):
        """
        Save evidence snapshot for a violation.
        If bbox is provided, saves a cropped ROI; otherwise saves full frame.

        Returns:
            - saved file path as string
            - (optional) thumbnail BGR image if return_thumbnail=True
        """
        if frame is None:
            return ("", None) if return_thumbnail else ""

        img = frame
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            h, w = frame.shape[:2]
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            if x2 > x1 and y2 > y1:
                img = frame[y1:y2, x1:x2]

        # Draw overlay text (plate, speed, etc.) onto the evidence image.
        if overlay_lines:
            try:
                img = img.copy()
                x0, y0 = 8, 22
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.65
                thickness = 2
                pad_x, pad_y = 8, 6
                # Compute background box size
                widths = []
                heights = []
                for line in overlay_lines[:4]:
                    (tw, th), _ = cv2.getTextSize(str(line), font, scale, thickness)
                    widths.append(int(tw))
                    heights.append(int(th))
                bw = (max(widths) if widths else 0) + pad_x * 2
                bh = (sum(heights) if heights else 0) + pad_y * (len(heights) + 1)
                cv2.rectangle(img, (0, 0), (bw, bh), (10, 10, 10), -1)
                y = y0
                for line in overlay_lines[:4]:
                    cv2.putText(
                        img,
                        str(line),
                        (x0, y),
                        font,
                        scale,
                        (255, 255, 255),
                        thickness,
                        cv2.LINE_AA,
                    )
                    (tw, th), _ = cv2.getTextSize(str(line), font, scale, thickness)
                    y += int(th) + pad_y
            except Exception:
                pass

        filename = f"{violation_type}_{violation_id}.jpg"
        out_path = self.output_dir / filename
        cv2.imwrite(str(out_path), img)
        if not return_thumbnail:
            return str(out_path)

        thumb = None
        try:
            tw, th = int(thumbnail_size[0]), int(thumbnail_size[1])
            thumb = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        except Exception:
            thumb = None
        return str(out_path), thumb

    def save_plate_crop(
        self,
        frame,
        violation_id: str,
        violation_type: str,
        bbox: Tuple[int, int, int, int],
        min_size: Tuple[int, int] = (80, 24),
        upscale_to_width: int = 420,
    ) -> str:
        """
        Save a heuristic plate-region crop derived from the vehicle bbox.
        This is intended for debugging OCR quality and improving plate extraction.
        """
        if frame is None:
            return ""

        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            return ""

        vehicle = frame[y1:y2, x1:x2]
        vh, vw = vehicle.shape[:2]
        if vh <= 0 or vw <= 0:
            return ""

        # Heuristic plate ROI (lower-middle band), similar to `models/anpr.py`.
        py1 = int(vh * 0.45)
        py2 = int(vh * 0.92)
        px1 = int(vw * 0.10)
        px2 = int(vw * 0.90)
        py1 = max(0, min(vh - 1, py1))
        py2 = max(py1 + 1, min(vh, py2))
        px1 = max(0, min(vw - 1, px1))
        px2 = max(px1 + 1, min(vw, px2))
        plate = vehicle[py1:py2, px1:px2]

        ph, pw = plate.shape[:2]
        if ph < int(min_size[1]) or pw < int(min_size[0]):
            return ""

        # Upscale small crops to help OCR and visual inspection.
        try:
            if pw < int(upscale_to_width):
                scale = float(upscale_to_width) / max(1.0, float(pw))
                new_w = int(pw * scale)
                new_h = int(ph * scale)
                plate = cv2.resize(plate, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        except Exception:
            pass

        plates_dir = self.output_dir / "plates"
        plates_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{violation_type}_{violation_id}_plate.jpg"
        out_path = plates_dir / filename
        cv2.imwrite(str(out_path), plate)
        return str(out_path)

    def get_recent_overlays(self) -> List[ViolationOverlayItem]:
        """
        Returns recent violation overlay items (newest first).
        """
        return list(self._recent_overlay)

    # Optional integration point used by `main.py` via safe-call
    def handle(
        self,
        frame,
        tracks: Any,
        speed_info: Any = None,
        rlvd_events: Any = None,
        plates: Any = None,
    ) -> None:
        """
        Best-effort conversion of module outputs into stored violations.
        This keeps `main.py` simple while you iterate on module APIs.
        """
        now = time.time()

        # Plate map: object_id -> plate_text
        plate_map: Dict[int, str] = {}
        if isinstance(plates, dict):
            for oid, p in plates.items():
                try:
                    plate_text = str(p.get("plate_text", "")).strip()
                except Exception:
                    plate_text = ""
                if plate_text:
                    plate_map[int(oid)] = plate_text

        # If we learned plate text later, retrofit it into recent overlay items (best-effort).
        if plate_map and self._recent_overlay:
            updated: List[ViolationOverlayItem] = []
            changed = False
            for it in list(self._recent_overlay):
                if (not (it.plate_text or "").strip()) and int(it.object_id) in plate_map:
                    updated.append(
                        ViolationOverlayItem(
                            violation_type=it.violation_type,
                            object_id=int(it.object_id),
                            timestamp=float(it.timestamp),
                            plate_text=str(plate_map[int(it.object_id)]),
                            image=it.image,
                        )
                    )
                    changed = True
                else:
                    updated.append(it)
            if changed:
                self._recent_overlay.clear()
                for it in updated:
                    self._recent_overlay.append(it)

        # Overspeed: speed_info is expected like dict[object_id] -> SpeedInfo
        if isinstance(speed_info, dict):
            for oid, info in speed_info.items():
                if isinstance(info, dict):
                    is_over = bool(info.get("is_overspeed", False))
                    speed_kmh = float(info.get("speed_kmh", 0.0))
                else:
                    is_over = bool(getattr(info, "is_overspeed", False))
                    speed_kmh = float(getattr(info, "speed_kmh", 0.0))
                if is_over:
                    bbox = _bbox_for_object(tracks, int(oid))
                    self.add_violation(
                        frame=frame,
                        violation_type="overspeed",
                        object_id=int(oid),
                        bbox=bbox,
                        meta={"speed_kmh": speed_kmh, "plate_text": plate_map.get(int(oid), "")},
                        timestamp=now,
                    )

        # RLVD: rlvd_events expected list of objects with object_id + violated
        if isinstance(rlvd_events, list):
            for e in rlvd_events:
                violated = bool(getattr(e, "violated", False) or (isinstance(e, dict) and e.get("violated")))
                if not violated:
                    continue
                oid = int(getattr(e, "object_id", -1) if not isinstance(e, dict) else e.get("object_id", -1))
                if oid < 0:
                    continue
                bbox = _bbox_for_object(tracks, oid)
                self.add_violation(
                    frame=frame,
                    violation_type="red_light",
                    object_id=oid,
                    bbox=bbox,
                    meta={"plate_text": plate_map.get(int(oid), "")},
                    timestamp=now,
                )

        # (optional) you can later create a dedicated "plate_seen" event here if needed.


def _bbox_for_object(tracks: Any, object_id: int) -> Optional[Tuple[int, int, int, int]]:
    for t in tracks or []:
        oid = int(getattr(t, "object_id", -1))
        if oid == int(object_id):
            bbox = getattr(t, "bbox", None)
            if bbox is None:
                return None
            x1, y1, x2, y2 = bbox
            return (int(x1), int(y1), int(x2), int(y2))
    return None

