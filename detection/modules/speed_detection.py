"""
Speed estimation for tracked vehicles (demo-friendly).

Core API required by the prompt:
- calculate_speed(object_id, positions) -> (speed_kmh, is_overspeed)

Notes:
- This is an approximate speed estimator because pixel-to-meter calibration is unknown.
- You can tune METERS_PER_PIXEL for your camera setup to improve realism.
"""

from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import cv2


Centroid = Tuple[int, int]

SPEED_LIMIT_KMH = float(os.getenv("SPEED_LIMIT_KMH", "60"))

# Approximate conversion (tune for your camera / scene).
# Example: if 1 pixel roughly equals 0.05 meters, then 20 px/s ≈ 3.6 km/h.
METERS_PER_PIXEL = float(os.getenv("METERS_PER_PIXEL", "0.20"))

# If no timestamps are provided in `positions`, assume this FPS.
DEFAULT_FPS = 30.0


def _as_point_and_time(p: Any) -> Tuple[Optional[Centroid], Optional[float]]:
    """
    Normalizes a position item into ((x,y), t_seconds).

    Supported shapes:
    - (x, y)
    - (x, y, t)
    - {"centroid": (x,y), "t": t}
    - {"x": x, "y": y, "t": t}
    """
    if p is None:
        return None, None

    if isinstance(p, dict):
        if "centroid" in p:
            c = p.get("centroid")
            t = p.get("t")
            if isinstance(c, (list, tuple)) and len(c) >= 2:
                return (int(c[0]), int(c[1])), (float(t) if t is not None else None)
        if "x" in p and "y" in p:
            t = p.get("t")
            return (int(p["x"]), int(p["y"])), (float(t) if t is not None else None)

    if isinstance(p, (list, tuple)):
        if len(p) >= 3:
            return (int(p[0]), int(p[1])), float(p[2])
        if len(p) >= 2:
            return (int(p[0]), int(p[1])), None

    return None, None


def calculate_speed(object_id: int, positions: List[Any]) -> Tuple[float, bool]:
    """
    Estimate speed for a given object from its recent positions.

    Args:
        object_id: tracked vehicle ID (not used in math, useful for logging/debugging)
        positions: list of positions over time. Each item can be:
            - (x, y)
            - (x, y, t_seconds)
            - dict with {"centroid": (x,y), "t": t_seconds}

    Returns:
        speed_kmh, is_overspeed
    """
    if not positions or len(positions) < 2:
        return 0.0, False

    # Use a window (first -> last) to reduce jitter.
    (p1, t1) = _as_point_and_time(positions[0])
    (p2, t2) = _as_point_and_time(positions[-1])
    if p1 is None or p2 is None or p1 == p2:
        return 0.0, False

    dx = float(p2[0] - p1[0])
    dy = float(p2[1] - p1[1])
    dist_px = (dx * dx + dy * dy) ** 0.5

    # Determine delta time
    if t1 is not None and t2 is not None and t2 > t1:
        dt = float(t2 - t1)
    else:
        dt = max((len(positions) - 1) / DEFAULT_FPS, 1.0 / DEFAULT_FPS)

    if dt <= 1e-6:
        return 0.0, False

    # px -> meters -> m/s -> km/h
    dist_m = dist_px * METERS_PER_PIXEL
    speed_mps = dist_m / dt
    speed_kmh = speed_mps * 3.6

    is_overspeed = speed_kmh > SPEED_LIMIT_KMH
    return float(speed_kmh), bool(is_overspeed)


@dataclass
class SpeedInfo:
    speed_kmh: float
    is_overspeed: bool


class SpeedDetector:
    """
    Convenience wrapper used by `detection/main.py`.
    Consumes tracker tracks (with `.object_id` and `.history`) and returns speed info.
    """

    def __init__(
        self,
        speed_limit_kmh: float = SPEED_LIMIT_KMH,
        meters_per_pixel: float = METERS_PER_PIXEL,
        window_s: float = 1.0,
        smooth_alpha: float = 0.35,
    ) -> None:
        self.speed_limit_kmh = float(speed_limit_kmh)
        self.meters_per_pixel = float(meters_per_pixel)
        self.window_s = float(window_s)
        self.smooth_alpha = float(smooth_alpha)

        # object_id -> deque[(t, (x,y))]
        self._history: Dict[int, Deque[Tuple[float, Centroid]]] = {}
        # object_id -> smoothed speed
        self._ema_speed: Dict[int, float] = {}

    def update(self, tracks: Iterable[Any], frame=None) -> Dict[int, SpeedInfo]:
        """
        Args:
            tracks: iterable of Track-like objects with fields:
                - object_id: int
                - history: list of (x,y) centroid points
        Returns:
            dict[object_id] = SpeedInfo
        """
        now = time.time()
        out: Dict[int, SpeedInfo] = {}

        for t in tracks:
            oid = int(getattr(t, "object_id"))
            centroid = getattr(t, "centroid", None)
            if centroid is None:
                continue

            dq = self._history.get(oid)
            if dq is None:
                dq = deque(maxlen=120)
                self._history[oid] = dq

            dq.append((now, (int(centroid[0]), int(centroid[1]))))

            # Drop entries older than window
            cutoff = now - self.window_s
            while len(dq) >= 2 and dq[0][0] < cutoff:
                dq.popleft()

            positions = [(pt[0], pt[1], ts) for ts, pt in dq]

            # Use configured meters-per-pixel by temporarily overriding the module constant
            global METERS_PER_PIXEL
            old_mpp = METERS_PER_PIXEL
            METERS_PER_PIXEL = self.meters_per_pixel
            try:
                speed_kmh, _ = calculate_speed(oid, positions)
            finally:
                METERS_PER_PIXEL = old_mpp

            # Smooth speed to reduce flicker
            prev = self._ema_speed.get(oid, speed_kmh)
            smoothed = (self.smooth_alpha * speed_kmh) + ((1.0 - self.smooth_alpha) * prev)
            self._ema_speed[oid] = smoothed

            is_over = bool(smoothed > self.speed_limit_kmh)
            out[oid] = SpeedInfo(float(smoothed), is_over)

        return out

    def draw(self, frame, tracks: Iterable[Any], speed_info: Dict[int, SpeedInfo]) -> None:
        for t in tracks:
            oid = int(getattr(t, "object_id"))
            bbox = getattr(t, "bbox", None)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox

            info = speed_info.get(oid, SpeedInfo(0.0, False))
            color = (0, 0, 255) if info.is_overspeed else (0, 255, 0)
            label = f"{info.speed_kmh:.1f} km/h" + (" OVER" if info.is_overspeed else "")

            cv2.putText(
                frame,
                label,
                (int(x1), int(y2) + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

