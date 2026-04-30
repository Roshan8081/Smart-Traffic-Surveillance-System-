"""
Red Light Violation Detection (RLVD) - demo-friendly.

Core API required by the prompt:
- check_rlvd(object_id, bbox, frame_time) -> bool

Logic:
- A hardcoded stop line is defined in image coordinates.
- A simulated traffic signal alternates RED/GREEN on a timer.
- If a vehicle's bottom-center point crosses the stop line during RED, it's a violation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2


BBoxXYXY = Tuple[int, int, int, int]
Point = Tuple[int, int]


# Stop line endpoints (x, y) in pixels.
# If not set manually, this module will auto-place it based on frame size.
STOP_LINE_P1: Point = (100, 400)
STOP_LINE_P2: Point = (540, 400)

# Signal simulation (seconds)
GREEN_DURATION_S = 10.0
RED_DURATION_S = 10.0


def _bottom_center(bbox: BBoxXYXY) -> Point:
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int(y2))


def _signal_is_red(frame_time: float) -> bool:
    """
    Timer-based signal:
    - cycle = GREEN then RED repeating.
    """
    cycle = GREEN_DURATION_S + RED_DURATION_S
    t = frame_time % cycle
    return t >= GREEN_DURATION_S


def _crossed_line(prev_pt: Optional[Point], curr_pt: Point, line_y: int) -> bool:
    """
    Checks if a point crossed a horizontal line at y=line_y between prev and curr.
    We consider crossing from above -> below.
    """
    if prev_pt is None:
        return False
    return prev_pt[1] < line_y <= curr_pt[1]

def _x_within_stop_segment(x: int) -> bool:
    x_min = min(STOP_LINE_P1[0], STOP_LINE_P2[0])
    x_max = max(STOP_LINE_P1[0], STOP_LINE_P2[0])
    return x_min <= int(x) <= x_max


_prev_points: Dict[int, Point] = {}
_violated_once: Dict[int, bool] = {}


def check_rlvd(object_id: int, bbox: BBoxXYXY, frame_time: float) -> bool:
    """
    Determine whether the object violates the red light rule at this frame.

    Args:
        object_id: track ID
        bbox: object's bbox (x1,y1,x2,y2) in pixels
        frame_time: seconds since some reference (e.g. time.time() or a running counter)

    Returns:
        True if this frame triggers a new red-light violation, else False.
    """
    oid = int(object_id)
    curr = _bottom_center(bbox)
    prev = _prev_points.get(oid)
    _prev_points[oid] = curr

    # Only trigger once per object to avoid spamming.
    if _violated_once.get(oid, False):
        return False

    is_red = _signal_is_red(frame_time)
    line_y = int(STOP_LINE_P1[1])  # horizontal line assumed
    crossed = _crossed_line(prev, curr, line_y)
    in_segment = _x_within_stop_segment(curr[0])

    if is_red and crossed and in_segment:
        _violated_once[oid] = True
        return True

    return False


@dataclass
class RLVDEvent:
    object_id: int
    violated: bool
    is_red: bool


class RedLightViolationDetector:
    """
    Convenience wrapper used by `detection/main.py`.
    """

    def __init__(
        self,
        stop_line_p1: Point = STOP_LINE_P1,
        stop_line_p2: Point = STOP_LINE_P2,
        green_s: float = GREEN_DURATION_S,
        red_s: float = RED_DURATION_S,
    ) -> None:
        global STOP_LINE_P1, STOP_LINE_P2, GREEN_DURATION_S, RED_DURATION_S
        STOP_LINE_P1 = stop_line_p1
        STOP_LINE_P2 = stop_line_p2
        GREEN_DURATION_S = float(green_s)
        RED_DURATION_S = float(red_s)

        self._start = time.time()
        self._auto_line_set = False

    def _maybe_autoset_line(self, frame) -> None:
        """
        Auto-place stop line if the defaults don't match the frame.
        Puts it around ~60% height and ~10%-90% width.
        """
        if self._auto_line_set or frame is None:
            return
        try:
            h, w = frame.shape[:2]
        except Exception:
            return

        global STOP_LINE_P1, STOP_LINE_P2
        y = int(h * 0.60)
        STOP_LINE_P1 = (int(w * 0.10), y)
        STOP_LINE_P2 = (int(w * 0.90), y)
        self._auto_line_set = True

    def update(self, tracks: Iterable[Any], frame=None) -> List[RLVDEvent]:
        self._maybe_autoset_line(frame)
        now = time.time() - self._start
        is_red = _signal_is_red(now)
        events: List[RLVDEvent] = []

        for t in tracks or []:
            oid = int(getattr(t, "object_id"))
            bbox = getattr(t, "bbox", None)
            if bbox is None:
                continue
            violated = check_rlvd(oid, bbox, now)
            events.append(RLVDEvent(object_id=oid, violated=bool(violated), is_red=bool(is_red)))

        return events

    def draw(self, frame, tracks: Iterable[Any], events: List[RLVDEvent]) -> None:
        if frame is None:
            return

        # Draw stop line
        cv2.line(frame, STOP_LINE_P1, STOP_LINE_P2, (0, 0, 255), 3)

        # Draw signal status
        now = time.time() - self._start
        is_red = _signal_is_red(now)
        label = "RED" if is_red else "GREEN"
        color = (0, 0, 255) if is_red else (0, 255, 0)
        cv2.putText(
            frame,
            f"Signal: {label}",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

        # Mark violations near the offending track
        ev_by_id = {e.object_id: e for e in (events or []) if e.violated}
        for t in tracks or []:
            oid = int(getattr(t, "object_id"))
            if oid not in ev_by_id:
                continue
            bbox = getattr(t, "bbox", None)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            cv2.putText(
                frame,
                "RLV!",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )

