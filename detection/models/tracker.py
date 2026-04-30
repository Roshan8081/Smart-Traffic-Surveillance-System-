"""
Vehicle tracking across frames.

Design goals:
- Demo-friendly and lightweight (no heavy deps required)
- Assign stable object IDs
- Maintain centroid history per object

If ByteTrack is available at runtime, this module can use it. Otherwise it falls back to
a simple IoU-based tracker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2


BBoxXYXY = Tuple[int, int, int, int]
CentroidXY = Tuple[int, int]


@dataclass
class Track:
    object_id: int
    bbox: BBoxXYXY
    centroid: CentroidXY
    history: List[CentroidXY] = field(default_factory=list)
    hits: int = 1
    age: int = 1


def _bbox_centroid(b: BBoxXYXY) -> CentroidXY:
    x1, y1, x2, y2 = b
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def _iou(a: BBoxXYXY, b: BBoxXYXY) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _as_bbox_list(detections: Any) -> List[BBoxXYXY]:
    """
    Accepts detections in a few common shapes and returns list of bbox xyxy ints.
    Supports:
    - list[Detection] where Detection has .bbox
    - list[dict] with 'bbox' key
    - list/tuple with 4 numeric values per item
    """
    out: List[BBoxXYXY] = []
    if detections is None:
        return out

    for d in detections:
        bbox = None
        if hasattr(d, "bbox"):
            bbox = getattr(d, "bbox")
        elif isinstance(d, dict) and "bbox" in d:
            bbox = d["bbox"]
        elif isinstance(d, (list, tuple)) and len(d) >= 4:
            bbox = d[:4]

        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox
        out.append((int(x1), int(y1), int(x2), int(y2)))

    return out


class _SimpleIoUTracker:
    """
    Minimal multi-object tracker using greedy IoU matching.
    Keeps state in-memory and is good enough for demos.
    """

    def __init__(self, iou_threshold: float = 0.25, max_lost: int = 20) -> None:
        self.iou_threshold = float(iou_threshold)
        self.max_lost = int(max_lost)

        self._next_id = 1
        self._tracks: Dict[int, Track] = {}
        self._lost: Dict[int, int] = {}  # frames since last match

    def update(self, bboxes: List[BBoxXYXY]) -> List[Track]:
        # Age all tracks each frame
        for t in self._tracks.values():
            t.age += 1

        # If no existing tracks, initialize all detections.
        if not self._tracks:
            for b in bboxes:
                self._add_track(b)
            return list(self._tracks.values())

        track_ids = list(self._tracks.keys())

        # Build IoU matrix
        iou_mat: List[List[float]] = []
        for tid in track_ids:
            row = [_iou(self._tracks[tid].bbox, b) for b in bboxes]
            iou_mat.append(row)

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()

        # Greedy matching by highest IoU pairs
        while True:
            best_iou = 0.0
            best_ti = -1
            best_di = -1
            for ti, tid in enumerate(track_ids):
                if tid in matched_tracks:
                    continue
                for di in range(len(bboxes)):
                    if di in matched_dets:
                        continue
                    v = iou_mat[ti][di] if len(bboxes) > 0 else 0.0
                    if v > best_iou:
                        best_iou = v
                        best_ti = ti
                        best_di = di

            if best_ti == -1 or best_di == -1 or best_iou < self.iou_threshold:
                break

            tid = track_ids[best_ti]
            self._update_track(tid, bboxes[best_di])
            matched_tracks.add(tid)
            matched_dets.add(best_di)

        # Unmatched existing tracks: increment lost
        for tid in track_ids:
            if tid in matched_tracks:
                self._lost[tid] = 0
                continue
            self._lost[tid] = self._lost.get(tid, 0) + 1

        # Remove tracks lost for too long
        for tid, lost in list(self._lost.items()):
            if lost > self.max_lost:
                self._tracks.pop(tid, None)
                self._lost.pop(tid, None)

        # Unmatched detections: add new tracks
        for di, b in enumerate(bboxes):
            if di not in matched_dets:
                self._add_track(b)

        return list(self._tracks.values())

    def _add_track(self, bbox: BBoxXYXY) -> None:
        oid = self._next_id
        self._next_id += 1
        c = _bbox_centroid(bbox)
        self._tracks[oid] = Track(object_id=oid, bbox=bbox, centroid=c, history=[c], hits=1, age=1)
        self._lost[oid] = 0

    def _update_track(self, object_id: int, bbox: BBoxXYXY) -> None:
        t = self._tracks[object_id]
        c = _bbox_centroid(bbox)
        t.bbox = bbox
        t.centroid = c
        t.history.append(c)
        t.hits += 1


class ObjectTracker:
    """
    Public tracker wrapper used by `detection/main.py`.

    - `update(detections, frame=...)` returns list[Track]
    - `draw(frame, tracks)` overlays bbox + ID + short trail
    """

    def __init__(
        self,
        prefer_bytetrack: bool = True,
        iou_threshold: float = 0.25,
        max_lost: int = 20,
        trail: int = 20,
        min_hits_to_draw: int = 3,
    ) -> None:
        self.trail = int(trail)
        self.min_hits_to_draw = int(min_hits_to_draw)

        self._use_bytetrack = False
        self._simple = _SimpleIoUTracker(iou_threshold=iou_threshold, max_lost=max_lost)

        # Optional ByteTrack support (kept best-effort / demo friendly).
        if prefer_bytetrack:
            try:
                # Common python package name in some environments:
                # - `yolox.tracker.byte_tracker` (from YOLOX)
                from yolox.tracker.byte_tracker import BYTETracker  # type: ignore

                self._BYTETracker = BYTETracker  # store class
                self._bt = BYTETracker(
                    track_thresh=0.5,
                    track_buffer=max_lost,
                    match_thresh=0.8,
                    frame_rate=30,
                )
                self._use_bytetrack = True
            except Exception:
                self._use_bytetrack = False

    def update(self, detections: Any, frame=None) -> List[Track]:
        """
        Args:
            detections: output from YOLO module (list with .bbox) or similar
            frame: optional frame, unused for simple tracker
        """
        bboxes = _as_bbox_list(detections)

        if self._use_bytetrack:
            # ByteTrack expects detections in tlbr + score + class. We only have boxes here.
            # For demo simplicity, we fall back to IoU tracker unless the caller passes richer dets.
            # You can extend this later to use confidence/class from your YOLO detections.
            return self._simple.update(bboxes)

        return self._simple.update(bboxes)

    def draw(self, frame, tracks: Iterable[Track]) -> None:
        for t in tracks:
            # Reduce clutter: only draw tracks that have been matched a few times.
            if int(getattr(t, "hits", 0)) < self.min_hits_to_draw:
                continue

            x1, y1, x2, y2 = t.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 0), 2)
            cv2.putText(
                frame,
                f"ID {t.object_id}",
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 180, 0),
                2,
                cv2.LINE_AA,
            )

            # Draw a short centroid trail
            pts = t.history[-self.trail :]
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], (255, 180, 0), 2)

