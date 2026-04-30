"""
Main entry point for the Smart Traffic Violation Detection System.

Pipeline (per frame):
OpenCV frame -> YOLOv8 detection -> tracking -> (speed, RLVD, ANPR) -> annotated display
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2

from models.yolo_model import YOLOModel
from models.tracker import ObjectTracker
from models.anpr import ANPR
from modules.speed_detection import SpeedDetector
from modules.rlvd import RedLightViolationDetector
from modules.violation_manager import ViolationManager


@dataclass(frozen=True)
class AppConfig:
    video_path: Path
    window_name: str = "Smart Traffic System"
    log_every_n_frames: int = 30
    draw_expensive_overlays_every_n_frames: int = 2


def _resolve_default_video_path() -> Path:
    """
    Returns the default input video path: detection/data/traffic.mp4
    This file is expected to exist in your repository structure.
    """
    # main.py lives in detection/, so data/ is next to it.
    return Path(__file__).resolve().parent / "data" / "traffic.mp4"

def _list_input_videos(data_dir: Path) -> List[Path]:
    """
    Returns all `.mp4` videos found in `detection/data/`, sorted by filename.
    This matches your setup where you have traffic1.mp4 ... traffic6.mp4.
    """
    return sorted(data_dir.glob("*.mp4"), key=lambda p: p.name.lower())


def _open_video(video_path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    return cap


def _compute_fps(prev_time_s: float) -> Tuple[float, float]:
    """
    Computes instantaneous FPS using the delta between now and prev_time.
    Returns (fps, now_time_s).
    """
    now = time.perf_counter()
    dt = max(now - prev_time_s, 1e-6)
    fps = 1.0 / dt
    return fps, now


def _put_fps(frame, fps: float) -> None:
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

def _draw_violation_panel(frame, items) -> None:
    """
    Draw a small interactive panel with recent violations:
    thumbnail + type + ID + plate.
    """
    if frame is None or not items:
        return

    x0, y0 = 15, 95
    pad = 8
    card_w, card_h = 320, 120

    for i, it in enumerate(items[:4]):  # show latest 4
        y = y0 + i * (card_h + pad)
        # Background card
        cv2.rectangle(frame, (x0, y), (x0 + card_w, y + card_h), (20, 20, 20), -1)
        cv2.rectangle(frame, (x0, y), (x0 + card_w, y + card_h), (80, 80, 80), 1)

        # Thumbnail
        if getattr(it, "image", None) is not None:
            thumb = it.image
            th, tw = thumb.shape[:2]
            tx, ty = x0 + 8, y + 8
            # Clamp paste region
            if ty + th < frame.shape[0] and tx + tw < frame.shape[1]:
                frame[ty : ty + th, tx : tx + tw] = thumb

        vtype = str(getattr(it, "violation_type", "violation")).upper()
        oid = int(getattr(it, "object_id", -1))
        plate = str(getattr(it, "plate_text", "")).strip() or "N/A"

        cv2.putText(
            frame,
            f"{vtype} | ID {oid}",
            (x0 + 200, y + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"PLATE: {plate}",
            (x0 + 200, y + 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 255, 200),
            2,
            cv2.LINE_AA,
        )


def _draw_track_summary_labels(frame, tracks: Any, speed_info: Any, plates: Any) -> None:
    """
    Draw compact per-vehicle labels above each bbox:
    [ID: <id> | <speed> km/h | <plate_or_N/A>]
    """
    if frame is None:
        return

    speed_map: Dict[int, float] = {}
    if isinstance(speed_info, dict):
        for oid, info in speed_info.items():
            try:
                if isinstance(info, dict):
                    speed_map[int(oid)] = float(info.get("speed_kmh", 0.0))
                else:
                    speed_map[int(oid)] = float(getattr(info, "speed_kmh", 0.0))
            except Exception:
                continue

    plate_map: Dict[int, str] = {}
    if isinstance(plates, dict):
        for oid, p in plates.items():
            try:
                txt = str((p or {}).get("plate_text", "")).strip()
            except Exception:
                txt = ""
            plate_map[int(oid)] = txt if txt else "N/A"

    for t in tracks or []:
        bbox = getattr(t, "bbox", None)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        oid = int(getattr(t, "object_id", -1))
        speed_kmh = float(speed_map.get(oid, 0.0))
        plate = plate_map.get(oid, "N/A")
        label = f"ID: {oid} | {speed_kmh:.0f} km/h | {plate}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        tx = max(0, int(x1))
        ty = max(th + 4, int(y1) - 8)
        bx2 = min(frame.shape[1] - 1, tx + tw + 10)
        by1 = max(0, ty - th - 8)
        cv2.rectangle(frame, (tx, by1), (bx2, ty + 4), (10, 10, 10), -1)
        cv2.putText(
            frame,
            label,
            (tx + 5, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def _safe_call(obj: Any, method_name: str, *args, **kwargs):
    """
    Calls obj.<method_name> if it exists, otherwise returns None.
    Keeps main loop resilient while you iterate on module APIs.
    """
    fn = getattr(obj, method_name, None)
    if callable(fn):
        return fn(*args, **kwargs)
    return None


def _flagged_object_ids(speed_info: Any, rlvd_events: Any) -> set[int]:
    """
    Determine which tracked object IDs are currently in violation.
    We use this to run OCR only for flagged vehicles (faster + better accuracy timing).
    """
    flagged: set[int] = set()

    # Overspeed
    if isinstance(speed_info, dict):
        for oid, info in speed_info.items():
            try:
                if isinstance(info, dict):
                    is_over = bool(info.get("is_overspeed", False))
                else:
                    is_over = bool(getattr(info, "is_overspeed", False))
                if is_over:
                    flagged.add(int(oid))
            except Exception:
                continue

    # Red light
    if isinstance(rlvd_events, list):
        for e in rlvd_events:
            try:
                violated = bool(
                    (isinstance(e, dict) and e.get("violated"))
                    or (not isinstance(e, dict) and getattr(e, "violated", False))
                )
                if not violated:
                    continue
                oid = int(e.get("object_id", -1) if isinstance(e, dict) else getattr(e, "object_id", -1))
                if oid >= 0:
                    flagged.add(oid)
            except Exception:
                continue

    return flagged


def _ocr_plates_for_ids(
    anpr: ANPR,
    frame,
    tracks: Any,
    object_ids: set[int],
    max_per_frame: int = 1,
    cooldown_s: float = 1.2,
) -> Dict[int, Dict[str, Any]]:
    """
    Run OCR for a subset of tracks (typically the currently flagged vehicles).
    Returns dict[object_id] = {"plate_text": str, "confidence": float}
    """
    out: Dict[int, Dict[str, Any]] = {}
    if frame is None or not object_ids:
        return out

    # Keep forced-OCR attempts throttled per object_id to avoid frame stalls.
    now_s = time.time()
    last_forced = getattr(_ocr_plates_for_ids, "_last_forced_ts", {})
    if not isinstance(last_forced, dict):
        last_forced = {}
    done = 0

    for t in tracks or []:
        try:
            oid = int(getattr(t, "object_id", -1))
        except Exception:
            continue
        if oid not in object_ids:
            continue
        if done >= max(1, int(max_per_frame)):
            break
        if (now_s - float(last_forced.get(oid, 0.0))) < float(cooldown_s):
            continue

        bbox = getattr(t, "bbox", None)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            continue

        vehicle_roi = frame[y1:y2, x1:x2]

        # For flagged vehicles we try harder:
        # 1) OCR a plate-region crop (helps a lot)
        # 2) fallback to ANPR's own heuristic/full ROI attempts
        plate_text, conf = ("", 0.0)
        try:
            vh, vw = vehicle_roi.shape[:2]
            py1 = int(vh * 0.45)
            py2 = int(vh * 0.92)
            px1 = int(vw * 0.10)
            px2 = int(vw * 0.90)
            plate_roi = vehicle_roi[py1:py2, px1:px2]

            old = getattr(anpr, "heuristic_crop", True)
            try:
                # plate_roi is already a "plate-ish" crop, so disable the extra heuristic crop.
                setattr(anpr, "heuristic_crop", False)
                plate_text, conf = anpr.extract_plate(plate_roi)
            finally:
                try:
                    setattr(anpr, "heuristic_crop", old)
                except Exception:
                    pass
        except Exception:
            plate_text, conf = ("", 0.0)

        if not plate_text:
            # Try with heuristic crop first (default), then fallback to full ROI.
            plate_text, conf = anpr.extract_plate(vehicle_roi)
        if not plate_text:
            old = getattr(anpr, "heuristic_crop", True)
            try:
                setattr(anpr, "heuristic_crop", False)
                plate_text2, conf2 = anpr.extract_plate(vehicle_roi)
            finally:
                try:
                    setattr(anpr, "heuristic_crop", old)
                except Exception:
                    pass
            if plate_text2:
                plate_text, conf = plate_text2, conf2

        if plate_text:
            out[oid] = {"plate_text": str(plate_text), "confidence": float(conf)}
        last_forced[oid] = now_s
        done += 1

    setattr(_ocr_plates_for_ids, "_last_forced_ts", last_forced)

    return out


def process_frame(
    frame,
    yolo: YOLOModel,
    tracker: ObjectTracker,
    speed_detector: SpeedDetector,
    rlvd: RedLightViolationDetector,
    anpr: ANPR,
    violation_manager: ViolationManager,
    draw_expensive_overlays: bool = True,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Process one frame through the full pipeline and return:
    - annotated_frame
    - a debug dict with intermediate outputs (useful for logging/testing)
    """
    debug: Dict[str, Any] = {}

    # 1) YOLO detection
    detections = yolo.detect(frame)
    debug["detections"] = detections

    # 2) Tracking (assign stable IDs)
    tracks = tracker.update(detections, frame=frame)
    debug["tracks"] = tracks

    # 3) Speed detection (updates per track over time)
    speed_info = speed_detector.update(tracks, frame=frame)
    debug["speed_info"] = speed_info

    # 4) Red light violation detection
    rlvd_events = rlvd.update(tracks, frame=frame)
    debug["rlvd_events"] = rlvd_events

    # 5) ANPR (plate OCR)
    # We prioritize OCR for currently-flagged vehicles to improve accuracy and performance.
    flagged_ids = _flagged_object_ids(speed_info, rlvd_events)

    plates: Any = {}
    # 5a) If you still want "background" recognition, keep it best-effort.
    base_plates = _safe_call(anpr, "recognize", frame, tracks)  # expected: (frame, tracks)
    if base_plates is None:
        base_plates = _safe_call(anpr, "update", tracks, frame=frame)  # alternate signature
    if isinstance(base_plates, dict):
        plates.update(base_plates)

    # 5b) Force a fresh OCR attempt for flagged IDs in this frame.
    flagged_plates = _ocr_plates_for_ids(
        anpr,
        frame,
        tracks,
        flagged_ids,
        max_per_frame=1,
        cooldown_s=1.2,
    )
    plates.update(flagged_plates)
    debug["plates"] = plates

    # 6) Central violation manager (save evidence + send to backend)
    # The manager can decide what constitutes a "violation" and how to persist it.
    _safe_call(
        violation_manager,
        "handle",
        frame=frame,
        tracks=tracks,
        speed_info=speed_info,
        rlvd_events=rlvd_events,
        plates=plates,
    )

    # 7) Draw annotations
    annotated = frame.copy()
    _safe_call(yolo, "draw", annotated, detections)  # optional helper if you add it later
    _safe_call(tracker, "draw", annotated, tracks)
    if draw_expensive_overlays:
        _safe_call(speed_detector, "draw", annotated, tracks, speed_info)
    _safe_call(rlvd, "draw", annotated, tracks, rlvd_events)
    if draw_expensive_overlays:
        _safe_call(anpr, "draw", annotated, tracks, plates)
    _draw_track_summary_labels(annotated, tracks, speed_info, plates)

    # 8) Interactive panel: show recent violations (thumbnail + plate)
    recent = _safe_call(violation_manager, "get_recent_overlays")
    if recent:
        _draw_violation_panel(annotated, recent)

    return annotated, debug


def run(video_path: Optional[Path] = None) -> None:
    """
    Runs the pipeline on a single video.
    """
    cfg = AppConfig(video_path=video_path or _resolve_default_video_path())

    # Initialize components (their internals will live in models/ and modules/)
    yolo = YOLOModel()
    tracker = ObjectTracker()
    speed_detector = SpeedDetector()
    rlvd = RedLightViolationDetector()
    # Throttled OCR: shows plates on bboxes without tanking FPS.
    # You can tune these if you want faster plate discovery:
    # - refresh_interval_s lower  -> more OCR attempts
    # - max_ocr_per_frame higher -> more vehicles get plates sooner (slower overall)
    anpr = ANPR(refresh_interval_s=1.5, max_ocr_per_frame=2, use_gpu=None)
    violation_manager = ViolationManager()
    print(f"[ANPR] EasyOCR GPU enabled: {getattr(anpr, 'use_gpu', False)}", flush=True)

    print(f"[INFO] Starting video: {cfg.video_path.name}", flush=True)
    cap = _open_video(cfg.video_path)
    prev_time = time.perf_counter()
    frame_idx = 0
    last_log_time = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print(f"[INFO] End of video: {cfg.video_path.name}", flush=True)
                break

            annotated, _ = process_frame(
                frame=frame,
                yolo=yolo,
                tracker=tracker,
                speed_detector=speed_detector,
                rlvd=rlvd,
                anpr=anpr,
                violation_manager=violation_manager,
                draw_expensive_overlays=(frame_idx % cfg.draw_expensive_overlays_every_n_frames == 0),
            )

            fps, prev_time = _compute_fps(prev_time)
            _put_fps(annotated, fps)

            cv2.imshow(cfg.window_name, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[INFO] 'q' pressed. Exiting current video.", flush=True)
                break

            frame_idx += 1
            if cfg.log_every_n_frames > 0 and (frame_idx % cfg.log_every_n_frames) == 0:
                now = time.perf_counter()
                elapsed = max(now - last_log_time, 1e-6)
                last_log_time = now
                print(
                    f"[PROGRESS] {cfg.video_path.name} | frame={frame_idx} | fps={fps:.1f} | dt_log={elapsed:.2f}s",
                    flush=True,
                )
    finally:
        cap.release()
        cv2.destroyAllWindows()


def run_all() -> None:
    """
    Runs the pipeline sequentially for each `.mp4` video in `detection/data/`.
    """
    data_dir = Path(__file__).resolve().parent / "data"
    videos = _list_input_videos(data_dir)

    if not videos:
        # Keep the error message explicit for easy debugging.
        raise FileNotFoundError(f"No input videos found in: {data_dir}")

    for idx, vp in enumerate(videos, start=1):
        print(f"[INFO] ({idx}/{len(videos)}) Queueing: {vp.name}", flush=True)
        run(video_path=vp)


if __name__ == "__main__":
    run_all()

