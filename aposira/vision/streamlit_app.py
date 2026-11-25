"""
Smart Store Vision AI - Streamlit App (v5 with fixed video/zone display)
======================================================================
- **Video Processing** mode: upload a video → process entire file → download
  results. Much faster than frame-by-frame streaming.
- **Live Stream** mode: webcam / RTSP with frame dropping for smooth
  real-time playback.
- Clean modern UI with processing stats.
- FIXED: Video and zone image display issues
"""
import torch
# Patch torch.classes.__path__ to prevent Streamlit's file watcher from raising warnings/errors
try:
    torch.classes.__path__ = []
except AttributeError:
    pass

import streamlit as st
import cv2, time, tempfile, os
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aposira.vision.src.utils.workspace import PROJECT_ROOT, runtime_directory
from datetime import datetime
from typing import List, Optional
from collections import deque
import threading
from aposira.vision.src.alerts.run_store import RunStore
import hashlib
import logging
import warnings

# Initialize Streamlit page config FIRST - before any other Streamlit calls
if not globals().get("APOSIRA_EMBEDDED"):
    st.set_page_config(page_title="APOSIRA | Store Vision", page_icon="🏪", layout="wide")

# NOW set up logging and warnings AFTER page config
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("streamlit.runtime.memory_media_file_storage").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.media_file_manager").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Initialize session state IMMEDIATELY after set_page_config
# ---------------------------------------------------------------------------
def _init_session():
    defaults = {
        "run": False,
        "stats": [],
        "am": None,
        "hm": None,
        "zA": None,
        "zones": None,
        "sus_detector": None,
        "pose_analyzer": None,
        "uploaded_video_path": None,
        "uploaded_video_first_frame": None,  # NEW: Store first frame for zone editor
        "activities_log": [],
        "frame_skip": 2,
        "last_processed_time": 0.0,
        "frame_count": 0,
        "processed_video_path": None,
        # Queue/live processing state
        "last_queue_status": [],
        "last_object_events": [],
        # Enable all advanced detection modules by default for live video detection
        "pp_enabled": True,
        "bg_enabled": True,
        "line_enabled": True,
        "queue_enabled": True,
        "obj_enabled": True,
        "pose_gesture": True,
        "pose_fall": True,
        "pose_fight": True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_session()

# Shared APOSIRA look: standalone vision uses the same theme as the unified app.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aposira.ui import apply_theme, page_header
if not globals().get('APOSIRA_EMBEDDED'):
    apply_theme()


# Add path for local imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Import all required modules AFTER session init
from aposira.vision.src.detector.yolo_detector import YOLODetector
from aposira.vision.src.tracker.bytetrack import ByteTracker
from aposira.vision.src.analytics.zone_analytics import ZoneAnalytics, Zone
from aposira.vision.src.analytics.heatmap import HeatmapGenerator
from aposira.vision.src.analytics.behavior_detector import BehaviorDetector
from aposira.vision.src.alerts.alert_manager import AlertManager
from aposira.vision.src.analytics.suspicious_activity import (
    SuspiciousActivityDetector, SuspiciousActivity, ActivityType, Severity, SEVERITY_RANK,
)
from aposira.vision.src.utils.zone_io import load_zones, save_zones, list_runtime_zone_files
from aposira.vision.dashboard.polygon_editor import render as render_polygon_editor
from aposira.vision.dashboard.alerts_panel import render as render_alerts_panel
from aposira.vision.dashboard.threshold_tuner import render as render_threshold_tuner
from aposira.vision.dashboard.advanced_analytics import (
    render as render_advanced_analytics,
    _get_preprocessor, _get_bg_subtractor, _get_line_detector,
    _get_queue_detector, _get_object_detector, _get_face_blurrer,
    _get_video_clipper, _get_pose_detectors,
)
from aposira.vision.src.analytics.pose_detectors import PoseEvent


# ---------------------------------------------------------------------------
# Resource singletons
# ---------------------------------------------------------------------------
@st.cache_resource
def get_det(ms="s", cf=0.5, dv="cpu"):
    return YOLODetector(model_size=ms, confidence_threshold=cf, device=dv)


# Retail COCO class IDs: person (0), backpack (24), umbrella (25), handbag (26),
# tie (27), suitcase (28), bottle (39), wine glass (40), cup (41), bowl (45),
# chair (56), laptop (63), cell phone (67), book (73), clock (74), vase (75), scissors (76)
RETAIL_CLASSES = [0, 24, 25, 26, 28, 39, 41, 45, 63, 67, 73, 76]


def _build_suspicious_detector(enable_pose: bool, loiter_seconds: Optional[float] = None) -> Optional[SuspiciousActivityDetector]:
    pose = None
    if enable_pose:
        try:
            from aposira.vision.src.utils.config_schema import PoseConfig
            from aposira.vision.src.analytics.pose_analyzer import PoseAnalyzer
            pose = PoseAnalyzer(PoseConfig(enabled=True))
            st.session_state["pose_analyzer"] = pose
        except Exception as error:
            pose = None
            st.session_state["pose_analyzer"] = None
            st.session_state["vision_initialization_errors"] = [str(error)]
    else:
        st.session_state["pose_analyzer"] = None
    try:
        override = runtime_directory(PROJECT_ROOT / "config") / "suspicious_activity_override.yaml"
        config_path = override if override.exists() else PROJECT_ROOT / "config/suspicious_activity.yaml"
        det = SuspiciousActivityDetector.from_config_path(str(config_path), pose_analyzer=pose)
        # Ensure LOW/INFO activities (erratic movement, shelf revisits, gestures, etc.) are surfaced
        det.config.alerts.min_severity_to_notify = "INFO"
        det.config.alerts.min_confidence = 0.45
        if loiter_seconds is not None:
            det.config.detectors.loitering.default_dwell_seconds = float(loiter_seconds)
        det.config.detectors.loitering.confirmation_frames = min(4, int(det.config.detectors.loitering.confirmation_frames))
        det.config.detectors.restricted_zone.confirmation_frames = 1
        det.config.detectors.shelf_tampering.confirmation_frames = 3
        det.config.detectors.shelf_tampering.min_interaction_seconds = 2.0
        det.config.detectors.movement.min_track_length = 6
        det.config.detectors.movement.running_speed_px_s = 160.0
        det.config.detectors.movement.erratic_heading_changes = 4
        det.config.detectors.abnormal_shelf.min_dwell_seconds = 5.0
        det.config.detectors.abnormal_shelf.max_dwell_seconds = max(15.0, float(loiter_seconds or 30.0))
        return det
    except Exception as e:
        st.error(f"Failed to build suspicious detector: {e}")
        return None


def _associate_carried_objects(tracks: List, object_dets: List) -> dict:
    """Map person track_id -> list of carried/nearby object class names."""
    carried = {}
    for t in tracks:
        tx1, ty1, tx2, ty2 = t.bbox
        tcx, tcy = t.center
        items = []
        for od in object_dets:
            ox1, oy1, ox2, oy2 = od.bbox
            ocx, ocy = od.center
            # Inside expanded person bbox or within 75px of torso
            in_box = (tx1 - 35 <= ocx <= tx2 + 35) and (ty1 - 20 <= ocy <= ty2 + 35)
            dist = ((tcx - ocx) ** 2 + (tcy - ocy) ** 2) ** 0.5
            if in_box or dist <= 75:
                items.append(f"{od.class_name} ({od.confidence:.0%})")
        if items:
            carried[t.track_id] = items
    return carried


def _make_activity_from_aux(
    activity_id: str,
    activity_type: ActivityType,
    severity: Severity,
    confidence: float,
    track_id: Optional[int],
    zone_id: Optional[str],
    zone_name: Optional[str],
    camera_id: str,
    message: str,
    ts: float,
    evidence: Optional[dict] = None,
    rule_violations: Optional[List[str]] = None,
    track_ids: Optional[List[int]] = None,
) -> SuspiciousActivity:
    return SuspiciousActivity(
        activity_id=activity_id,
        activity_type=activity_type,
        severity=severity,
        confidence=float(confidence),
        track_id=track_id,
        track_ids=track_ids or ([track_id] if track_id is not None else []),
        zone_id=zone_id,
        zone_name=zone_name,
        camera_id=camera_id,
        message=message,
        timestamp=ts,
        evidence=evidence or {},
        rule_violations=rule_violations or [],
    )


def _draw_rich_live_annotations(
    frame: np.ndarray,
    det,
    trk,
    person_dets: List,
    object_dets: List,
    tracks: List,
    zA: ZoneAnalytics,
    zones: List[Zone],
    hm: HeatmapGenerator,
    show_hm: bool,
    bg_stats,
    line_det,
    queue_det,
    last_queue_status: List,
    obj_shelves: List,
    pose_ests: dict,
    pose_gestures: dict,
    carried_map: dict,
    activities_log: List[SuspiciousActivity],
    fc: int,
    total_frames: int,
    fps_val: float,
    ts: float,
    extra_label: str = "",
    error_callback=None,
) -> np.ndarray:
    """Draw comprehensive live computer-vision overlays on a video frame."""
    from aposira.vision.src.analytics.movement_features import compute_features, heading_degrees

    out = frame.copy()
    if show_hm and hm is not None:
        try:
            out = hm.get_overlay(out, 0.28)
        except Exception as error:
            if error_callback:
                error_callback("Annotation overlay", error)

    # 1. Draw Zones
    out = zA.draw_zones(out)

    # 2. Draw Shelf Polygons (Object Monitor)
    for s in obj_shelves or []:
        try:
            pts = np.array(s.polygon, dtype=np.int32)
            cv2.polylines(out, [pts], True, (220, 120, 255), 2)
            cv2.putText(out, f"Monitor: {s.name}", (pts[0][0] + 4, pts[0][1] + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 120, 255), 1)
        except Exception as error:
            if error_callback:
                error_callback("Annotation overlay", error)

    # 3. Draw Queue Polygons + Live Queue Status
    if queue_det is not None:
        q_status_map = {qs.queue_id: qs for qs in (last_queue_status or [])}
        for qid, q in queue_det.queues.items():
            try:
                pts = np.array(q.polygon, dtype=np.int32)
                qs = q_status_map.get(qid)
                q_col = (0, 140, 255) if (qs and qs.length >= q.min_queue_length) else (255, 200, 0)
                cv2.polylines(out, [pts], True, q_col, 2)
                q_txt = f"{q.name}: {qs.length if qs else 0} ppl | Wait {qs.estimated_wait_seconds if qs else 0:.0f}s"
                cv2.putText(out, q_txt, (pts[0][0] + 4, max(18, pts[0][1] - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, q_col, 2)
            except Exception as error:
                if error_callback:
                    error_callback("Annotation overlay", error)

    # 4. Draw Virtual Line Crossing Lines + Live In/Out Counts
    if line_det is not None:
        l_counts = line_det.get_counts()
        for lid, line in line_det.lines.items():
            try:
                cv2.line(out, line.p1, line.p2, (0, 255, 255), 2)
                c_in = l_counts.get((lid, line.left_label), 0)
                c_out = l_counts.get((lid, line.right_label), 0)
                lbl = f"{line.name} [{line.left_label.upper()}:{c_in} | {line.right_label.upper()}:{c_out}]"
                cv2.putText(out, lbl, (line.p1[0] + 6, max(20, line.p1[1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            except Exception as error:
                if error_callback:
                    error_callback("Annotation overlay", error)

    # 5. Draw Retail / Carried Objects (bags, backpacks, handbags, bottles, phones, etc.)
    for od in object_dets:
        x1, y1, x2, y2 = od.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 180, 255), 2)
        olbl = f"OBJ: {od.class_name.upper()} {od.confidence:.0%}"
        (lw, lh), _ = cv2.getTextSize(olbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x1, max(0, y1 - lh - 8)), (x1 + lw + 6, y1), (0, 140, 230), -1)
        cv2.putText(out, olbl, (x1 + 3, max(10, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # 6. Draw Person Tracks, Trajectories, Pose Skeletons, Movement State, Dwell & Carried Bags
    out = trk.draw_tracks(out, tracks)
    zone_lookup = {z.zone_id: z for z in zones}

    for t in tracks:
        x1, y1, x2, y2 = t.bbox
        tid = t.track_id
        feats = compute_features(t.history)
        spd = feats.get("avg_speed", 0.0)
        pk_spd = feats.get("peak_speed", 0.0)
        turns = feats.get("heading_change_count", 0)

        # Compass direction
        compass = ""
        if len(t.history) >= 2:
            hd = (heading_degrees(t.history[-2][1], t.history[-1][1]) + 360) % 360
            compass = "S" if 45 <= hd < 135 else ("W" if 135 <= hd < 225 else ("N" if 225 <= hd < 315 else "E"))

        if pk_spd >= 160:
            mv_state = f"RUNNING {pk_spd:.0f}px/s {compass}"
            mv_col = (0, 60, 255)
        elif turns >= 4 and feats.get("duration_seconds", 0.0) <= 10.0:
            mv_state = f"ERRATIC ({turns} turns)"
            mv_col = (0, 140, 255)
        elif spd >= 18:
            mv_state = f"WALKING {spd:.0f}px/s {compass}"
            mv_col = (80, 230, 80)
        else:
            mv_state = "STATIONARY / BROWSING"
            mv_col = (255, 210, 60)

        # Active zone & dwell time for this track
        zone_tag = ""
        for zid, occ_map in zA.zone_occ.items():
            if tid in occ_map:
                zname = zone_lookup[zid].name if zid in zone_lookup else zid
                dwell_s = max(0.0, ts - occ_map[tid])
                zone_tag = f"{zname}: {dwell_s:.1f}s"
                break

        # Pose skeleton & posture tags
        pose = (pose_ests or {}).get(tid)
        posture_tags = []
        if pose:
            lw_pt = pose.get("left_wrist")
            rw_pt = pose.get("right_wrist")
            le_pt = pose.get("left_elbow")
            re_pt = pose.get("right_elbow")
            ls_pt = pose.get("left_shoulder")
            rs_pt = pose.get("right_shoulder")
            hd_pt = pose.get("head")
            hp_pt = pose.get("hip_center")

            # Draw skeletal links
            skel_pairs = [
                (hd_pt, ls_pt), (hd_pt, rs_pt), (ls_pt, rs_pt),
                (ls_pt, le_pt), (le_pt, lw_pt),
                (rs_pt, re_pt), (re_pt, rw_pt),
                (ls_pt, hp_pt), (rs_pt, hp_pt),
            ]
            for p_a, p_b in skel_pairs:
                if p_a and p_b:
                    cv2.line(out, p_a, p_b, (255, 255, 100), 1)
            for kp, kp_col in [(lw_pt, (0, 255, 255)), (rw_pt, (255, 255, 0)),
                               (le_pt, (0, 200, 200)), (re_pt, (200, 200, 0)),
                               (hd_pt, (255, 180, 0))]:
                if kp:
                    cv2.circle(out, kp, 4, kp_col, -1)

            if pose.get("is_crouching"):
                posture_tags.append("CROUCHING")
            if pose.get("is_leaning"):
                posture_tags.append("LEANING")

        if tid in (pose_gestures or {}):
            posture_tags.append(f"POSE:{pose_gestures[tid].upper()}")

        # Draw rich header tag above person bbox
        top_y = max(36, y1 - 6)
        info_line1 = f"ID:{tid} ({t.confidence:.0%}) | {mv_state}"
        (tw1, th1), _ = cv2.getTextSize(info_line1, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        cv2.rectangle(out, (x1, top_y - th1 - 6), (x1 + tw1 + 8, top_y), (20, 25, 35), -1)
        cv2.putText(out, info_line1, (x1 + 4, top_y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, mv_col, 1)

        # Second sub-badge for Zone Dwell + Posture + Carried Items
        sub_parts = []
        if zone_tag:
            sub_parts.append(zone_tag)
        if posture_tags:
            sub_parts.append(" ".join(posture_tags))
        if tid in (carried_map or {}):
            sub_parts.append("CARRYING: " + ", ".join(carried_map[tid]))
        if sub_parts:
            info_line2 = " | ".join(sub_parts)
            by = min(out.shape[0] - 8, y2 + 16)
            (tw2, th2), _ = cv2.getTextSize(info_line2, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(out, (x1, by - th2 - 4), (x1 + tw2 + 8, by + 2), (35, 20, 60), -1)
            cv2.putText(out, info_line2, (x1 + 4, by - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 220, 120), 1)

    # 7. Background Subtraction Motion Mask (Top-Right PiP)
    motion_pct = 0.0
    if bg_stats is not None and bg_stats.fg_mask is not None and bg_stats.fg_mask.size > 0:
        try:
            motion_pct = bg_stats.motion_density * 100.0
            small = cv2.resize(bg_stats.fg_mask, (160, 90))
            rgb_mask = cv2.applyColorMap(small, cv2.COLORMAP_JET)
            out[8:98, out.shape[1] - 168 : out.shape[1] - 8] = rgb_mask
            cv2.rectangle(out, (out.shape[1] - 168, 8), (out.shape[1] - 8, 98), (255, 255, 255), 1)
            cv2.putText(out, f"Motion Map: {motion_pct:.1f}%",
                        (out.shape[1] - 166, 114),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1)
        except Exception as error:
            if error_callback:
                error_callback("Annotation overlay", error)

    # 8. Top-Left Multi-Metric Live HUD Panel
    hud_overlay = out.copy()
    cv2.rectangle(hud_overlay, (6, 6), (520, 108), (15, 23, 42), -1)
    cv2.addWeighted(hud_overlay, 0.72, out, 0.28, 0, out)

    line_total_in, line_total_out = 0, 0
    if line_det is not None:
        for (lid, d_lbl), cnt in line_det.get_counts().items():
            if d_lbl.lower() in ("in", "left", "enter"):
                line_total_in += cnt
            else:
                line_total_out += cnt

    q_len = sum(qs.length for qs in (last_queue_status or []))
    pose_active_cnt = len(pose_ests or {})

    cv2.putText(
        out,
        f"LIVE AI VISION | FPS: {fps_val:.1f} | People: {len(tracks)} | Objects/Bags: {len(object_dets)} | Footfall: {hm.total_footfall}",
        (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 160), 1,
    )
    cv2.putText(
        out,
        f"Pose Skeletons: {pose_active_cnt} | Line In/Out: {line_total_in}/{line_total_out} | Queue: {q_len} ppl | Motion: {motion_pct:.0f}%",
        (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 220, 255), 1,
    )
    cv2.putText(
        out,
        f"Detected Events: {len(activities_log)} | Carried Items: {sum(len(v) for v in (carried_map or {}).values())}",
        (14, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 255), 1,
    )
    frame_info_str = f"Frame: {fc}/{total_frames}" if total_frames > 0 else f"Frame: {fc}"
    if extra_label:
        frame_info_str += f" | {extra_label}"
    cv2.putText(
        out,
        frame_info_str,
        (14, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 210, 225), 1,
    )

    # 9. Bottom Live Event Ticker (shows latest 3 detected events right on the video)
    if activities_log:
        recent_acts = activities_log[-3:][::-1]
        bar_h = 22 * len(recent_acts) + 12
        y_top = max(120, out.shape[0] - bar_h - 6)
        ticker_bg = out.copy()
        cv2.rectangle(ticker_bg, (6, y_top), (out.shape[1] - 6, out.shape[0] - 6), (20, 15, 30), -1)
        cv2.addWeighted(ticker_bg, 0.78, out, 0.22, 0, out)
        for idx, act in enumerate(recent_acts):
            sev_str = act.severity.value if hasattr(act.severity, "value") else str(act.severity)
            col = (0, 70, 255) if sev_str in ("HIGH", "CRITICAL") else ((0, 180, 255) if sev_str == "MEDIUM" else (120, 240, 120))
            msg_txt = f"[{sev_str}] {act.message[:95]} (conf {act.confidence:.0%})"
            cv2.putText(out, msg_txt, (14, y_top + 18 + idx * 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1)

    return out


def _default_zones(width: int = 960, height: int = 540) -> List[Zone]:
    sx, sy = width / 1920, height / 1080
    def s(p): return (int(p[0] * sx), int(p[1] * sy))
    return [
        Zone("entrance", "Store Entrance", "entrance",
             [s((100, 800)), s((400, 800)), s((400, 1080)), s((100, 1080))], 10, 120),
        Zone("checkout", "Checkout", "checkout",
             [s((700, 900)), s((1200, 900)), s((1200, 1080)), s((700, 1080))], 8, 180),
        Zone("restricted", "Restricted", "restricted",
             [s((800, 100)), s((950, 100)), s((950, 250)), s((800, 250))], 0, 0),
        Zone("shelf_a", "Shelf A", "shelf",
             [s((450, 200)), s((800, 200)), s((800, 650)), s((450, 650))], 5, 240),
    ]


def _ensure_opencv_compatible(video_path: str) -> Optional[str]:
    """
    Check if OpenCV can open the video. If not, use FFmpeg to re-encode
    it to a format compatible with OpenCV's FFmpeg build.

    Returns the path to a compatible video file (could be the original
    if it was already compatible).
    """
    import subprocess
    import shutil

    path = Path(video_path)
    if not path.exists():
        return None

    # Quick check: can OpenCV open this video?
    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None and frame.size > 0:
                return video_path  # Already compatible
        cap.release()
    except Exception:
        pass

    # Need to convert using FFmpeg
    if not shutil.which("ffmpeg"):
        print(f"[ERROR] OpenCV cannot decode '{path.name}' and ffmpeg is not available for conversion")
        return None

    converted_path = str(path.with_name(f"converted_{path.name}"))
    print(f"[INFO] Re-encoding video for OpenCV compatibility: {path.name}")

    try:
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        # Use libx264 for maximum compatibility with OpenCV builds
        cmd = [
            "ffmpeg", "-y",
            "-i", str(path),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "fast",
            "-crf", "23",
            "-an",  # No audio - not needed for processing
            str(converted_path)
        ]
        result = subprocess.run(cmd, capture_output=True, startupinfo=startupinfo)

        if Path(converted_path).exists() and Path(converted_path).stat().st_size > 0:
            print(f"[INFO] Video successfully re-encoded: {converted_path}")
            return str(converted_path)
        else:
            print(f"[ERROR] FFmpeg conversion failed: {result.stderr[:300]}")
            return None
    except Exception as e:
        print(f"[ERROR] FFmpeg conversion failed: {e}")
        return None


def _extract_first_frame(video_path: str) -> Optional[np.ndarray]:
    """Extract the first frame from a video file for zone editor display.
    Uses OpenCV first, falls back to FFmpeg if OpenCV can't decode."""
    import subprocess
    import shutil

    # Try OpenCV first
    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                return cv2.resize(frame, (960, 540))
        cap.release()
    except Exception:
        pass

    # Fallback: Use FFmpeg to extract first frame
    if shutil.which("ffmpeg"):
        try:
            temp_frame = tempfile.mktemp(suffix=".png")
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vframes", "1",
                "-q:v", "2",
                temp_frame
            ]
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            result = subprocess.run(cmd, capture_output=True, startupinfo=startupinfo)
            if Path(temp_frame).exists():
                frame = cv2.imread(temp_frame)
                os.unlink(temp_frame)
                if frame is not None:
                    print(f"[INFO] Extracted first frame using FFmpeg fallback")
                    return cv2.resize(frame, (960, 540))
            else:
                if result.stderr:
                    print(f"[WARN] FFmpeg frame extraction failed: {result.stderr[:200]}")
        except Exception as e:
            print(f"[WARN] FFmpeg fallback failed: {e}")

    return None


def _ensure_h264_compatibility(file_path):
    """
    Checks if ffmpeg is available on the path and re-encodes the video to H.264/avc1
    to ensure it can be streamed inline in standard web browsers.
    """
    import subprocess
    import shutil
    from pathlib import Path

    path = Path(file_path)
    if not path.exists():
        return

    # Check if ffmpeg is available
    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg is unavailable. Download the output video for external playback.")

    temp_path = path.with_name(f"temp_encode_{path.name}")
    try:
        # Re-encode to H.264 with AAC audio (if any) or just video
        # -pix_fmt yuv420p is required for browser compatibility
        cmd = [
            "ffmpeg", "-y",
            "-i", str(path),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "fast",
            "-crf", "23",
            str(temp_path)
        ]
        # Run ffmpeg with hidden console on Windows to avoid flashing command window
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        subprocess.run(cmd, check=True, startupinfo=startupinfo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Replace the original file with the re-encoded one
        if temp_path.exists() and temp_path.stat().st_size > 0:
            os.replace(temp_path, path)
            print(f"[INFO] Video successfully re-encoded to web-compatible H.264: {path}")
    except Exception as e:
        logging.getLogger(__name__).error("Video encoding failed: %s", e)
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        raise RuntimeError(f"Browser video encoding failed: {e}") from e


# ---------------------------------------------------------------------------
# Batch Video Processing (for uploaded files, with Live Preview & Full Detection)
# ---------------------------------------------------------------------------
def process_video_batch(
    video_path: str,
    output_path: str,
    det,
    trk,
    zones: List[Zone],
    zA: ZoneAnalytics,
    hm: HeatmapGenerator,
    am: AlertManager,
    sus_detector,
    bhv: BehaviorDetector,
    show_hm: bool,
    enable_pose: bool,
    preproc,
    bg_sub,
    line_det,
    queue_det,
    olb_det, orm_det, obj_shelves,
    face_blur,
    video_clip,
    gest_d, fall_d, fight_d,
    progress_callback=None,
    frame_callback=None,
    error_callback=None,
    camera_id="v1",
) -> dict:
    """
    Process a video file while streaming live annotated preview frames and
    real-time telemetry across all computer-vision detection subsystems.
    Returns stats dict.
    """
    fc = 0
    reported_errors = set()

    def report_error(stage, error):
        key = (stage, str(error))
        if key not in reported_errors:
            reported_errors.add(key)
            logging.getLogger(__name__).warning("%s at frame %s: %s", stage, fc, error)
            if error_callback:
                error_callback(stage, fc, str(error))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Failed to open video. Check the file codec or upload another video.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 0 or fps > 120:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output video writer - use widely supported codecs
    out_w, out_h = 960, 540
    writer = None

    codec_priority = ["mp4v", "avc1", "XVID", "MJPG"]
    for codec in codec_priority:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
        if writer.isOpened():
            print(f"[INFO] Video writer initialized with codec: {codec}")
            break
        else:
            writer.release()
            writer = None
            if codec == "avc1":
                print(f"[WARN] avc1/H264 codec not available, trying next codec...")

    if writer is None:
        cap.release()
        raise RuntimeError("Failed to initialize video writer with any available codec")

    fc = 0
    start_time = time.time()
    base_video_ts = start_time - (max(total_frames, 1) / fps)
    last_frame_time = 0.0
    activities_log: List[SuspiciousActivity] = []
    last_queue_status = []
    last_object_events = []
    pose_gestures_active = {}
    total_objects_seen = 0
    unique_track_ids = set()
    crowd_last_emit = 0.0
    carried_alerted = set()

    try:
        while True:
            ret, fr = cap.read()
            if not ret:
                break

            fc += 1
            fr = cv2.resize(fr, (out_w, out_h))
            # Use consistent video timeline timestamp so speeds (px/s) & dwell times are accurate
            ts = base_video_ts + (fc / fps)

            # --- Preprocessing ---
            if preproc is not None:
                try: fr = preproc(fr)
                except Exception as error: report_error("Frame preprocessing", error)
            if face_blur is not None:
                try: fr = face_blur.blur(fr)
                except Exception as error: report_error("Face privacy filter", error)
            if video_clip is not None:
                try: video_clip.add_frame(fr, ts)
                except Exception as error: report_error("Event video buffer", error)
            bg_stats = None
            if bg_sub is not None:
                try: bg_stats = bg_sub.apply(fr)
                except Exception as error: report_error("Motion analysis", error)

            # --- Core Multi-Class Detection (Persons + Bags/Retail Objects) & Tracking ---
            dets = det.detect(fr, classes=RETAIL_CLASSES)
            person_dets = [d for d in dets if d.class_id == 0]
            object_dets = [d for d in dets if d.class_id != 0]
            total_objects_seen += len(object_dets)

            res = trk.update(fr, person_dets, ts)
            for t in res.tracks:
                unique_track_ids.add(t.track_id)

            carried_map = _associate_carried_objects(res.tracks, object_dets)

            zA.update(res.tracks, ts)
            hm.update(res.tracks, ts)

            cfg = {z.zone_id: {"name": z.name, "dwell_threshold": z.dwell_threshold, "capacity": z.capacity}
                   for z in zones}

            # Line crossing
            crossings = []
            if line_det is not None:
                try: crossings = line_det.update(res.tracks, ts) or []
                except Exception as error: report_error("Line crossings", error)
            for cr in crossings:
                act = _make_activity_from_aux(
                    activity_id=f"{camera_id}_line_{cr.line_id}_{cr.track_id}_{fc}",
                    activity_type=ActivityType.CUSTOM_RULE,
                    severity=Severity.INFO,
                    confidence=cr.confidence,
                    track_id=cr.track_id,
                    zone_id=cr.line_id,
                    zone_name=cr.line_id,
                    camera_id=camera_id,
                    message=f"Line crossed ({cr.line_id}): person #{cr.track_id} moved {cr.direction.upper()}",
                    ts=ts,
                    evidence={"line_id": cr.line_id, "direction": cr.direction},
                    rule_violations=[f"line_crossing:{cr.direction}"],
                )
                activities_log.append(act)

            # Queue detection
            if queue_det is not None:
                try: last_queue_status = queue_det.update(res.tracks, zA.zone_occ, ts)
                except Exception as error: report_error("Queue analysis", error)

            # Object monitor (left-behind & shelf removal)
            obj_events = []
            if olb_det is not None and obj_shelves:
                try: obj_events.extend(olb_det.update(fr, res.tracks, zA.zone_occ, obj_shelves, ts))
                except Exception as error: report_error("Object monitoring", error)
            if orm_det is not None and obj_shelves:
                try: obj_events.extend(orm_det.update(fr, res.tracks, zA.zone_occ, obj_shelves, ts))
                except Exception as error: report_error("Object monitoring", error)
            if obj_events:
                last_object_events.extend(obj_events)
                last_object_events = last_object_events[-200:]
                for oe in obj_events:
                    act = _make_activity_from_aux(
                        activity_id=f"{camera_id}_obj_{oe.shelf_id}_{fc}",
                        activity_type=ActivityType.UNATTENDED_CHECKOUT if oe.event_type == "object_left" else ActivityType.SHELF_TAMPERING,
                        severity=Severity.HIGH if oe.severity == "high" else Severity.MEDIUM,
                        confidence=0.82,
                        track_id=oe.track_id,
                        zone_id=oe.shelf_id,
                        zone_name=oe.shelf_id,
                        camera_id=camera_id,
                        message=oe.message,
                        ts=ts,
                        evidence={"shelf_id": oe.shelf_id, "event_type": oe.event_type},
                        rule_violations=[f"object_monitor:{oe.event_type}"],
                    )
                    am.create_from_activity(act, camera_id, fr, None)
                    activities_log.append(act)

            # Pose estimation & Pose-based detectors (Gesture / Fall / Fight / Posture)
            pose_ests = {}
            if enable_pose and st.session_state.get("pose_analyzer"):
                try: pose_ests = st.session_state["pose_analyzer"].estimate(fr, res.tracks, ts) or {}
                except Exception as error: report_error("Pose estimation", error)

            pose_events: List[PoseEvent] = []
            if gest_d:
                try: pose_events.extend(gest_d.update(pose_ests, res.tracks, ts) or [])
                except Exception as error: report_error("Pose activity analysis", error)
            if fall_d:
                try: pose_events.extend(fall_d.update(res.tracks, pose_ests, ts) or [])
                except Exception as error: report_error("Pose activity analysis", error)
            if fight_d:
                try: pose_events.extend(fight_d.update(res.tracks, pose_ests, ts) or [])
                except Exception as error: report_error("Pose activity analysis", error)

            for pev in pose_events:
                if pev.gesture and pev.track_id is not None:
                    pose_gestures_active[pev.track_id] = pev.gesture
                sev = Severity.HIGH if pev.event_type in ("fall", "fight") else Severity.MEDIUM
                act = _make_activity_from_aux(
                    activity_id=f"{camera_id}_pose_{pev.event_type}_{pev.track_id or 0}_{fc}",
                    activity_type=ActivityType.CUSTOM_RULE,
                    severity=sev,
                    confidence=pev.confidence,
                    track_id=pev.track_id,
                    zone_id=None,
                    zone_name="Store Floor",
                    camera_id=camera_id,
                    message=f"[{pev.event_type.upper()}] {pev.message}",
                    ts=ts,
                    evidence={"pose_event": pev.event_type, "gesture": pev.gesture},
                    rule_violations=[f"pose:{pev.event_type}"],
                    track_ids=pev.track_ids,
                )
                track_obj = next((t for t in res.tracks if t.track_id == pev.track_id), None)
                am.create_from_activity(act, camera_id, fr, track_obj.bbox if track_obj else None)
                activities_log.append(act)

            # Carried Bag / Object in Shelf or Restricted Zone Corroboration
            for tid, items in carried_map.items():
                for zid, occ in zA.zone_occ.items():
                    if tid in occ and (tid, zid) not in carried_alerted:
                        z_obj = next((z for z in zones if z.zone_id == zid), None)
                        if z_obj and z_obj.zone_type in ("shelf", "restricted", "checkout"):
                            carried_alerted.add((tid, zid))
                            act = _make_activity_from_aux(
                                activity_id=f"{camera_id}_bag_{tid}_{zid}_{fc}",
                                activity_type=ActivityType.ABNORMAL_SHELF_INTERACTION if z_obj.zone_type == "shelf" else ActivityType.CUSTOM_RULE,
                                severity=Severity.HIGH if z_obj.zone_type == "restricted" else Severity.MEDIUM,
                                confidence=0.84,
                                track_id=tid,
                                zone_id=zid,
                                zone_name=z_obj.name,
                                camera_id=camera_id,
                                message=f"Person #{tid} carrying {', '.join(items)} in {z_obj.name}",
                                ts=ts,
                                evidence={"carried_items": items, "zone": z_obj.name},
                                rule_violations=["object_association:carried_item_in_zone"],
                            )
                            track_obj = next((t for t in res.tracks if t.track_id == tid), None)
                            am.create_from_activity(act, camera_id, fr, track_obj.bbox if track_obj else None)
                            activities_log.append(act)

            # Crowd / Group Gathering Detection ("and more")
            if len(res.tracks) >= 3 and (ts - crowd_last_emit) >= 15.0:
                # Check if >= 3 tracks are clustered within 180px of each other
                centers = [t.center for t in res.tracks]
                clustered_ids = []
                for idx_a, t_a in enumerate(res.tracks):
                    nearby = [
                        t_b.track_id for t_b in res.tracks
                        if ((t_a.center[0] - t_b.center[0]) ** 2 + (t_a.center[1] - t_b.center[1]) ** 2) ** 0.5 <= 180
                    ]
                    if len(nearby) >= 3:
                        clustered_ids = nearby
                        break
                if clustered_ids:
                    crowd_last_emit = ts
                    act = _make_activity_from_aux(
                        activity_id=f"{camera_id}_crowd_{fc}",
                        activity_type=ActivityType.CUSTOM_RULE,
                        severity=Severity.MEDIUM,
                        confidence=0.80,
                        track_id=clustered_ids[0],
                        zone_id=None,
                        zone_name="Store Floor",
                        camera_id=camera_id,
                        message=f"Crowd / group gathering detected ({len(clustered_ids)} people in close proximity)",
                        ts=ts,
                        evidence={"group_size": len(clustered_ids), "track_ids": clustered_ids},
                        rule_violations=["crowd:cluster_detected"],
                        track_ids=clustered_ids,
                    )
                    am.create_from_activity(act, camera_id, fr, None)
                    activities_log.append(act)

            # Core Suspicious Activity Detector (Loitering, Restricted Zone, Running, Erratic, Shelf Tampering, Unattended Checkout)
            if sus_detector is not None:
                try:
                    activities = sus_detector.detect(
                        tracks=res.tracks,
                        zones={z.zone_id: z for z in zones},
                        zone_occ=zA.zone_occ,
                        timestamp=ts, frame=fr, detections=dets,
                        camera_id=camera_id,
                    )
                    for a in activities:
                        track = next((t for t in res.tracks if t.track_id == a.track_id), None)
                        bbox = track.bbox if track else None
                        am.create_from_activity(a, camera_id, fr, bbox)
                        if video_clip is not None:
                            try: video_clip.save_clip(a.activity_id, fr)
                            except Exception as error: report_error("video_clip.save_clip", error)
                    activities_log.extend(activities)
                except Exception as error:
                    report_error("Suspicious activity analysis", error)

            # Legacy Behavior Detector (Congestion checks without duplicating sus_detector)
            for alert in bhv._check_congestion(cfg, zA.zone_occ, ts):
                act = _make_activity_from_aux(
                    activity_id=f"{camera_id}_cong_{alert.zone_id}_{fc}",
                    activity_type=ActivityType.CUSTOM_RULE,
                    severity=Severity.MEDIUM,
                    confidence=0.85,
                    track_id=None,
                    zone_id=alert.zone_id,
                    zone_name=cfg.get(alert.zone_id, {}).get("name", alert.zone_id),
                    camera_id=camera_id,
                    message=alert.message,
                    ts=ts,
                    evidence={"zone_id": alert.zone_id},
                    rule_violations=["zone:congestion"],
                )
                am.create_from_activity(act, camera_id, fr, None)
                activities_log.append(act)

            if len(activities_log) > 500:
                activities_log = activities_log[-500:]

            # --- Rich Live Drawing ---
            s_stats = det.get_performance_stats()
            elapsed_now = max(time.time() - start_time, 0.001)
            live_fps = fc / elapsed_now

            out = _draw_rich_live_annotations(
                frame=fr,
                det=det,
                trk=trk,
                person_dets=person_dets,
                object_dets=object_dets,
                tracks=res.tracks,
                zA=zA,
                zones=zones,
                hm=hm,
                show_hm=show_hm,
                bg_stats=bg_stats,
                line_det=line_det,
                queue_det=queue_det,
                last_queue_status=last_queue_status,
                obj_shelves=obj_shelves,
                pose_ests=pose_ests,
                pose_gestures=pose_gestures_active,
                carried_map=carried_map,
                activities_log=activities_log,
                fc=fc,
                total_frames=total_frames,
                fps_val=live_fps,
                ts=ts,
                error_callback=report_error,
            )

            writer.write(out)

            live_telemetry = {
                "frame": fc,
                "total": max(total_frames, fc),
                "fps": live_fps,
                "tracks": len(res.tracks),
                "unique_people": len(unique_track_ids),
                "objects_now": len(object_dets),
                "objects_total": total_objects_seen,
                "carried_now": sum(len(v) for v in carried_map.values()),
                "pose_active": len(pose_ests),
                "footfall": hm.total_footfall,
                "suspicious": len(activities_log),
                "motion_pct": (bg_stats.motion_density * 100.0) if (bg_stats and bg_stats.fg_mask is not None) else 0.0,
                "line_counts": line_det.get_counts() if line_det else {},
                "queue_status": last_queue_status,
                "activities_log": activities_log,
                "zA": zA,
                "zones": zones,
                "hm": hm,
            }

            # Bound browser updates to ten per second while analyzing every frame.
            if frame_callback:
                now = time.monotonic()
                if fc == 1 or now - last_frame_time >= 0.1:
                    frame_callback(out, live_telemetry)
                    last_frame_time = now

            # Progress update every 5 frames for responsive live metrics
            if progress_callback and (fc == 1 or fc % 5 == 0 or fc == total_frames):
                progress_callback(min(1.0, fc / max(total_frames, 1)), live_telemetry)


        if fc == 0:
            raise RuntimeError("The video contains no decodable frames.")
        if total_frames > 0 and fc < total_frames:
            raise RuntimeError(f"Video decoding stopped at frame {fc} of {total_frames}.")
    finally:
        cap.release()
        writer.release()

    # Ensure the finished video is H.264-compatible for browser streaming
    try:
        _ensure_h264_compatibility(output_path)
    except Exception as error:
        report_error("Browser video encoding", error)

    elapsed = time.time() - start_time

    return {
        "total_frames": total_frames,
        "processed_frames": fc,
        "elapsed_seconds": elapsed,
        "avg_fps": fc / max(elapsed, 0.001),
        "total_footfall": hm.total_footfall,
        "unique_people": len(unique_track_ids),
        "total_objects_seen": total_objects_seen,
        "suspicious_count": len(activities_log),
        "output_path": output_path,
        "activities_log": activities_log,
        "last_queue_status": last_queue_status,
        "last_object_events": last_object_events,
    }

# ---------------------------------------------------------------------------
# Adaptive frame processing (for live streams)
# ---------------------------------------------------------------------------
def _calculate_target_fps(processing_time, min_fps=5, max_fps=30):
    if processing_time <= 0:
        return max_fps
    target = 1.0 / processing_time
    return max(min_fps, min(max_fps, target))


def _should_process_frame(frame_idx, frame_skip):
    return frame_idx % (frame_skip + 1) == 0


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
page_header("Store Vision", "YOLOv8 + ByteTrack detection, zone analytics, suspicious-activity alerts and a 30-day event timeline.", "APOSIRA · STORE VISION")

# Sidebar
with st.sidebar:
    st.markdown("---")
    st.subheader("Store Vision settings")
    ms = st.selectbox("Model", ["n", "s", "m"], 0, key="cfg_model")
    cf = st.slider("Confidence", 0.2, 0.9, 0.5, 0.05, key="cfg_conf")
    device_options = ["cpu"]
    if torch.cuda.is_available():
        device_options.append("cuda")
    default_device_idx = device_options.index("cuda") if "cuda" in device_options else 0
    dv = st.selectbox("Device", device_options, default_device_idx, key="cfg_device")

    if "cuda" not in device_options:
        st.warning("⚠️ CUDA GPU not detected. Running on CPU (slow).")

    loiter = st.slider("Loiter (s)", 10, 300, 30, 5, key="cfg_loiter")
    show_hm = st.checkbox("Show Heatmap", True, key="cfg_showhm")
    enable_pose = st.checkbox("Enable Pose & Posture AI", True, key="cfg_pose")
    enable_multi = st.checkbox("Multi-Camera Re-ID", False, key="cfg_mcam")

    st.markdown("---")
    st.subheader("⚡ Performance")
    frame_skip = st.slider("Frame Skip (live only)", 0, 10, 2,
                           key="cfg_frame_skip",
                           help="For live streams only. Higher = smoother display.")
    st.session_state["frame_skip"] = frame_skip

# Tabs
tab_live, tab_zones, tab_adv, tab_sus, tab_tune, tab_dash = st.tabs(
    ["📹 Live", "🗺️ Zone Editor", "🛠️ Advanced",
     "🚨 Activity", "🎛️ Tuner", "📊 Dashboard"]
)

# --- Tab: Live ---
with tab_live:
    # Mode selector
    mode = st.radio(
        "Processing Mode",
        ["🎬 Batch Video Processing", "📡 Live Stream (Webcam/RTSP)"],
        horizontal=True,
        key="live_mode",
        help="Batch: upload a video file and watch live real-time detections as it processes. "
             "Live: real-time webcam or RTSP stream."
    )

    run_store = RunStore(PROJECT_ROOT / "data/events.sqlite3",
                         st.session_state.get("aposira_org", "local"),
                         st.session_state.get("aposira_branch", "main-store"))
    with st.expander("Processing history and errors"):
        history = run_store.recent()
        if history:
            st.dataframe(history, hide_index=True, width="stretch")
            selected_run = st.selectbox("Processing run", [row["id"] for row in history], key="vision_history_run")
            diagnostics = run_store.errors(selected_run)
            if diagnostics:
                st.dataframe(diagnostics, hide_index=True, width="stretch")
            st.caption("A running record with an old update time may have been interrupted by a Pod restart.")
        else:
            st.caption("Completed runs and processing errors will appear here.")

    if mode == "🎬 Batch Video Processing":
        # ===== VIDEO PROCESSING WITH LIVE DETECTION PREVIEW =====
        st.markdown("""
        <div class="result-card">
        <h3>🎬 Video Live Detection & Processing</h3>
        <p>Upload a video file to watch a <strong>real-time live preview</strong> as the AI pipeline detects
        <strong>people, carried bags/objects, multi-object trajectories, speed & heading, zone occupancy & dwell time,
        virtual line crossings (In/Out), checkout queues & wait time, shelf object removal/left-behind,
        pose skeletons & posture (crouching, leaning, gestures, falls, fights), crowd gatherings, motion masks, and heatmaps</strong>—and saves the annotated video.</p>
        </div>
        """, unsafe_allow_html=True)

        automatic = st.checkbox("Start live analysis automatically after upload", value=True, key="vision_auto_start")
        up = st.file_uploader("Choose video file", type=["mp4", "avi", "mov"],
                              key="batch_uploader")
        if up:
            upload_id = getattr(up, "file_id", None)
            previous_upload = st.session_state.get("vision_upload_identity")
            if upload_id and previous_upload and previous_upload["id"] == upload_id:
                upload_key = previous_upload["key"]
            else:
                upload_key = hashlib.sha256(up.getbuffer()).hexdigest()
                st.session_state["vision_upload_identity"] = {"id": upload_id, "key": upload_key}
            prepared = st.session_state.get("vision_prepared_upload")
            if not prepared or prepared["key"] != upload_key:
                with st.status("Preparing uploaded video…", expanded=True) as preparation:
                    st.write("Saving upload and checking video compatibility…")
                    try:
                        # Save uploaded file to a PERSISTENT location (not temp that gets deleted)
                        data_dir = runtime_directory(PROJECT_ROOT / "data/uploads")
                        data_dir.mkdir(parents=True, exist_ok=True)

                        # Create a unique filename based on timestamp
                        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        safe_name = f"{timestamp_str}_{Path(up.name).name}"
                        input_path = str(data_dir / safe_name)

                        # Save the uploaded file
                        with open(input_path, "wb") as f:
                            f.write(up.getbuffer())

                        # Convert incompatible videos to OpenCV-compatible format
                        compatible_path = _ensure_opencv_compatible(input_path)
                        if compatible_path is not None and compatible_path != input_path:
                            # Use the converted version for OpenCV operations
                            st.session_state["uploaded_video_path"] = compatible_path
                            print(f"[INFO] Using re-encoded video for processing: {compatible_path}")
                        else:
                            st.session_state["uploaded_video_path"] = input_path

                        # Extract and store the first frame for zone editor
                        first_frame = _extract_first_frame(compatible_path or input_path)
                        if first_frame is not None:
                            st.session_state["uploaded_video_first_frame"] = first_frame
                            # Save the first frame as a PNG for direct use in zone editor
                            data_dir.mkdir(parents=True, exist_ok=True)
                            frame_png_path = str(data_dir / "zone_editor_frame.png")
                            cv2.imwrite(frame_png_path, first_frame)
                            st.session_state["uploaded_video_first_frame_path"] = frame_png_path
                            print(f"[INFO] Saved first frame for zone editor: {input_path}")
                        else:
                            st.session_state["uploaded_video_first_frame"] = None
                            st.session_state["uploaded_video_first_frame_path"] = None
                            print(f"[WARN] Could not extract first frame from: {input_path}")

                        # Output path - also in data directory for persistence
                        output_path = str(data_dir / f"processed_{timestamp_str}.mp4")

                        if first_frame is None:
                            raise RuntimeError("No readable video frame was found in this upload.")
                        st.session_state["vision_prepared_upload"] = {
                            "key": upload_key, "input": input_path,
                            "compatible": compatible_path, "output": output_path,
                            "directory": str(data_dir), "first_frame": first_frame,
                        }
                        preparation.update(label="Video ready for live analysis", state="complete")
                    except Exception as error:
                        preparation.update(label="Video preparation failed", state="error")
                        failed_run = run_store.start("upload", up.name)
                        run_store.error(failed_run, "Upload preparation", 0, str(error))
                        run_store.update(failed_run, status="failed", error=str(error))
                        st.error(f"Video preparation failed: {error}")
                        st.exception(error)
                        st.stop()
            prepared = st.session_state["vision_prepared_upload"]
            input_path = prepared["input"]
            compatible_path = prepared["compatible"]
            output_path = prepared["output"]
            data_dir = Path(prepared["directory"])
            first_frame = prepared["first_frame"]

            # Show input info
            process_path = compatible_path or input_path
            cap_check = cv2.VideoCapture(process_path)
            if cap_check.isOpened():
                total = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT))
                vid_fps = cap_check.get(cv2.CAP_PROP_FPS) or 25.0
                dur = total / vid_fps if vid_fps > 0 else 0
                st.info(f"📹 **{up.name}** — {total} frames, {vid_fps:.1f} FPS, "
                        f"{dur:.1f}s duration")
                cap_check.release()
            else:
                cap_check.release()
                # Try FFprobe to get video info as fallback
                import subprocess, shutil
                if shutil.which("ffprobe"):
                    try:
                        startupinfo = None
                        if os.name == 'nt':
                            startupinfo = subprocess.STARTUPINFO()
                            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                        probe = subprocess.run([
                            "ffprobe", "-v", "error",
                            "-select_streams", "v:0",
                            "-show_entries", "stream=nb_frames,r_frame_rate",
                            "-of", "csv=p=0",
                            process_path
                        ], capture_output=True, text=True, startupinfo=startupinfo)
                        if probe.stdout.strip():
                            parts = probe.stdout.strip().split(",")
                            if len(parts) >= 2:
                                total_str, fps_str = parts[0], parts[1]
                                total = int(float(total_str)) if total_str != "N/A" else 0
                                if "/" in fps_str:
                                    num, den = fps_str.split("/")
                                    vid_fps = float(num) / float(den) if float(den) > 0 else 25.0
                                else:
                                    vid_fps = float(fps_str)
                                dur = total / vid_fps if vid_fps > 0 else 0
                                st.info(f"📹 **{up.name}** — {total} frames, {vid_fps:.1f} FPS, "
                                        f"{dur:.1f}s duration")
                    except Exception:
                        pass
                if not st.session_state.get("_video_info_shown"):
                    st.info(f"📹 **{up.name}** — uploaded successfully")
                    st.session_state["_video_info_shown"] = True

            # Zone loading
            cam_id = st.text_input("Camera ID for zone loading", value="v1",
                                   key="batch_cam")
            existing_zones = []
            try:
                existing_zones = load_zones(camera_id=cam_id)
            except Exception:
                pass
            if not existing_zones:
                try:
                    existing_zones = load_zones(path=PROJECT_ROOT / "config/zones.yaml")
                except Exception:
                    pass
            if not existing_zones:
                existing_zones = _default_zones(960, 540)
                st.caption("Using default zones (Store Entrance, Checkout, Restricted, Shelf A)")

            # Live Preview & Detection options (ENABLED BY DEFAULT)
            st.markdown("##### ⚙️ Live Preview & Detection Options")
            opt_c1, opt_c2 = st.columns(2)
            with opt_c1:
                show_preview = st.checkbox(
                    "📺 Show Live Preview While Processing",
                    value=True,
                    key="batch_show_preview",
                    help="Stream real-time annotated video frames, live zone occupancy, "
                         "heatmap, pose skeletons, and security alerts while processing."
                )
            with opt_c2:
                preview_every = st.slider(
                    "Live Preview Refresh (every N frames)",
                    1, 30, 2,
                    key="batch_preview_every",
                    help="1-2 = smooth real-time live playback; higher = faster batch completion."
                )

            # Show initial first-frame preview with zones before clicking Process
            if first_frame is not None and not st.session_state.get("_is_processing_video"):
                with st.expander("👁️ Preview Uploaded Video Frame & Active Zones", expanded=True):
                    init_zA = ZoneAnalytics(existing_zones, 960, 540)
                    preview_init = init_zA.draw_zones(first_frame.copy())
                    st.image(cv2.cvtColor(preview_init, cv2.COLOR_BGR2RGB),
                             caption="Uploaded video frame with configured monitoring zones — click '▶️ Start Live Video Detection & Process' below to watch live detections.",
                             use_container_width=True)

            # Process button
            start_requested = st.button("▶️ Start Live Video Detection & Process", type="primary", width='stretch')
            if start_requested or (automatic and st.session_state.get("vision_attempted_upload") != upload_key):
                st.session_state["vision_attempted_upload"] = upload_key
                run_id = run_store.start(cam_id, up.name)
                st.session_state["_is_processing_video"] = True
                st.session_state["vision_initialization_errors"] = []
                finished = False
                last_saved = {"time": 0.0, "frame": 0, "total": 0}
                try:
                    st.info("Loading the detection model and initializing analysis modules…")
                    try:
                        det = get_det(ms, cf, dv)
                    except Exception as e:
                        st.error(f"Model error: {e}")
                        raise

                    if det:
                        # Initialize components
                        trk = ByteTracker()
                        zones = existing_zones
                        zA = ZoneAnalytics(zones, 960, 540)
                        hm = HeatmapGenerator(960, 540)
                        am = AlertManager(organization_id=st.session_state.get("aposira_org", "local"), store_id=st.session_state.get("aposira_branch", "main-store"))
                        sus_det = _build_suspicious_detector(enable_pose, loiter_seconds=loiter)
                        bhv = BehaviorDetector(loiter_threshold=loiter,
                                               suspicious_detector=sus_det)

                        preproc = _get_preprocessor() if st.session_state.get("pp_enabled", True) else None
                        bg_sub = _get_bg_subtractor()
                        line_det = _get_line_detector()
                        queue_det = _get_queue_detector()
                        olb_det, orm_det, obj_shelves = _get_object_detector()
                        face_blur = _get_face_blurrer()
                        video_clip = _get_video_clipper()
                        gest_d, fall_d, fight_d = _get_pose_detectors()

                        # Progress bar & top live metrics
                        live_errors = []
                        error_placeholder = st.empty()
                        seen_errors = set()

                        def on_error(stage, frame, message):
                            if (stage, message) in seen_errors:
                                return
                            seen_errors.add((stage, message))
                            run_store.error(run_id, stage, frame, message)
                            live_errors.append(f"Frame {frame} · {stage}: {message}")
                            error_placeholder.warning("Processing warnings\n\n" + "\n\n".join(live_errors[-8:]))

                        for initialization_error in st.session_state.get("vision_initialization_errors", []):
                            on_error("Pose initialization", 0, initialization_error)
                        pose_model = st.session_state.get("pose_analyzer")
                        if enable_pose and pose_model is not None and not pose_model.is_available:
                            on_error("Pose initialization", 0, "MediaPipe pose model is unavailable; posture uses geometric estimates.")
                        if sus_det is None:
                            on_error("Activity initialization", 0, "Suspicious activity detection could not be initialized.")

                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        stats_cols = st.columns(6)
                        fps_m = stats_cols[0].empty()
                        frame_m = stats_cols[1].empty()
                        tracks_m = stats_cols[2].empty()
                        objs_m = stats_cols[3].empty()
                        foot_m = stats_cols[4].empty()
                        sus_m = stats_cols[5].empty()

                        # Side-by-side Live Video Preview & Real-Time Telemetry Console
                        if show_preview:
                            st.markdown("#### 🔴 Live Detection Stream (Video Playback)")
                            vid_col, live_col = st.columns([3, 1])
                            with vid_col:
                                video_placeholder = st.empty()
                            with live_col:
                                st.markdown("**🎯 Live Subsystems**")
                                subsys_placeholder = st.empty()
                                st.markdown("**📍 Live Zone Occupancy**")
                                zones_live_placeholder = st.empty()
                                st.markdown("**🚨 Live Event Feed**")
                                alerts_live_placeholder = st.empty()
                                st.markdown("**🗺️ Live Footfall Heatmap**")
                                hm_live_placeholder = st.empty()
                        else:
                            video_placeholder = None
                            subsys_placeholder = None
                            zones_live_placeholder = None
                            alerts_live_placeholder = None
                            hm_live_placeholder = None

                        preview_counter = {"n": 0}

                        def _update_live_side_panels(info: dict):
                            if not show_preview:
                                return
                            try:
                                if subsys_placeholder is not None:
                                    l_counts = info.get("line_counts", {})
                                    l_in = sum(c for (_, d), c in l_counts.items() if d.lower() in ("in", "left", "enter"))
                                    l_out = sum(c for (_, d), c in l_counts.items() if d.lower() not in ("in", "left", "enter"))
                                    q_list = info.get("queue_status", [])
                                    q_ppl = sum(q.length for q in q_list)
                                    q_wait = max((q.estimated_wait_seconds for q in q_list), default=0.0)
                                    subsys_placeholder.markdown(
                                        f"- 👥 **Active People:** `{info.get('tracks', 0)}` (Total: `{info.get('unique_people', 0)}`)\n"
                                        f"- 🎒 **Bags / Objects:** `{info.get('objects_now', 0)}` (`{info.get('carried_now', 0)}` carried)\n"
                                        f"- 🦴 **Pose Skeletons:** `{info.get('pose_active', 0)}` active\n"
                                        f"- ➡️ **Line Crossings:** In `{l_in}` / Out `{l_out}`\n"
                                        f"- 🧍 **Queue:** `{q_ppl}` ppl (Wait `{q_wait:.0f}s`)\n"
                                        f"- 🌊 **Motion Density:** `{info.get('motion_pct', 0.0):.1f}%`"
                                    )
                                if zones_live_placeholder is not None and info.get("zA") and info.get("zones"):
                                    zA_cur = info["zA"]
                                    z_lines = []
                                    for zz in info["zones"]:
                                        st_z = zA_cur.stats.get(zz.zone_id)
                                        occ = st_z.current_occupancy if st_z else 0
                                        entries = st_z.total_entries if st_z else 0
                                        icon = "🔴" if zz.zone_type == "restricted" else ("🟣" if zz.zone_type == "checkout" else ("🔵" if zz.zone_type == "shelf" else "🟢"))
                                        cap_str = f"/{zz.capacity}" if zz.capacity else ""
                                        z_lines.append(f"{icon} **{zz.name}**: `{occ}{cap_str}` *(entries: {entries})*")
                                    zones_live_placeholder.markdown("\n\n".join(z_lines))
                                if alerts_live_placeholder is not None:
                                    acts = info.get("activities_log", [])
                                    if acts:
                                        a_lines = []
                                        for a in acts[-5:][::-1]:
                                            sev_v = a.severity.value if hasattr(a.severity, "value") else str(a.severity)
                                            badge = "🔴" if sev_v in ("HIGH", "CRITICAL") else ("🟠" if sev_v == "MEDIUM" else "🔵")
                                            a_lines.append(f"{badge} **[{sev_v}]** {a.message}")
                                        alerts_live_placeholder.markdown("\n\n".join(a_lines))
                                    else:
                                        alerts_live_placeholder.caption("Scanning video frames for events...")
                                if hm_live_placeholder is not None and info.get("hm") is not None:
                                    hm_img = info["hm"].get_image()
                                    hm_live_placeholder.image(
                                        cv2.cvtColor(hm_img, cv2.COLOR_BGR2RGB),
                                        use_container_width=True,
                                    )
                            except Exception as error:
                                on_error("Live preview", last_saved["frame"], str(error))

                        def on_progress(pct, info):
                            last_saved["frame"] = info["frame"]
                            last_saved["total"] = info["total"]
                            now = time.monotonic()
                            if now - last_saved["time"] >= 5:
                                run_store.update(run_id, info["frame"], info["total"])
                                last_saved["time"] = now
                            progress_bar.progress(pct)
                            status_text.markdown(
                                f"🔴 **LIVE DETECTING** — Frame **{info['frame']}/{info['total']}** "
                                f"| Speed: **{info['fps']:.1f} FPS** | "
                                f"People: **{info['tracks']}** (Unique: **{info.get('unique_people', 0)}**) | "
                                f"Objects/Bags: **{info.get('objects_now', 0)}** | "
                                f"Footfall: **{info['footfall']}** | "
                                f"Events: **{info['suspicious']}**"
                            )
                            fps_m.metric("Processing FPS", f"{info['fps']:.1f}")
                            frame_m.metric("Frame", f"{info['frame']}/{info['total']}")
                            tracks_m.metric("Active People", f"{info['tracks']} ({info.get('unique_people', 0)} tot)")
                            objs_m.metric("Objects / Bags", f"{info.get('objects_now', 0)}")
                            foot_m.metric("Footfall", info["footfall"])
                            sus_m.metric("Detected Events", info["suspicious"])
                            _update_live_side_panels(info)

                        def on_frame(frame, info=None):
                            if not show_preview or video_placeholder is None:
                                return
                            preview_counter["n"] += 1
                            if preview_counter["n"] != 1 and preview_counter["n"] % max(1, preview_every) != 0:
                                return
                            try:
                                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
                                if info is not None:
                                    _update_live_side_panels(info)
                            except Exception as error:
                                on_error("Live preview", last_saved["frame"], str(error))

                        # Run batch processing using the compatible video path
                        process_video_path = compatible_path or input_path
                        result = process_video_batch(
                            video_path=process_video_path,
                            output_path=output_path,
                            det=det, trk=trk,
                            zones=zones, zA=zA, hm=hm, am=am,
                            sus_detector=sus_det, bhv=bhv,
                            show_hm=show_hm, enable_pose=enable_pose,
                            preproc=preproc, bg_sub=bg_sub,
                            line_det=line_det, queue_det=queue_det,
                            olb_det=olb_det, orm_det=orm_det, obj_shelves=obj_shelves,
                            face_blur=face_blur, video_clip=video_clip,
                            gest_d=gest_d, fall_d=fall_d, fight_d=fight_d,
                            progress_callback=on_progress,
                            frame_callback=on_frame,
                            error_callback=on_error,
                            camera_id=cam_id,
                        )

                        run_store.update(run_id, result["processed_frames"], result["total_frames"],
                                         status="completed", summary={
                                             "fps": result["avg_fps"], "events": result["suspicious_count"],
                                             "output": output_path,
                                         })
                        progress_bar.progress(1.0)
                        status_text.success("✅ Live Video Detection & Batch Processing Complete!")

                        # Store results - KEEP PATHS FOR DISPLAY
                        st.session_state["processed_video_path"] = output_path
                        st.session_state["stats"] = [{
                            "fps": result["avg_fps"],
                            "tr": result.get("unique_people", 0),
                            "fh": result["total_footfall"],
                            "proc_ms": (result["elapsed_seconds"] / max(result["processed_frames"], 1)) * 1000,
                        }]
                        st.session_state["hm"] = hm
                        st.session_state["am"] = am
                        st.session_state["activities_log"] = result["activities_log"]
                        st.session_state["zones"] = zones
                        st.session_state["zA"] = zA
                        st.session_state["last_queue_status"] = result.get("last_queue_status", [])
                        st.session_state["last_object_events"] = result.get("last_object_events", [])

                        # Show results
                        st.markdown("---")
                        st.markdown("### 📊 Live Video Detection Summary")
                        r1, r2, r3, r4, r5, r6 = st.columns(6)
                        r1.metric("Frames Analyzed", result["processed_frames"])
                        r2.metric("Avg Speed", f"{result['avg_fps']:.1f} FPS")
                        r3.metric("Unique People", result.get("unique_people", 0))
                        r4.metric("Footfall", result["total_footfall"])
                        r5.metric("Objects / Bags Sightings", result.get("total_objects_seen", 0))
                        r6.metric("Events & Alerts", result["suspicious_count"])

                        # Show detected events breakdown table if any events occurred
                        if result["activities_log"]:
                            with st.expander(f"🚨 View All {len(result['activities_log'])} Detected Activities & Events from Video", expanded=True):
                                for act_item in result["activities_log"][-15:][::-1]:
                                    sev_v = act_item.severity.value if hasattr(act_item.severity, "value") else str(act_item.severity)
                                    badge = "🔴" if sev_v in ("HIGH", "CRITICAL") else ("🟠" if sev_v == "MEDIUM" else "🔵")
                                    st.markdown(
                                        f"{badge} **[{sev_v}]** `{act_item.activity_type.value}` — {act_item.message} "
                                        f"*(Confidence: {act_item.confidence:.0%})*"
                                    )

                        # Show processed video
                        st.markdown("### 🎬 Processed Output Video (Full Annotations)")
                        try:
                            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                        except OSError:
                            file_size_mb = 0

                        # Check if output file exists and is readable
                        if os.path.exists(output_path) and file_size_mb > 0:
                            try:
                                if file_size_mb < 30:
                                    st.success(f"📦 Output size: {file_size_mb:.1f} MB — streaming inline")
                                    st.video(output_path)
                                else:
                                    st.warning(f"📦 Output size: {file_size_mb:.1f} MB — large video")
                                    st.video(output_path)
                            except Exception as video_err:
                                st.error(f"⚠️ Video cannot be streamed. Download directly below.")
                                print(f"[WARN] Video streaming failed: {video_err}")
                        else:
                            st.error(f"⚠️ Output video not found or empty. Path: {output_path}")

                        # Download button - ALWAYS show prominently
                        if os.path.exists(output_path):
                            with open(output_path, "rb") as f:
                                st.download_button(
                                    "⬇️ Download Processed Video",
                                    f,
                                    file_name=f"processed_{up.name}",
                                    mime="video/mp4",
                                    width='stretch',
                                )
                        else:
                            st.error("⚠️ Processed video file not available for download")
                    finished = True
                except Exception as error:
                    run_store.error(run_id, "Processing", last_saved["frame"], str(error))
                    run_store.update(run_id, last_saved["frame"], last_saved["total"], status="failed", error=str(error))
                    st.error(f"Video processing failed: {error}")
                    st.exception(error)
                    finished = True
                finally:
                    st.session_state["_is_processing_video"] = False
                    if not finished:
                        run_store.update(run_id, last_saved["frame"], last_saved["total"], status="interrupted", error="Processing was stopped before completion.")
            elif st.session_state.get("processed_video_path") == output_path and Path(output_path).is_file():
                st.success("This upload has been analyzed. Start again to apply changed settings.")
                st.video(output_path)
                with open(output_path, "rb") as video_file:
                    st.download_button("Download analyzed video", video_file, file_name="analyzed.mp4", mime="video/mp4")

        else:
            st.info("👆 Upload a video file above to watch the live detection preview and generate a processed output video.")

    else:
        # ===== LIVE STREAM MODE =====
        st.markdown("""
        <div class="result-card">
        <h3>📡 Live Stream</h3>
        <p>Connect to a webcam or RTSP camera for real-time analysis.
        Use <strong>Frame Skip</strong> in the sidebar to adjust performance.</p>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns([3, 1])
        with c2:
            st.subheader("📊 Stats")
            fps_metric = st.empty()
            tracks_metric = st.empty()
            footfall_metric = st.empty()
            alerts_metric = st.empty()
            sus_metric = st.empty()
            proc_time_metric = st.empty()
            fps_metric.metric("Display FPS", "0.0")
            tracks_metric.metric("Tracks", "0")
            footfall_metric.metric("Footfall", "0")
            alerts_metric.metric("Alerts", "0")
            sus_metric.metric("Suspicious", "0")
            proc_time_metric.metric("Proc Time", "0ms")

            st.subheader("📍 Zones")
            zones_container = st.empty()
            st.subheader("⚠️ Alerts")
            alerts_container = st.empty()
            st.subheader("🗺️ Heatmap")
            heatmap_container = st.empty()

        with c1:
            src_type = st.radio("Source Type", ["Webcam", "RTSP URL"],
                                horizontal=True, key="live_src_type")

            video_source = None
            if src_type == "Webcam":
                camera_idx = st.number_input("Camera index", 0, 10, 0, 1, key="cam_idx")
                video_source = camera_idx
                st.caption(f"Webcam #{camera_idx}")
            else:
                rtsp_url = st.text_input("RTSP URL",
                                          "rtsp://admin:password@192.168.1.100:554/stream",
                                          key="rtsp_url")
                if rtsp_url and rtsp_url.startswith("rtsp://"):
                    video_source = rtsp_url
                else:
                    st.info("Enter an RTSP URL to stream from an IP camera")

            b1, b2 = st.columns(2)
            with b1:
                start_enabled = video_source is not None
                if st.button("▶️ Start", type="primary", width='stretch',
                             key="start_live", disabled=not start_enabled):
                    st.session_state["run"] = True
            with b2:
                if st.button("⏹️ Stop", width='stretch', key="stop_live"):
                    st.session_state["run"] = False

            fp = st.empty()

            if st.session_state["run"] and video_source is not None:
                try:
                    det = get_det(ms, cf, dv)
                except Exception as e:
                    st.error(f"Model error: {e}")
                    det = None

                if det:
                    st.session_state["stats"] = []
                    st.session_state["zones"] = _default_zones(960, 540)
                    st.session_state["zA"] = ZoneAnalytics(st.session_state["zones"], 960, 540)
                    st.session_state["hm"] = HeatmapGenerator(960, 540)
                    st.session_state["am"] = AlertManager(organization_id=st.session_state.get("aposira_org", "local"), store_id=st.session_state.get("aposira_branch", "main-store"))
                    st.session_state["sus_detector"] = _build_suspicious_detector(enable_pose, loiter_seconds=loiter)
                    st.session_state["activities_log"] = []
                    st.session_state["last_queue_status"] = []
                    st.session_state["last_object_events"] = []

                    preproc = _get_preprocessor() if st.session_state.get("pp_enabled", True) else None
                    bg_sub = _get_bg_subtractor()
                    line_det = _get_line_detector()
                    queue_det = _get_queue_detector()
                    olb_det, orm_det, obj_shelves = _get_object_detector()
                    face_blur = _get_face_blurrer()
                    video_clip = _get_video_clipper()
                    gest_d, fall_d, fight_d = _get_pose_detectors()

                    trk = ByteTracker()
                    bhv = BehaviorDetector(loiter_threshold=loiter,
                                           suspicious_detector=st.session_state["sus_detector"])

                    cap = cv2.VideoCapture(video_source)
                    if not cap.isOpened():
                        st.error("Failed to open video source")
                        st.session_state["run"] = False
                    else:
                        fc = 0
                        last_display_time = time.time()
                        frame_times = deque(maxlen=30)
                        proc_times = deque(maxlen=30)
                        last_frame = None
                        pose_gestures_live = {}

                        while st.session_state["run"]:
                            loop_start = time.time()
                            ret, fr = cap.read()
                            if not ret:
                                st.warning("Lost connection, reconnecting...")
                                time.sleep(1)
                                cap.release()
                                cap = cv2.VideoCapture(video_source)
                                if not cap.isOpened():
                                    break
                                continue

                            fc += 1
                            fr = cv2.resize(fr, (960, 540))
                            ts = time.time()

                            avg_proc = np.mean(proc_times) if proc_times else 0.033
                            target_fps = _calculate_target_fps(avg_proc)
                            should_process = _should_process_frame(fc, st.session_state["frame_skip"])

                            if should_process:
                                proc_start = time.time()

                                if preproc is not None:
                                    try: fr = preproc(fr)
                                    except Exception: pass
                                if face_blur is not None:
                                    try: fr = face_blur.blur(fr)
                                    except Exception: pass
                                if video_clip is not None:
                                    try: video_clip.add_frame(fr, ts)
                                    except Exception: pass
                                bg_stats = None
                                if bg_sub is not None:
                                    try: bg_stats = bg_sub.apply(fr)
                                    except Exception: pass

                                dets = det.detect(fr, classes=RETAIL_CLASSES)
                                person_dets = [d for d in dets if d.class_id == 0]
                                object_dets = [d for d in dets if d.class_id != 0]
                                res = trk.update(fr, person_dets, ts)
                                carried_map = _associate_carried_objects(res.tracks, object_dets)
                                st.session_state["zA"].update(res.tracks, ts)
                                st.session_state["hm"].update(res.tracks, ts)

                                cfg = {z.zone_id: {"name": z.name, "dwell_threshold": z.dwell_threshold, "capacity": z.capacity}
                                       for z in st.session_state["zones"]}

                                if line_det is not None:
                                    try: line_det.update(res.tracks, ts)
                                    except Exception: pass
                                if queue_det is not None:
                                    try: st.session_state["last_queue_status"] = queue_det.update(
                                        res.tracks, st.session_state["zA"].zone_occ, ts)
                                    except Exception: pass
                                obj_events = []
                                if olb_det is not None and obj_shelves:
                                    try: obj_events.extend(olb_det.update(
                                        fr, res.tracks, st.session_state["zA"].zone_occ, obj_shelves, ts))
                                    except Exception: pass
                                if orm_det is not None and obj_shelves:
                                    try: obj_events.extend(orm_det.update(
                                        fr, res.tracks, st.session_state["zA"].zone_occ, obj_shelves, ts))
                                    except Exception: pass
                                if obj_events:
                                    st.session_state["last_object_events"].extend(obj_events)
                                    st.session_state["last_object_events"] = st.session_state["last_object_events"][-200:]

                                pose_ests = {}
                                if enable_pose and st.session_state.get("pose_analyzer"):
                                    try: pose_ests = st.session_state["pose_analyzer"].estimate(fr, res.tracks, ts) or {}
                                    except Exception: pass
                                pose_events_live = []
                                if gest_d:
                                    try: pose_events_live.extend(gest_d.update(pose_ests, res.tracks, ts) or [])
                                    except Exception: pass
                                if fall_d:
                                    try: pose_events_live.extend(fall_d.update(res.tracks, pose_ests, ts) or [])
                                    except Exception: pass
                                if fight_d:
                                    try: pose_events_live.extend(fight_d.update(res.tracks, pose_ests, ts) or [])
                                    except Exception: pass
                                for pev in pose_events_live:
                                    if pev.gesture and pev.track_id is not None:
                                        pose_gestures_live[pev.track_id] = pev.gesture
                                    act = _make_activity_from_aux(
                                        activity_id=f"v1_pose_{pev.event_type}_{pev.track_id or 0}_{fc}",
                                        activity_type=ActivityType.CUSTOM_RULE,
                                        severity=Severity.HIGH if pev.event_type in ("fall", "fight") else Severity.MEDIUM,
                                        confidence=pev.confidence,
                                        track_id=pev.track_id,
                                        zone_id=None,
                                        zone_name="Store Floor",
                                        camera_id="v1",
                                        message=f"[{pev.event_type.upper()}] {pev.message}",
                                        ts=ts,
                                        evidence={"pose_event": pev.event_type, "gesture": pev.gesture},
                                        rule_violations=[f"pose:{pev.event_type}"],
                                        track_ids=pev.track_ids,
                                    )
                                    st.session_state["am"].create_from_activity(act, "v1", fr, None)
                                    st.session_state["activities_log"].append(act)

                                sus_d = st.session_state["sus_detector"]
                                if sus_d is not None:
                                    try:
                                        activities = sus_d.detect(
                                            tracks=res.tracks,
                                            zones={z.zone_id: z for z in st.session_state["zones"]},
                                            zone_occ=st.session_state["zA"].zone_occ,
                                            timestamp=ts, frame=fr, detections=dets,
                                            camera_id="v1",
                                        )
                                        for a in activities:
                                            track = next((t for t in res.tracks if t.track_id == a.track_id), None)
                                            bbox = track.bbox if track else None
                                            st.session_state["am"].create_from_activity(a, "v1", fr, bbox)
                                            if video_clip is not None:
                                                try: video_clip.save_clip(a.activity_id, fr)
                                                except Exception: pass
                                        st.session_state["activities_log"].extend(activities)
                                        if len(st.session_state["activities_log"]) > 500:
                                            st.session_state["activities_log"] = st.session_state["activities_log"][-500:]
                                    except Exception:
                                        pass

                                for alert in bhv._check_congestion(cfg, st.session_state["zA"].zone_occ, ts):
                                    st.session_state["am"].create(alert.alert_type.value, "v1", alert.track_id,
                                                                  alert.message, alert.severity.value,
                                                                  alert.zone_id, fr, None)

                                s_stats = det.get_performance_stats()
                                proc_time = (time.time() - proc_start) * 1000
                                proc_times.append(proc_time / 1000.0)

                                out = _draw_rich_live_annotations(
                                    frame=fr,
                                    det=det,
                                    trk=trk,
                                    person_dets=person_dets,
                                    object_dets=object_dets,
                                    tracks=res.tracks,
                                    zA=st.session_state["zA"],
                                    zones=st.session_state["zones"],
                                    hm=st.session_state["hm"],
                                    show_hm=show_hm,
                                    bg_stats=bg_stats,
                                    line_det=line_det,
                                    queue_det=queue_det,
                                    last_queue_status=st.session_state.get("last_queue_status", []),
                                    obj_shelves=obj_shelves,
                                    pose_ests=pose_ests,
                                    pose_gestures=pose_gestures_live,
                                    carried_map=carried_map,
                                    activities_log=st.session_state["activities_log"],
                                    fc=fc,
                                    total_frames=0,
                                    fps_val=s_stats.get("fps", 0.0),
                                    ts=ts,
                                    extra_label=f"Proc: {proc_time:.0f}ms | Skip: {st.session_state['frame_skip']}",
                                )

                                last_frame = out
                                st.session_state["last_processed_time"] = time.time()

                            display_frame = last_frame if last_frame is not None else fr
                            fp.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB),
                                     channels="RGB", use_container_width=True)

                            frame_times.append(time.time() - last_display_time)
                            last_display_time = time.time()

                            if should_process:
                                st.session_state["stats"].append({
                                    "fps": s_stats["fps"], "tr": len(res.tracks),
                                    "fh": st.session_state["hm"].total_footfall,
                                    "proc_ms": proc_time,
                                })

                            if fc % 5 == 0 and should_process:
                                avg_display_fps = len(frame_times) / max(sum(frame_times), 0.001) if frame_times else 0
                                avg_proc_ms = np.mean(proc_times) * 1000 if proc_times else 0

                                fps_metric.metric("Display FPS", f"{avg_display_fps:.1f}")
                                tracks_metric.metric("Tracks", len(res.tracks))
                                footfall_metric.metric("Footfall", st.session_state["hm"].total_footfall)
                                alerts_metric.metric("Alerts", len(st.session_state["am"].get_recent(60)))
                                sus_metric.metric("Suspicious", len(st.session_state["activities_log"]))
                                proc_time_metric.metric("Proc Time", f"{avg_proc_ms:.0f}ms")

                                with zones_container.container():
                                    for zid, st_ in st.session_state["zA"].stats.items():
                                        z = next((zz for zz in st.session_state["zones"]
                                                  if zz.zone_id == zid), None)
                                        if z:
                                            c = "🔴" if z.zone_type == "restricted" else "🟡"
                                            st.markdown(f'<div class="zone">{c} {z.name}: {st_.current_occupancy}</div>',
                                                        unsafe_allow_html=True)
                                cur = st.session_state["am"].get_recent(60)
                                if cur:
                                    with alerts_container.container():
                                        for a in cur[:4]:
                                            sev_cls = f"severity-{a.severity.upper()}"
                                            st.markdown(
                                                f"**{datetime.fromtimestamp(a.timestamp).strftime('%H:%M:%S')}** "
                                                f"<span class='{sev_cls}'>[{a.severity}]</span> {a.message}",
                                                unsafe_allow_html=True)
                                else:
                                    alerts_container.info("No recent alerts")
                                if st.session_state["hm"]:
                                    heatmap_container.image(
                                        cv2.cvtColor(st.session_state["hm"].get_image(), cv2.COLOR_BGR2RGB),
                                        use_column_width=True)

                            elapsed = time.time() - loop_start
                            target_interval = 1.0 / target_fps
                            if elapsed < target_interval:
                                time.sleep(max(0.001, target_interval - elapsed))

                        cap.release()
                        st.session_state["run"] = False

# --- Tab: Zone Editor ---
with tab_zones:
    st.markdown("### 🗺️ Zone Editor")
    st.caption("Define monitoring zones for tracking people in specific areas.")

    cam_id = st.text_input("Camera ID", value="v1", key="zone_cam")

    # FIXED: Get video path from session state - check both uploaded path and first frame
    video_for_frame = st.session_state.get("uploaded_video_path")
    first_frame = st.session_state.get("uploaded_video_first_frame")

    # Validate the path exists if set
    if video_for_frame and not Path(video_for_frame).exists():
        video_for_frame = None

    # If we have a first frame stored but no valid video path, use the frame directly
    if first_frame is not None:
        st.info("📹 Showing first frame from uploaded video for zone editing")

    existing = []
    try:
        existing = load_zones(camera_id=cam_id)
    except Exception:
        pass
    if not existing:
        try:
            existing = load_zones(path=PROJECT_ROOT / "config/zones.yaml")
        except Exception:
            pass

    # Pass the first frame to the polygon editor - save it as an image file
    try:
        if first_frame is not None:
            # Save the first frame to a temp file that persists
            data_dir = runtime_directory(PROJECT_ROOT / "data/uploads")
            data_dir.mkdir(parents=True, exist_ok=True)
            frame_path = str(data_dir / "zone_editor_frame.png")
            cv2.imwrite(frame_path, first_frame)
            result = render_polygon_editor(camera_id=cam_id, video_path=frame_path,
                                          existing_zones=existing)
        else:
            result = render_polygon_editor(camera_id=cam_id, video_path=video_for_frame,
                                          existing_zones=existing)
    except Exception as e:
        st.error(f"Zone editor error: {e}")
        st.info("Try switching to the Live tab first and upload a video, then return here.")
        result = {"zones": existing or [], "saved_to": None}
    if result.get("saved_to"):
        st.success(f"Saved to {result['saved_to']}")
    st.markdown("---")
    st.subheader("📁 Saved Runtime Zone Files")
    files = list_runtime_zone_files()
    if files:
        for p in files:
            st.markdown(f"- `{p}`")
    else:
        st.caption("No runtime zone files yet.")

# --- Tab: Advanced Analytics ---
with tab_adv:
    render_advanced_analytics()

# --- Tab: Suspicious Activity ---
with tab_sus:
    st.markdown("### 🚨 Suspicious Activity Feed")
    activities: List[SuspiciousActivity] = st.session_state.get("activities_log", [])
    window = st.slider("Time window (s)", 30, 1800, 300, 30)
    cutoff = time.time() - window
    activities = [a for a in activities if a.timestamp >= cutoff]
    render_alerts_panel(activities)

    line_det = _get_line_detector()
    if line_det is not None:
        st.markdown("---")
        st.markdown("### ➡️ Line Crossing Counts")
        counts = line_det.get_counts()
        if counts:
            for (lid, d), c in counts.items():
                st.markdown(f"- **{lid}** ({d}): `{c}`")
        else:
            st.caption("No crossings yet.")

    queue_det = _get_queue_detector()
    if queue_det is not None and st.session_state.get("last_queue_status"):
        st.markdown("---")
        st.markdown("### 🧍 Queue Status")
        for s in st.session_state["last_queue_status"]:
            color = "🟢" if not s.is_queue else "🟠" if s.length < 4 else "🔴"
            st.markdown(
                f"{color} **{s.name}** — {s.length} ppl, "
                f"est wait **{s.estimated_wait_seconds:.0f}s**, "
                f"avg **{s.avg_wait_seconds:.0f}s**"
            )

    if st.session_state.get("last_object_events"):
        st.markdown("---")
        st.markdown("### 📦 Object Monitor Events")
        for e in st.session_state["last_object_events"][-10:]:
            icon = "🟡" if e.event_type == "object_left" else "🔴"
            st.markdown(f"{icon} {e.message}")

# --- Tab: Tuner ---
with tab_tune:
    render_threshold_tuner()

# --- Tab: Dashboard ---
with tab_dash:
    st.markdown("### 📊 Analytics Dashboard")
    st.caption("Quick view of recent activity.")
    if st.session_state.get("stats"):
        df_stats = st.session_state["stats"][-200:]
        fps_vals = [s["fps"] for s in df_stats]
        proc_vals = [s.get("proc_ms", 0) for s in df_stats]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg FPS", f"{sum(fps_vals) / max(1, len(fps_vals)):.1f}")
        c2.metric("Peak FPS", f"{max(fps_vals) if fps_vals else 0:.1f}")
        c3.metric("Total Footfall", st.session_state["hm"].total_footfall if st.session_state["hm"] else 0)
        c4.metric("Suspicious (recent)",
                  sum(1 for a in st.session_state.get("activities_log", [])
                      if a.timestamp >= time.time() - 300))
        if proc_vals:
            c5, c6 = st.columns(2)
            c5.metric("Avg Proc Time", f"{sum(proc_vals) / max(1, len(proc_vals)):.0f}ms")
            c6.metric("Frame Skip", st.session_state.get("frame_skip", 2))
    else:
        st.info("Run the live analysis to populate dashboard stats.")

    st.markdown("---")
    st.markdown("### 📦 Module Status")
    cm1, cm2, cm3 = st.columns(3)
    cm1.markdown("**Detection / Tracking**")
    cm1.markdown("- ✅ YOLOv8 person detection")
    cm1.markdown("- ✅ ByteTrack tracking")
    cm1.markdown("- ✅ Re-ID (optional)")
    cm1.markdown("- ✅ Heatmap / footfall")
    cm2.markdown("**Advanced Analytics**")
    cm2.markdown(f"- {'✅' if st.session_state.get('pp_enabled', True) else '⬜'} Frame preprocessor")
    cm2.markdown(f"- {'✅' if st.session_state.get('bg_enabled') else '⬜'} Background subtraction")
    cm2.markdown(f"- {'✅' if st.session_state.get('line_enabled') else '⬜'} Line crossing")
    cm2.markdown(f"- {'✅' if st.session_state.get('queue_enabled') else '⬜'} Queue detection")
    cm2.markdown(f"- {'✅' if st.session_state.get('obj_enabled') else '⬜'} Object monitor")
    cm3.markdown("**Privacy / Output**")
    cm3.markdown(f"- {'✅' if st.session_state.get('face_blur_enabled') else '⬜'} Face blur")
    cm3.markdown(f"- {'✅' if st.session_state.get('clip_enabled') else '⬜'} Video clipper")
    cm3.markdown(f"- {'✅' if st.session_state.get('pose_gesture') else '⬜'} Gesture detection")
    cm3.markdown(f"- {'✅' if st.session_state.get('pose_fall') else '⬜'} Fall detection")
    cm3.markdown(f"- {'✅' if st.session_state.get('pose_fight') else '⬜'} Fight detection")

    clips_dir = Path("data/clips")
    if clips_dir.exists():
        clips = sorted(clips_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        if clips:
            st.markdown("---")
            st.markdown("### 🎥 Recent Clips")
            for c in clips[:5]:
                st.markdown(f"- `{c.name}` ({c.stat().st_size // 1024} KB)")

    # Show processed video in dashboard if available
    processed_path = st.session_state.get("processed_video_path")
    if processed_path and os.path.exists(processed_path):
        st.markdown("---")
        st.markdown("### 🎬 Processed Video (from last batch)")
        try:
            file_size_mb = os.path.getsize(processed_path) / (1024 * 1024)
            st.info(f"📦 Video file size: {file_size_mb:.1f} MB")
            st.video(processed_path)

            # Download button
            with open(processed_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Processed Video",
                    f,
                    file_name="processed_video.mp4",
                    mime="video/mp4",
                    width='stretch',
                )
        except Exception as e:
            st.error(f"Could not load processed video: {e}")


# Persisted event timeline, available independently of the current video session.
if globals().get("APOSIRA_EMBEDDED"):
    from aposira.vision_events import render_timeline
    render_timeline(st.session_state.aposira_org, st.session_state.aposira_branch)
