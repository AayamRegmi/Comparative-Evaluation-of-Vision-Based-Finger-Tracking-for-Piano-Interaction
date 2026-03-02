# live_preview.py
# MediaPipe Hands live preview & performance stats with live model switching.
#
# Key bindings:
#   M   – cycle model: MediaPipe Hands → MoveNet Lightning → OpenPose COCO → …
#   ESC – quit and print final statistics

import cv2
import time
import numpy as np
from pathlib import Path

try:
    from . import config
    from .stats_collector import StatsCollector
    from .camera_setup import init_camera, RESIZE_INTERPOLATION
    from .model_manager import ModelManager
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config
    from stats_collector import StatsCollector
    from camera_setup import init_camera, RESIZE_INTERPOLATION
    from model_manager import ModelManager


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------

def _draw_loading_overlay(frame: np.ndarray, model_name: str) -> None:
    h, w = frame.shape[:2]
    dim  = frame.copy()
    cv2.rectangle(dim, (w // 4, h // 2 - 55), (3 * w // 4, h // 2 + 55),
                  (20, 20, 20), -1)
    cv2.addWeighted(dim, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame, f"Loading {model_name}...",
                (w // 4 + 20, h // 2 + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, config.TEXT_COLOR_LOADING, 2)
    cv2.putText(frame, "(window may freeze briefly)",
                (w // 4 + 20, h // 2 + 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)


def _draw_model_label(frame: np.ndarray, model_name: str) -> None:
    label = f"Model: {model_name}  [M to switch]"
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    x = frame.shape[1] - tw - 10
    cv2.putText(frame, label, (x, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.TEXT_COLOR_MODEL_LABEL, 2)


def _draw_stats_overlay(frame: np.ndarray, stats: StatsCollector,
                        inference_ms: float, total_latency_ms: float) -> None:
    if not stats.warmup_done:
        return
    avg_fps = np.mean(stats.fps_history)
    y = 35
    if config.SHOW_FPS:
        cv2.putText(frame, f"FPS: {avg_fps:4.1f}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, config.TEXT_COLOR_FPS, 2)
        y += 35
    if config.SHOW_INFERENCE_TIME:
        cv2.putText(frame, f"Inf: {inference_ms:4.1f} ms", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.TEXT_COLOR_INF, 2)
        y += 35
    if config.SHOW_TOTAL_LATENCY:
        cv2.putText(frame, f"Lat: {total_latency_ms:4.1f} ms", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.TEXT_COLOR_LAT, 2)


# ---------------------------------------------------------------------------
# Main preview
# ---------------------------------------------------------------------------

def run_preview():
    project_root = str(Path(__file__).parent.parent)

    mgr = ModelManager(config, project_root)
    mgr.ensure_loaded()   # load MediaPipe immediately — fast, no download needed

    cap = init_camera(config)

    print(f"Starting with model: {mgr.current.name}")
    print("M: cycle model   ESC: quit\n")

    stats = StatsCollector(config.WARMUP_FRAMES, config.STAT_FRAMES)

    WIN = "MediaPipe Hand Tracking - Live Preview"
    process_frame = None

    while cap.isOpened():
        loop_start = time.perf_counter()

        # Key input first — keeps the UI responsive
        key = cv2.waitKey(1) & 0xFF
        if key == config.EXIT_KEY:
            break
        elif key == config.MODEL_SWITCH_KEY:
            mgr.cycle()

        # Lazy-load current model if not yet initialised
        # (blocks briefly on first switch to MoveNet / OpenPose)
        needs_load = not mgr.current.loaded

        success, frame = cap.read()
        if not success:
            print("Frame capture failed")
            break

        # Show loading overlay and skip inference this frame
        if needs_load:
            _draw_loading_overlay(frame, mgr.current.name)
            _draw_model_label(frame, mgr.current.name)
            cv2.imshow(WIN, frame)
            mgr.ensure_loaded()   # blocks here — window shows overlay first
            continue

        # Resize for inference
        if config.RESIZE_WIDTH is not None:
            process_frame = cv2.resize(
                frame,
                (config.RESIZE_WIDTH, config.RESIZE_HEIGHT),
                interpolation=RESIZE_INTERPOLATION,
            )
        else:
            process_frame = frame

        # Inference
        t0           = time.perf_counter()
        result       = mgr.current.infer(process_frame)
        t1           = time.perf_counter()
        inference_ms = (t1 - t0) * 1000

        # Draw landmarks
        if config.DRAW_LANDMARKS:
            mgr.current.draw(frame, result)

        # Overlays
        total_latency_ms = (time.perf_counter() - loop_start) * 1000
        stats.update(loop_start, inference_ms, total_latency_ms)

        _draw_stats_overlay(frame, stats, inference_ms, total_latency_ms)
        _draw_model_label(frame, mgr.current.name)

        cv2.imshow(WIN, frame)

    # Final stats use process_frame dimensions (may be None if no frame captured)
    if process_frame is not None:
        stats.print_final_stats(
            process_frame.shape[1],
            process_frame.shape[0],
            config.MODEL_COMPLEXITY,
        )

    mgr.close_all()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_preview()
