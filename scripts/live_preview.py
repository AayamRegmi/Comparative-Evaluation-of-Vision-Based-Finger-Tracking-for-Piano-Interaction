# live_preview.py
# MediaPipe Hands live preview & performance stats
# All files are in the same folder → simple imports

import cv2
import mediapipe as mp
import time
import numpy as np

# Import local modules (same folder, no sys.path needed)
from . import config
from .stats_collector import StatsCollector
from .camera_setup import init_camera, RESIZE_INTERPOLATION

def run_preview():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Initialize MediaPipe Hands with config values
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=config.MAX_NUM_HANDS,
        model_complexity=config.MODEL_COMPLEXITY,
        min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
    )

    # Initialize camera using config
    cap = init_camera(config)

    print(f"MediaPipe config: complexity={config.MODEL_COMPLEXITY}, "
          f"resize={config.RESIZE_WIDTH}x{config.RESIZE_HEIGHT if config.RESIZE_WIDTH else 'native'}, "
          f"max_hands={config.MAX_NUM_HANDS}\n")

    # Initialize stats collector
    stats = StatsCollector(config.WARMUP_FRAMES, config.STAT_FRAMES)

    print("Press ESC to stop and show statistics\n")

    while cap.isOpened():
        loop_start = time.perf_counter()

        success, frame = cap.read()
        if not success:
            print("Frame capture failed")
            break

        # Resize if configured
        if config.RESIZE_WIDTH is not None:
            process_frame = cv2.resize(
                frame,
                (config.RESIZE_WIDTH, config.RESIZE_HEIGHT),
                interpolation=RESIZE_INTERPOLATION
            )
        else:
            process_frame = frame.copy()

        rgb = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        # Inference
        t0 = time.perf_counter()
        results = hands.process(rgb)
        t1 = time.perf_counter()
        inference_ms = (t1 - t0) * 1000

        # Draw landmarks if enabled
        if config.DRAW_LANDMARKS and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        # Calculate total latency and update stats
        total_latency_ms = (time.perf_counter() - loop_start) * 1000
        stats.update(loop_start, inference_ms, total_latency_ms)

        # Live overlay
        if stats.warmup_done:
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
                cv2.putText(frame, f"Total lat: {total_latency_ms:4.1f} ms", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.TEXT_COLOR_LAT, 2)

        cv2.imshow("MediaPipe Hand Tracking - Live Preview", frame)

        if cv2.waitKey(1) & 0xFF == config.EXIT_KEY:
            break

    # Print final statistics
    stats.print_final_stats(
        process_frame.shape[1],
        process_frame.shape[0],
        config.MODEL_COMPLEXITY
    )

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    run_preview()