# stats_collector.py
from collections import deque
import numpy as np

class StatsCollector:
    def __init__(self, warmup_frames, stat_frames):
        self.warmup_frames = warmup_frames
        self.stat_frames = stat_frames
        
        self.fps_history = deque(maxlen=stat_frames)
        self.inference_history_ms = deque(maxlen=stat_frames)
        self.total_latency_history_ms = deque(maxlen=stat_frames)
        
        self.frame_count = 0
        self.warmup_done = False
        self.prev_time = None

    def update(self, loop_start, inference_ms, total_latency_ms):
        now = loop_start
        
        if self.prev_time is None:
            self.prev_time = now
            return

        if self.frame_count >= self.warmup_frames:
            if not self.warmup_done:
                self.warmup_done = True
                print("Warmup complete — collecting statistics...")
            
            fps = 1 / (now - self.prev_time) if self.prev_time > 0 else 0
            self.fps_history.append(fps)
            self.inference_history_ms.append(inference_ms)
            self.total_latency_history_ms.append(total_latency_ms)

        self.prev_time = now
        self.frame_count += 1

    def print_final_stats(self, processed_width, processed_height, model_complexity):
        if not self.warmup_done or len(self.fps_history) == 0:
            print("No statistics collected (not enough frames after warmup)")
            return

        print("\n" + "="*70)
        print(f"Statistics over {len(self.fps_history)} frames (after {self.warmup_frames} warmup)")
        print(f"Processed resolution: {processed_width}×{processed_height}")
        print(f"Model complexity:    {model_complexity}")
        print("-"*60)
        print(f"Avg FPS:              {np.mean(self.fps_history):6.2f} ± {np.std(self.fps_history):5.2f}")
        print(f"Median FPS:           {np.median(self.fps_history):6.2f}")
        p10, p90 = np.percentile(self.fps_history, [10, 90])
        print(f"P10 / P90 FPS:        {p10:6.2f} – {p90:6.2f}")
        print(f"Avg inference time:   {np.mean(self.inference_history_ms):6.2f} ms")
        print(f"Avg total loop lat:   {np.mean(self.total_latency_history_ms):6.2f} ms")
        print("="*70)