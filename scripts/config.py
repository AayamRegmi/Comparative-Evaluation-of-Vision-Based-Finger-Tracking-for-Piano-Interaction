# config.py – all tunable parameters (no cv2 imports here)

# Camera & Capture
CAMERA_INDEX = 0
CAP_WIDTH = 1280
CAP_HEIGHT = 720
CAP_FPS = 60
CAP_BUFFERSIZE = 1

# MediaPipe
MODEL_COMPLEXITY = 1                  # 0 = lite, 1 = full
MAX_NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6

# Processing
RESIZE_WIDTH = 640
RESIZE_HEIGHT = 480

# Stats & Display
WARMUP_FRAMES = 60
STAT_FRAMES = 400
DRAW_LANDMARKS = True
SHOW_FPS = True
SHOW_INFERENCE_TIME = True
SHOW_TOTAL_LATENCY = True

TEXT_COLOR_FPS = (0, 220, 0)
TEXT_COLOR_INF = (0, 200, 255)
TEXT_COLOR_LAT = (100, 100, 255)

EXIT_KEY = 27   # ESC

# Lux Classification Thresholds
LUX_DIM_THRESHOLD    = 100   # below → Dim
LUX_BRIGHT_THRESHOLD = 500   # at or above → Bright; between → Indoor

# Hand Size Input Validation (cm)
HAND_SIZE_MIN_CM = 10.0
HAND_SIZE_MAX_CM = 30.0

# Session Overlay Text Colours
TEXT_COLOR_FITZPATRICK = (100, 200, 255)   # light blue
TEXT_COLOR_SESSION     = (200, 200, 200)   # light grey
TEXT_COLOR_SETUP       = (255, 255, 255)   # white