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

# ---------------------------------------------------------------------------
# Multi-model tracking (live_preview.py)
# ---------------------------------------------------------------------------

MODEL_SWITCH_KEY = ord("m")   # press M to cycle models

MODEL_MEDIAPIPE     = "MediaPipe Hands"
MODEL_OPENPOSE_HAND = "OpenPose Hand"
MODEL_ORDER         = [MODEL_MEDIAPIPE, MODEL_OPENPOSE_HAND]

# OpenPose Hand — 21 finger keypoints (downloaded via download_openpose_model.py)
OPENPOSE_HAND_PROTOTXT             = "models/openpose_hand.prototxt"
OPENPOSE_HAND_CAFFEMODEL           = "models/openpose_hand.caffemodel"
OPENPOSE_HAND_CONFIDENCE_THRESHOLD = 0.1
OPENPOSE_HAND_INPUT_WIDTH          = 368
OPENPOSE_HAND_INPUT_HEIGHT         = 368

TEXT_COLOR_MODEL_LABEL = (255, 200,  50)   # amber
TEXT_COLOR_LOADING     = (200,  80,  80)   # muted red