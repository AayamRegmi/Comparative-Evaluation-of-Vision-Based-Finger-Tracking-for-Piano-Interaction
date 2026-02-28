# camera_setup.py
import cv2
import sys

# OpenCV-specific constants live here
CAP_API = cv2.CAP_DSHOW
RESIZE_INTERPOLATION = cv2.INTER_AREA

def init_camera(config):
    cap = cv2.VideoCapture(config.CAMERA_INDEX, CAP_API)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAP_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.CAP_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, config.CAP_BUFFERSIZE)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        sys.exit(1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera opened: {actual_w}x{actual_h} @ {config.CAP_FPS} FPS "
          f"(requested {config.CAP_WIDTH}x{config.CAP_HEIGHT})")
    
    return cap