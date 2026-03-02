"""
download_openpose_model.py
--------------------------
Downloads the OpenPose Hand model files into models/ (one-time setup).

Files downloaded:
  models/openpose_hand.prototxt   (~5 KB  — network architecture)
  models/openpose_hand.caffemodel (~55 MB — pre-trained weights)

The hand model detects 21 finger keypoints per hand — the same joints as
MediaPipe Hands, making it a direct comparator for the MJMPE research.

Usage:
    python download_openpose_model.py

After this completes, press M in live_preview.py to switch to OpenPose Hand.
"""

import sys
import urllib.request
import pathlib

PROTOTXT_URL = (
    "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/"
    "openpose/master/models/hand/pose_deploy.prototxt"
)
# CMU server (posefs1) is permanently offline — use Hugging Face mirrors
CAFFEMODEL_URL = (
    "https://huggingface.co/camenduru/openpose/resolve/main/"
    "models/hand/pose_iter_102000.caffemodel"
)
CAFFEMODEL_URL_MIRROR = (
    "https://huggingface.co/gaijingeek/openpose-models/resolve/main/"
    "models/hand/pose_iter_102000.caffemodel"
)

MODELS_DIR   = pathlib.Path("models")
PROTO_DEST   = MODELS_DIR / "openpose_hand.prototxt"
CAFFE_DEST   = MODELS_DIR / "openpose_hand.caffemodel"
CAFFE_MIN_MB = 50   # sanity check: hand model is ~55 MB


def _progress(count, block_size, total_size):
    if total_size <= 0:
        return
    pct = min(int(count * block_size * 100 / total_size), 100)
    bar = "#" * (pct // 2) + "-" * (50 - pct // 2)
    mb  = count * block_size / 1_048_576
    sys.stdout.write(f"\r  [{bar}] {pct:3d}%  {mb:6.1f} MB")
    sys.stdout.flush()


def _download(url: str, dest: pathlib.Path, label: str) -> bool:
    print(f"\nDownloading {label}...")
    print(f"  From : {url}")
    print(f"  To   : {dest}")
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        print()   # newline after progress bar
        return True
    except Exception as exc:
        print(f"\n  ERROR: {exc}")
        if dest.exists():
            dest.unlink()
        return False


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- prototxt --------------------------------------------------------
    if PROTO_DEST.exists():
        print(f"[OK] Prototxt already present: {PROTO_DEST}")
    else:
        ok = _download(PROTOTXT_URL, PROTO_DEST, "OpenPose Hand prototxt (~5 KB)")
        if not ok:
            print("  Could not download prototxt. Check your internet connection.")
            sys.exit(1)
        print(f"  Saved: {PROTO_DEST}")

    # ---- caffemodel ------------------------------------------------------
    if CAFFE_DEST.exists():
        size_mb = CAFFE_DEST.stat().st_size / 1_048_576
        if size_mb >= CAFFE_MIN_MB:
            print(f"[OK] Caffemodel already present: {CAFFE_DEST}  ({size_mb:.0f} MB)")
            print("\nAll files ready. Run live_preview.py and press M for OpenPose Hand.")
            return
        else:
            print(f"  Existing caffemodel looks incomplete ({size_mb:.1f} MB). Re-downloading.")
            CAFFE_DEST.unlink()

    ok = _download(CAFFEMODEL_URL, CAFFE_DEST, "OpenPose Hand caffemodel (~55 MB)")
    if not ok:
        print(f"\n  Primary server failed. Trying mirror...")
        ok = _download(CAFFEMODEL_URL_MIRROR, CAFFE_DEST, "OpenPose Hand caffemodel (mirror)")

    if not ok or not CAFFE_DEST.exists():
        print("\n  Download failed on all sources.")
        print("  You can manually download the file from:")
        print(f"    {CAFFEMODEL_URL}")
        print(f"  and place it at:  {CAFFE_DEST}")
        sys.exit(1)

    size_mb = CAFFE_DEST.stat().st_size / 1_048_576
    if size_mb < CAFFE_MIN_MB:
        print(f"\n  Warning: file is only {size_mb:.1f} MB — may be incomplete.")
    else:
        print(f"  Saved: {CAFFE_DEST}  ({size_mb:.0f} MB)")

    print("\nAll files ready. Run live_preview.py and press M for OpenPose Hand.")


if __name__ == "__main__":
    main()
