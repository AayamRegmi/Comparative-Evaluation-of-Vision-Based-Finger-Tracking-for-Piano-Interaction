# lux_calculator.py
# Converts a manually entered lux value to a lighting category label.
# Thresholds are read from config so they stay in one place.

try:
    from . import config
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config


def lux_to_label(lux: float) -> str:
    """
    Map a lux measurement to a lighting category string.

    Thresholds (set in config.py):
        Bright  >= LUX_BRIGHT_THRESHOLD  (default 500 lux)
        Indoor  >= LUX_DIM_THRESHOLD     (default 100 lux)
        Dim      < LUX_DIM_THRESHOLD
    """
    if lux >= config.LUX_BRIGHT_THRESHOLD:
        return "Bright"
    if lux >= config.LUX_DIM_THRESHOLD:
        return "Indoor"
    return "Dim"
