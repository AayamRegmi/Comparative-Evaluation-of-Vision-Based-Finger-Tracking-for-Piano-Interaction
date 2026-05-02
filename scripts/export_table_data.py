# export_table_data.py
# Compute every number that plot_results.py would draw, and write them to
# data/table_data.json so the documentation can be filled without reading images.
#
# Usage:
#   python -m scripts.export_table_data
#   python -m scripts.export_table_data --out data/my_table_data.json

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

_ROOT      = Path(__file__).parent.parent
_PROCESSED = _ROOT / "data" / "processed"
_OUT_FILE  = _ROOT / "data" / "table_data.json"

_FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
_MODELS       = ["mediapipe", "openpose"]
_LUX_ORDER    = ["Dim", "Indoor", "Bright"]
_FITZ_LABELS  = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI"}


def _lux_label(lux):
    if lux is None:
        return "Unknown"
    if lux < 100:
        return "Dim"
    if lux < 500:
        return "Indoor"
    return "Bright"


def _r2(xs, ys):
    """Coefficient of determination for a linear fit."""
    if len(xs) < 2:
        return None
    xs, ys = np.array(xs), np.array(ys)
    corr = np.corrcoef(xs, ys)
    return float(round(corr[0, 1] ** 2, 4))


def _stats(vals):
    if not vals:
        return None
    a = np.array(vals)
    return {
        "mean":   float(round(float(np.mean(a)), 3)),
        "sd":     float(round(float(np.std(a)),  3)),
        "median": float(round(float(np.median(a)), 3)),
        "q1":     float(round(float(np.percentile(a, 25)), 3)),
        "q3":     float(round(float(np.percentile(a, 75)), 3)),
        "min":    float(round(float(np.min(a)), 3)),
        "max":    float(round(float(np.max(a)), 3)),
        "n":      len(vals),
    }


def load_records(processed_dir):
    records = []
    for path in sorted(processed_dir.glob("*_results.json")):
        try:
            with open(path) as f:
                r = json.load(f)
            r["lux_label"] = _lux_label(r.get("lux"))
            records.append(r)
        except Exception as e:
            print(f"  Warning: could not load {path.name}: {e}")
    return records


def _finger_vals(records, side, finger_idx, key="mjmpe"):
    vals = []
    for r in records:
        fd = (r.get("per_hand", {}).get(side, {})
               .get("fingers", {}).get(str(finger_idx)))
        if fd and fd.get(key) is not None:
            vals.append(fd[key])
    return vals


def _finger_vals_combined(records, finger_idx, key="mjmpe"):
    return (_finger_vals(records, "L", finger_idx, key) +
            _finger_vals(records, "R", finger_idx, key))


# ---------------------------------------------------------------------------

def chart_00(records):
    seen = set()
    participants = []
    for r in records:
        if r["pid"] not in seen:
            seen.add(r["pid"])
            participants.append(r)

    fitz_counts = {}
    for r in participants:
        ft = r.get("fitzpatrick")
        if ft is not None:
            label = f"Type {_FITZ_LABELS.get(ft, str(ft))}"
            fitz_counts[label] = fitz_counts.get(label, 0) + 1

    lux_counts = {l: 0 for l in _LUX_ORDER}
    for r in participants:
        lx = r["lux_label"]
        lux_counts[lx] = lux_counts.get(lx, 0) + 1

    hand_sizes = [r["hand_size_cm"] for r in participants if r.get("hand_size_cm") is not None]

    return {
        "total_participants": len(participants),
        "fitzpatrick_counts": dict(sorted(fitz_counts.items())),
        "lighting_counts": {k: lux_counts[k] for k in _LUX_ORDER},
        "hand_size_stats": _stats(hand_sizes),
    }


def chart_01(records):
    by_model = defaultdict(list)
    for r in records:
        if r.get("mjmpe_px") is not None:
            by_model[r["model"]].append(r["mjmpe_px"])
    out = {}
    for m in _MODELS:
        vals = by_model[m]
        if vals:
            s = _stats(vals)
            out[m] = {"mean_px": s["mean"], "sd_px": s["sd"], "n": s["n"]}
    return out


def chart_02(records):
    mp_recs = [r for r in records if r["model"] == "mediapipe"]
    op_recs = [r for r in records if r["model"] == "openpose"]
    out = {"mediapipe": {"L": {}, "R": {}}, "openpose": {"combined": {}}}
    for side in ["L", "R"]:
        for fi, name in enumerate(_FINGER_NAMES):
            vals = _finger_vals(mp_recs, side, fi)
            out["mediapipe"][side][name] = {
                "mean_px": round(float(np.mean(vals)), 3) if vals else None,
                "n": len(vals),
            }
    for fi, name in enumerate(_FINGER_NAMES):
        vals = _finger_vals_combined(op_recs, fi)
        out["openpose"]["combined"][name] = {
            "mean_px": round(float(np.mean(vals)), 3) if vals else None,
            "n": len(vals),
        }
    return out


def chart_03(records):
    by_model = defaultdict(lambda: {"matched": 0, "detection_fail": 0, "missed": 0})
    for r in records:
        m = r["model"]
        by_model[m]["matched"]        += r.get("notes_matched", 0) or 0
        by_model[m]["detection_fail"] += r.get("notes_detection_fail", 0) or 0
        by_model[m]["missed"]         += r.get("notes_missed", 0) or 0
    out = {}
    for m in _MODELS:
        if m not in by_model:
            continue
        d = by_model[m]
        total = d["matched"] + d["detection_fail"] + d["missed"]
        out[m] = {
            "matched_count":        d["matched"],
            "detection_fail_count": d["detection_fail"],
            "missed_count":         d["missed"],
            "total_count":          total,
            "matched_pct":          round(100 * d["matched"] / total, 2) if total else 0,
            "detection_fail_pct":   round(100 * d["detection_fail"] / total, 2) if total else 0,
            "missed_pct":           round(100 * d["missed"] / total, 2) if total else 0,
        }
    return out


def chart_04(records):
    by_lux_model = defaultdict(list)
    for r in records:
        if r.get("mjmpe_px") is not None:
            by_lux_model[(r["lux_label"], r["model"])].append(r["mjmpe_px"])
    out = {}
    for lbl in _LUX_ORDER:
        out[lbl] = {}
        for m in _MODELS:
            vals = by_lux_model.get((lbl, m), [])
            out[lbl][m] = {
                "mean_px": round(float(np.mean(vals)), 3) if vals else None,
                "sd_px":   round(float(np.std(vals)),  3) if len(vals) > 1 else None,
                "n": len(vals),
            }
    return out


def chart_05(records):
    by_fitz_model = defaultdict(list)
    for r in records:
        if r.get("mjmpe_px") is not None and r.get("fitzpatrick"):
            by_fitz_model[(r["fitzpatrick"], r["model"])].append(r["mjmpe_px"])
    all_types = sorted({r.get("fitzpatrick") for r in records if r.get("fitzpatrick")})
    out = {}
    for ft in all_types:
        label = f"Type {_FITZ_LABELS.get(ft, str(ft))}"
        out[label] = {}
        for m in _MODELS:
            vals = by_fitz_model.get((ft, m), [])
            out[label][m] = {
                "mean_px": round(float(np.mean(vals)), 3) if vals else None,
                "sd_px":   round(float(np.std(vals)),  3) if len(vals) > 1 else None,
                "n": len(vals),
            }
    return out


def chart_06(records):
    out = {}
    for m in _MODELS:
        recs = [r for r in records
                if r["model"] == m
                and r.get("mjmpe_px") is not None
                and r.get("hand_size_cm") is not None]
        if not recs:
            continue
        xs = [r["hand_size_cm"] for r in recs]
        ys = [r["mjmpe_px"]     for r in recs]
        if len(xs) >= 2:
            slope, intercept = np.polyfit(xs, ys, 1)
            r2 = _r2(xs, ys)
        else:
            slope, intercept, r2 = None, None, None
        out[m] = {
            "slope_px_per_cm":  round(float(slope), 4) if slope is not None else None,
            "intercept_px":     round(float(intercept), 4) if intercept is not None else None,
            "r_squared":        r2,
            "n": len(xs),
        }
    return out


def chart_07(records):
    out = {}
    for m in _MODELS:
        model_recs = [r for r in records if r["model"] == m]
        out[m] = {}
        if m == "mediapipe":
            for side in ["L", "R"]:
                out[m][side] = {}
                for fi, name in enumerate(_FINGER_NAMES):
                    vals = _finger_vals(model_recs, side, fi, "mjmpe")
                    out[m][side][name] = _stats(vals)
        else:
            out[m]["combined"] = {}
            for fi, name in enumerate(_FINGER_NAMES):
                vals = _finger_vals_combined(model_recs, fi, "mjmpe")
                out[m]["combined"][name] = _stats(vals)
    return out


def chart_08(records):
    out = {}
    for m in _MODELS:
        model_recs = [r for r in records if r["model"] == m]
        pids = sorted({r["pid"] for r in model_recs})
        out[m] = {}
        for pid in pids:
            r = next((x for x in model_recs if x["pid"] == pid), None)
            if r is None:
                continue
            out[m][pid] = {}
            if m == "mediapipe":
                for side in ["L", "R"]:
                    out[m][pid][side] = {}
                    for fi, name in enumerate(_FINGER_NAMES):
                        fd = (r.get("per_hand", {}).get(side, {})
                               .get("fingers", {}).get(str(fi)))
                        out[m][pid][side][name] = (
                            round(fd["mjmpe"], 3) if fd and fd.get("mjmpe") is not None else None
                        )
            else:
                out[m][pid]["combined"] = {}
                for fi, name in enumerate(_FINGER_NAMES):
                    vals = []
                    for side in ["L", "R"]:
                        fd = (r.get("per_hand", {}).get(side, {})
                               .get("fingers", {}).get(str(fi)))
                        if fd and fd.get("mjmpe") is not None:
                            vals.append(fd["mjmpe"])
                    out[m][pid]["combined"][name] = (
                        round(float(np.mean(vals)), 3) if vals else None
                    )
    return out


def chart_09(records):
    by_fitz_model = defaultdict(list)
    for r in records:
        ft = r.get("fitzpatrick")
        if ft is None:
            continue
        total = ((r.get("notes_matched", 0) or 0) +
                 (r.get("notes_detection_fail", 0) or 0) +
                 (r.get("notes_missed", 0) or 0))
        df = r.get("notes_detection_fail", 0) or 0
        by_fitz_model[(ft, r["model"])].append(100 * df / total if total else 0)
    all_types = sorted({r.get("fitzpatrick") for r in records if r.get("fitzpatrick")})
    out = {}
    for ft in all_types:
        label = f"Type {_FITZ_LABELS.get(ft, str(ft))}"
        out[label] = {}
        for m in _MODELS:
            vals = by_fitz_model.get((ft, m), [])
            out[label][m] = {
                "mean_pct": round(float(np.mean(vals)), 3) if vals else None,
                "sd_pct":   round(float(np.std(vals)),  3) if len(vals) > 1 else None,
                "n": len(vals),
            }
    return out


def chart_10(records):
    by_lux_model = defaultdict(list)
    for r in records:
        total = ((r.get("notes_matched", 0) or 0) +
                 (r.get("notes_detection_fail", 0) or 0) +
                 (r.get("notes_missed", 0) or 0))
        df = r.get("notes_detection_fail", 0) or 0
        by_lux_model[(r["lux_label"], r["model"])].append(100 * df / total if total else 0)
    out = {}
    for lbl in _LUX_ORDER:
        out[lbl] = {}
        for m in _MODELS:
            vals = by_lux_model.get((lbl, m), [])
            out[lbl][m] = {
                "mean_pct": round(float(np.mean(vals)), 3) if vals else None,
                "sd_pct":   round(float(np.std(vals)),  3) if len(vals) > 1 else None,
                "n": len(vals),
            }
    return out


def chart_11(records):
    mp_recs = [r for r in records if r["model"] == "mediapipe"]
    by_side = defaultdict(list)
    for r in mp_recs:
        ph = r.get("per_hand", {})
        for side in ["L", "R"]:
            v = ph.get(side, {}).get("mjmpe_px")
            if v is not None:
                by_side[side].append(v)
    out = {}
    for side, label in [("L", "Left"), ("R", "Right")]:
        vals = by_side[side]
        out[label] = {
            "mean_px": round(float(np.mean(vals)), 3) if vals else None,
            "sd_px":   round(float(np.std(vals)),  3) if len(vals) > 1 else None,
            "n": len(vals),
        }
    return out


def _accuracy_by_finger(records):
    accurate   = [0] * 5
    inaccurate = [0] * 5
    for r in records:
        ph = r.get("per_hand", {})
        for side in ["L", "R"]:
            fingers = ph.get(side, {}).get("fingers", {})
            for fi in range(5):
                f = fingers.get(str(fi), {})
                count   = f.get("count", 0) or 0
                acc_pct = f.get("accuracy_pct", 0) or 0
                acc  = round(count * acc_pct / 100)
                inacc = count - acc
                accurate[fi]   += acc
                inaccurate[fi] += inacc
    out = {}
    for fi, name in enumerate(_FINGER_NAMES):
        total = accurate[fi] + inaccurate[fi]
        out[name] = {
            "within_threshold_count":  accurate[fi],
            "above_threshold_count":   inaccurate[fi],
            "total_matched":           total,
            "within_threshold_pct":    round(100 * accurate[fi] / total, 2) if total else None,
            "above_threshold_pct":     round(100 * inaccurate[fi] / total, 2) if total else None,
        }
    return out


def chart_12(records):
    return _accuracy_by_finger([r for r in records if r["model"] == "mediapipe"])


def chart_13(records):
    return _accuracy_by_finger([r for r in records if r["model"] == "openpose"])


def chart_14(records):
    mp_recs = [r for r in records if r["model"] == "mediapipe"]
    totals  = {"L": {"matched": 0, "detection_fail": 0},
               "R": {"matched": 0, "detection_fail": 0}}
    for r in mp_recs:
        ph = r.get("per_hand", {})
        for side in ["L", "R"]:
            totals[side]["matched"]        += ph.get(side, {}).get("matched", 0) or 0
            totals[side]["detection_fail"] += ph.get(side, {}).get("detection_fail", 0) or 0
    out = {}
    for side, label in [("L", "Left"), ("R", "Right")]:
        m     = totals[side]["matched"]
        d     = totals[side]["detection_fail"]
        denom = m + d
        out[label] = {
            "matched_count":        m,
            "detection_fail_count": d,
            "hit_rate_pct":         round(100.0 * m / denom, 2) if denom else None,
        }
    return out


def chart_15(records):
    mp_recs    = [r for r in records if r["model"] == "mediapipe"]
    mjmpe_vals = {"L": [], "R": []}
    for r in mp_recs:
        ph = r.get("per_hand", {})
        for side in ["L", "R"]:
            m = ph.get(side, {}).get("matched", 0) or 0
            v = ph.get(side, {}).get("mjmpe_px")
            if v is not None and m > 0:
                mjmpe_vals[side].append(v)
    out = {}
    for side, label in [("L", "Left"), ("R", "Right")]:
        vals = mjmpe_vals[side]
        out[label] = {
            "mjmpe_mean_px":           round(float(np.mean(vals)), 3) if vals else None,
            "mjmpe_sd_px":             round(float(np.std(vals, ddof=0)), 3) if len(vals) > 1 else None,
            "n_sessions_with_matches": len(vals),
        }
    return out


def chart_16(records):
    """MJMPE vs lux scatter: per-category stats + regression per model."""
    def _lux_cat(lux):
        if lux < 100:  return "Dim"
        if lux < 500:  return "Indoor"
        return "Bright"

    out = {}
    for model in _MODELS:
        recs = [r for r in records
                if r["model"] == model
                and r.get("mjmpe_px") is not None
                and r.get("lux") is not None]
        if not recs:
            continue

        xs = np.array([r["lux"]      for r in recs])
        ys = np.array([r["mjmpe_px"] for r in recs])

        if len(xs) >= 2:
            slope, intercept = np.polyfit(xs, ys, 1)
            corr = np.corrcoef(xs, ys)[0, 1]
            r2   = float(corr ** 2)
        else:
            slope = intercept = r2 = None

        by_cat = {}
        for cat in _LUX_ORDER:
            cat_ys = [r["mjmpe_px"] for r in recs if _lux_cat(r["lux"]) == cat]
            by_cat[cat] = {
                "n":       len(cat_ys),
                "mean_px": round(float(np.mean(cat_ys)), 3) if cat_ys else None,
                "sd_px":   round(float(np.std(cat_ys)),  3) if len(cat_ys) > 1 else None,
            }

        out[model] = {
            "n": len(recs),
            "lux_range": [float(xs.min()), float(xs.max())],
            "regression_slope":     round(float(slope),     4) if slope     is not None else None,
            "regression_intercept": round(float(intercept), 4) if intercept is not None else None,
            "r_squared":            round(r2, 4)                if r2        is not None else None,
            "by_category": by_cat,
        }
    return out


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", default=str(_PROCESSED))
    parser.add_argument("--out", default=str(_OUT_FILE))
    args = parser.parse_args()

    records = load_records(Path(args.processed))
    if not records:
        print("No result files found.")
        return

    pids   = sorted({r["pid"]   for r in records})
    models = sorted({r["model"] for r in records})
    print(f"Loaded {len(records)} record(s) — {len(pids)} participant(s), models: {models}")

    data = {
        "n_records":      len(records),
        "n_participants": len(pids),
        "pids":           pids,
        "models":         models,
        "chart_00_participant_composition":       chart_00(records),
        "chart_01_model_comparison_mjmpe":        chart_01(records),
        "chart_02_per_finger_mjmpe":              chart_02(records),
        "chart_03_detection_breakdown":           chart_03(records),
        "chart_04_mjmpe_by_lux":                  chart_04(records),
        "chart_05_mjmpe_by_fitzpatrick":          chart_05(records),
        "chart_06_mjmpe_vs_handsize_regression":  chart_06(records),
        "chart_07_per_finger_distribution":       chart_07(records),
        "chart_08_heatmap_per_participant":        chart_08(records),
        "chart_09_detection_fail_by_fitzpatrick": chart_09(records),
        "chart_10_detection_fail_by_lux":         chart_10(records),
        "chart_11_mjmpe_by_hand":                 chart_11(records),
        "chart_12_accuracy_by_finger_mediapipe":  chart_12(records),
        "chart_13_accuracy_by_finger_openpose":   chart_13(records),
        "chart_14_mediapipe_per_hand_hitrate":     chart_14(records),
        "chart_15_mediapipe_per_hand_mjmpe":       chart_15(records),
        "chart_16_mjmpe_vs_lux_scatter":           chart_16(records),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
