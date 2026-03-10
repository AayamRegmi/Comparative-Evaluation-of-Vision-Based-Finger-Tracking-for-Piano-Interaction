# plot_results.py
# Visualise all analyse.py JSON outputs with matplotlib.
#
# Usage:
#   python -m scripts.plot_results                      # all sessions
#   python -m scripts.plot_results --pid p001 p003      # filter participants
#   python -m scripts.plot_results --out figures/       # custom output folder
#
# Output PNGs are written to data/plots/ (or --out).
# Charts generated:
#   01_model_comparison_mjmpe.png   – MediaPipe vs OpenPose MJMPE per participant
#   02_per_finger_mjmpe.png         – per-finger MJMPE (L/R subplots, both models)
#   03_detection_breakdown.png      – matched / detection-fail / missed proportions
#   04_mjmpe_by_lux.png             – MJMPE by lighting condition (Dim/Indoor/Bright)
#   05_mjmpe_by_fitzpatrick.png     – MJMPE by Fitzpatrick skin type
#   06_mjmpe_vs_handsize.png        – scatter MJMPE vs hand size cm with regression
#   07_finger_distribution_<model>.png – box plots of per-finger MJMPE across sessions
#   08_heatmap_<model>.png          – per-finger MJMPE heatmap across participants

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

_ROOT       = Path(__file__).parent.parent
_PROCESSED  = _ROOT / "data" / "processed"
_PLOTS_DIR  = _ROOT / "data" / "plots"

_FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
_MODELS       = ["mediapipe", "openpose"]
_MODEL_COLORS = {"mediapipe": "#4CAF50", "openpose": "#2196F3"}
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(processed_dir=_PROCESSED, pids=None):
    """Return list of result dicts loaded from all *_results.json files."""
    records = []
    for path in sorted(processed_dir.glob("*_results.json")):
        try:
            with open(path) as f:
                r = json.load(f)
            if pids and r.get("pid") not in pids:
                continue
            r["_path"]     = str(path)
            r["lux_label"] = _lux_label(r.get("lux"))
            records.append(r)
        except Exception as e:
            print(f"  Warning: could not load {path.name}: {e}")
    return records


def _finger_vals(records, side, finger_idx, key="mjmpe"):
    """Collect per-session stat values for one finger/side across records."""
    vals = []
    for r in records:
        fdata = (r.get("per_hand", {})
                  .get(side, {})
                  .get("fingers", {})
                  .get(str(finger_idx)))
        if fdata and fdata.get(key) is not None:
            vals.append(fdata[key])
    return vals


def _finger_vals_combined(records, finger_idx, key="mjmpe"):
    """Collect per-session stat values for one finger pooling both hands (L+R)."""
    return (_finger_vals(records, "L", finger_idx, key) +
            _finger_vals(records, "R", finger_idx, key))


# ---------------------------------------------------------------------------
# Chart 1 — MediaPipe vs OpenPose MJMPE per participant
# ---------------------------------------------------------------------------

def plot_model_comparison(records, out_dir):
    by_pid = defaultdict(dict)
    for r in records:
        if r.get("mjmpe_px") is not None:
            by_pid[r["pid"]][r["model"]] = r["mjmpe_px"]

    pids = sorted(by_pid.keys())
    if not pids:
        print("  01: no data — skipping")
        return

    x     = np.arange(len(pids))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(pids) * 1.4), 5))

    for i, model in enumerate(_MODELS):
        vals  = [by_pid[p].get(model) for p in pids]
        valid = [v is not None for v in vals]
        plot  = [v if v is not None else 0 for v in vals]
        bars  = ax.bar(x + (i - 0.5) * width, plot, width,
                       label=model.capitalize(),
                       color=_MODEL_COLORS[model], alpha=0.85, edgecolor="white")
        for bar, v, ok in zip(bars, plot, valid):
            if ok and v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.2,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Participant")
    ax.set_ylabel("MJMPE (px)")
    ax.set_title("MediaPipe vs OpenPose — Overall MJMPE per Participant")
    ax.set_xticks(x)
    ax.set_xticklabels(pids, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "01_model_comparison_mjmpe.png")


# ---------------------------------------------------------------------------
# Chart 2 — Per-finger MJMPE
#   MediaPipe: L/R split (physical hands via handedness detection)
#   OpenPose:  combined L+R (hand split uses keyboard position only,
#              not physically reliable at scene level)
# ---------------------------------------------------------------------------

def plot_per_finger_mjmpe(records, out_dir):
    x     = np.arange(len(_FINGER_NAMES))
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- MediaPipe: Left Hand ---
    mp_recs = [r for r in records if r["model"] == "mediapipe"]
    for ax, side, title in zip(axes[:2], ["L", "R"],
                                ["MediaPipe — Left Hand", "MediaPipe — Right Hand"]):
        means = [np.mean(_finger_vals(mp_recs, side, fi)) if _finger_vals(mp_recs, side, fi) else 0
                 for fi in range(5)]
        bars = ax.bar(x, means, 0.5, color=_MODEL_COLORS["mediapipe"],
                      alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, means):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.1,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("Finger")
        if side == "L":
            ax.set_ylabel("Mean MJMPE (px) across participants")
        ax.set_xticks(x)
        ax.set_xticklabels(_FINGER_NAMES)
        ax.grid(axis="y", alpha=0.3)

    # --- OpenPose: combined L+R ---
    ax_op  = axes[2]
    op_recs = [r for r in records if r["model"] == "openpose"]
    if op_recs:
        means = [np.mean(_finger_vals_combined(op_recs, fi)) if _finger_vals_combined(op_recs, fi) else 0
                 for fi in range(5)]
        bars = ax_op.bar(x, means, 0.5, color=_MODEL_COLORS["openpose"],
                         alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, means):
            if v > 0:
                ax_op.text(bar.get_x() + bar.get_width() / 2, v + 0.1,
                           f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    else:
        ax_op.text(0.5, 0.5, "No OpenPose data", transform=ax_op.transAxes,
                   ha="center", va="center", color="grey")
    ax_op.set_title("OpenPose — Both Hands Combined\n"
                    "(hand split uses keyboard position, not physical handedness)")
    ax_op.set_xlabel("Finger")
    ax_op.set_xticks(x)
    ax_op.set_xticklabels(_FINGER_NAMES)
    ax_op.grid(axis="y", alpha=0.3)

    fig.suptitle("Per-Finger MJMPE by Model")
    fig.tight_layout()
    _save(fig, out_dir, "02_per_finger_mjmpe.png")


# ---------------------------------------------------------------------------
# Chart 3 — Detection outcome breakdown (matched / fail / missed)
# ---------------------------------------------------------------------------

def plot_detection_breakdown(records, out_dir):
    by_model = defaultdict(lambda: {"matched": 0, "detection_fail": 0, "missed": 0})
    for r in records:
        m = r["model"]
        by_model[m]["matched"]        += r.get("notes_matched", 0) or 0
        by_model[m]["detection_fail"] += r.get("notes_detection_fail", 0) or 0
        by_model[m]["missed"]         += r.get("notes_missed", 0) or 0

    models = [m for m in _MODELS if m in by_model]
    if not models:
        print("  03: no data — skipping")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(len(models))

    matched = [by_model[m]["matched"]        for m in models]
    fails   = [by_model[m]["detection_fail"] for m in models]
    missed  = [by_model[m]["missed"]         for m in models]
    totals  = [a + b + c for a, b, c in zip(matched, fails, missed)]

    pct_m = [100 * a / t if t else 0 for a, t in zip(matched, totals)]
    pct_f = [100 * b / t if t else 0 for b, t in zip(fails,   totals)]
    pct_x = [100 * c / t if t else 0 for c, t in zip(missed,  totals)]

    ax.bar(x, pct_m, label="Matched",          color="#4CAF50", alpha=0.85)
    ax.bar(x, pct_f, bottom=pct_m,             label="Detection fail", color="#FF9800", alpha=0.85)
    ax.bar(x, pct_x, bottom=[a+b for a,b in zip(pct_m, pct_f)],
           label="Missed (no hands)", color="#F44336", alpha=0.85)

    ax.set_ylabel("% of note events")
    ax.set_title("Detection Outcome Breakdown by Model")
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in models])
    ax.legend(loc="lower right")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    _save(fig, out_dir, "03_detection_breakdown.png")


# ---------------------------------------------------------------------------
# Chart 4 — MJMPE by lighting condition
# ---------------------------------------------------------------------------

def plot_by_lux(records, out_dir):
    by_lux_model = defaultdict(list)
    for r in records:
        if r.get("mjmpe_px") is not None:
            by_lux_model[(r["lux_label"], r["model"])].append(r["mjmpe_px"])

    fig, ax = plt.subplots(figsize=(9, 5))
    x     = np.arange(len(_LUX_ORDER))
    width = 0.35

    for i, model in enumerate(_MODELS):
        means, errs = [], []
        for lbl in _LUX_ORDER:
            vals = by_lux_model.get((lbl, model), [])
            means.append(np.mean(vals) if vals else 0)
            errs.append(np.std(vals)   if len(vals) > 1 else 0)
        ax.bar(x + (i - 0.5) * width, means, width, yerr=errs, capsize=4,
               label=model.capitalize(),
               color=_MODEL_COLORS[model], alpha=0.85, edgecolor="white")

    ax.set_xlabel("Lighting condition")
    ax.set_ylabel("Mean MJMPE (px)  ±SD")
    ax.set_title("MJMPE by Lighting Condition")
    ax.set_xticks(x)
    ax.set_xticklabels(_LUX_ORDER)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "04_mjmpe_by_lux.png")


# ---------------------------------------------------------------------------
# Chart 5 — MJMPE by Fitzpatrick type
# ---------------------------------------------------------------------------

def plot_by_fitzpatrick(records, out_dir):
    by_fitz_model = defaultdict(list)
    for r in records:
        if r.get("mjmpe_px") is not None and r.get("fitzpatrick"):
            by_fitz_model[(r["fitzpatrick"], r["model"])].append(r["mjmpe_px"])

    all_types = sorted({r.get("fitzpatrick") for r in records if r.get("fitzpatrick")})
    if not all_types:
        print("  05: no Fitzpatrick data — skipping")
        return

    fig, ax = plt.subplots(figsize=(max(7, len(all_types) * 1.6), 5))
    x     = np.arange(len(all_types))
    width = 0.35

    for i, model in enumerate(_MODELS):
        means, errs = [], []
        for ft in all_types:
            vals = by_fitz_model.get((ft, model), [])
            means.append(np.mean(vals) if vals else 0)
            errs.append(np.std(vals)   if len(vals) > 1 else 0)
        ax.bar(x + (i - 0.5) * width, means, width, yerr=errs, capsize=4,
               label=model.capitalize(),
               color=_MODEL_COLORS[model], alpha=0.85, edgecolor="white")

    ax.set_xlabel("Fitzpatrick Type")
    ax.set_ylabel("Mean MJMPE (px)  ±SD")
    ax.set_title("MJMPE by Fitzpatrick Skin Type")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Type {t}\n({_FITZ_LABELS.get(t, '')})" for t in all_types])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "05_mjmpe_by_fitzpatrick.png")


# ---------------------------------------------------------------------------
# Chart 6 — Scatter: MJMPE vs hand size with regression line
# ---------------------------------------------------------------------------

def plot_mjmpe_vs_handsize(records, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))

    for model in _MODELS:
        recs = [r for r in records
                if r["model"] == model
                and r.get("mjmpe_px") is not None
                and r.get("hand_size_cm") is not None]
        if not recs:
            continue
        xs = [r["hand_size_cm"] for r in recs]
        ys = [r["mjmpe_px"]     for r in recs]
        ax.scatter(xs, ys, label=model.capitalize(),
                   color=_MODEL_COLORS[model], alpha=0.8, s=70, zorder=3)
        if len(xs) >= 2:
            z   = np.polyfit(xs, ys, 1)
            xp  = np.linspace(min(xs), max(xs), 50)
            ax.plot(xp, np.polyval(z, xp),
                    color=_MODEL_COLORS[model], linestyle="--", alpha=0.6)

    ax.set_xlabel("Hand size (cm)")
    ax.set_ylabel("MJMPE (px)")
    ax.set_title("MJMPE vs Hand Size")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "06_mjmpe_vs_handsize.png")


# ---------------------------------------------------------------------------
# Chart 7 — Box plot: per-finger MJMPE distribution across sessions
# ---------------------------------------------------------------------------

def plot_finger_distribution(records, out_dir):
    """
    One figure per model — box plots where each box covers sessions.
    MediaPipe: two subplots (L / R physical hands).
    OpenPose:  single subplot, L+R combined (hand split is keyboard-position only).
    """
    for model in _MODELS:
        model_recs = [r for r in records if r["model"] == model]
        if not model_recs:
            continue

        if model == "mediapipe":
            fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
            for ax, side, title in zip(axes, ["L", "R"], ["Left Hand", "Right Hand"]):
                data = [_finger_vals(model_recs, side, fi, "mjmpe") or [0]
                        for fi in range(5)]
                bp = ax.boxplot(data, labels=_FINGER_NAMES, patch_artist=True,
                                medianprops={"color": "white", "linewidth": 2})
                for patch in bp["boxes"]:
                    patch.set_facecolor(_MODEL_COLORS[model])
                    patch.set_alpha(0.65)
                ax.set_title(title)
                ax.set_xlabel("Finger")
                if side == "L":
                    ax.set_ylabel("MJMPE (px) — mean per session")
                ax.grid(axis="y", alpha=0.3)
            fig.suptitle("MediaPipe — Per-Finger MJMPE Distribution (across sessions)")
        else:
            # OpenPose: combine L+R — hand split not physically reliable at scene level
            fig, ax = plt.subplots(figsize=(7, 5))
            data = [_finger_vals_combined(model_recs, fi, "mjmpe") or [0]
                    for fi in range(5)]
            bp = ax.boxplot(data, labels=_FINGER_NAMES, patch_artist=True,
                            medianprops={"color": "white", "linewidth": 2})
            for patch in bp["boxes"]:
                patch.set_facecolor(_MODEL_COLORS[model])
                patch.set_alpha(0.65)
            ax.set_xlabel("Finger")
            ax.set_ylabel("MJMPE (px) — mean per session")
            ax.set_title("Both hands combined\n"
                         "(hand split uses keyboard position, not physical handedness)")
            ax.grid(axis="y", alpha=0.3)
            fig.suptitle("OpenPose — Per-Finger MJMPE Distribution (across sessions)")

        fig.tight_layout()
        _save(fig, out_dir, f"07_finger_distribution_{model}.png")


# ---------------------------------------------------------------------------
# Chart 8 — Heatmap: per-finger MJMPE across participants
# ---------------------------------------------------------------------------

def plot_finger_heatmap(records, out_dir):
    """
    MediaPipe: 10-column heatmap (L-Thumb … L-Pinky | R-Thumb … R-Pinky).
    OpenPose:   5-column heatmap, L+R combined per finger
                (hand split uses keyboard position, not physical handedness).
    """
    for model in _MODELS:
        model_recs = [r for r in records if r["model"] == model]
        pids = sorted({r["pid"] for r in model_recs})
        if not pids:
            continue

        if model == "mediapipe":
            col_labels = ([f"L-{n[:3]}" for n in _FINGER_NAMES] +
                          [f"R-{n[:3]}" for n in _FINGER_NAMES])
            matrix = np.full((len(pids), 10), np.nan)
            for row_i, pid in enumerate(pids):
                r = next((x for x in model_recs if x["pid"] == pid), None)
                if r is None:
                    continue
                for fi in range(5):
                    for col_off, side in enumerate(["L", "R"]):
                        fdata = (r.get("per_hand", {}).get(side, {})
                                  .get("fingers", {}).get(str(fi)))
                        if fdata and fdata.get("mjmpe") is not None:
                            matrix[row_i, col_off * 5 + fi] = fdata["mjmpe"]
            draw_separator = True
            fig_w = 13
            title_suffix = ""
        else:
            col_labels = [n[:5] for n in _FINGER_NAMES]
            matrix = np.full((len(pids), 5), np.nan)
            for row_i, pid in enumerate(pids):
                r = next((x for x in model_recs if x["pid"] == pid), None)
                if r is None:
                    continue
                for fi in range(5):
                    vals = []
                    for side in ["L", "R"]:
                        fdata = (r.get("per_hand", {}).get(side, {})
                                  .get("fingers", {}).get(str(fi)))
                        if fdata and fdata.get("mjmpe") is not None:
                            vals.append(fdata["mjmpe"])
                    if vals:
                        matrix[row_i, fi] = float(np.mean(vals))
            draw_separator = False
            fig_w = 7
            title_suffix = "\n(both hands combined — hand split not physically reliable)"

        vmax  = float(np.nanmax(matrix)) if not np.all(np.isnan(matrix)) else 20
        fig_h = max(4, len(pids) * 0.7 + 1.5)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right")
        ax.set_yticks(range(len(pids)))
        ax.set_yticklabels(pids)

        for row_i in range(len(pids)):
            for col_j in range(len(col_labels)):
                v = matrix[row_i, col_j]
                if not np.isnan(v):
                    txt_col = "black" if v < vmax * 0.65 else "white"
                    ax.text(col_j, row_i, f"{v:.1f}",
                            ha="center", va="center", fontsize=8, color=txt_col)

        if draw_separator:
            ax.axvline(4.5, color="white", linewidth=2.5)

        plt.colorbar(im, ax=ax, label="MJMPE (px)")
        ax.set_title(f"{model.capitalize()} — Per-Finger MJMPE Heatmap (px){title_suffix}")
        fig.tight_layout()
        _save(fig, out_dir, f"08_heatmap_{model}.png")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig, out_dir, filename):
    path = out_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filename}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot MJMPE analysis results")
    parser.add_argument("--processed", default=str(_PROCESSED),
                        help="Directory with *_results.json files")
    parser.add_argument("--out", default=str(_PLOTS_DIR),
                        help="Output directory for plots")
    parser.add_argument("--pid", nargs="+", default=None,
                        help="Filter to specific PIDs e.g. p001 p002")
    args = parser.parse_args()

    processed_dir = Path(args.processed)
    out_dir       = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_results(processed_dir, pids=args.pid)
    if not records:
        print("No result files found in", processed_dir)
        print("Run analyse.py on your sessions first.")
        return

    pids   = sorted({r["pid"]   for r in records})
    models = sorted({r["model"] for r in records})
    print(f"Loaded {len(records)} result(s) — {len(pids)} participant(s), models: {models}")
    print(f"Output → {out_dir}\n")

    plot_model_comparison(records, out_dir)
    plot_per_finger_mjmpe(records, out_dir)
    plot_detection_breakdown(records, out_dir)
    plot_by_lux(records, out_dir)
    plot_by_fitzpatrick(records, out_dir)
    plot_mjmpe_vs_handsize(records, out_dir)
    plot_finger_distribution(records, out_dir)
    plot_finger_heatmap(records, out_dir)

    print(f"\nDone — {len(list(out_dir.glob('*.png')))} chart(s) in {out_dir}")


if __name__ == "__main__":
    main()
