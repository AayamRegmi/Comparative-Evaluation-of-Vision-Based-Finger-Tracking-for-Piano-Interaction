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
#   00_participant_composition.png  – Fitzpatrick, lighting condition, hand size distributions
#   01_model_comparison_mpjpe.png   – MediaPipe vs OpenPose MPJPE per participant
#   02_per_finger_mpjpe.png         – per-finger MPJPE (L/R subplots, both models)
#   03_detection_breakdown.png      – matched / detection-fail / missed proportions
#   04_mpjpe_by_lux.png             – MPJPE by lighting condition (Dim/Indoor/Bright)
#   05_mpjpe_by_fitzpatrick.png     – MPJPE by Fitzpatrick skin type
#   06_mpjpe_vs_handsize.png        – scatter MPJPE vs hand size cm with regression
#   07_finger_distribution_<model>.png – box plots of per-finger MPJPE across sessions
#   08_heatmap_<model>.png          – per-finger MPJPE heatmap across participants
#   09_detection_fail_by_fitzpatrick.png – detection-fail rate by Fitzpatrick skin type
#   10_detection_fail_by_lux.png    – detection-fail rate by lighting condition
#   11_mpjpe_by_hand.png            – overall MPJPE by hand (Left vs Right), MediaPipe only
#   12_accuracy_by_finger.png       – per-finger accuracy breakdown (within/above threshold), MediaPipe only
#   13_accuracy_by_finger_openpose.png – same breakdown for OpenPose

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
# Chart 0 — Participant composition (Fitzpatrick, lighting, hand size)
# ---------------------------------------------------------------------------

def plot_participant_composition(records, out_dir):
    """
    Three-panel summary of participant demographics:
      Left  — Fitzpatrick skin type counts
      Middle — Lighting condition counts (Dim / Indoor / Bright)
      Right  — Hand size distribution (cm histogram)
    One record per participant (model duplicates removed).
    """
    seen = set()
    participants = []
    for r in records:
        if r["pid"] not in seen:
            seen.add(r["pid"])
            participants.append(r)

    if not participants:
        print("  00: no data — skipping")
        return

    # ── Fitzpatrick counts ───────────────────────────────────────────────────
    fitz_counts = {}
    for r in participants:
        ft = r.get("fitzpatrick")
        if ft is not None:
            label = f"Type {_FITZ_LABELS.get(ft, str(ft))}"
            fitz_counts[label] = fitz_counts.get(label, 0) + 1
    fitz_labels  = sorted(fitz_counts.keys())
    fitz_vals    = [fitz_counts[l] for l in fitz_labels]

    # ── Lighting counts ──────────────────────────────────────────────────────
    lux_counts = {l: 0 for l in _LUX_ORDER}
    for r in participants:
        lux_counts[r["lux_label"]] = lux_counts.get(r["lux_label"], 0) + 1
    lux_labels = [l for l in _LUX_ORDER if lux_counts[l] > 0]
    lux_vals   = [lux_counts[l] for l in lux_labels]

    # ── Hand sizes ───────────────────────────────────────────────────────────
    hand_sizes = [r["hand_size_cm"] for r in participants if r.get("hand_size_cm") is not None]

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(f"Participant Composition  (n={len(participants)})",
                 fontsize=14, fontweight="bold", y=1.02)

    BAR_CLR  = "#5C6BC0"
    EDGE_CLR = "white"

    # Left — Fitzpatrick
    ax = axes[0]
    xf = np.arange(len(fitz_labels))
    bars = ax.bar(xf, fitz_vals, color=BAR_CLR, edgecolor=EDGE_CLR, alpha=0.88)
    for bar, v in zip(bars, fitz_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.15, str(v),
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xticks(xf)
    ax.set_xticklabels(fitz_labels, fontsize=10)
    ax.set_ylabel("Number of participants")
    ax.set_title("Fitzpatrick Skin Type")
    ax.set_ylim(0, max(fitz_vals) + 2)
    ax.grid(axis="y", alpha=0.3)

    # Middle — Lighting
    ax = axes[1]
    xl = np.arange(len(lux_labels))
    LUX_COLORS = {"Dim": "#78909C", "Indoor": "#FFA726", "Bright": "#FFEE58"}
    colors = [LUX_COLORS.get(l, BAR_CLR) for l in lux_labels]
    bars = ax.bar(xl, lux_vals, color=colors, edgecolor=EDGE_CLR, alpha=0.88)
    for bar, v in zip(bars, lux_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.15, str(v),
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xticks(xl)
    ax.set_xticklabels(lux_labels, fontsize=10)
    ax.set_ylabel("Number of participants")
    ax.set_title("Lighting Condition")
    ax.set_ylim(0, max(lux_vals) + 2)
    ax.grid(axis="y", alpha=0.3)

    # Right — Hand size histogram
    ax = axes[2]
    ax.hist(hand_sizes, bins=8, color=BAR_CLR, edgecolor=EDGE_CLR, alpha=0.88)
    ax.set_xlabel("Hand size (cm)")
    ax.set_ylabel("Number of participants")
    ax.set_title("Hand Size Distribution")
    ax.grid(axis="y", alpha=0.3)
    mean_hs = np.mean(hand_sizes)
    ax.axvline(mean_hs, color="#E53935", linewidth=1.5, linestyle="--",
               label=f"Mean: {mean_hs:.1f} cm")
    ax.legend(fontsize=9)

    fig.tight_layout()
    _save(fig, out_dir, "00_participant_composition.png")


# ---------------------------------------------------------------------------
# Chart 1 — MediaPipe vs OpenPose MPJPE per participant
# ---------------------------------------------------------------------------

def plot_model_comparison(records, out_dir):
    by_model = defaultdict(list)
    for r in records:
        if r.get("mjmpe_px") is not None:
            by_model[r["model"]].append(r["mjmpe_px"])

    models = [m for m in _MODELS if m in by_model]
    if not models:
        print("  01: no data — skipping")
        return

    means = [np.mean(by_model[m]) for m in models]
    sds   = [np.std(by_model[m])  for m in models]
    ns    = [len(by_model[m])     for m in models]

    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(len(models))

    bars = ax.bar(x, means, 0.45, yerr=sds, capsize=6,
                  color=[_MODEL_COLORS[m] for m in models],
                  alpha=0.85, edgecolor="white",
                  error_kw={"linewidth": 1.5, "ecolor": "#444"})

    for bar, mean, sd, n in zip(bars, means, sds, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + sd + 0.15,
                f"{mean:.2f} px\n(n={n})", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Mean MPJPE (px)  ±SD")
    ax.set_title("Overall MPJPE by Model")
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in models], fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(means) + max(sds) + 1.5)
    fig.tight_layout()
    _save(fig, out_dir, "01_model_comparison_mpjpe.png")


# ---------------------------------------------------------------------------
# Chart 2 — Per-finger MPJPE
#   MediaPipe: L/R split (physical hands via handedness detection)
#   OpenPose:  combined L+R (hand split uses keyboard position only,
#              not physically reliable at scene level)
# ---------------------------------------------------------------------------

def plot_per_finger_mpjpe(records, out_dir):
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
            ax.set_ylabel("Mean MPJPE (px) across participants")
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

    fig.suptitle("Per-Finger MPJPE by Model")
    fig.tight_layout()
    _save(fig, out_dir, "02_per_finger_mpjpe.png")

    # MediaPipe-only variant (2 panels: L and R)
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    for ax, side, title in zip(axes2, ["L", "R"],
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
            ax.set_ylabel("Mean MPJPE (px) across participants")
        ax.set_xticks(x)
        ax.set_xticklabels(_FINGER_NAMES)
        ax.grid(axis="y", alpha=0.3)
    fig2.suptitle("Per-Finger MPJPE — MediaPipe")
    fig2.tight_layout()
    _save(fig2, out_dir, "02_per_finger_mpjpe_mediapipe.png")


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
# Chart 4 — MPJPE by lighting condition
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
    ax.set_ylabel("Mean MPJPE (px)  ±SD")
    ax.set_title("MPJPE by Lighting Condition")
    ax.set_xticks(x)
    ax.set_xticklabels(_LUX_ORDER)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "04_mpjpe_by_lux.png")


# ---------------------------------------------------------------------------
# Chart 5 — MPJPE by Fitzpatrick type
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
    ax.set_ylabel("Mean MPJPE (px)  ±SD")
    ax.set_title("MPJPE by Fitzpatrick Skin Type")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Type {t}\n({_FITZ_LABELS.get(t, '')})" for t in all_types])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "05_mpjpe_by_fitzpatrick.png")


# ---------------------------------------------------------------------------
# Chart 6 — Scatter: MPJPE vs hand size with regression line
# ---------------------------------------------------------------------------

def plot_mpjpe_vs_handsize(records, out_dir):
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
    ax.set_ylabel("MPJPE (px)")
    ax.set_title("MPJPE vs Hand Size")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "06_mpjpe_vs_handsize.png")


# ---------------------------------------------------------------------------
# Chart 7 — Box plot: per-finger MPJPE distribution across sessions
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
                    ax.set_ylabel("MPJPE (px) — mean per session")
                ax.grid(axis="y", alpha=0.3)
            fig.suptitle("MediaPipe — Per-Finger MPJPE Distribution (across sessions)")
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
            ax.set_ylabel("MPJPE (px) — mean per session")
            ax.set_title("Both hands combined\n"
                         "(hand split uses keyboard position, not physical handedness)")
            ax.grid(axis="y", alpha=0.3)
            fig.suptitle("OpenPose — Per-Finger MPJPE Distribution (across sessions)")

        fig.tight_layout()
        _save(fig, out_dir, f"07_finger_distribution_{model}.png")


# ---------------------------------------------------------------------------
# Chart 8 — Heatmap: per-finger MPJPE across participants
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

        plt.colorbar(im, ax=ax, label="MPJPE (px)")
        ax.set_title(f"{model.capitalize()} — Per-Finger MPJPE Heatmap (px){title_suffix}")
        fig.tight_layout()
        _save(fig, out_dir, f"08_heatmap_{model}.png")


# ---------------------------------------------------------------------------
# Chart 9 — Individual participant report
# ---------------------------------------------------------------------------

def plot_participant_report(pid, records, out_dir):
    """
    One-page performance report for a single participant.
    Saved to out_dir/reports/{pid}_report.png

    Layout (2 rows × 3 cols):
      [session info | detection bar | overall stats]
      [L-hand fingers | R-hand fingers | highlights]
    """
    mp_rec = next((r for r in records if r["pid"] == pid and r["model"] == "mediapipe"), None)
    op_rec = next((r for r in records if r["pid"] == pid and r["model"] == "openpose"), None)

    if mp_rec is None:
        print(f"  {pid} report: no MediaPipe data — skipping")
        return

    reports_dir = out_dir / "reports"
    reports_dir.mkdir(exist_ok=True)

    # ── colours ──────────────────────────────────────────────────────────
    BG        = "#f7f9fc"
    PANEL_BG  = "#ffffff"
    BORDER    = "#dde3ed"
    TXT       = "#1a1a2e"
    MUTED     = "#6b7280"
    C_GREEN   = "#22c55e"
    C_YELLOW  = "#f59e0b"
    C_RED     = "#ef4444"
    C_BLUE    = "#3b82f6"
    C_GREY    = "#cbd5e1"
    C_MP      = _MODEL_COLORS["mediapipe"]
    C_OP      = _MODEL_COLORS["openpose"]

    fig = plt.figure(figsize=(15, 9))
    fig.patch.set_facecolor(BG)

    gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.35,
                          left=0.05, right=0.97, top=0.88, bottom=0.09)

    ax_info    = fig.add_subplot(gs[0, 0])
    ax_detect  = fig.add_subplot(gs[0, 1])
    ax_overall = fig.add_subplot(gs[0, 2])
    ax_left    = fig.add_subplot(gs[1, 0])
    ax_right   = fig.add_subplot(gs[1, 1])
    ax_hi      = fig.add_subplot(gs[1, 2])

    def _style_ax(ax, keep_axes=True):
        ax.set_facecolor(PANEL_BG)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        if not keep_axes:
            ax.axis("off")

    # ── Panel 1 : session metadata ────────────────────────────────────────
    _style_ax(ax_info, keep_axes=False)

    fitz      = mp_rec.get("fitzpatrick")
    lux       = mp_rec.get("lux")
    hand_size = mp_rec.get("hand_size_cm")
    lux_lbl   = mp_rec.get("lux_label", "Unknown")
    fitz_lbl  = (f"Type {fitz}  ({_FITZ_LABELS.get(fitz, '?')})"
                 if fitz else "Not recorded")

    info_rows = [
        ("Participant",  pid.upper()),
        ("Skin Type",    fitz_lbl),
        ("Lighting",     f"{lux_lbl}  ({lux} lux)" if lux is not None else lux_lbl),
        ("Hand Size",    f"{hand_size} cm" if hand_size else "Not recorded"),
        ("Notes Played", str(mp_rec.get("notes_total", "?"))),
    ]

    ax_info.text(0.06, 0.97, "Session Info",
                 transform=ax_info.transAxes,
                 fontsize=11, fontweight="bold", color=TXT, va="top")
    for i, (lbl, val) in enumerate(info_rows):
        y = 0.83 - i * 0.17
        ax_info.text(0.06, y, lbl,
                     transform=ax_info.transAxes,
                     fontsize=8, color=MUTED, va="top")
        ax_info.text(0.06, y - 0.07, val,
                     transform=ax_info.transAxes,
                     fontsize=10, fontweight="semibold", color=TXT, va="top")

    # ── Panel 2 : detection outcomes ──────────────────────────────────────
    _style_ax(ax_detect)

    matched  = mp_rec.get("notes_matched", 0) or 0
    det_fail = mp_rec.get("notes_detection_fail", 0) or 0
    missed   = mp_rec.get("notes_missed", 0) or 0
    total    = matched + det_fail + missed

    if total > 0:
        cats   = ["Matched", "Det. Fail", "Missed"]
        pcts   = [100 * matched / total, 100 * det_fail / total, 100 * missed / total]
        raws   = [matched, det_fail, missed]
        colors = [C_GREEN, C_YELLOW, C_RED]
        bars   = ax_detect.bar(cats, pcts, color=colors, alpha=0.85,
                               edgecolor=BORDER, width=0.5)
        for bar, pct, raw in zip(bars, pcts, raws):
            ax_detect.text(bar.get_x() + bar.get_width() / 2, pct + 1.5,
                           f"{pct:.0f}%\n({raw})",
                           ha="center", va="bottom", fontsize=8, color=TXT)
        ax_detect.set_ylim(0, 115)
    else:
        ax_detect.text(0.5, 0.5, "No data", transform=ax_detect.transAxes,
                       ha="center", va="center", color=MUTED)

    ax_detect.set_title("Detection Outcomes (MediaPipe)", fontsize=9, color=TXT, pad=6)
    ax_detect.set_ylabel("% of note events", fontsize=8, color=MUTED)
    ax_detect.tick_params(colors=TXT, labelsize=8)
    ax_detect.grid(axis="y", alpha=0.25, color=BORDER)

    # ── Panel 3 : overall stats text ──────────────────────────────────────
    _style_ax(ax_overall, keep_axes=False)

    mp_mpjpe = mp_rec.get("mjmpe_px")
    mp_acc   = mp_rec.get("accuracy_pct")
    mp_dr    = mp_rec.get("detection_rate_pct")
    op_mpjpe = op_rec.get("mjmpe_px")          if op_rec else None
    op_dr    = op_rec.get("detection_rate_pct") if op_rec else None

    stat_rows = [
        ("MediaPipe MPJPE",      f"{mp_mpjpe:.2f} px" if mp_mpjpe else "—",  C_MP),
        ("MediaPipe Accuracy",   f"{mp_acc:.1f}%"     if mp_acc   else "—",  C_MP),
        ("MediaPipe Det. Rate",  f"{mp_dr:.1f}%"      if mp_dr    else "—",  C_MP),
        ("OpenPose MPJPE",       f"{op_mpjpe:.2f} px" if op_mpjpe else "—",  C_OP),
        ("OpenPose Det. Rate",   f"{op_dr:.1f}%"      if op_dr    else "—",  C_OP),
    ]

    ax_overall.text(0.06, 0.97, "Overall Results",
                    transform=ax_overall.transAxes,
                    fontsize=11, fontweight="bold", color=TXT, va="top")
    for i, (lbl, val, col) in enumerate(stat_rows):
        y = 0.83 - i * 0.17
        ax_overall.text(0.06, y, lbl,
                        transform=ax_overall.transAxes,
                        fontsize=8, color=MUTED, va="top")
        ax_overall.text(0.06, y - 0.07, val,
                        transform=ax_overall.transAxes,
                        fontsize=10, fontweight="semibold", color=col, va="top")

    # ── Panels 4 & 5 : per-finger MPJPE L and R ───────────────────────────
    finger_short = ["Thumb", "Index", "Mid", "Ring", "Pinky"]
    xf = np.arange(5)

    for ax, side, title in [(ax_left, "L", "Left Hand"),
                             (ax_right, "R", "Right Hand")]:
        _style_ax(ax)
        ph      = mp_rec.get("per_hand", {}).get(side, {})
        fingers = ph.get("fingers", {})

        mpjpe_vals = []
        counts     = []
        for fi in range(5):
            fd = fingers.get(str(fi))
            mpjpe_vals.append(fd["mjmpe"] if fd and fd.get("mjmpe") is not None else None)
            counts.append(fd.get("count", 0) if fd else 0)

        bar_vals   = [v if v is not None else 0 for v in mpjpe_vals]
        bar_colors = []
        for v in mpjpe_vals:
            if v is None:
                bar_colors.append(C_GREY)
            elif v <= 4:
                bar_colors.append(C_GREEN)
            elif v <= 7:
                bar_colors.append(C_YELLOW)
            else:
                bar_colors.append(C_RED)

        bars = ax.bar(xf, bar_vals, color=bar_colors, alpha=0.88,
                      edgecolor=BORDER, width=0.6)
        for bar, v in zip(bars, mpjpe_vals):
            if v is not None:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.15,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=8, color=TXT)
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, 0.3,
                        "N/A", ha="center", va="bottom", fontsize=7, color=MUTED)

        hand_mpjpe   = ph.get("mjmpe_px")
        hand_matched = ph.get("matched", 0)
        subtitle = f"({hand_matched} notes"
        if hand_mpjpe:
            subtitle += f",  avg {hand_mpjpe:.1f} px)"
        else:
            subtitle += ")"

        ax.set_title(f"{title}  {subtitle}", fontsize=9, color=TXT, pad=5)
        ax.set_ylabel("MPJPE (px)", fontsize=8, color=MUTED)
        ax.set_xticks(xf)
        ax.set_xticklabels(finger_short, fontsize=8)
        ax.tick_params(colors=TXT, labelsize=8)
        ax.grid(axis="y", alpha=0.25, color=BORDER)
        max_y = max(max(bar_vals) * 1.3, 5)
        ax.set_ylim(0, max_y)

    # ── Panel 6 : highlights ──────────────────────────────────────────────
    _style_ax(ax_hi, keep_axes=False)

    # Collect reliable finger values (count >= 5)
    finger_scores = {}
    for side in ["L", "R"]:
        for fi in range(5):
            fd = (mp_rec.get("per_hand", {}).get(side, {})
                  .get("fingers", {}).get(str(fi)))
            if fd and fd.get("mjmpe") is not None and fd.get("count", 0) >= 5:
                finger_scores[f"{'L' if side == 'L' else 'R'}-{_FINGER_NAMES[fi]}"] = fd["mjmpe"]

    highlights = []
    if finger_scores:
        best_k  = min(finger_scores, key=finger_scores.get)
        worst_k = max(finger_scores, key=finger_scores.get)
        highlights.append(("Most Accurate Finger",
                            f"{best_k}  ({finger_scores[best_k]:.2f} px)", C_GREEN))
        highlights.append(("Least Accurate Finger",
                            f"{worst_k}  ({finger_scores[worst_k]:.2f} px)", C_RED))

    if mp_dr is not None:
        if mp_dr >= 90:
            dr_txt, dr_col = f"Excellent — {mp_dr:.1f}%", C_GREEN
        elif mp_dr >= 70:
            dr_txt, dr_col = f"Good — {mp_dr:.1f}%",      C_YELLOW
        else:
            dr_txt, dr_col = f"Low — {mp_dr:.1f}%",       C_RED
        highlights.append(("Detection Rate", dr_txt, dr_col))

    if mp_mpjpe is not None:
        if mp_mpjpe <= 4:
            acc_txt, acc_col = f"Great — {mp_mpjpe:.2f} px", C_GREEN
        elif mp_mpjpe <= 7:
            acc_txt, acc_col = f"Good — {mp_mpjpe:.2f} px",  C_YELLOW
        else:
            acc_txt, acc_col = f"Needs work — {mp_mpjpe:.2f} px", C_RED
        highlights.append(("Overall Accuracy", acc_txt, acc_col))

    ax_hi.text(0.06, 0.97, "Highlights",
               transform=ax_hi.transAxes,
               fontsize=11, fontweight="bold", color=TXT, va="top")
    for i, (lbl, val, col) in enumerate(highlights):
        y = 0.83 - i * 0.22
        ax_hi.text(0.06, y, lbl,
                   transform=ax_hi.transAxes,
                   fontsize=8, color=MUTED, va="top")
        ax_hi.text(0.06, y - 0.08, val,
                   transform=ax_hi.transAxes,
                   fontsize=10, fontweight="semibold", color=col, va="top")

    # ── legend for bar colours ────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(color=C_GREEN,  label="≤ 4 px  (great)"),
        mpatches.Patch(color=C_YELLOW, label="4–7 px  (ok)"),
        mpatches.Patch(color=C_RED,    label="> 7 px  (poor)"),
        mpatches.Patch(color=C_GREY,   label="No data"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=4,
               facecolor=PANEL_BG, edgecolor=BORDER, fontsize=8,
               bbox_to_anchor=(0.5, 0.01))

    fig.suptitle(f"Performance Report — {pid.upper()}  ·  MediaPipe Hand Tracking",
                 fontsize=14, fontweight="bold", color=TXT, y=0.96)
    fig.patch.set_facecolor(BG)

    _save(fig, reports_dir, f"{pid}_report.png")


# ---------------------------------------------------------------------------
# Chart 9 — Detection-fail rate by Fitzpatrick type
# ---------------------------------------------------------------------------

def plot_detection_fail_by_fitzpatrick(records, out_dir):
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
    if not all_types:
        print("  09: no Fitzpatrick data — skipping")
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
        bars = ax.bar(x + (i - 0.5) * width, means, width, yerr=errs, capsize=4,
                      label=model.capitalize(),
                      color=_MODEL_COLORS[model], alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, means):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
                        f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Fitzpatrick Type")
    ax.set_ylabel("Mean Detection-Fail Rate (%)  ±SD")
    ax.set_title("Detection-Fail Rate by Fitzpatrick Skin Type\n"
                 "(tip visible but landed outside key polygon)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Type {t}\n({_FITZ_LABELS.get(t, '')})" for t in all_types])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "09_detection_fail_by_fitzpatrick.png")


# ---------------------------------------------------------------------------
# Chart 10 — Detection-fail rate by lighting condition
# ---------------------------------------------------------------------------

def plot_detection_fail_by_lux(records, out_dir):
    by_lux_model = defaultdict(list)
    for r in records:
        total = ((r.get("notes_matched", 0) or 0) +
                 (r.get("notes_detection_fail", 0) or 0) +
                 (r.get("notes_missed", 0) or 0))
        df = r.get("notes_detection_fail", 0) or 0
        by_lux_model[(r["lux_label"], r["model"])].append(100 * df / total if total else 0)

    present_lux = [l for l in _LUX_ORDER if any(
        (l, m) in by_lux_model for m in _MODELS)]
    if not present_lux:
        print("  10: no lux data — skipping")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    x     = np.arange(len(present_lux))
    width = 0.35

    for i, model in enumerate(_MODELS):
        means, errs = [], []
        for lbl in present_lux:
            vals = by_lux_model.get((lbl, model), [])
            means.append(np.mean(vals) if vals else 0)
            errs.append(np.std(vals)   if len(vals) > 1 else 0)
        bars = ax.bar(x + (i - 0.5) * width, means, width, yerr=errs, capsize=4,
                      label=model.capitalize(),
                      color=_MODEL_COLORS[model], alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, means):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
                        f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Lighting condition")
    ax.set_ylabel("Mean Detection-Fail Rate (%)  ±SD")
    ax.set_title("Detection-Fail Rate by Lighting Condition\n"
                 "(tip visible but landed outside key polygon)")
    ax.set_xticks(x)
    ax.set_xticklabels(present_lux)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "10_detection_fail_by_lux.png")


# ---------------------------------------------------------------------------
# Chart 11 — Overall MPJPE by hand (Left vs Right), both models
# ---------------------------------------------------------------------------

def plot_mpjpe_by_hand(records, out_dir):
    """
    Bar chart: MediaPipe Left vs Right hand mean MPJPE.
    OpenPose excluded — it has no physical handedness identifier.
    """
    mp_recs = [r for r in records if r["model"] == "mediapipe"]
    by_side = defaultdict(list)
    for r in mp_recs:
        ph = r.get("per_hand", {})
        for side in ["L", "R"]:
            v = ph.get(side, {}).get("mjmpe_px")
            if v is not None:
                by_side[side].append(v)

    if not by_side:
        print("  11: no MediaPipe per-hand data — skipping")
        return

    sides       = ["L", "R"]
    side_labels = ["Left Hand", "Right Hand"]
    means = [np.mean(by_side[s]) if by_side[s] else 0 for s in sides]
    errs  = [np.std(by_side[s])  if len(by_side[s]) > 1 else 0 for s in sides]
    ns    = [len(by_side[s]) for s in sides]

    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(len(sides))
    bars = ax.bar(x, means, 0.45, yerr=errs, capsize=6,
                  color=_MODEL_COLORS["mediapipe"], alpha=0.85, edgecolor="white",
                  error_kw={"linewidth": 1.5, "ecolor": "#444"})
    for bar, v, e, n in zip(bars, means, errs, ns):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, v + e + 0.15,
                    f"{v:.2f} px\n(n={n})", ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("Hand")
    ax.set_ylabel("Mean MPJPE (px)  ±SD")
    ax.set_title("MediaPipe — Overall MPJPE by Hand\n(physical handedness)")
    ax.set_xticks(x)
    ax.set_xticklabels(side_labels, fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(means) + max(errs) + 1.5)
    fig.tight_layout()
    _save(fig, out_dir, "11_mpjpe_by_hand.png")


# ---------------------------------------------------------------------------
# Chart 12 — Per-finger accuracy breakdown (MediaPipe only)
# ---------------------------------------------------------------------------

def plot_accuracy_by_finger(records, out_dir):
    """
    Stacked bar chart: for each finger, % of matched events that were accurate
    (within threshold) vs above threshold. Aggregated across L+R hands and all
    sessions. MediaPipe only — detection_fail/missed are not tracked per finger.
    """
    mp_recs = [r for r in records if r["model"] == "mediapipe"]
    if not mp_recs:
        print("  12: no MediaPipe data — skipping")
        return

    # Accumulate total accurate and total inaccurate counts per finger
    accurate   = [0] * 5   # within threshold
    inaccurate = [0] * 5   # matched but above threshold

    for r in mp_recs:
        ph = r.get("per_hand", {})
        for side in ["L", "R"]:
            fingers = ph.get(side, {}).get("fingers", {})
            for fi in range(5):
                f = fingers.get(str(fi), {})
                count   = f.get("count", 0) or 0
                acc_pct = f.get("accuracy_pct", 0) or 0
                acc     = round(count * acc_pct / 100)
                inacc   = count - acc
                accurate[fi]   += acc
                inaccurate[fi] += inacc

    totals = [a + b for a, b in zip(accurate, inaccurate)]
    if not any(totals):
        print("  12: no per-finger count data — skipping")
        return

    pct_acc   = [100 * a / t if t else 0 for a, t in zip(accurate,   totals)]
    pct_inacc = [100 * b / t if t else 0 for b, t in zip(inaccurate, totals)]

    x = np.arange(5)
    fig, ax = plt.subplots(figsize=(8, 5))

    bars_acc = ax.bar(x, pct_acc,   label="Within threshold",        color="#4CAF50", alpha=0.85)
    bars_inacc = ax.bar(x, pct_inacc, bottom=pct_acc,
                        label="Above threshold (matched)", color="#FF9800", alpha=0.85)

    for i, (pa, pi, t) in enumerate(zip(pct_acc, pct_inacc, totals)):
        ax.text(i, pa / 2, f"{pa:.1f}%", ha="center", va="center",
                fontsize=9, fontweight="bold", color="white")
        if pi > 3:
            ax.text(i, pa + pi / 2, f"{pi:.1f}%", ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white")
        ax.text(i, 101, f"n={t}", ha="center", va="bottom", fontsize=8, color="#555")

    ax.set_ylabel("% of matched events")
    ax.set_title("MediaPipe — Per-Finger Accuracy Breakdown\n(matched events only; L+R combined)")
    ax.set_xticks(x)
    ax.set_xticklabels(_FINGER_NAMES, fontsize=11)
    ax.set_ylim(0, 112)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "12_accuracy_by_finger.png")


# ---------------------------------------------------------------------------
# Chart 14 — Per-hand detection rate (MediaPipe, specified participants)
# ---------------------------------------------------------------------------

def plot_mediapipe_per_hand_hitrate(records, out_dir):
    """
    Chart 14 — MediaPipe per-hand polygon hit-rate stacked bar.
    matched% + detection-fail% as % of (matched + fail) per hand.
    """
    mp_recs = [r for r in records if r["model"] == "mediapipe"]
    if not mp_recs:
        print("  14: no MediaPipe data — skipping")
        return

    totals = {"L": {"matched": 0, "detection_fail": 0},
              "R": {"matched": 0, "detection_fail": 0}}
    for r in mp_recs:
        ph = r.get("per_hand", {})
        for side in ["L", "R"]:
            totals[side]["matched"]        += ph.get(side, {}).get("matched", 0) or 0
            totals[side]["detection_fail"] += ph.get(side, {}).get("detection_fail", 0) or 0

    side_labels = ["Left Hand", "Right Hand"]
    x         = np.arange(2)
    hit_pcts  = []
    fail_pcts = []
    ns        = []
    for side in ["L", "R"]:
        m     = totals[side]["matched"]
        d     = totals[side]["detection_fail"]
        denom = m + d
        ns.append(denom)
        hit_pcts.append(100.0 * m / denom if denom else 0.0)
        fail_pcts.append(100.0 * d / denom if denom else 0.0)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(x, hit_pcts,  label="Matched (on key)",  color="#4CAF50", alpha=0.85, edgecolor="white")
    ax.bar(x, fail_pcts, bottom=hit_pcts,
           label="Detection fail", color="#FF9800", alpha=0.85, edgecolor="white")

    for i, (hr, fr, n) in enumerate(zip(hit_pcts, fail_pcts, ns)):
        if hr > 2:
            ax.text(i, hr / 2, f"{hr:.1f}%\n(n={n})",
                    ha="center", va="center", fontsize=10, fontweight="bold", color="white")
        if fr > 2:
            ax.text(i, hr + fr / 2, f"{fr:.1f}%",
                    ha="center", va="center", fontsize=10, fontweight="bold", color="white")

    ax.set_title("MediaPipe — Polygon Hit-Rate by Hand")
    ax.set_ylabel("% of attributable events (matched + fail)")
    ax.set_xlabel("n = matched + detection-fail events per hand", fontsize=9, color="#555")
    ax.set_xticks(x)
    ax.set_xticklabels(side_labels, fontsize=11)
    ax.set_ylim(0, 108)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "14_mediapipe_per_hand_hitrate.png")


def plot_mediapipe_per_hand_mpjpe(records, out_dir):
    """
    Chart 15 — MediaPipe per-hand MPJPE ±1 SD across sessions.
    Only sessions where that hand had at least one matched event are included.
    """
    mp_recs = [r for r in records if r["model"] == "mediapipe"]
    if not mp_recs:
        print("  15: no MediaPipe data — skipping")
        return

    mpjpe_vals = {"L": [], "R": []}
    for r in mp_recs:
        ph = r.get("per_hand", {})
        for side in ["L", "R"]:
            m = ph.get(side, {}).get("matched", 0) or 0
            v = ph.get(side, {}).get("mjmpe_px")
            if v is not None and m > 0:
                mpjpe_vals[side].append(v)

    side_labels = ["Left Hand", "Right Hand"]
    x     = np.arange(2)
    means = [np.mean(mpjpe_vals[s]) if mpjpe_vals[s] else 0 for s in ["L", "R"]]
    sds   = [np.std(mpjpe_vals[s], ddof=0) if len(mpjpe_vals[s]) > 1 else 0
             for s in ["L", "R"]]
    ns    = [len(mpjpe_vals[s]) for s in ["L", "R"]]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(x, means, 0.45, yerr=sds, capsize=6,
                  color=_MODEL_COLORS["mediapipe"], alpha=0.85, edgecolor="white",
                  error_kw={"linewidth": 1.5, "ecolor": "#444"})
    for bar, v, e, n in zip(bars, means, sds, ns):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, v + e + 0.15,
                    f"{v:.2f} px\n(n={n})", ha="center", va="bottom", fontsize=10)

    ax.set_title("MediaPipe — MPJPE by Hand (matched events)")
    ax.set_ylabel("Mean MPJPE (px)  ±SD")
    ax.set_xticks(x)
    ax.set_xticklabels(side_labels, fontsize=11)
    ax.set_ylim(0, max(means) + max(sds) + 2 if any(means) else 10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "15_mediapipe_per_hand_mpjpe.png")


# ---------------------------------------------------------------------------
# Chart 16 — Scatter: MPJPE vs raw lux value with regression line
# ---------------------------------------------------------------------------

def plot_mpjpe_vs_lux_scatter(records, out_dir):
    """
    Scatter + regression of MPJPE against raw lux values (continuous),
    one series per model. Vertical dashed lines mark the Dim/Indoor/Bright
    category boundaries (100 lux, 500 lux) for reference.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    # Shaded bands for lighting categories
    ax.axvspan(0,   100, color="#78909C", alpha=0.07, zorder=0)
    ax.axvspan(100, 500, color="#FFA726", alpha=0.07, zorder=0)
    ax.axvline(100, color="#aaa", linewidth=1, linestyle="--", zorder=1)
    ax.axvline(500, color="#aaa", linewidth=1, linestyle="--", zorder=1)

    # Category labels along the top
    ax.text(50,  0.25, "Dim",     ha="center", va="bottom", fontsize=8, color="#888")
    ax.text(300, 0.25, "Indoor",  ha="center", va="bottom", fontsize=8, color="#888")

    all_ys = []
    for model in _MODELS:
        recs = [r for r in records
                if r["model"] == model
                and r.get("mjmpe_px") is not None
                and r.get("lux") is not None]
        if not recs:
            continue
        xs = [r["lux"]     for r in recs]
        ys = [r["mjmpe_px"] for r in recs]
        all_ys.extend(ys)
        ax.scatter(xs, ys, label=model.capitalize(),
                   color=_MODEL_COLORS[model], alpha=0.8, s=70, zorder=3)
        if len(xs) >= 2:
            z  = np.polyfit(xs, ys, 1)
            xp = np.linspace(min(xs), max(xs), 200)
            ax.plot(xp, np.polyval(z, xp),
                    color=_MODEL_COLORS[model], linestyle="--", alpha=0.65,
                    label=f"{model.capitalize()} trend")

    ax.set_xlabel("Lux (measured)")
    ax.set_ylabel("MPJPE (px)")
    ax.set_title("MPJPE vs Lux Level")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0, top=max(all_ys) * 1.15 if all_ys else 15)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "16_mpjpe_vs_lux_scatter.png")


# ---------------------------------------------------------------------------
# Chart 13 — Per-finger accuracy breakdown (OpenPose only)
# ---------------------------------------------------------------------------

def plot_accuracy_by_finger_openpose(records, out_dir):
    """
    Stacked bar chart: per-finger accuracy breakdown for OpenPose.
    Same logic as chart 12 but for OpenPose. Low sample counts expected
    given OpenPose's ~8-9% overall detection rate.
    """
    op_recs = [r for r in records if r["model"] == "openpose"]
    if not op_recs:
        print("  13: no OpenPose data — skipping")
        return

    accurate   = [0] * 5
    inaccurate = [0] * 5

    for r in op_recs:
        ph = r.get("per_hand", {})
        for side in ["L", "R"]:
            fingers = ph.get(side, {}).get("fingers", {})
            for fi in range(5):
                f = fingers.get(str(fi), {})
                count   = f.get("count", 0) or 0
                acc_pct = f.get("accuracy_pct", 0) or 0
                acc     = round(count * acc_pct / 100)
                inacc   = count - acc
                accurate[fi]   += acc
                inaccurate[fi] += inacc

    totals = [a + b for a, b in zip(accurate, inaccurate)]
    if not any(totals):
        print("  13: no per-finger count data — skipping")
        return

    pct_acc   = [100 * a / t if t else 0 for a, t in zip(accurate,   totals)]
    pct_inacc = [100 * b / t if t else 0 for b, t in zip(inaccurate, totals)]

    x = np.arange(5)
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(x, pct_acc,   label="Within threshold",          color="#4CAF50", alpha=0.85)
    ax.bar(x, pct_inacc, bottom=pct_acc,
           label="Above threshold (matched)", color="#FF9800", alpha=0.85)

    for i, (pa, pi, t) in enumerate(zip(pct_acc, pct_inacc, totals)):
        ax.text(i, pa / 2, f"{pa:.1f}%", ha="center", va="center",
                fontsize=9, fontweight="bold", color="white")
        if pi > 3:
            ax.text(i, pa + pi / 2, f"{pi:.1f}%", ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white")
        ax.text(i, 101, f"n={t}", ha="center", va="bottom", fontsize=8, color="#555")

    ax.set_ylabel("% of matched events")
    ax.set_title("OpenPose — Per-Finger Accuracy Breakdown\n(matched events only; L+R combined)")
    ax.set_xticks(x)
    ax.set_xticklabels(_FINGER_NAMES, fontsize=11)
    ax.set_ylim(0, 112)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "13_accuracy_by_finger_openpose.png")


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
    parser = argparse.ArgumentParser(description="Plot MPJPE analysis results")
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
    print(f"Output -> {out_dir}\n")

    plot_participant_composition(records, out_dir)
    plot_model_comparison(records, out_dir)
    plot_per_finger_mpjpe(records, out_dir)
    plot_detection_breakdown(records, out_dir)
    plot_by_lux(records, out_dir)
    plot_by_fitzpatrick(records, out_dir)
    plot_mpjpe_vs_handsize(records, out_dir)
    plot_finger_distribution(records, out_dir)
    plot_finger_heatmap(records, out_dir)
    plot_detection_fail_by_fitzpatrick(records, out_dir)
    plot_detection_fail_by_lux(records, out_dir)
    plot_mpjpe_by_hand(records, out_dir)
    plot_accuracy_by_finger(records, out_dir)
    plot_accuracy_by_finger_openpose(records, out_dir)
    plot_mediapipe_per_hand_hitrate(records, out_dir)
    plot_mediapipe_per_hand_mpjpe(records, out_dir)
    plot_mpjpe_vs_lux_scatter(records, out_dir)

    print("\nGenerating per-participant reports...")
    for pid in pids:
        plot_participant_report(pid, records, out_dir)

    n_group   = len(list(out_dir.glob("*.png")))
    n_reports = len(list((out_dir / "reports").glob("*.png"))) if (out_dir / "reports").exists() else 0
    print(f"\nDone — {n_group} group chart(s) in {out_dir}"
          f"  +  {n_reports} report(s) in {out_dir / 'reports'}")


if __name__ == "__main__":
    main()
