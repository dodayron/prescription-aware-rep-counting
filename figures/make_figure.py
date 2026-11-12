"""P5 figure: KIMORE squat rep-counting with fixed / prescription / auto-calibrated thresholds.
Panel A: rate of recordings counted within +-1 rep of expected, patients vs healthy, 95% bootstrap CIs (side view).
Panel B: patients counted as zero reps, simulated side view vs frontal projection.
Reads calib_eval.py outputs; prints every plotted number so they are traceable.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "kimore", "results")
side = pd.read_csv(f"{OUT}/side_i05_m25_summary.csv")
front = pd.read_csv(f"{OUT}/front_i05_m25_summary.csv")

COL = {"fixed": "#2a78d6", "presc": "#eb6834", "calib": "#1baf7a"}
LBL = {"fixed": "Fixed gym\nthresholds (160/90°)", "presc": "Prescription\nthresholds (160/120°)", "calib": "Auto-calibrated\nthresholds (first 2 reps)"}
TXT, TXT2, GRID = "#262625", "#5f5e5a", "#e6e5e1"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": TXT2, "axes.labelcolor": TXT,
                     "xtick.color": TXT, "ytick.color": TXT, "axes.spines.top": False, "axes.spines.right": False})
fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 3.1), dpi=300, gridspec_kw=dict(width_ratios=[2.0, 1.15], wspace=0.35))

# ---- Panel A
cohorts = ["patient", "healthy"]; methods = ["fixed", "presc", "calib"]
w = 0.26
print("Panel A: within +-1 rep rate [95% CI], side view")
for j, m in enumerate(methods):
    for i, c in enumerate(cohorts):
        r = side[(side.method == m) & (side.cohort == c)].iloc[0]
        v, lo, hi = 100 * r.within1_rate, 100 * r.within1_lo, 100 * r.within1_hi
        x = i + (j - 1) * (w + 0.02)
        ax.bar(x, v, w, color=COL[m], label=LBL[m] if i == 0 else None, linewidth=0)
        ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], color=TXT, capsize=2, lw=0.8, capthick=0.8)
        ax.text(x, hi + 2.5, f"{v:.0f}", ha="center", va="bottom", fontsize=7, color=TXT)
        print(f"  {c:8s} {m:6s} {v:5.1f} [{lo:4.1f}, {hi:5.1f}] n={int(r.n)}")
n_p = int(side[(side.cohort == "patient")].n.iloc[0]); n_h = int(side[(side.cohort == "healthy")].n.iloc[0])
ax.set_xticks([0, 1]); ax.set_xticklabels([f"Patients (n = {n_p})", f"Healthy (n = {n_h})"])
ax.set_ylim(0, 115); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_ylabel("Recordings counted within ±1 rep (%)")
ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, handlelength=1.0, columnspacing=1.5)
ax.set_title("A  KIMORE squat, simulated side view", loc="left", fontsize=8.5, color=TXT, fontweight="bold")

# ---- Panel B
print("Panel B: patients with zero reps [95% CI]")
views = [("Side view\n(simulated)", side), ("Frontal\nprojection", front)]
for j, m in enumerate(methods):
    for i, (name, df) in enumerate(views):
        r = df[(df.method == m) & (df.cohort == "patient")].iloc[0]
        v, lo, hi = 100 * r.zero_rate, 100 * r.zero_lo, 100 * r.zero_hi
        x = i + (j - 1) * (w + 0.02)
        bx.bar(x, v, w, color=COL[m], linewidth=0)
        bx.errorbar(x, v, yerr=[[v - lo], [hi - v]], color=TXT, capsize=2, lw=0.8, capthick=0.8)
        bx.text(x, hi + 2.5, f"{v:.0f}", ha="center", va="bottom", fontsize=7, color=TXT)
        print(f"  {name.split()[0]:8s} {m:6s} {v:5.1f} [{lo:4.1f}, {hi:5.1f}] n={int(r.n)}")
bx.set_xticks([0, 1]); bx.set_xticklabels([v[0] for v in views])
bx.set_ylim(0, 115); bx.set_yticks([0, 25, 50, 75, 100]); bx.set_ylabel("Patients with zero reps counted (%)")
bx.yaxis.grid(True, color=GRID, lw=0.6); bx.set_axisbelow(True)
bx.set_title("B  Missed entirely, by view", loc="left", fontsize=8.5, color=TXT, fontweight="bold")

fig.savefig(os.path.join(HERE, "fig1_kimore_counting.png"), bbox_inches="tight", dpi=300)
fig.savefig(os.path.join(HERE, "fig1_kimore_counting.pdf"), bbox_inches="tight")
print("saved")
