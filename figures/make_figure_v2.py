"""Figure 1 (v2 framing): what each counter reports on the KIMORE squat, and fidelity by group.
Panel A: mean repetitions per recording reported by the fixed counter vs the calibrated counter
         (attempts split into at-prescription / below-prescription), patients and healthy.
Panel B: fidelity (share of attempts at prescribed depth) by group, subject-level bootstrap 95% CI.
Reads fidelity_eval.py's aggregate output (fidelity_summary.json); prints every plotted number.
"""
import json, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "kimore", "results")
S = json.load(open(f"{OUT}/fidelity_summary.json"))
COL = {"fixed": "#2a78d6", "met": "#1baf7a", "below": "#eb6834"}
TXT, TXT2, GRID = "#262625", "#5f5e5a", "#e6e5e1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": TXT2, "axes.labelcolor": TXT,
                     "xtick.color": TXT, "ytick.color": TXT, "axes.spines.top": False, "axes.spines.right": False})
fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 3.1), dpi=300, gridspec_kw=dict(width_ratios=[1.45, 1.0], wspace=0.38))

w = 0.34
print("Panel A: mean repetitions per recording")
for i, c in enumerate(["patient", "healthy"]):
    v = S["cohort"][c]; n = v["n"]
    fx, met, below = v["fixed_reps"] / n, v["met"] / n, (v["attempts"] - v["met"]) / n
    x1, x2 = i - w / 2 - 0.02, i + w / 2 + 0.02
    ax.bar(x1, fx, w, color=COL["fixed"], linewidth=0, label="Fixed counter (160/90°): repetitions logged" if i == 0 else None)
    ax.bar(x2, met, w, color=COL["met"], linewidth=0, label="Calibrated counter: attempts at prescribed depth" if i == 0 else None)
    ax.bar(x2, below, w, bottom=met, color=COL["below"], edgecolor="white", linewidth=1.2,
           label="Calibrated counter: attempts below prescription" if i == 0 else None)
    ax.text(x1, fx + 0.07, f"{fx:.1f}", ha="center", va="bottom", fontsize=7, color=TXT)
    ax.text(x2, met + below + 0.07, f"{met + below:.1f}", ha="center", va="bottom", fontsize=7, color=TXT)
    if met > 0.5:
        ax.text(x2, met / 2, f"{met:.1f}", ha="center", va="center", fontsize=7, color="white", fontweight="bold")
    print(f"  {c:8s} n={n} fixed {fx:.2f} ({v['fixed_reps']}) | attempts {met+below:.2f} ({v['attempts']}) = met {met:.2f} ({v['met']}) + below {below:.2f}")
ax.axhline(3, color=TXT2, lw=0.8, ls=(0, (4, 3)))
ax.text(-0.52, 3.04, "expected: 3", ha="left", va="bottom", fontsize=6.5, color=TXT2)
ax.set_xticks([0, 1]); ax.set_xticklabels([f"Patients (n = {S['cohort']['patient']['n']})", f"Healthy (n = {S['cohort']['healthy']['n']})"])
ax.set_xlim(-0.55, 1.55); ax.set_ylim(0, 4.4); ax.set_yticks([0, 1, 2, 3, 4]); ax.set_ylabel("Mean repetitions per recording")
ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(-0.02, -0.13), ncol=1, handlelength=1.0, labelspacing=0.35)
ax.set_title("A  What each counter reports", loc="left", fontsize=8.5, color=TXT, fontweight="bold")

print("Panel B: fidelity % [95% CI]")
groups = [("Parkinson's", S["subgroup"]["Parkinson"]), ("Stroke", S["subgroup"]["Stroke"]),
          ("Low back pain", S["subgroup"]["BackPain"]), ("Healthy", S["cohort"]["healthy"])]
for i, (name, v) in enumerate(groups):
    f, (lo, hi) = 100 * v["fidelity"], [100 * x for x in v["fidelity_ci"]]
    y = len(groups) - 1 - i
    bx.plot([lo, hi], [y, y], color=TXT, lw=0.9, solid_capstyle="butt")
    bx.plot(f, y, "o", ms=6, color=COL["met"], markeredgecolor="white", markeredgewidth=1.0, zorder=3)
    bx.text(min(hi, 100) + 2.5, y, f"{f:.0f}", va="center", ha="left", fontsize=7, color=TXT)
    print(f"  {name:14s} n={v['n']:2d} {f:5.1f} [{lo:4.1f}, {hi:5.1f}]")
bx.set_yticks(range(len(groups))); bx.set_yticklabels([f"{n}\n(n = {v['n']})" for n, v in groups][::-1], fontsize=7)
bx.set_xlim(0, 112); bx.set_xticks([0, 25, 50, 75, 100]); bx.set_ylim(-0.6, len(groups) - 0.4)
bx.set_xlabel("Attempts at prescribed depth (%)")
bx.xaxis.grid(True, color=GRID, lw=0.6); bx.set_axisbelow(True)
bx.set_title("B  Fidelity by group", loc="left", fontsize=8.5, color=TXT, fontweight="bold")
fig.savefig(os.path.join(HERE, "fig1_v2_adherence_fidelity.png"), bbox_inches="tight", dpi=300)
fig.savefig(os.path.join(HERE, "fig1_v2_adherence_fidelity.pdf"), bbox_inches="tight")
print("saved")
