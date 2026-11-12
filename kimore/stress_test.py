"""Pose-error stress test (12 Nov 2026): does the KIMORE squat result survive BlazePose-like angle error?

The KIMORE analysis uses Kinect-derived knee angles. Here those angles are passed through the error model measured
for BlazePose lite against motion capture on REHAB24-6 (profile view):
    true = SLOPE * blazepose + INTERCEPT   (all-data fit: 1.162, -36.9)  ->  blazepose = (true - INTERCEPT) / SLOPE
plus AR(1) noise whose SD and lag-1 autocorrelation are estimated from the REHAB24-6 residuals.
Then the same counters are run on the distorted angles:
  fixed 160/90, calibrated (first 2 reps, 25% margin), and the fidelity (>= 60 deg flexion) check
  (a) on the distorted angle as-is and (b) after applying the linear correction.
SIMULATION on top of a simulated side view: labelled as such. Monte Carlo over noise seeds.

Usage: python -I stress_test.py <Kimore root> <rehab lite_frames.csv> <out_prefix> [--inventory inventory.csv] [--seeds 20]
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kimore_io import inventory, load_joint_csv, knee_angle_series  # noqa: E402
from pi_rule import count_reps  # noqa: E402
from calib_eval import fill_none, detect_cycles, calibrate, BAND, CALIB_CYCLES  # noqa: E402
from fidelity_eval import reps_with_depth  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("frames"); ap.add_argument("out_prefix")
ap.add_argument("--inventory", default=None); ap.add_argument("--seeds", type=int, default=20)
a = ap.parse_args()

# ---- error model from REHAB24-6 (profile view, lite)
fr = pd.read_csv(a.frames); p = fr[fr.view == "profile"].sort_values(["video", "cam", "frame"])
SLOPE, INTERCEPT = np.polyfit(p.bp, p.mocap, 1)
res_bp = p.bp - (p.mocap - INTERCEPT) / SLOPE                 # residual in BlazePose-angle units
SD = float(res_bp.std())
same = (p.video.values[1:] == p.video.values[:-1]) & (p.cam.values[1:] == p.cam.values[:-1]) & (np.diff(p.frame.values) == 1)
RHO = float(np.corrcoef(res_bp.values[:-1][same], res_bp.values[1:][same])[0, 1])
print(f"error model: true = {SLOPE:.3f}*bp + {INTERCEPT:.1f}; residual SD {SD:.2f} deg (bp units), lag-1 autocorrelation {RHO:.3f}")


def distort(ang, rng):
    n = len(ang); e = np.empty(n); e[0] = rng.normal(0, SD)
    w = rng.normal(0, SD * np.sqrt(1 - RHO ** 2), n)
    for i in range(1, n):
        e[i] = RHO * e[i - 1] + w[i]
    return np.clip((ang - INTERCEPT) / SLOPE + e, 0, 180)


inv = pd.read_csv(a.inventory) if a.inventory and os.path.exists(a.inventory) else inventory(a.root)
e5 = inv[(inv.exercise == 5) & inv.parse_ok & inv.cPO.notna()]
recs = []
for row in e5.itertuples():
    pos, st = load_joint_csv(row.joint_file)
    raw = knee_angle_series(pos, st, "side", 0.5)
    ang = fill_none(raw)
    if np.isnan(ang).all():
        continue
    mins, f = detect_cycles(ang)
    if len(mins) < CALIB_CYCLES + 1:
        continue                                               # same inclusion as the main analysis (clean signal)
    _, _, _, _, end = calibrate(f, mins, 0.25)
    recs.append((row.subject, row.cohort, row.subgroup, raw, ang, end))

rows = []
for seed in range(a.seeds):
    rng = np.random.default_rng(seed)
    for subject, cohort, subgroup, raw, ang, end in recs:
        d = distort(ang, rng)
        draw = [None if r is None else float(v) for r, v in zip(raw, d)]       # keep the unusable frames unusable
        mins, f = detect_cycles(d)
        out = dict(seed=seed, subject=subject, cohort=cohort, subgroup=subgroup, fixed=count_reps(draw[end:], 160, 90)[0],
                   presc_nominal=count_reps(draw[end:], 160, 120)[0], n_cycles=len(mins))
        if len(mins) >= CALIB_CYCLES + 1:
            rest, peak, depth, stand, end_d = calibrate(f, mins, 0.25)
            dep = reps_with_depth(draw[end_d:], f[end_d:], rest, peak)
            out.update(calib_ok=1, attempts=len(dep), fixed=count_reps(draw[end_d:], 160, 90)[0],
                       presc_nominal=count_reps(draw[end_d:], 160, 120)[0],
                       met_uncorrected=sum((180 - x) >= BAND[0] for x in dep),
                       met_corrected=sum((180 - (SLOPE * x + INTERCEPT)) >= BAND[0] for x in dep))
        else:
            out.update(calib_ok=0, attempts=0, met_uncorrected=0, met_corrected=0)
        rows.append(out)
df = pd.DataFrame(rows); df.to_csv(a.out_prefix + "_runs.csv", index=False)
summ = []
for c, g in df.groupby("cohort"):
    per_seed = g.groupby("seed").apply(lambda x: pd.Series(dict(
        n=len(x), fixed_zero=(x.fixed == 0).mean(), calib_zero=(x.attempts == 0).mean(), calib_fail=(x.calib_ok == 0).mean(),
        fixed_reps=x.fixed.sum(), attempts=x.attempts.sum(), met_uncorrected=x.met_uncorrected.sum(), met_corrected=x.met_corrected.sum(),
        fidelity_uncorrected=x.met_uncorrected.sum() / max(1, x.attempts.sum()), fidelity_corrected=x.met_corrected.sum() / max(1, x.attempts.sum()))))
    s = per_seed.mean().to_dict(); s.update(cohort=c, seeds=a.seeds, **{k + "_sd": float(per_seed[k].std()) for k in ("fixed_zero", "calib_zero", "fidelity_uncorrected", "fidelity_corrected", "attempts", "fixed_reps")})
    summ.append(s)
json.dump(dict(model=dict(slope=float(SLOPE), intercept=float(INTERCEPT), sd=SD, rho=RHO), summary=summ), open(a.out_prefix + "_summary.json", "w"), indent=1)
for s in summ:
    print(f"{s['cohort']:8s} n={s['n']:.0f} | fixed logs zero {100*s['fixed_zero']:.0f}% (sd {100*s['fixed_zero_sd']:.1f}) | calibrated zero {100*s['calib_zero']:.1f}% | "
          f"fixed reps {s['fixed_reps']:.0f} | attempts {s['attempts']:.0f} (sd {s['attempts_sd']:.1f}) | fidelity uncorrected {100*s['fidelity_uncorrected']:.0f}% | corrected {100*s['fidelity_corrected']:.0f}% (sd {100*s['fidelity_corrected_sd']:.1f})")
