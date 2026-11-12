"""KIMORE Ex5 (squat) offline evaluation of the Pi rep rule with fixed, prescription and auto-calibrated
thresholds, plus subject-level bootstrap 95% CIs. P2 of the ISVR 2027 plan.

Usage: python -I calib_eval.py <Kimore root> <out_prefix> [--view side|front] [--inferred 0.5] [--margin 0.25]
                               [--nboot 2000] [--seed 0]

Everything here is OFFLINE on Kinect v2 skeletons. The 'side' view is an orthographic projection of the
Kinect 3D joints onto the sagittal (z, y) plane: a SIMULATED side camera, not a real one.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, medfilt
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kimore_io import inventory, load_joint_csv, knee_angle_series  # noqa: E402
from pi_rule import count_reps  # noqa: E402

PROTOCOL_REPS = 5          # KIMORE Ex5 protocol: 5 squats
CALIB_CYCLES = 2           # calibrate on the first 2 squats
EXPECTED_TEST = PROTOCOL_REPS - CALIB_CYCLES   # 3 reps expected on the remainder
BAND = (60.0, 70.0)        # prescribed knee flexion band (deg); joint angle = 180 - flexion


def fill_none(series):
    a = np.array([np.nan if v is None else v for v in series], float)
    if np.isnan(a).all():
        return a
    idx = np.arange(len(a))
    good = ~np.isnan(a)
    a[~good] = np.interp(idx[~good], idx[good], a[good])
    return a


def detect_cycles(angle, prominence=15.0, distance=15, window=5):
    """Squat minima on the median-filtered knee-angle series. Returns (minima_idx, filtered)."""
    f = medfilt(angle, kernel_size=window)
    mins, _ = find_peaks(-f, prominence=prominence, distance=distance)
    return mins, f


def calibrate(angle, mins, margin):
    """First CALIB_CYCLES minima -> (rest, peak, depth, stand, calib_end_frame)."""
    m = mins[:CALIB_CYCLES]
    depth = float(np.mean(angle[m]))
    # calibration window: start to midway between the 2nd and 3rd minima (or end)
    end = int((mins[CALIB_CYCLES - 1] + mins[CALIB_CYCLES]) // 2) if len(mins) > CALIB_CYCLES else len(angle)
    stand = float(np.percentile(angle[:end], 90))
    rom = stand - depth
    peak = depth + margin * rom
    rest = stand - margin * rom
    return rest, peak, depth, stand, end


def evaluate_recording(row, view, inferred, margin):
    pos, st = load_joint_csv(row.joint_file)
    raw = knee_angle_series(pos, st, view, inferred)
    usable = np.mean([v is not None for v in raw])
    ang = fill_none(raw)
    out = dict(subject=row.subject, subgroup=row.subgroup, cohort=row.cohort, cPO=row.cPO, cTS=row.cTS,
               cCF=row.cCF, n_frames=len(ang), frac_usable=usable)
    if np.isnan(ang).all():
        out["excluded"] = "no usable frames"
        return out
    mins, f = detect_cycles(ang)
    out["n_cycles"] = len(mins)
    out["deepest_angle"] = float(np.nanmin(f))
    # whole-recording counts with fixed and prescription thresholds (as in 3a)
    out["reps_fixed_full"] = count_reps(raw, 160, 90)[0]
    out["reps_presc_full"] = count_reps(raw, 160, 120)[0]
    if len(mins) < CALIB_CYCLES + 1:
        out["excluded"] = f"only {len(mins)} detectable cycles"
        return out
    rest, peak, depth, stand, end = calibrate(f, mins, margin)
    out.update(calib_rest=rest, calib_peak=peak, calib_depth=depth, calib_stand=stand, calib_end=end)
    test = raw[end:]
    out["reps_fixed"] = count_reps(test, 160, 90)[0]
    out["reps_presc"] = count_reps(test, 160, 120)[0]
    out["reps_calib"] = count_reps(test, rest, peak)[0]
    out["n_cycles_test"] = int(np.sum(mins >= end))
    for m in ("fixed", "presc", "calib"):
        out[f"err_vs_cycles_{m}"] = out[f"reps_{m}"] - out["n_cycles_test"]
    out["excluded"] = ""
    # depth relative to the prescribed band, in flexion terms
    flex = 180.0 - depth
    out["calib_flexion"] = flex
    out["band_class"] = "inside" if BAND[0] <= flex <= BAND[1] else ("shallower" if flex < BAND[0] else "deeper")
    return out


def rate(x):
    return float(np.mean(x)) if len(x) else np.nan


def boot_ci(values, stat, nboot, rng, groups=None):
    """Percentile bootstrap over recordings (one recording per subject here, so subject-level)."""
    values = np.asarray(values)
    n = len(values)
    if n == 0:
        return (np.nan, np.nan)
    res = []
    for _ in range(nboot):
        idx = rng.integers(0, n, n)
        res.append(stat(values[idx]))
    return tuple(np.nanpercentile(res, [2.5, 97.5]))


def boot_ci_rho(x, y, nboot, rng):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    res = []
    for _ in range(nboot):
        idx = rng.integers(0, n, n)
        if np.std(x[idx]) == 0 or np.std(y[idx]) == 0:
            res.append(np.nan)
        else:
            res.append(spearmanr(x[idx], y[idx]).statistic)
    return tuple(np.nanpercentile(res, [2.5, 97.5]))


def summarise(df, nboot, seed):
    rng = np.random.default_rng(seed)
    ok = df[df.excluded == ""].copy()
    rows = []
    for method in ("fixed", "presc", "calib"):
        col = f"reps_{method}"
        for cohort in ("patient", "healthy", "all"):
            sub = ok if cohort == "all" else ok[ok.cohort == cohort]
            r = sub[col].to_numpy()
            zero = (r == 0)
            exact = (r == EXPECTED_TEST)
            near = (np.abs(r - EXPECTED_TEST) <= 1)
            rec = dict(method=method, cohort=cohort, n=len(sub),
                       zero_rate=rate(zero), exact_rate=rate(exact), within1_rate=rate(near),
                       mean_abs_err=float(np.mean(np.abs(r - EXPECTED_TEST))) if len(r) else np.nan)
            rec["zero_lo"], rec["zero_hi"] = boot_ci(zero, np.mean, nboot, rng)
            rec["exact_lo"], rec["exact_hi"] = boot_ci(exact, np.mean, nboot, rng)
            rec["within1_lo"], rec["within1_hi"] = boot_ci(near, np.mean, nboot, rng)
            ec = sub[f"err_vs_cycles_{method}"].to_numpy()
            rec["match_cycles_rate"] = rate(ec == 0)
            rec["match_lo"], rec["match_hi"] = boot_ci(ec == 0, np.mean, nboot, rng)
            rec["missed_per_cycle"] = float(-np.sum(ec[ec < 0]) / sub["n_cycles_test"].sum()) if len(sub) else np.nan
            if cohort == "all":
                rho = spearmanr(sub[col], sub.cPO)
                rec["rho_reps_cPO"], rec["rho_p"] = rho.statistic, rho.pvalue
                rec["rho_lo"], rec["rho_hi"] = boot_ci_rho(sub[col], sub.cPO, nboot, rng)
            rows.append(rec)
    summ = pd.DataFrame(rows)
    # depth vs clinical score
    extra = {}
    for cohort in ("all", "patient", "healthy"):
        sub = ok if cohort == "all" else ok[ok.cohort == cohort]
        rho = spearmanr(sub.calib_depth, sub.cPO)
        lo, hi = boot_ci_rho(sub.calib_depth, sub.cPO, nboot, rng)
        extra[f"rho_depth_cPO_{cohort}"] = dict(n=len(sub), rho=rho.statistic, p=rho.pvalue, lo=lo, hi=hi,
                                              median_depth=float(sub.calib_depth.median()))
    # band classification vs cPO
    extra["band"] = ok.groupby(["cohort", "band_class"]).agg(n=("cPO", "size"), cPO_median=("cPO", "median"),
                                                              cPO_mean=("cPO", "mean")).reset_index().to_dict("records")
    # patients vs non-experts only (expert inflation check)
    ne = ok[(ok.subgroup == "NotExpert")]
    pt = ok[ok.cohort == "patient"]
    extra["nonexpert_vs_patient"] = {m: dict(nonexpert_exact=rate(ne[f"reps_{m}"] == EXPECTED_TEST),
                                             patient_exact=rate(pt[f"reps_{m}"] == EXPECTED_TEST))
                                     for m in ("fixed", "presc", "calib")}
    # by pathology (exploratory)
    extra["by_subgroup"] = ok.groupby("subgroup").agg(n=("cPO", "size"),
                                                      fixed_zero=("reps_fixed", lambda r: rate(r == 0)),
                                                      presc_zero=("reps_presc", lambda r: rate(r == 0)),
                                                      calib_zero=("reps_calib", lambda r: rate(r == 0)),
                                                      calib_exact=("reps_calib", lambda r: rate(r == EXPECTED_TEST)),
                                                      depth_median=("calib_depth", "median"),
                                                      cPO_median=("cPO", "median")).reset_index().to_dict("records")
    return summ, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out_prefix")
    ap.add_argument("--view", default="side", choices=["side", "front"])
    ap.add_argument("--inferred", type=float, default=0.5)
    ap.add_argument("--margin", type=float, default=0.25)
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inventory", default=None, help="cached inventory.csv")
    a = ap.parse_args()

    inv = pd.read_csv(a.inventory) if a.inventory and os.path.exists(a.inventory) else inventory(a.root)
    e5 = inv[(inv.exercise == 5) & inv.parse_ok & inv.cPO.notna()]
    recs = [evaluate_recording(r, a.view, a.inferred, a.margin) for r in e5.itertuples()]
    df = pd.DataFrame(recs)
    df.to_csv(f"{a.out_prefix}_per_recording.csv", index=False)
    summ, extra = summarise(df, a.nboot, a.seed)
    summ.to_csv(f"{a.out_prefix}_summary.csv", index=False)
    import json
    with open(f"{a.out_prefix}_extra.json", "w") as f:
        json.dump(extra, f, indent=1, default=float)
    print(f"view={a.view} inferred={a.inferred} margin={a.margin} nboot={a.nboot}")
    print(f"recordings: {len(df)}, excluded: {(df.excluded != '').sum()} -> {df[df.excluded != ''][['subject', 'excluded']].values.tolist()}")
    pd.set_option("display.width", 200)
    print(summ.round(3).to_string(index=False))
    for k, v in extra.items():
        print(k, json.dumps(v, default=lambda x: round(float(x), 3)) if not isinstance(v, list) else "")
        if isinstance(v, list):
            print(pd.DataFrame(v).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
