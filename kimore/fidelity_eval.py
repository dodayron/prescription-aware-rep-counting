"""Per-repetition adherence vs fidelity on KIMORE Ex5 (squat), simulated side view. Added 11 Nov 2026.

For each recording: calibrate on the first 2 squats (as calib_eval.py), then run the Pi two-state rule with the
calibrated thresholds on the remainder and record the deepest knee angle of every counted repetition.
  attempts  = repetitions counted by the calibrated counter           (adherence)
  met       = attempts whose depth reaches the prescription, flexion >= 60 deg   (fidelity)
  inside / deeper / shallower = position relative to the 60-70 deg band
Compared with what the fixed 160/90 counter logs for the same window.
OFFLINE, Kinect skeletons, sagittal projection = SIMULATED side camera. No band clamp is applied.

Usage: python -I fidelity_eval.py <Kimore root> <out_prefix> [--inventory inventory.csv] [--nboot 2000]
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kimore_io import inventory, load_joint_csv, knee_angle_series  # noqa: E402
from pi_rule import count_reps  # noqa: E402
from calib_eval import fill_none, detect_cycles, calibrate, BAND, CALIB_CYCLES  # noqa: E402


def reps_with_depth(raw, filt, rest, peak):
    """Pi rule (direction 'decrease') returning the deepest filtered angle of each counted rep."""
    stage, cur, out = "rest", None, []
    for a, f in zip(raw, filt):
        if a is None:
            continue
        if a < peak:
            stage = "peak"
        if stage == "peak":
            cur = f if cur is None else min(cur, f)
        if a > rest and stage == "peak":
            stage = "rest"; out.append(cur); cur = None
    return out


def one(row, inferred=0.5, margin=0.25):
    pos, st = load_joint_csv(row.joint_file)
    raw = knee_angle_series(pos, st, "side", inferred)
    ang = fill_none(raw)
    if np.isnan(ang).all():
        return None
    mins, f = detect_cycles(ang)
    if len(mins) < CALIB_CYCLES + 1:
        return None
    rest, peak, depth, stand, end = calibrate(f, mins, margin)
    d = reps_with_depth(raw[end:], f[end:], rest, peak)
    assert len(d) == count_reps(raw[end:], rest, peak)[0]
    flex = [180.0 - x for x in d]
    return dict(subject=row.subject, subgroup=row.subgroup, cohort=row.cohort, cPO=row.cPO,
                fixed=count_reps(raw[end:], 160, 90)[0], attempts=len(d),
                met=sum(x >= BAND[0] for x in flex), inside=sum(BAND[0] <= x <= BAND[1] for x in flex),
                deeper=sum(x > BAND[1] for x in flex), shallower=sum(x < BAND[0] for x in flex),
                rep_flexion=";".join(f"{x:.1f}" for x in flex))


def boot(df, fn, nboot, rng):
    v = [fn(df.iloc[rng.integers(0, len(df), len(df))]) for _ in range(nboot)]
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


def summ(df, nboot, rng):
    fns = dict(
        fidelity=lambda d: d.met.sum() / max(1, d.attempts.sum()),                       # share of attempts at prescription
        rec_all_met=lambda d: float(((d.attempts > 0) & (d.met == d.attempts)).mean()),  # every attempt met
        rec_none_met=lambda d: float(((d.attempts > 0) & (d.met == 0)).mean()),          # attempts but none met
        fixed_zero=lambda d: float((d.fixed == 0).mean()),
        misreported=lambda d: float(((d.fixed == 0) & (d.met > 0)).mean()),              # fixed logs 0, yet >=1 rep met prescription
    )
    out = dict(n=len(df), attempts=int(df.attempts.sum()), met=int(df.met.sum()), inside=int(df.inside.sum()),
               deeper=int(df.deeper.sum()), shallower=int(df.shallower.sum()), fixed_reps=int(df.fixed.sum()))
    for k, fn in fns.items():
        out[k] = float(fn(df)); out[k + "_ci"] = boot(df, fn, nboot, rng) if len(df) >= 5 else None
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("out_prefix")
    ap.add_argument("--inventory", default=None); ap.add_argument("--nboot", type=int, default=2000)
    a = ap.parse_args()
    inv = pd.read_csv(a.inventory) if a.inventory and os.path.exists(a.inventory) else inventory(a.root)
    e5 = inv[(inv.exercise == 5) & inv.parse_ok & inv.cPO.notna()]
    df = pd.DataFrame([r for r in (one(x) for x in e5.itertuples()) if r])
    df.to_csv(a.out_prefix + "_per_recording.csv", index=False)
    rng = np.random.default_rng(0)
    res = {"cohort": {c: summ(g, a.nboot, rng) for c, g in df.groupby("cohort")},
           "subgroup": {c: summ(g, a.nboot, rng) for c, g in df.groupby("subgroup")}}
    from scipy.stats import spearmanr
    df["fid"] = df.met / df.attempts.replace(0, np.nan)
    for name, g in [("all", df), ("patient", df[df.cohort == "patient"])]:
        r = spearmanr(g.fid, g.cPO, nan_policy="omit")
        res[f"rho_fidelity_cPO_{name}"] = dict(rho=float(r.statistic), p=float(r.pvalue), n=int(g.fid.notna().sum()))
    json.dump(res, open(a.out_prefix + "_summary.json", "w"), indent=1)
    for lvl in ("cohort", "subgroup"):
        print(f"\n== by {lvl} ==")
        for k, v in res[lvl].items():
            ci = lambda key: "" if v[key + "_ci"] is None else f" [{v[key+'_ci'][0]*100:.0f}, {v[key+'_ci'][1]*100:.0f}]"
            print(f"{k:10s} n={v['n']:2d} attempts={v['attempts']:3d} met={v['met']:3d} (inside {v['inside']}, deeper {v['deeper']}, shallower {v['shallower']}) "
                  f"fixed-counter reps={v['fixed_reps']:3d} | fidelity {v['fidelity']*100:.0f}%{ci('fidelity')} | "
                  f"fixed logs 0: {v['fixed_zero']*100:.0f}%{ci('fixed_zero')} | misreported {v['misreported']*100:.0f}%{ci('misreported')} | "
                  f"none met {v['rec_none_met']*100:.0f}%{ci('rec_none_met')}")
    print({k: v for k, v in res.items() if k.startswith("rho")})


if __name__ == "__main__":
    main()
