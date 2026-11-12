"""REHAB24-6 squat (Ex6) evaluation of Aaron's Pi rule with BlazePose against motion capture and annotated reps.

Usage: python -I analyze_rehab.py <rehab24 root> <blazepose out dir> <out_prefix> [--suffix _c1] [--margin 0.25]
       [--nboot 2000] [--slack 15]

Views: Segmentation gives each rep's orientation towards Camera17 (front / half-profile).
       Camera18 is orthogonal: front -> profile (side), half-profile -> half-profile.
Sessions: consecutive reps of one video with the same orientation form a segment; each segment x camera is
       one session (calibrate on its first 2 annotated reps, count on the remainder).
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kimore"))  # shared pi_rule.py
from pi_rule import count_reps  # noqa: E402

JOINTS = dict(L=(16, 17, 18), R=(21, 22, 23))   # LeftUpLeg, LeftLeg, LeftFoot / Right...


def mocap_knee(pos):
    out = {}
    for s, (h, k, a) in JOINTS.items():
        v1, v2 = pos[:, h, :3] - pos[:, k, :3], pos[:, a, :3] - pos[:, k, :3]
        cos = (v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-9)
        out[s] = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    return out


def mocap_knee_2d(p2):
    """Image-plane knee angle (Aaron's 2D convention) from the projected mocap joints, per side."""
    from pi_rule import angle_2d
    out = {}
    for s, (h, k, a) in JOINTS.items():
        out[s] = np.array([angle_2d(p2[t, h], p2[t, k], p2[t, a]) for t in range(len(p2))])
    return out


def interp_nan(a):
    a = np.array(a, float)
    good = ~np.isnan(a)
    if good.sum() < 2:
        return a
    idx = np.arange(len(a))
    return np.interp(idx, idx[good], a[good])


def best_lag(x, y, maxlag=10):
    """lag l such that x[t] ~ y[t+l]; both nan-free, same length."""
    best, bl = -np.inf, 0
    for l in range(-maxlag, maxlag + 1):
        if l >= 0:
            a, b = x[:len(x) - l], y[l:]
        else:
            a, b = x[-l:], y[:len(y) + l]
        c = np.corrcoef(a, b)[0, 1]
        if c > best:
            best, bl = c, l
    return bl, best


def calibrate_on_reps(angle, reps, margin):
    """reps: DataFrame rows of the first 2 annotated reps. angle: nan-filled BlazePose Pi angle."""
    mins = [np.nanmin(angle[r.first_frame:r.last_frame + 1]) for r in reps.itertuples()]
    if any(np.isnan(m) for m in mins):
        return None
    depth = float(np.mean(mins))
    stand = float(np.nanpercentile(angle[reps.first_frame.min():reps.last_frame.max() + 1], 90))
    rom = stand - depth
    return dict(rest=stand - margin * rom, peak=depth + margin * rom, depth=depth, stand=stand)


def score_events(events, reps, slack):
    """Match counted-rep frames to annotated reps. Returns tp, fp, fn."""
    used = set()
    tp = 0
    for e in events:
        hit = None
        for r in reps.itertuples():
            if r.Index in used:
                continue
            if r.first_frame <= e <= r.last_frame + slack:
                hit = r.Index
                break
        if hit is not None:
            used.add(hit); tp += 1
    return tp, len(events) - tp, len(reps) - tp


def boot_mean_ci(x, nboot, rng):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    r = [np.mean(x[rng.integers(0, len(x), len(x))]) for _ in range(nboot)]
    return tuple(np.percentile(r, [2.5, 97.5]))


def boot_ratio_ci(num, den, nboot, rng):
    """CI for sum(num)/sum(den) resampling sessions."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    n = len(num)
    r = []
    for _ in range(nboot):
        i = rng.integers(0, n, n)
        d = den[i].sum()
        r.append(num[i].sum() / d if d > 0 else np.nan)
    return tuple(np.nanpercentile(r, [2.5, 97.5]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("bp_dir"); ap.add_argument("out_prefix")
    ap.add_argument("--suffix", default="_c1")
    ap.add_argument("--margin", type=float, default=0.25)
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--slack", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    seg = pd.read_csv(os.path.join(a.root, "Segmentation.csv"), sep=";")
    seg = seg[seg.exercise_id == 6].reset_index(drop=True)

    frame_rows, rep_rows, sess_rows, sync_rows = [], [], [], []
    for vid, g in seg.groupby("video_id"):
        pos = np.load(os.path.join(a.root, "joints", "Ex6", f"{vid}-30fps.npy"))
        mk = mocap_knee(pos)
        mocap_max = np.maximum(mk["L"], mk["R"])
        for cam, camname in ((17, "Camera17-30fps"), (18, "Camera18-30fps-transposed")):
            f = os.path.join(a.bp_dir, f"{vid}-{camname}{a.suffix}.csv")
            if not os.path.exists(f):
                print("missing", f); continue
            bp = pd.read_csv(f)
            p2 = np.load(os.path.join(a.root, "joints2d", "Ex6", f"{vid}-c{cam}-30fps.npy"))
            mk2 = mocap_knee_2d(p2)
            mocap2d_max = np.maximum(mk2["L"], mk2["R"])
            n = min(len(bp), len(pos))
            bp = bp.iloc[:n]
            ang = bp.angle_pi.to_numpy(float)          # nan where not usable
            # sync check on the whole video
            x = interp_nan(ang)[:n]; y = mocap_max[:n]
            if np.isfinite(x).all():
                lag, c = best_lag(x - x.mean(), y - y.mean())
                sync_rows.append(dict(video=vid, cam=cam, lag=lag, corr=c))
            usable = ~np.isnan(ang)
            # ---- per-frame and per-rep accuracy
            for r in g.itertuples():
                view = r.cam17_orientation if cam == 17 else {"front": "profile", "half-profile": "half-profile"}[r.cam17_orientation]
                sl = slice(r.first_frame, r.last_frame + 1)
                e = ang[sl] - mocap_max[sl]
                det = bp.detected.to_numpy()[sl]
                row = dict(video=vid, cam=cam, view=view, person=r.person_id, rep=r.repetition_number,
                           correct=r.correctness, lights=r.lights_on, mocap_err=r.mocap_erroneous,
                           n_frames=r.last_frame - r.first_frame + 1, det_rate=float(det.mean()),
                           usable_rate=float(usable[sl].mean()),
                           bias=float(np.nanmean(e)), rmse=float(np.sqrt(np.nanmean(e ** 2))),
                           mocap_depth=float(mocap_max[sl].min()), mocap_depth_L=float(mk["L"][sl].min()),
                           mocap_depth_R=float(mk["R"][sl].min()),
                           bp_depth=float(np.nanmin(ang[sl])) if usable[sl].any() else np.nan)
                row["depth_err"] = row["bp_depth"] - row["mocap_depth"]
                e2 = ang[sl] - mocap2d_max[sl]
                row["bias_vs2d"] = float(np.nanmean(e2)); row["rmse_vs2d"] = float(np.sqrt(np.nanmean(e2 ** 2)))
                row["mocap2d_depth"] = float(mocap2d_max[sl].min())
                row["depth_err_vs2d"] = row["bp_depth"] - row["mocap2d_depth"]
                row["proj_depth_err"] = row["mocap2d_depth"] - row["mocap_depth"]
                # per-side error on the same side (2D BlazePose vs 3D mocap)
                for s, col in (("L", "angle_left"), ("R", "angle_right")):
                    bs = bp[col].to_numpy(float)[sl]
                    vs = bp["vis_left" if s == "L" else "vis_right"].to_numpy(float)[sl]
                    ok = (vs >= 0.6) & ~np.isnan(bs)
                    row[f"rmse_{s}"] = float(np.sqrt(np.mean((bs[ok] - mk[s][sl][ok]) ** 2))) if ok.any() else np.nan
                    row[f"usable_{s}"] = float(ok.mean())
                rep_rows.append(row)
                for t in range(r.first_frame, r.last_frame + 1):
                    if usable[t]:
                        frame_rows.append(dict(video=vid, cam=cam, view=view, frame=t, bp=ang[t], mocap=mocap_max[t], mocap2d=mocap2d_max[t]))
            # ---- sessions: orientation segments
            seg_id = (g.cam17_orientation != g.cam17_orientation.shift()).cumsum()
            for sid, gs in g.groupby(seg_id):
                gs = gs.sort_values("first_frame")
                view = gs.cam17_orientation.iloc[0] if cam == 17 else {"front": "profile", "half-profile": "half-profile"}[gs.cam17_orientation.iloc[0]]
                if len(gs) < 3:
                    continue
                cal = calibrate_on_reps(ang, gs.iloc[:2], a.margin)
                test = gs.iloc[2:]
                start = int(gs.iloc[1].last_frame) + 1
                end = int(test.last_frame.max()) + a.slack + 1
                series = [None if np.isnan(v) else v for v in ang[start:end]]
                base = dict(video=vid, cam=cam, view=view, person=gs.person_id.iloc[0], lights=gs.lights_on.iloc[0],
                            n_test_reps=len(test), calib_ok=cal is not None,
                            calib_depth=cal["depth"] if cal else np.nan, calib_stand=cal["stand"] if cal else np.nan,
                            usable_rate=float(usable[start:end].mean()))
                for name, (rest, peak) in dict(fixed=(160, 90), presc=(160, 120),
                                               calib=(cal["rest"], cal["peak"]) if cal else (160, 90)).items():
                    cnt, ev = count_reps(series, rest, peak)
                    ev = [e + start for e in ev]
                    tp, fp, fn = score_events(ev, test, a.slack)
                    base.update({f"{name}_count": cnt, f"{name}_tp": tp, f"{name}_fp": fp, f"{name}_fn": fn,
                                 f"{name}_exact": int(cnt == len(test))})
                sess_rows.append(base)

    reps = pd.DataFrame(rep_rows); sess = pd.DataFrame(sess_rows); frames = pd.DataFrame(frame_rows)
    sync = pd.DataFrame(sync_rows)
    reps.to_csv(f"{a.out_prefix}_reps.csv", index=False); sess.to_csv(f"{a.out_prefix}_sessions.csv", index=False)
    frames.to_csv(f"{a.out_prefix}_frames.csv", index=False); sync.to_csv(f"{a.out_prefix}_sync.csv", index=False)

    # ---------------- summaries ----------------
    pd.set_option("display.width", 250)
    print("SYNC lag (frames) by camera:", sync.groupby("cam").lag.agg(["median", "min", "max"]).to_dict())
    print("\n== Angle accuracy per view (frames inside annotated reps where the Pi angle is usable) ==")
    acc = []
    for view, gf in frames.groupby("view"):
        e = gf.bp - gf.mocap
        gr = reps[reps.view == view]
        de = gr.depth_err.dropna()
        acc.append(dict(view=view, n_frames=len(gf), n_reps=len(gr),
                        frame_bias=e.mean(), frame_rmse=np.sqrt((e ** 2).mean()), frame_mae=e.abs().mean(),
                        frame_loa_lo=e.mean() - 1.96 * e.std(), frame_loa_hi=e.mean() + 1.96 * e.std(),
                        depth_bias=de.mean(), depth_rmse=np.sqrt((de ** 2).mean()),
                        depth_loa_lo=de.mean() - 1.96 * de.std(), depth_loa_hi=de.mean() + 1.96 * de.std(),
                        depth_bias_ci=boot_mean_ci(de, a.nboot, rng),
                        frame_rmse_vs2d=np.sqrt(((gf.bp - gf.mocap2d) ** 2).mean()), frame_bias_vs2d=(gf.bp - gf.mocap2d).mean(),
                        proj_rmse=np.sqrt(((gf.mocap2d - gf.mocap) ** 2).mean()), proj_bias=(gf.mocap2d - gf.mocap).mean(),
                        depth_rmse_vs2d=np.sqrt((gr.depth_err_vs2d.dropna() ** 2).mean()), depth_bias_vs2d=gr.depth_err_vs2d.mean(),
                        proj_depth_bias=gr.proj_depth_err.mean(),
                        det_rate=gr.det_rate.mean(), usable_rate=gr.usable_rate.mean(),
                        usable_rate_ci=boot_mean_ci(gr.usable_rate, a.nboot, rng),
                        rmse_same_side_L=gr.rmse_L.mean(), rmse_same_side_R=gr.rmse_R.mean()))
    acc = pd.DataFrame(acc); print(acc.round(2).to_string(index=False))
    acc.to_csv(f"{a.out_prefix}_accuracy.csv", index=False)

    print("\n== Rep counting per view (sessions = orientation segment x camera; calibrate on first 2 reps) ==")
    cnt = []
    for view, gs in sess.groupby("view"):
        for m in ("fixed", "presc", "calib"):
            tp, fp, fn = gs[f"{m}_tp"].sum(), gs[f"{m}_fp"].sum(), gs[f"{m}_fn"].sum()
            rec = dict(view=view, method=m, sessions=len(gs), test_reps=int(gs.n_test_reps.sum()), tp=int(tp), fp=int(fp), fn=int(fn),
                       recall=tp / (tp + fn) if tp + fn else np.nan, precision=tp / (tp + fp) if tp + fp else np.nan,
                       exact_rate=gs[f"{m}_exact"].mean(), zero_rate=(gs[f"{m}_count"] == 0).mean())
            rec["recall_ci"] = boot_ratio_ci(gs[f"{m}_tp"], gs[f"{m}_tp"] + gs[f"{m}_fn"], a.nboot, rng)
            rec["precision_ci"] = boot_ratio_ci(gs[f"{m}_tp"], gs[f"{m}_tp"] + gs[f"{m}_fp"], a.nboot, rng)
            cnt.append(rec)
    cnt = pd.DataFrame(cnt); print(cnt.round(3).to_string(index=False)); cnt.to_csv(f"{a.out_prefix}_counting.csv", index=False)

    print("\n== Quality: per-rep depth vs physio correct/incorrect label ==")
    q = []
    for view, gr in reps.groupby("view"):
        for src in ("mocap_depth", "bp_depth"):
            d = gr.dropna(subset=[src])
            if d.correct.nunique() < 2:
                continue
            auc = roc_auc_score(d.correct, -d[src])  # deeper (smaller angle) predicted correct
            q.append(dict(view=view, source=src, n=len(d), n_correct=int(d.correct.sum()), auc=auc,
                          depth_correct=d[d.correct == 1][src].mean(), depth_incorrect=d[d.correct == 0][src].mean()))
    q = pd.DataFrame(q); print(q.round(3).to_string(index=False)); q.to_csv(f"{a.out_prefix}_quality.csv", index=False)
    rho = spearmanr(reps.dropna(subset=["bp_depth"]).bp_depth, reps.dropna(subset=["bp_depth"]).mocap_depth)
    print(f"\nBlazePose depth vs mocap depth over all reps: rho={rho.statistic:.3f}")
    print("\nper-person lights-off reps:", reps[reps.lights == 0].groupby("view").size().to_dict())


if __name__ == "__main__":
    main()
