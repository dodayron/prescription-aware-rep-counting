"""Can a simple correction reduce the BlazePose knee-angle error? REHAB24-6 squat, offline. Added 11 Nov 2026.

Corrections tested on the per-frame output of analyze_rehab.py (<prefix>_frames.csv), per view:
  raw        : no correction
  loso_const : one constant offset per view, learned on the OTHER participants (leave-one-subject-out).
               Deployable with no per-user step.
  loso_lin   : linear map (slope, intercept) per view, learned on the other participants.
  user_const : constant offset learned on the same session's first 2 repetitions (needs a reference
               measurement of the user, e.g. goniometer; mocap stands in for it here), tested on the rest.
  user_lin   : linear map from the session's first 2 repetitions, tested on the rest.
Test frames are the same for all methods: frames inside repetitions 3+ of each session.
Usage: python -I offset_test.py <rehab root> <prefix e.g. out/lite> [--nboot 2000]
"""
import argparse, os
import numpy as np, pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("prefix"); ap.add_argument("--nboot", type=int, default=2000)
a = ap.parse_args()
fr = pd.read_csv(a.prefix + "_frames.csv")
seg = pd.read_csv(os.path.join(a.root, "Segmentation.csv"), sep=";")
seg = seg[seg.exercise_id == 6].sort_values(["video_id", "first_frame"]).reset_index(drop=True)
seg["segment"] = seg.groupby("video_id").cam17_orientation.transform(lambda s: (s != s.shift()).cumsum())
seg["rep_in_seg"] = seg.groupby(["video_id", "segment"]).cumcount()
# map each frame to its repetition
parts = []
for vid, g in fr.groupby("video"):
    s = seg[seg.video_id == vid]
    idx = pd.IntervalIndex.from_arrays(s.first_frame, s.last_frame, closed="both")
    loc = idx.get_indexer(g.frame)
    g = g[loc >= 0].copy(); loc = loc[loc >= 0]
    for c in ("person_id", "segment", "rep_in_seg"):
        g[c] = s[c].to_numpy()[loc]
    parts.append(g)
fr = pd.concat(parts)
fr["session"] = fr.video + "_" + fr.cam.astype(str) + "_" + fr.segment.astype(str)
rng = np.random.default_rng(0)


def stats(e):
    e = np.asarray(e, float)
    return dict(bias=e.mean(), rmse=np.sqrt((e ** 2).mean()), mae=np.abs(e).mean(), sd=e.std(ddof=1),
                loa_lo=e.mean() - 1.96 * e.std(ddof=1), loa_hi=e.mean() + 1.96 * e.std(ddof=1))


rows = []
for view, g in fr.groupby("view"):
    test = g[g.rep_in_seg >= 2].copy()
    cal = g[g.rep_in_seg < 2]
    pred = {"raw": test.bp.to_numpy()}
    lc, ll = np.full(len(test), np.nan), np.full(len(test), np.nan)
    for p in test.person_id.unique():
        tr = g[g.person_id != p]; m = (test.person_id == p).to_numpy()
        lc[m] = test.bp[m] - (tr.bp - tr.mocap).mean()
        k, b = np.polyfit(tr.bp, tr.mocap, 1); ll[m] = k * test.bp[m] + b
    pred["loso_const"], pred["loso_lin"] = lc, ll
    uc, ul = np.full(len(test), np.nan), np.full(len(test), np.nan)
    for s_id, c in cal.groupby("session"):
        m = (test.session == s_id).to_numpy()
        if not m.any() or len(c) < 20:
            continue
        uc[m] = test.bp[m] - (c.bp - c.mocap).mean()
        k, b = np.polyfit(c.bp, c.mocap, 1); ul[m] = k * test.bp[m] + b
    pred["user_const"], pred["user_lin"] = uc, ul
    for name, y in pred.items():
        ok = ~np.isnan(y)
        e = y[ok] - test.mocap.to_numpy()[ok]
        st = stats(e)
        # person-level bootstrap CI of RMSE
        pe = pd.DataFrame(dict(p=test.person_id.to_numpy()[ok], e=e))
        groups = [x.e.to_numpy() for _, x in pe.groupby("p")]
        bs = []
        for _ in range(a.nboot):
            pick = rng.integers(0, len(groups), len(groups))
            ee = np.concatenate([groups[i] for i in pick]); bs.append(np.sqrt((ee ** 2).mean()))
        st.update(view=view, method=name, n_frames=int(ok.sum()), n_persons=len(groups),
                  rmse_lo=np.percentile(bs, 2.5), rmse_hi=np.percentile(bs, 97.5))
        rows.append(st)
out = pd.DataFrame(rows)[["view", "method", "n_frames", "n_persons", "bias", "rmse", "rmse_lo", "rmse_hi", "mae", "sd", "loa_lo", "loa_hi"]]
out.to_csv(a.prefix + "_offset.csv", index=False)
pd.set_option("display.width", 200)
print(out.round(2).to_string(index=False))
