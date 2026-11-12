"""IntelliRehabDS (IRDS; Miron et al., Data 2021; Zenodo 4610859, CC BY 4.0): does a fixed-threshold counter
register patients' repetitions? Second cohort and second/third exercise for the ISVR 2027 abstract.

Each IRDS file is ONE repetition (an attempt) with an annotator label (1 correct, 2 incorrect).
Device rules reproduced from exercise_v2.py / exercises_v2.json (commit 23ffae2):
  arm curl      : SHOULDER-ELBOW-WRIST angle, peak when angle < 45 and wrist above elbow; rest 160
  lateral raise : HIP-SHOULDER-ELBOW angle, peak when angle > 90 and elbow at/above shoulder and
                  wrist not more than 0.05 (normalised image height) above elbow; rest 20
IRDS gestures used: 0/1 elbow flexion L/R (curl), 4/5 shoulder abduction L/R (lateral raise).
An attempt is "registered" when the frame-wise peak condition is met at least once in the file (files are cut
per repetition, so the return-to-rest half of the state machine cannot be tested reliably).
Views: '2d' = the Kinect's own image-plane joint coordinates (a real frontal camera view, depth image 512x424);
       '3d' = true 3D joint angle (view-independent reference; position checks still use image y).
Visibility: Tracked = 1.0, Inferred = 0.5, NotTracked = 0; a frame is usable when min of the 3 joints >= 0.6.
Calibration: per subject x gesture, first 2 repetitions (file order) -> depth = mean extreme angle,
start = 90th (curl) / 10th (raise) percentile; threshold = depth + margin * ROM towards start; tested on the rest.
OFFLINE on Kinect skeletons, not BlazePose and not the device.

Usage: python -I irds_eval.py <SkeletonData/RawData dir> <out_prefix> [--margin 0.25] [--nboot 2000]
"""
import argparse, glob, json, os, re
import numpy as np, pandas as pd

JR = re.compile(r"\((\w+),(\w+),([-\d.eE]+),([-\d.eE]+),([-\d.eE]+),([-\d.eE]+|-?Infinity|NaN),([-\d.eE]+|-?Infinity|NaN)\)")
VIS = {"Tracked": 1.0, "Inferred": 0.5, "NotTracked": 0.0}
# The Kinect skeleton stream in IRDS is mirrored: "Elbow Flexion Left" (gesture 0) moves the joints named *Right*
# (checked on the healthy controls: median minimum 3D elbow angle 26 deg on the Right joints vs 151 deg on the Left).
GEST = {0: ("curl", "Right"), 1: ("curl", "Left"), 4: ("raise", "Right"), 5: ("raise", "Left")}
IMG_H = 424.0


def load(path):
    frames = []
    for line in open(path, errors="ignore"):
        js = JR.findall(line)
        if len(js) < 20:
            continue
        d = {}
        for n, st, x, y, z, u, v in js:
            try:
                d[n] = (float(x), float(y), float(z), float(u), float(v), VIS.get(st, 0.0))
            except ValueError:
                d[n] = (float(x), float(y), float(z), np.nan, np.nan, 0.0)
        frames.append(d)
    return frames


def ang(a, b, c):
    a, b, c = np.asarray(a, float), np.asarray(b, float), np.asarray(c, float)
    if len(a) == 2:
        r = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        x = abs(np.degrees(r)); return 360 - x if x > 180 else x
    v1, v2 = a - b, c - b
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9), -1, 1))))


def series(frames, kind, side, view):
    """Per frame: (angle or nan, position-condition bool)."""
    names = ([f"Shoulder{side}", f"Elbow{side}", f"Wrist{side}"] if kind == "curl"
             else [f"Hip{side}", f"Shoulder{side}", f"Elbow{side}"])
    A, C = [], []
    for d in frames:
        try:
            p = [d[n] for n in names]; sh, el, wr = d[f"Shoulder{side}"], d[f"Elbow{side}"], d[f"Wrist{side}"]
        except KeyError:
            A.append(np.nan); C.append(False); continue
        if min(q[5] for q in p) < 0.6 or any(np.isnan(q[3]) or np.isnan(q[4]) for q in p):
            A.append(np.nan); C.append(False); continue
        A.append(ang(*[(q[3], q[4]) for q in p]) if view == "2d" else ang(*[q[:3] for q in p]))
        if kind == "curl":
            C.append(wr[4] < el[4])                                    # image y grows downwards
        else:
            C.append(el[4] <= sh[4] and wr[4] >= el[4] - 0.05 * IMG_H)
    return np.array(A), np.array(C)


def boot_ci(df, fn, nboot, rng):
    subs = df.subject.unique(); groups = {s: g for s, g in df.groupby("subject")}
    v = []
    for _ in range(nboot):
        pick = rng.choice(subs, len(subs)); v.append(fn(pd.concat([groups[s] for s in pick])))
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("raw_dir"); ap.add_argument("out_prefix")
    ap.add_argument("--margin", type=float, default=0.25); ap.add_argument("--nboot", type=int, default=2000)
    a = ap.parse_args()
    rows = []
    for f in sorted(glob.glob(os.path.join(a.raw_dir, "*.txt"))):
        sid, date, g, rep, lab, pos = os.path.basename(f)[:-4].split("_")[:6]
        g, lab, sid = int(g), int(lab), int(sid)
        if g not in GEST or lab not in (1, 2):
            continue
        kind, side = GEST[g]; fr = load(f)
        row = dict(file=os.path.basename(f), subject=sid, cohort="patient" if 200 <= sid < 300 else "healthy",
                   date=int(date), gesture=g, kind=kind, side=side, rep=int(rep), correct=int(lab == 1), position=pos, n_frames=len(fr))
        for view in ("2d", "3d"):
            A, C = series(fr, kind, side, view)
            ok = ~np.isnan(A)
            row[f"usable_{view}"] = float(ok.mean()) if len(A) else 0.0
            if not ok.any():
                row[f"ext_{view}"] = np.nan; row[f"start_{view}"] = np.nan; row[f"fixed_{view}"] = 0; row[f"fixed_angle_only_{view}"] = 0
                continue
            if kind == "curl":
                row[f"ext_{view}"] = float(np.nanmin(A)); row[f"start_{view}"] = float(np.nanpercentile(A, 90))
                row[f"fixed_angle_only_{view}"] = int(np.nanmin(A) < 45); row[f"fixed_{view}"] = int(np.any((A < 45) & C))
                row[f"cond_at_ext_{view}"] = int(C[np.nanargmin(A)])
            else:
                row[f"ext_{view}"] = float(np.nanmax(A)); row[f"start_{view}"] = float(np.nanpercentile(A, 10))
                row[f"fixed_angle_only_{view}"] = int(np.nanmax(A) > 90); row[f"fixed_{view}"] = int(np.any((A > 90) & C))
                row[f"cond_at_ext_{view}"] = int(C[np.nanargmax(A)])
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["subject", "gesture", "date", "rep"]).reset_index(drop=True)
    # calibration per subject x gesture on the first 2 repetitions
    df["order"] = df.groupby(["subject", "gesture"]).cumcount()
    for view in ("2d", "3d"):
        df[f"calib_{view}"] = np.nan; df[f"calib_thr_{view}"] = np.nan
        for (s, g), grp in df.groupby(["subject", "gesture"]):
            cal = grp[grp.order < 2].dropna(subset=[f"ext_{view}"])
            if len(cal) < 2 or len(grp) < 3:
                continue
            depth, start = cal[f"ext_{view}"].mean(), cal[f"start_{view}"].mean()
            thr = depth + a.margin * (start - depth)
            t = grp[grp.order >= 2]
            hit = (t[f"ext_{view}"] < thr) if GEST[g][0] == "curl" else (t[f"ext_{view}"] > thr)
            df.loc[t.index, f"calib_{view}"] = hit.astype(float); df.loc[t.index, f"calib_thr_{view}"] = thr
    df.to_csv(a.out_prefix + "_per_rep.csv", index=False)
    rng = np.random.default_rng(0); res = []
    from sklearn.metrics import roc_auc_score
    for kind in ("curl", "raise"):
        for view in ("2d", "3d"):
            for cohort in ("patient", "healthy"):
                d = df[(df.kind == kind) & (df.cohort == cohort)]
                t = d.dropna(subset=[f"calib_{view}"])                 # test reps (3rd onwards, calibrated subjects)
                tc = t[t.correct == 1]
                r = dict(kind=kind, view=view, cohort=cohort, subjects=int(d.subject.nunique()), reps=len(d),
                         labelled_correct=float(d.correct.mean()), test_reps=len(t), test_correct_reps=len(tc),
                         usable=float(d[f"usable_{view}"].mean()))
                for name, col in (("fixed", f"fixed_{view}"), ("fixed_angle_only", f"fixed_angle_only_{view}"), ("calib", f"calib_{view}")):
                    r[name] = float(t[col].mean()); r[name + "_ci"] = boot_ci(t, lambda x, c=col: x[c].mean(), a.nboot, rng)
                    r[name + "_on_correct"] = float(tc[col].mean()) if len(tc) else np.nan
                    r[name + "_on_correct_ci"] = boot_ci(tc, lambda x, c=col: x[c].mean(), a.nboot, rng) if tc.subject.nunique() >= 3 else None
                # subjects for whom the fixed counter registers nothing at all in the test reps
                per = t.groupby("subject")[[f"fixed_{view}", f"calib_{view}"]].sum()
                r["subjects_test"] = len(per); r["subjects_fixed_zero"] = int((per[f"fixed_{view}"] == 0).sum()); r["subjects_calib_zero"] = int((per[f"calib_{view}"] == 0).sum())
                dd = d.dropna(subset=[f"ext_{view}"])
                if dd.correct.nunique() == 2:
                    sc = -dd[f"ext_{view}"] if kind == "curl" else dd[f"ext_{view}"]
                    r["auc_range_vs_correct"] = float(roc_auc_score(dd.correct, sc))
                r["median_ext"] = float(dd[f"ext_{view}"].median()); r["median_ext_correct"] = float(dd[dd.correct == 1][f"ext_{view}"].median())
                res.append(r)
    out = pd.DataFrame(res); out.to_csv(a.out_prefix + "_summary.csv", index=False)
    json.dump(res, open(a.out_prefix + "_summary.json", "w"), indent=1)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    show = out[["kind", "view", "cohort", "subjects", "reps", "labelled_correct", "test_reps", "fixed", "fixed_angle_only", "calib",
                "fixed_on_correct", "calib_on_correct", "subjects_test", "subjects_fixed_zero", "subjects_calib_zero", "auc_range_vs_correct", "median_ext", "usable"]]
    print(show.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
