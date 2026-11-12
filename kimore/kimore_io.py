"""KIMORE skeleton loader (Kinect v2 JointPosition CSVs) and clinical-score inventory.

Layout: Kimore/<CG|GPP>/<Expert|NotExpert|Stroke|Parkinson|BackPain>/<ID>/Es<N>/
          Raw/JointPosition<ts>.csv   (25 joints x (x,y,z,state) = 100 cols + trailing empty)
          Label/ClinicalAssessment_<ID>.xlsx  (clinical TS/PO/CF Ex#1..5)
Kinect joint indices: hips 12 (L) / 16 (R), knees 13/17, ankles 14/18.
Kinect camera space: x right, y UP, z toward camera depth. We flip y to image convention.
Tracking state: 2 = tracked, 1 = inferred, 0 = not tracked.
"""
import glob
import os
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="Unknown extension")

J = dict(HIP_L=12, KNEE_L=13, ANKLE_L=14, HIP_R=16, KNEE_R=17, ANKLE_R=18,
         SHOULDER_L=4, ELBOW_L=5, WRIST_L=6, SHOULDER_R=8, ELBOW_R=9, WRIST_R=10,
         SPINE_BASE=0, SPINE_MID=1, NECK=2, HEAD=3)

GROUPS = {"Expert": ("CG", "healthy"), "NotExpert": ("CG", "healthy"),
          "Stroke": ("GPP", "patient"), "Parkinson": ("GPP", "patient"), "BackPain": ("GPP", "patient")}


def load_joint_csv(path):
    """Returns pos (T,25,3) in Kinect metres and state (T,25) ints. Raises on parse failure."""
    df = pd.read_csv(path, header=None)
    df = df.dropna(axis=1, how="all")
    arr = df.to_numpy(dtype=float)
    if arr.shape[1] != 100:
        raise ValueError(f"{path}: expected 100 columns, got {arr.shape[1]}")
    arr = arr.reshape(len(arr), 25, 4)
    pos = arr[:, :, :3].copy()
    state = arr[:, :, 3].astype(int)
    return pos, state


def state_to_visibility(state, inferred=0.5):
    """Map Kinect tracking state to a BlazePose-like visibility. 2->1.0, 1->inferred, 0->0."""
    v = np.zeros(state.shape, float)
    v[state == 2] = 1.0
    v[state == 1] = inferred
    return v


def project(pos, view):
    """Return (T,25,2) image-convention 2D points.
    view='front': (x, -y)  Kinect frontal projection (what the Kinect camera saw).
    view='side' : (z, -y)  orthographic projection onto the sagittal plane = SIMULATED side camera.
    """
    if view == "front":
        return np.stack([pos[:, :, 0], -pos[:, :, 1]], axis=-1)
    if view == "side":
        return np.stack([pos[:, :, 2], -pos[:, :, 1]], axis=-1)
    raise ValueError(view)


def inventory(root):
    """One row per recording folder Es<N> with a parseable JointPosition and a clinical sheet."""
    rows = []
    for sub in GROUPS:
        for idp in sorted(glob.glob(os.path.join(root, GROUPS[sub][0], sub, "*"))):
            sid = os.path.basename(idp)
            clin = glob.glob(os.path.join(idp, "Es*", "Label", "ClinicalAssessment_*.xlsx"))
            scores = None
            if clin:
                try:
                    c = pd.read_excel(clin[0]).iloc[0]
                    scores = c
                except Exception as e:  # noqa
                    scores = None
            for esp in sorted(glob.glob(os.path.join(idp, "Es*"))):
                ex = int(re.search(r"Es(\d+)", esp).group(1))
                jp = glob.glob(os.path.join(esp, "Raw", "JointPosition*.csv"))
                if not jp:
                    continue
                ts = re.search(r"JointPosition(\d+_\d+)", jp[0]).group(1)
                ok = True
                try:
                    pos, st = load_joint_csv(jp[0])
                    nframes = len(pos)
                except Exception:
                    ok, nframes = False, 0
                row = dict(subject=sid, subgroup=sub, cohort=GROUPS[sub][1], exercise=ex, ts=ts,
                           joint_file=jp[0], parse_ok=ok, n_frames=nframes)
                for k in ("TS", "PO", "CF"):
                    col = f"clinical {k} Ex#{ex}"
                    row[f"c{k}"] = float(scores[col]) if scores is not None and col in scores else np.nan
                rows.append(row)
    return pd.DataFrame(rows)


def knee_angle_series(pos, state, view, inferred=0.5):
    """Per-frame knee angle by the Pi rule (max over sides with visibility gate). None where unusable."""
    from pi_rule import frame_angle
    p2 = project(pos, view)
    vis = state_to_visibility(state, inferred)
    out = []
    for t in range(len(pos)):
        sides = []
        for s in ("L", "R"):
            idx = (J[f"HIP_{s}"], J[f"KNEE_{s}"], J[f"ANKLE_{s}"])
            sides.append(([p2[t, i] for i in idx], [vis[t, i] for i in idx]))
        out.append(frame_angle(sides))
    return out
