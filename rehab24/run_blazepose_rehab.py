"""BlazePose (legacy mp.solutions.pose) over REHAB24-6 squat videos, Aaron's settings.

Per frame it writes the inputs the Pi rule needs: 2D image-normalised hip/knee/ankle for both sides with
visibility, the Pi-rule angle (max over sides, visibility >= 0.6), and BlazePose's 3D world knee angle for
reference. Rep logic is applied afterwards (analyze_rehab.py), so thresholds can be varied without rerunning.

Usage: python run_blazepose_rehab.py <videos/Ex6> <out_dir> [--complexity 1] [--size 640x480] [--videos PM_022,...]
Preprocessing: Camera17 (1920x1080) -> centre crop to 4:3 -> resize; Camera18 (1080x1920 portrait) -> pad to
4:3 with black bars -> resize. Pose(min_detection_confidence=0.8, min_tracking_confidence=0.5) as in main_v2.py.
"""
import argparse
import csv
import glob
import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
LM = mp_pose.PoseLandmark
VIS = 0.6


def angle_2d(a, b, c):
    r = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    ang = abs(r * 180.0 / np.pi)
    return 360.0 - ang if ang > 180.0 else ang


def angle_3d(a, b, c):
    v1, v2 = np.asarray(a) - np.asarray(b), np.asarray(c) - np.asarray(b)
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def to_4_3(frame, size):
    h, w = frame.shape[:2]
    tw, th = size
    if w / h > 4 / 3:          # wider than 4:3 -> centre crop width
        nw = int(h * 4 / 3)
        x0 = (w - nw) // 2
        frame = frame[:, x0:x0 + nw]
    elif w / h < 4 / 3:        # taller than 4:3 -> pad width
        nw = int(h * 4 / 3)
        pad = (nw - w) // 2
        frame = cv2.copyMakeBorder(frame, 0, 0, pad, nw - w - pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--complexity", type=int, default=1)
    ap.add_argument("--size", default="640x480")
    ap.add_argument("--videos", default=None, help="comma-separated PM ids to run")
    ap.add_argument("--min_det", type=float, default=0.8)
    a = ap.parse_args()
    size = tuple(int(v) for v in a.size.split("x"))
    os.makedirs(a.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.video_dir, "*.mp4")))
    if a.videos:
        keep = set(a.videos.split(","))
        files = [f for f in files if os.path.basename(f).split("-")[0] in keep]
    sides = {"left": (LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE), "right": (LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE)}
    for f in files:
        base = os.path.basename(f)[:-4]
        out = os.path.join(a.out_dir, base + f"_c{a.complexity}.csv")
        if os.path.exists(out):
            print("skip", out); continue
        cap = cv2.VideoCapture(f)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        t0 = time.time()
        with mp_pose.Pose(min_detection_confidence=a.min_det, min_tracking_confidence=0.5,
                          model_complexity=a.complexity) as pose, open(out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "detected", "angle_pi", "angle_left", "angle_right", "vis_left", "vis_right",
                        "angle3d_left", "angle3d_right", "mean_vis_all", "hip_y_left", "knee_y_left", "ankle_y_left"]
                       + [f"{s}_{j}_{c}" for s in ("l", "r") for j in ("hip", "knee", "ankle") for c in ("x", "y", "v")])
            i = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                img = cv2.cvtColor(to_4_3(frame, size), cv2.COLOR_BGR2RGB)
                img.flags.writeable = False
                res = pose.process(img)
                row = [i, 0] + [""] * 11 + [""] * 18
                if res.pose_landmarks:
                    lm = res.pose_landmarks.landmark
                    wl = res.pose_world_landmarks.landmark if res.pose_world_landmarks else None
                    angs, per = [], {}
                    coords = []
                    for s, (h, k, an) in sides.items():
                        p = [lm[h.value], lm[k.value], lm[an.value]]
                        v = min(q.visibility for q in p)
                        per[s] = (angle_2d((p[0].x, p[0].y), (p[1].x, p[1].y), (p[2].x, p[2].y)), v)
                        if v >= VIS:
                            angs.append(per[s][0])
                        per[s + "3d"] = angle_3d([wl[h.value].x, wl[h.value].y, wl[h.value].z],
                                                 [wl[k.value].x, wl[k.value].y, wl[k.value].z],
                                                 [wl[an.value].x, wl[an.value].y, wl[an.value].z]) if wl else ""
                        coords += [f"{q.x:.4f}" for q in p[:1]] + [f"{p[0].y:.4f}", f"{p[0].visibility:.3f}",
                                                                 f"{p[1].x:.4f}", f"{p[1].y:.4f}", f"{p[1].visibility:.3f}",
                                                                 f"{p[2].x:.4f}", f"{p[2].y:.4f}", f"{p[2].visibility:.3f}"]
                    row = [i, 1, f"{max(angs):.2f}" if angs else "",
                           f"{per['left'][0]:.2f}", f"{per['right'][0]:.2f}", f"{per['left'][1]:.3f}", f"{per['right'][1]:.3f}",
                           per["left3d"] if per["left3d"] == "" else f"{per['left3d']:.2f}",
                           per["right3d"] if per["right3d"] == "" else f"{per['right3d']:.2f}",
                           f"{np.mean([q.visibility for q in lm]):.3f}",
                           f"{lm[LM.LEFT_HIP.value].y:.4f}", f"{lm[LM.LEFT_KNEE.value].y:.4f}", f"{lm[LM.LEFT_ANKLE.value].y:.4f}"] + coords
                w.writerow(row)
                i += 1
                if i % 500 == 0:
                    fh.flush()
                    print(f"{base}: {i}/{n} frames, {i / (time.time() - t0):.1f} fps", flush=True)
        cap.release()
        print(f"done {base}: {i} frames in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
