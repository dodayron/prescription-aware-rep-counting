"""Replay REHAB24-6 squat videos through the deployed pose pipeline on the Raspberry Pi 5 (no camera, no LEDs,
no person in front of the device). Run it ON THE PI.

What it measures
  1. Reproduction: with --mode all every video frame is processed; the per-frame CSVs have the same format as the
     offline workstation run, so the two can be compared number for number (x86 vs ARM inference).
  2. Live-rate behaviour: with --mode realtime frames are dropped the way a live camera would drop them. A simulated
     clock advances by the measured processing time of each frame (resize + BlazePose + angle); the next frame
     processed is the one a 30 fps camera would be delivering at that moment. Skipped frames are written with
     skipped=1 so the rep logic can be re-run on exactly what the device would have seen.
  3. Device load: per-frame processing time, effective fps, peak RAM (RSS), CPU temperature, throttling flags.

Pipeline settings are those of main_v2.py --tested: 640x480 input, mp.solutions.pose.Pose(model_complexity=0,
min_detection_confidence=0.8, min_tracking_confidence=0.5), knee angle = max over sides with visibility >= 0.6.

Usage (from the folder holding this file):
    python pi_replay_bench.py <folder with the Ex6 .mp4 files> <output folder> --mode all
    python pi_replay_bench.py <folder with the Ex6 .mp4 files> <output folder> --mode realtime
Optional: --videos PM_022,PM_105   --max-frames 900   --complexity 0   --size 640x480
Send back the whole output folder (zip it).
"""
import argparse
import csv
import glob
import json
import os
import platform
import resource
import subprocess
import time

import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
LM = mp_pose.PoseLandmark
VIS = 0.6
SIDES = {"left": (LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE), "right": (LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE)}
HEADER = (["frame", "detected", "angle_pi", "angle_left", "angle_right", "vis_left", "vis_right",
           "angle3d_left", "angle3d_right", "mean_vis_all", "hip_y_left", "knee_y_left", "ankle_y_left"]
          + [f"{s}_{j}_{c}" for s in ("l", "r") for j in ("hip", "knee", "ankle") for c in ("x", "y", "v")]
          + ["skipped", "proc_ms", "sim_t"])


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
    if w / h > 4 / 3:
        nw = int(h * 4 / 3)
        x0 = (w - nw) // 2
        frame = frame[:, x0:x0 + nw]
    elif w / h < 4 / 3:
        nw = int(h * 4 / 3)
        pad = (nw - w) // 2
        frame = cv2.copyMakeBorder(frame, 0, 0, pad, nw - w - pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)


def landmark_row(i, res):
    row = [i, 0] + [""] * 11 + [""] * 18
    if not res.pose_landmarks:
        return row
    lm = res.pose_landmarks.landmark
    wl = res.pose_world_landmarks.landmark if res.pose_world_landmarks else None
    angs, per, coords = [], {}, []
    for s, (h, k, an) in SIDES.items():
        p = [lm[h.value], lm[k.value], lm[an.value]]
        v = min(q.visibility for q in p)
        per[s] = (angle_2d((p[0].x, p[0].y), (p[1].x, p[1].y), (p[2].x, p[2].y)), v)
        if v >= VIS:
            angs.append(per[s][0])
        per[s + "3d"] = angle_3d([wl[h.value].x, wl[h.value].y, wl[h.value].z],
                                 [wl[k.value].x, wl[k.value].y, wl[k.value].z],
                                 [wl[an.value].x, wl[an.value].y, wl[an.value].z]) if wl else ""
        for q in p:
            coords += [f"{q.x:.4f}", f"{q.y:.4f}", f"{q.visibility:.3f}"]
    return [i, 1, f"{max(angs):.2f}" if angs else "",
            f"{per['left'][0]:.2f}", f"{per['right'][0]:.2f}", f"{per['left'][1]:.3f}", f"{per['right'][1]:.3f}",
            per["left3d"] if per["left3d"] == "" else f"{per['left3d']:.2f}",
            per["right3d"] if per["right3d"] == "" else f"{per['right3d']:.2f}",
            f"{np.mean([q.visibility for q in lm]):.3f}",
            f"{lm[LM.LEFT_HIP.value].y:.4f}", f"{lm[LM.LEFT_KNEE.value].y:.4f}", f"{lm[LM.LEFT_ANKLE.value].y:.4f}"] + coords


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def cpu_temp():
    t = sh(["vcgencmd", "measure_temp"])           # "temp=48.3'C" on a Pi
    if t.startswith("temp="):
        return float(t[5:].split("'")[0])
    try:
        return int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000.0
    except Exception:
        return None


def system_info():
    model = ""
    try:
        model = open("/proc/device-tree/model").read().strip("\x00")
    except Exception:
        pass
    mem = ""
    try:
        mem = [l for l in open("/proc/meminfo") if l.startswith("MemTotal")][0].split()[1]
    except Exception:
        pass
    return dict(board=model, machine=platform.machine(), python=platform.python_version(),
                os=platform.platform(), mediapipe=mp.__version__, opencv=cv2.__version__, numpy=np.__version__,
                mem_total_kb=mem, cpu_count=os.cpu_count(), throttled_start=sh(["vcgencmd", "get_throttled"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video_dir"); ap.add_argument("out_dir")
    ap.add_argument("--mode", choices=["all", "realtime"], default="all")
    ap.add_argument("--complexity", type=int, default=0)
    ap.add_argument("--size", default="640x480")
    ap.add_argument("--videos", default=None, help="comma-separated PM ids, e.g. PM_022,PM_105")
    ap.add_argument("--max-frames", type=int, default=0, help="stop each video after N frames (0 = whole video)")
    ap.add_argument("--min_det", type=float, default=0.8)
    ap.add_argument("--extra-ms", type=float, default=0.0, help="testing only: add this delay to every frame")
    a = ap.parse_args()
    size = tuple(int(v) for v in a.size.split("x"))
    out_dir = os.path.join(a.out_dir, f"{a.mode}_c{a.complexity}")
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.video_dir, "*.mp4")))
    if a.videos:
        keep = set(a.videos.split(","))
        files = [f for f in files if os.path.basename(f).split("-")[0] in keep]
    if not files:
        raise SystemExit(f"no .mp4 files found in {a.video_dir}")
    info = system_info()
    info.update(mode=a.mode, complexity=a.complexity, size=a.size, min_detection_confidence=a.min_det,
                min_tracking_confidence=0.5, started=time.strftime("%Y-%m-%d %H:%M:%S"), temp_start_c=cpu_temp())
    print(json.dumps(info, indent=1), flush=True)
    summary, temps = [], []
    for f in files:
        base = os.path.basename(f)[:-4]
        out = os.path.join(out_dir, base + f"_c{a.complexity}.csv")
        if os.path.exists(out):
            print("skip (exists)", out, flush=True); continue
        cap = cv2.VideoCapture(f)
        vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        proc, n_proc, n_skip, sim_t, i = [], 0, 0, 0.0, 0
        wall0 = time.time()
        tmp = out + ".part"
        with mp_pose.Pose(min_detection_confidence=a.min_det, min_tracking_confidence=0.5,
                          model_complexity=a.complexity) as pose, open(tmp, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(HEADER)
            while True:
                if a.max_frames and i >= a.max_frames:
                    break
                due = int(sim_t * vfps) if a.mode == "realtime" else i   # frame the camera would deliver now
                if i < due:                                             # the device was busy: this frame is lost
                    if not cap.grab():
                        break
                    w.writerow([i, ""] + [""] * 29 + [1, "", f"{sim_t:.4f}"])
                    n_skip += 1; i += 1
                    continue
                ok, frame = cap.read()
                if not ok:
                    break
                t0 = time.perf_counter()
                img = cv2.cvtColor(to_4_3(frame, size), cv2.COLOR_BGR2RGB)
                img.flags.writeable = False
                res = pose.process(img)
                row = landmark_row(i, res)
                if a.extra_ms:
                    time.sleep(a.extra_ms / 1000.0)
                dt = time.perf_counter() - t0
                proc.append(dt); n_proc += 1
                if a.mode == "realtime":
                    sim_t = max(sim_t, i / vfps) + dt                   # frame i arrived at i/vfps, done dt later
                w.writerow(row + [0, f"{dt * 1000:.2f}", f"{sim_t:.4f}"])
                i += 1
                if n_proc % 300 == 0:
                    fh.flush(); temps.append(cpu_temp())
                    print(f"{base}: frame {i}/{n}, {1.0 / np.mean(proc[-300:]):.1f} fps, {temps[-1]} C", flush=True)
        cap.release()
        os.replace(tmp, out)
        p = np.array(proc) * 1000 if proc else np.array([np.nan])
        summary.append(dict(video=base, frames=i, processed=n_proc, skipped=n_skip, video_fps=vfps,
                            proc_ms_mean=float(p.mean()), proc_ms_p50=float(np.percentile(p, 50)),
                            proc_ms_p95=float(np.percentile(p, 95)), proc_ms_max=float(p.max()),
                            pipeline_fps=float(1000.0 / p.mean()), wall_s=time.time() - wall0))
        print(f"done {base}: {n_proc} processed, {n_skip} skipped, {summary[-1]['pipeline_fps']:.1f} fps", flush=True)
    tt = [t for t in temps if t is not None]
    info.update(finished=time.strftime("%Y-%m-%d %H:%M:%S"), temp_end_c=cpu_temp(),
                temp_max_c=max(tt) if tt else None, throttled_end=sh(["vcgencmd", "get_throttled"]),
                peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, videos=summary)
    if summary:
        allp = sum(s["processed"] for s in summary)
        info["overall_pipeline_fps"] = float(allp / sum(s["processed"] * s["proc_ms_mean"] / 1000 for s in summary))
        info["overall_skip_fraction"] = float(sum(s["skipped"] for s in summary) / max(1, sum(s["frames"] for s in summary)))
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(info, fh, indent=1)
    print(f"\nwrote {out_dir}/summary.json  | overall {info.get('overall_pipeline_fps', float('nan')):.1f} fps, "
          f"peak RAM {info['peak_rss_mb']:.0f} MB, skipped {100 * info.get('overall_skip_fraction', 0):.1f}%")


if __name__ == "__main__":
    main()
