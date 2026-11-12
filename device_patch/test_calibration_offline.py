"""Offline tests for the auto-calibration and prescription-band code in exercise_v2.py.

Runs without a camera or LEDs. Needs mediapipe installed (exercise_v2 imports it for the landmark enum).
    python test_calibration_offline.py                       # synthetic squats
    python test_calibration_offline.py <JointPosition.csv>   # also a KIMORE Kinect recording, side projection
"""
import json
import sys
from types import SimpleNamespace

import numpy as np

from exercise_v2 import Exercise, mp_pose

N_LM = len(mp_pose.PoseLandmark)


def landmarks_from_knee_angles(left_deg, right_deg, vis=1.0):
    """Build a 33-landmark list where each leg's hip-knee-ankle angle equals the requested value.
    Hip at (0.5, 0.3), knee at (0.5, 0.5), ankle rotated by (180 - angle) from straight."""
    lm = [SimpleNamespace(x=0.0, y=0.0, z=0.0, visibility=0.0) for _ in range(N_LM)]

    def put(name, x, y, v=vis):
        lm[getattr(mp_pose.PoseLandmark, name).value] = SimpleNamespace(x=x, y=y, z=0.0, visibility=v)

    for side, ang in (("LEFT", left_deg), ("RIGHT", right_deg)):
        hip, knee = (0.5, 0.3), (0.5, 0.5)
        th = np.radians(180.0 - ang)       # 0 = straight down
        ankle = (knee[0] + 0.2 * np.sin(th), knee[1] + 0.2 * np.cos(th))
        put(f"{side}_HIP", *hip)
        put(f"{side}_KNEE", *knee)
        put(f"{side}_ANKLE", *ankle)
    return lm


def squat_series(depths, stand=175.0, half_period=20, hold=10):
    """Angle series: stand -> each depth -> stand, cosine profile."""
    out = [stand] * hold
    for d in depths:
        t = np.linspace(0, np.pi, half_period)
        out += list(stand - (stand - d) * (1 - np.cos(t)) / 2)
        out += list(d + (stand - d) * (1 - np.cos(t)) / 2)
        out += [stand] * hold
    return np.array(out)


def run(ex, series):
    for a in series:
        ex.update(landmarks_from_knee_angles(a, a))
    return ex


def squat_cfg(**over):
    cfg = json.load(open("exercises_v2.json"))["squat"]
    cfg = json.loads(json.dumps(cfg))
    cfg.update(over)
    return cfg


def test_default_unchanged():
    ex = Exercise(squat_cfg(calibration=False))
    assert ex.mode == "counting" and ex.rest_thresh == 160 and ex.peak_thresh == 90
    run(ex, squat_series([80, 80, 100, 80]))     # 100 deg is shallower than the fixed 90 deg peak
    assert ex.reps == 3, ex.reps
    print("default (fixed 160/90): 4 squats at 80/80/100/80 deg -> reps", ex.reps, "(expected 3)")


def test_calibration_counts_user_depth():
    ex = Exercise(squat_cfg(calibration={"enabled": True, "margin": 0.25, "cycles": 2, "clamp_to_prescription": False}))
    assert ex.mode == "calibrating"
    run(ex, squat_series([112, 114, 113, 111, 115, 110]))
    print(f"calibrated (no clamp): depth {ex.calib_depth:.1f} stand {ex.calib_stand:.1f} "
          f"-> rest {ex.rest_thresh:.1f} peak {ex.peak_thresh:.1f}; reps {ex.reps} (expected 4)")
    assert ex.mode == "counting" and ex.reps == 4, (ex.mode, ex.reps)
    # the fixed rule would have missed all of these 112-115 deg squats
    ex2 = run(Exercise(squat_cfg(calibration=False)), squat_series([112, 114, 113, 111, 115, 110]))
    assert ex2.reps == 0
    print("same series with fixed 160/90 -> reps", ex2.reps)


def test_clamp_to_prescription():
    # user calibrates far shallower than the 60-70 deg flexion band (joint angle 110-120): clamp to 120
    ex = run(Exercise(squat_cfg(calibration={"enabled": True, "margin": 0.25, "cycles": 2})),
             squat_series([140, 140, 140, 140, 140]))
    print(f"clamp: learned depth {ex.calib_depth:.1f}, clamped={ex.calib_clamped}, peak thresh {ex.peak_thresh:.1f}, reps {ex.reps}")
    assert ex.calib_clamped and ex.peak_thresh < 140 and ex.reps == 0
    # user calibrates deeper than the band: clamp to 110 but the deep reps still count (deeper = amber, not a fail)
    ex = run(Exercise(squat_cfg(calibration={"enabled": True, "margin": 0.25, "cycles": 2})),
             squat_series([85, 85, 85, 85, 85]))
    print(f"clamp deep: learned {ex.calib_depth:.1f}, clamped={ex.calib_clamped}, peak thresh {ex.peak_thresh:.1f}, reps {ex.reps}")
    assert ex.calib_clamped and ex.reps == 3


def test_band_status():
    ex = Exercise(squat_cfg())
    assert ex.band_status(150) == "above" and ex.band_status(115) == "inside" and ex.band_status(100) == "deeper"
    assert Exercise(squat_cfg(prescription=None)).band_status(100) is None
    print("band status: 150 above / 115 inside / 100 deeper; no prescription -> None")


def test_visibility_gate_and_max():
    ex = Exercise(squat_cfg(calibration=False))
    lm = landmarks_from_knee_angles(80, 120)
    assert abs(ex.get_angle(lm) - 120) < 1e-6        # max over sides, as before
    lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].visibility = 0.3
    assert abs(ex.get_angle(lm) - 80) < 1e-6         # right side gated out
    print("max-over-sides and 0.6 visibility gate unchanged")


def test_calibration_timeout():
    ex = Exercise(squat_cfg(calibration={"enabled": True, "max_frames": 60}))
    run(ex, np.full(100, 175.0))
    assert ex.mode == "counting" and ex.rest_thresh == 160 and ex.peak_thresh == 90
    print("no reps within max_frames -> falls back to fixed thresholds")


def test_kimore(path):
    import pandas as pd
    df = pd.read_csv(path, header=None).dropna(axis=1, how="all").to_numpy(float).reshape(-1, 25, 4)
    pos, st = df[:, :, :3], df[:, :, 3]
    vis = np.where(st == 2, 1.0, np.where(st == 1, 0.5, 0.0))
    idx = dict(L=(12, 13, 14), R=(16, 17, 18))
    results = {}
    for label, cal in (("fixed 160/90", False), ("auto-calibrated", {"enabled": True, "clamp_to_prescription": False})):
        ex = Exercise(squat_cfg(calibration=cal))
        for t in range(len(pos)):
            lm = [SimpleNamespace(x=0.0, y=0.0, z=0.0, visibility=0.0) for _ in range(N_LM)]
            for side, (h, k, a) in idx.items():
                for name, j in (("HIP", h), ("KNEE", k), ("ANKLE", a)):
                    nm = f"{'LEFT' if side == 'L' else 'RIGHT'}_{name}"
                    lm[getattr(mp_pose.PoseLandmark, nm).value] = SimpleNamespace(
                        x=pos[t, j, 2], y=-pos[t, j, 1], z=0.0, visibility=vis[t, j])   # simulated side view (z, -y)
            ex.update(lm)
        results[label] = ex
        extra = "" if ex.calib_depth is None else f" depth {ex.calib_depth:.1f} stand {ex.calib_stand:.1f} thresholds {ex.rest_thresh:.1f}/{ex.peak_thresh:.1f}"
        print(f"KIMORE {path.split('/')[-3]}: {label}: reps {ex.reps}{extra}")


if __name__ == "__main__":
    test_default_unchanged()
    test_calibration_counts_user_depth()
    test_clamp_to_prescription()
    test_band_status()
    test_visibility_gate_and_max()
    test_calibration_timeout()
    for p in sys.argv[1:]:
        test_kimore(p)
    print("all offline tests passed")
