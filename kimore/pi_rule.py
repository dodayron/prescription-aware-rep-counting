"""Re-implementation of Aaron's Pi rep-counting rule (exercise_v2.py, CV-Fitness-Tracker-Upgrade).

Checked line by line against exercise_v2.py (commit 23ffae2):
  - angle = |atan2(c-b) - atan2(a-b)| in degrees, folded to [0,180]
  - a side is used only if min(visibility of the 3 points) >= 0.6
  - returns max(angles) over the usable sides; None if no side usable
  - two-state machine rest -> peak -> rest; rep counted on return to rest
  - for direction "decrease": peak if angle < peak_thresh; rest if angle > rest_thresh
This file does NOT depend on mediapipe, so it runs on Kinect joints or BlazePose output alike.
"""
import numpy as np

VIS_THRESH = 0.6


def angle_2d(a, b, c):
    """a, b, c: (x, y) arrays. Returns the angle at b in degrees, in [0, 180]."""
    a, b, c = np.asarray(a, float), np.asarray(b, float), np.asarray(c, float)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    ang = abs(radians * 180.0 / np.pi)
    if ang > 180.0:
        ang = 360.0 - ang
    return ang


def frame_angle(sides, vis_thresh=VIS_THRESH):
    """sides: list of ((p1,p2,p3), (v1,v2,v3)) per side. Mirrors Exercise.get_angle."""
    angles = []
    for pts, vis in sides:
        if min(vis) < vis_thresh:
            continue
        angles.append(angle_2d(*pts))
    if not angles:
        return None
    return max(angles)


class RepCounter:
    """Mirrors Exercise.update for a non-curl, non-lateral-raise exercise (i.e. squat)."""

    def __init__(self, rest, peak, direction="decrease"):
        self.rest_thresh = rest
        self.peak_thresh = peak
        self.direction = direction
        self.stage = "rest"
        self.reps = 0

    def update(self, angle):
        if angle is None:
            return None
        if self.direction == "increase":
            if angle > self.peak_thresh:
                self.stage = "peak"
            if angle < self.rest_thresh and self.stage == "peak":
                self.stage = "rest"
                self.reps += 1
        else:
            if angle < self.peak_thresh:
                self.stage = "peak"
            if angle > self.rest_thresh and self.stage == "peak":
                self.stage = "rest"
                self.reps += 1
        return angle


def count_reps(angle_series, rest, peak, direction="decrease"):
    """angle_series: iterable of float or None (per frame). Returns (reps, rep_frame_indices)."""
    rc = RepCounter(rest, peak, direction)
    frames = []
    for i, a in enumerate(angle_series):
        before = rc.reps
        rc.update(a)
        if rc.reps > before:
            frames.append(i)
    return rc.reps, frames
