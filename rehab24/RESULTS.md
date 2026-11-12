# P1 — REHAB24-6 squat (Ex6): BlazePose + Pi rule vs motion capture and annotated reps

Generated 10 Nov 2026 by `run_blazepose_rehab.py` + `analyze_rehab.py` (outputs in `out/full_*.csv`).
**Offline evaluation**. Headline numbers below are the BlazePose **full** model (`model_complexity=1`, `out/full_*`);
the **lite** rerun (`model_complexity=0`, 11 Nov, `out/lite_*`, model file from Aaron's Pi) is the one quoted in the abstract — see §0. `mediapipe==0.10.14`, legacy `mp.solutions.pose`,
`min_detection_confidence=0.8`, `min_tracking_confidence=0.5`, frames preprocessed to 640×480 4:3
(Camera17 landscape centre-cropped; Camera18 portrait padded with black bars → subject ≈290 px tall, similar to
the Pi Wide camera at 3 m). Rule = `exercise_v2.py` re-implemented (`pi_rule.py`): 2D image angle, max over
sides with all three landmarks ≥ 0.6 visibility, rest→peak→rest.

Data: 9 videos, 9 participants (ids 2–9 and 1), 195 annotated reps (69 % correct), 30 fps, 2 cameras. Views
per rep: Camera17 front (98 reps) / half-profile (97); Camera18 is orthogonal → profile (98) / half-profile (97).
Two videos (PM_008, PM_022; 27 profile reps) were recorded with lights off. Subjects wear a dark mocap suit
with markers. Sync: video frame i ↔ joint sample i; cross-correlation lag 0 (cam18) / −1 frame (cam17), not
corrected (33 ms). Mocap reference = 3D hip–knee–ankle angle from the OptiTrack skeleton, max over legs
(same combination rule as the system). 2D-projected mocap joints (dataset-supplied) give a second reference
that isolates projection from pose error.

## 0. Lite vs full (profile view, the deployed configuration; 11 Nov)

| Metric (profile) | Full c1 | **Lite c0** |
|---|---|---|
| Frame bias / RMSE vs mocap 3D | +11.9° / 13.1° | **+14.3° / 15.7°** |
| Per-rep depth bias [95% CI] | +15.8° [15.1, 16.6] | **+19.5° [18.7, 20.3]** |
| Fixed 160/90 recall | 63% [47, 83] | **54% [36, 76]** |
| Prescription 160/120 recall | 100% | **99%** |
| Auto-calibrated recall / precision | 92% [80, 100] / 100% | **97% [93, 100] / 100%** |
| Calibrated exact-count sessions | 9/10 | **9/10** |
| Depth AUC (deeper → incorrect), BP | 0.86 | **0.86** |
| Detection / usable rate | 100% | **100%** |

Half-profile lite: calibrated recall 95% [89, 99], precision 97%; front lite: 81% [66, 96], precision 93%.
Lite reads the knee ~2.5° straighter than full; counting is unchanged or slightly better.

## 0b. Angle correction (11 Nov evening; `offset_test.py`, `out/lite_offset.csv`, `out/full_offset.csv`)

The profile-view error is mostly systematic and close to linear in the angle, so a fixed linear map removes most of it.
Test frames: repetitions 3+ of every session; person-level bootstrap CIs. Lite model:

| Profile view, lite | Bias | RMSE [95% CI] | 95% limits of agreement |
|---|---|---|---|
| Raw 2D BlazePose angle | +14.4° | 15.8° [14.6, 16.9] | +1.7 to +27.1 |
| Constant offset, learned on the other participants (LOSO) | +0.1° | 6.6° [5.9, 7.3] | −12.7 to +13.0 |
| **Linear map, learned on the other participants (LOSO)** | **+0.1°** | **4.5° [4.2, 4.9]** | **−8.8 to +8.9** |
| Constant offset from the user's own first 2 reps (needs a reference) | +1.0° | 6.4° | −11.4 to +13.5 |
| Linear map from the user's own first 2 reps (needs a reference) | +0.3° | 4.5° [3.7, 5.6] | −8.6 to +9.2 |

- Full model, LOSO linear: 13.2° → 4.1° [3.5, 4.7].
- Per-rep depth after LOSO linear (lite): bias −2.2°, RMSE 6.0° (raw +19.5°, 19.9°), n = 98.
- All-data profile fit (lite): `true ≈ 1.162 × blazepose − 36.9`. BlazePose compresses depth: a BlazePose reading of
  90° is a true knee angle of about 68° (112° flexion); 120° reads as about 103°. So the nominal "90°" fixed threshold
  demands a far deeper squat than its label says, which is part of why fixed thresholds under-count.
- A population-level map needs **no per-user step** and does as well as a per-user one here.
- Off-axis views stay poor even after correction (half-profile 21°, front 28° RMSE): correction does not replace side-on placement.
- **Limits:** learned and tested on one dataset (9 healthy adults, two fixed lab cameras); transfer to the Pi Camera
  Module 3 Wide and to patients is untested. Counting with calibrated thresholds does not depend on this correction,
  but the band/fidelity check on the device does (it compares absolute angles), so the device needs it.

## 1. Knee-angle accuracy (frames inside annotated reps; detection 100 %, usable angle 100 % in all views)

| View (camera) | Frames | Bias (BP − mocap 3D) | RMSE | Bland–Altman LoA | RMSE vs 2D-projected mocap | Projection-only RMSE |
|---|---|---|---|---|---|---|
| **Profile (cam18)** | 9,825 | **+11.9°** | **13.1°** | +1.4 to +22.4° | 11.1° | 4.0° |
| Half-profile (both) | 19,096 | +30.9° | 37.0° | −9.0 to +70.7° | 12.7° | 33.5° |
| Front (cam17) | 9,825 | +44.4° | 52.5° | −10.4 to +99.3° | 5.4° | 50.8° |

Per-rep depth (minimum angle in the rep), profile: bias +15.8° [95 % CI 15.1, 16.6], RMSE 16.2°, LoA +8.7 to
+22.9°, n = 98 reps. BlazePose reads the knee straighter than the mocap skeleton by a near-constant offset
(per-person bias 8–14°; lights off 12.4° vs on 11.6°). In profile the far leg passes the visibility gate in only
6 % of frames, so the system is effectively using the near leg. Front and half-profile errors are dominated by
projection (the 2D image angle is not the knee angle), not by BlazePose.

## 2. Rep counting (sessions = orientation segment × camera; calibrate on the first 2 annotated reps, count the rest; event within [rep start, rep end + 0.5 s] = TP)

| View | Sessions / test reps | Fixed 160/90 recall | Prescription 160/120 recall | **Auto-calibrated recall** | Calib. precision | Calib. exact-count sessions |
|---|---|---|---|---|---|---|
| **Profile** | 10 / 78 | 63 % [47, 83] | 100 % | **92 % [80, 100]** | 100 % | 9/10 |
| Half-profile | 18 / 158 | 20 % [9, 31] | 63 % [47, 78] | **96 % [93, 99]** | 99 % | 13/18 |
| Front | 10 / 78 | 3 % [0, 5] | 15 % [4, 31] | **76 % [63, 91]** | 86 % | 4/10 |

Sessions with zero reps counted: fixed 0 % profile / 56 % half-profile / 80 % front; calibrated 0 % everywhere.
No false positives with fixed or prescription thresholds in any view; calibrated: 1 FP half-profile, 10 FP front.
All 6 calibrated misses in profile come from one session (PM_008, lights off, 15 test reps): calibration reps at
~94°, then seven reps at 114–119° sitting on the calibrated peak threshold (115°) — a drift-shallower case
the 25 % margin does not absorb. Prescription thresholds (peak 120°) got 100 % in profile because these healthy
subjects all squat past 120°; the KIMORE patients did not (P2).

## 3. Quality: per-rep depth vs the physiotherapist's correct/incorrect label

In this dataset the **incorrect squats are deeper**, not shallower: within-person, incorrect reps bottom out
20.1° [14.2, 27.2] deeper than correct ones by mocap (6 persons with both labels; 3 persons have no incorrect
reps). BlazePose profile depth reproduces this: within-person difference −20.4° [−26.8, −14.4], AUC 0.86
(mocap 0.87) for deeper → incorrect. Half-profile: AUC 0.67 (mocap 0.73). Front: AUC 0.53 (no information).
Implication for the abstract: depth carries the quality signal, but its *direction* is prescription-specific —
support for a physio-set band with an "deeper than prescribed" state rather than "deeper is better".
(KIMORE patients: deeper ↔ higher clinician score; REHAB24-6 healthy: deeper ↔ labelled incorrect.)

BlazePose per-rep depth vs mocap depth across all reps and views: Spearman ρ = 0.49 (profile only: see §1).

## 4. Robustness
Detection and usable-angle rates were 100 % in every view, including lights-off videos — a single subject in a
lab with a plain background. This dataset does not test clutter, occlusion or distance; the thesis robustness
tests and the P4 lux sweep cover those.

## Caveats to state
- Full model, not lite; offline; 640×480 preprocessed frames, not the Pi camera.
- Mocap skeleton joint centres ≠ BlazePose landmark definitions: part of the +12° offset is definitional.
- 9 participants, healthy, mocap suit; two lights-off recordings.
- Calibration reps are the first two annotated reps (always labelled correct); on-device the user chooses them.
- CC BY-NC 4.0; cite Černek, Sedmidubský & Budíková, SISAP 2024 (and the 2025 Information Systems paper).
