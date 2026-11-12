# P2 — KIMORE Ex5 (squat) offline evaluation with subject-level bootstrap 95% CIs

Generated 10 Nov 2026 by `calib_eval.py` (2,000 percentile-bootstrap resamples over recordings; one recording
per subject, so subject-level). All runs are **offline on Kinect v2 skeletons**. "Side" = orthographic
projection of Kinect 3D joints onto the sagittal plane = **simulated** side camera. Rule = Aaron's
`exercise_v2.py` (max over sides, visibility ≥ 0.6, rest→peak→rest), re-implemented in `pi_rule.py`.

Inventory: 385 recording folders, 381 parse OK ([subject] Es1, [subject] Es5, [subject] Es1, [subject] Es1 fail), [subject] has no
clinical sheet. Ex5 usable = 75 (42 healthy: 16 Expert + 26 NotExpert; 33 patients: 8 BackPain, 16 Parkinson,
9 Stroke). Side view excludes [subject] (no frame with all three knee landmarks usable) and [subject] (no detectable
cycle) → **n = 73 (42 healthy, 31 patients)**. Repo subject counts 17 Expert / 27 NotExpert differ from the
paper's 12 / 32 — still unresolved (ask Luca Romeo).

Protocol: calibrate on the first 2 detected squat minima (median-filter 5, `find_peaks` prominence 15°,
distance 15), test on the remainder. KIMORE protocol is 5 squats, so 3 expected; several Experts do 6 cycles,
so the ±1 rate and the match-to-detected-cycles rate are the fair metrics.

## Main table — side view, inferred joints → 0.5, margin 25% (`out/side_i05_m25_*`)

| Thresholds | Cohort | n | Zero reps | 2–4 reps (±1 of 3) | Count = detected cycles | Missed reps / cycle |
|---|---|---|---|---|---|---|
| Fixed 160/90 | patients | 31 | **58% [42, 74]** | 26% [10, 42] | 16% [3, 29] | 0.69 |
| Fixed 160/90 | healthy | 42 | 7% [0, 17] | 88% [79, 98] | 74% [60, 86] | 0.14 |
| Prescription 160/120 | patients | 31 | 35% [19, 55] | 55% [39, 71] | 45% [29, 65] | 0.42 |
| Prescription 160/120 | healthy | 42 | 0% [0, 0] | 93% [86, 100] | 88% [79, 98] | 0.04 |
| Auto-calibrated | patients | 31 | **0% [0, 0]** | **81% [65, 94]** | 77% [61, 90] | 0.08 |
| Auto-calibrated | healthy | 42 | 0% [0, 0] | 95% [88, 100] | 90% [81, 98] | 0.03 |

Spearman ρ(reps counted, clinical PO), all 73: fixed 0.38 [0.16, 0.59] p = 0.001; prescription 0.26 [0.01, 0.50]
p = 0.028; calibrated 0.01 [−0.22, 0.27] p = 0.92. (Counting with fixed thresholds "tracks" the clinician score
only because it fails on shallow squatters; calibration removes that confound, as intended.)

**Quality carrier — calibrated depth vs clinical PO**: ρ = **−0.31 [−0.54, −0.07]**, p = 0.008, n = 73
(deeper squat = higher score). Median calibrated knee angle: healthy 80°, patients 101°. Within cohort the
correlation is not significant (patients ρ = −0.14 [−0.55, 0.29], n = 31; healthy −0.09, n = 42) — the
association is largely between-cohort; say so.

Band classification (60–70° flexion = 110–120° joint angle, calibrated depth): healthy 40 deeper / 1 inside /
1 shallower; patients 18 deeper / 4 inside / 9 shallower. Median cPO: patients shallower 7.0, inside 9.0,
deeper 10.8. Almost everyone squats deeper than the KIMORE prescription, so the band is a poor binary split
on this dataset; report continuous depth as the quality metric and the band as the clinical mechanism.

Expert-inflation check (exact-3 rate): NotExpert vs patients — fixed 73% vs 13%; prescription 77% vs 39%;
calibrated 77% vs 45%. The healthy–patient gap survives without Experts.

By pathology (exploratory, n = 7–26): zero-rep rate fixed → calibrated: BackPain 29% → 0%, Parkinson 75% → 0%,
Stroke 50% → 0%. Median calibrated depth: BackPain 80°, Parkinson 110°, Stroke 112°.

## Sensitivity

| Run | Calib. patients zero | Calib. patients ±1 | Calib. healthy ±1 | ρ depth–cPO |
|---|---|---|---|---|
| side, 0.5, 15% | 6% [0, 16] | 61% [45, 77] | 95% [88, 100] | −0.31 |
| **side, 0.5, 25%** | 0% | 81% [65, 94] | 95% [88, 100] | −0.31 [−0.54, −0.07] |
| side, 0.5, 35% | 0% | 74% [58, 87] | 93% [83, 100] | −0.31 |
| side, 0.7, 15% | 9% [0, 22] | 62% [47, 78] | 93% [83, 100] | −0.28 |
| side, 0.7, 25% | 0% | 84% [72, 97] | 95% [88, 100] | −0.28 [−0.52, −0.01] |
| side, 0.7, 35% | 0% | 84% [72, 97] | 90% [81, 98] | −0.28 |
| front, 0.5, 25% | 5% [0, 14] | 57% [38, 76] | 83% [69, 94] | −0.10 [−0.41, 0.20] n.s. |

Fixed and prescription rows do not depend on the margin. Front view: only 56 recordings have ≥3 detectable
cycles (n = 21 patients, 35 healthy); fixed-threshold zero-rate 76% patients / 60% healthy.

## Limitations to state
- Side view is simulated from Kinect (orthographic, no far-leg occlusion, no BlazePose error) → upper bound.
- KIMORE was filmed front-on in a lab; clinician scores are questionnaire-based (EAAQ), not kinematic truth.
- Calibration and test windows come from the same recording.
- Expert healthy subjects may inflate cohort gaps; NotExpert-vs-patient comparison given above.
- 2 of 75 recordings excluded for the side view; 19 of 75 for the front view.

## Count-and-flag check (added 11 Nov; from `out/side_i05_m25_per_recording.csv`)

Question: does calibration simply count shallow squats as good reps? No. Of the 18 patient recordings in which
fixed 160/90 thresholds counted **zero** reps, the calibrated depth meets the KIMORE prescription (≥60° flexion)
in **10** — fixed thresholds rejected squats that met the prescription — and is shallower in **8** (6 Parkinson
<50°, [subject] 57°, [subject] 60°). With calibration all 18 are counted; the band flags the 8 shallow ones.
Across all 31 patients, 9 (29% [13, 45]) are counted-and-flagged shallow; median cPO 7.0 vs 10.8 for the rest.
Healthy: 1 of 42 flagged. The abstract wording: shallow attempts are counted as attempts and flagged below
prescription, not credited as prescribed repetitions.

## Per-repetition adherence vs fidelity (11 Nov evening; `fidelity_eval.py`, `out/fidelity_*`)

This SUPERSEDES the "10 of 18" count above (that used the mean depth of the two calibration squats; this uses
every counted repetition in the test window). Simulated side view, inferred = 0.5, margin 25%, no band clamp.

| Cohort | n | Fixed-counter reps logged | Attempts (calibrated) | Attempts at prescription (≥60° flexion) | Fidelity [95% CI] |
|---|---|---|---|---|---|
| Patients | 31 | 35 | 112 | 87 (12 inside band, 75 deeper) | 78% [64, 89] |
| Healthy | 42 | 123 | 139 | 137 | 99% [96, 100] |

- Patients the fixed counter logged as **zero**: 18 of 31 (58% [42, 74]). They made 50 attempts; **28 met the
  prescription**. 9 of the 18 had ≥1 repetition at prescription (7 had all), 9 had none.
- "Misreported" (fixed logs 0, yet ≥1 rep at prescription): 29% [13, 45] of patients, 7% [0, 14] of healthy.
- Patients with attempts but none at prescription: 29% [13, 45] (9: 7 Parkinson, 1 stroke, 1 back pain).
- Fidelity (share of attempts at prescription) vs clinician cPO: ρ = 0.44 [0.19, 0.62], p < 0.001, n = 73;
  **within patients ρ = 0.38 [0.00, 0.71], p = 0.03, n = 31** (exploratory, CI touches 0, no multiplicity correction).
  Stronger than calibrated depth (ρ = −0.31 overall, −0.14 within patients).

Exploratory by pathology (small groups, descriptive only):

| Group | n | Attempts | At prescription | Fidelity | Fixed logs 0 | None at prescription |
|---|---|---|---|---|---|---|
| Parkinson | 16 | 43 | 27 | 63% [38, 84] | 75% [50, 94] | 44% [19, 69] |
| Stroke | 8 | 29 | 25 | 86% [65, 98] | 50% [12, 88] | 12% |
| Back pain | 7 | 40 | 35 | 88% [64, 100] | 29% | 14% |
| Healthy non-expert | 26 | 82 | 82 | 100% | 4% | 0% |
| Healthy expert | 16 | 57 | 55 | 96% | 12% | 0% |
