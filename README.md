# Prescription-aware repetition counting for home rehabilitation

Offline evaluation code, results and figures for an ISVR 2027 (RehabWeek, Lausanne) abstract submission:

> **Fixed-threshold repetition counting misreports home-exercise adherence in patients: a prescription-aware counter evaluated against clinician scores and motion capture**
> Aaron Abraham (presenting); co-authors and affiliations to be confirmed.

The device itself (Raspberry Pi 5, MediaPipe BlazePose lite, LED and audio feedback) lives in
[dodayron/CV-Fitness-Tracker-Upgrade](https://github.com/dodayron/CV-Fitness-Tracker-Upgrade). This repository holds
everything used to evaluate its repetition-counting rule, plus the calibration patch for that repository.

**Status:** preliminary offline technical evaluation on public datasets. Not a clinical validation. Nothing here
was run on the device except the frame-rate figures quoted from the thesis; `pi_benchmark/` is the script for that.

## Main results (squat)

| Question | Dataset | Result |
|---|---|---|
| Does a fixed (160°/90°) counter misreport patients? | KIMORE, 31 patients, simulated side view | Logged zero reps for 58% [42, 74] of patients; those 18 made 50 attempts, 28 at prescribed depth |
| Adherence vs fidelity | KIMORE | Patients: fixed counter 35 reps; calibrated counter 112 attempts, 87 at prescription (fidelity 78% [64, 89]); healthy 99% |
| Does fidelity track clinician scores? | KIMORE, n = 73 | Spearman ρ = 0.44 [0.19, 0.62] |
| Counting accuracy vs annotated reps | REHAB24-6, profile view, lite model | Recall 54% (fixed) → 97% [93, 100] (calibrated), no false positives |
| Knee-angle accuracy vs motion capture | REHAB24-6, profile view, lite model | RMSE 15.8° raw → 4.5° [4.2, 4.9] with a linear correction learned on other participants |

Intervals are subject-level bootstrap 95% CIs (2,000 resamples). Full tables, sensitivity analyses and limitations:
[`kimore/RESULTS.md`](kimore/RESULTS.md) and [`rehab24/RESULTS.md`](rehab24/RESULTS.md).

## Layout

```
kimore/            KIMORE (clinical transfer): rep rule, calibration, fidelity
  pi_rule.py         re-implementation of the device rule (checked against exercise_v2.py); no mediapipe needed
  kimore_io.py       Kinect v2 skeleton loading, side/front projection, knee angle
  calib_eval.py      fixed vs prescription vs auto-calibrated thresholds, bootstrap CIs
  fidelity_eval.py   per-repetition attempts vs attempts at prescribed depth
  results/           aggregate outputs only (see Data below)
  RESULTS.md
rehab24/           REHAB24-6 (measurement accuracy): BlazePose vs motion capture and annotated reps
  run_blazepose_rehab.py   video -> BlazePose -> per-frame knee angles (device settings)
  analyze_rehab.py         angle accuracy, rep counting, depth vs correctness labels
  offset_test.py           angle-correction methods (constant / linear, LOSO / per-user)
  results/                 lite_* (model_complexity=0, as deployed) and full_* (model_complexity=1)
  RESULTS.md
pi_benchmark/      replay the REHAB24-6 videos on the Pi 5 (no camera, no person): fps, RAM, reproduction
device_patch/      auto-calibration, prescription band and per-frame logging for CV-Fitness-Tracker-Upgrade
figures/           figure scripts and outputs (v2 = current abstract figure)
docs/              abstract text drafts, bench measurement protocol
```

## Reproducing

Python 3.11, `mediapipe==0.10.14` (last series with `mp.solutions.pose`), `opencv-python-headless`, `numpy`,
`pandas`, `scipy`, `scikit-learn`, `matplotlib`, `openpyxl`. See `requirements.txt`.

```bash
# KIMORE (needs the Kinect skeleton release; see Data)
python kimore/calib_eval.py    <Kimore root> kimore/results/side_i05_m25 --view side --inferred 0.5 --margin 0.25
python kimore/fidelity_eval.py <Kimore root> kimore/results/fidelity

# REHAB24-6 (needs videos.zip, 3d_joints.zip, 2d_joints.zip, Segmentation.csv from Zenodo)
python rehab24/run_blazepose_rehab.py <rehab root>/videos/Ex6 <bp_out> --complexity 0 --size 640x480
python rehab24/analyze_rehab.py <rehab root> <bp_out> rehab24/results/lite --suffix _c0
python rehab24/offset_test.py   <rehab root> rehab24/results/lite

# Figures (read the aggregate results in this repo)
python figures/make_figure_v2.py
```
Both analyses were re-run from this repository's copy and reproduce the committed results exactly.

## Data and licences

- **KIMORE** (Capecci et al., IEEE TNSRE 2019) is released under an academic, non-commercial EULA that does not allow
  redistribution. This repository contains **no KIMORE data and no per-subject outputs**, only aggregate
  statistics. Request access from the dataset authors. The Kinect skeletons used here came from the copy linked by
  the authors at `ulkudilemre/a-deep-learning-framework-for-assesing-physical-rehabilitation-exercises`.
- **REHAB24-6** (Černek, Sedmidubský & Budíková, Zenodo [13305826](https://zenodo.org/records/13305826)) is
  CC BY-NC 4.0. The derived per-frame and per-repetition tables in `rehab24/results/` are shared under the same
  licence with attribution. Raw videos and motion capture are not included; download them from Zenodo.
- Code: see `LICENSE` (to be chosen before the repository is made public).

## Citation

Abstract under review at ISVR 2027. Cite the tagged release once one exists.
