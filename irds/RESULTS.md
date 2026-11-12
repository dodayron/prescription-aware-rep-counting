# IntelliRehabDS (IRDS): second patient cohort, arm exercises (12 Nov 2026)

**Offline, Kinect skeletons.** Not BlazePose, not the device. Dataset: Miron et al., *Data* 2021;6(5):46; Zenodo 4610859; CC BY 4.0.
15 patients (stroke, spinal cord injury, brain injury and others) and 14 healthy controls (physiotherapists and
students); one file per repetition; two annotators labelled each repetition correct or incorrect.
Script: `irds_eval.py`; outputs `out/irds_summary.{csv,json}`, `out/irds_per_rep.csv`.

Device rules reproduced from `exercise_v2.py` at commit 23ffae2: arm curl (elbow angle < 45° and wrist above elbow)
on IRDS elbow flexion (gestures 0/1); lateral raise (hip-shoulder-elbow angle > 90°, elbow at shoulder height) on
IRDS shoulder abduction (gestures 4/5). An attempt is "registered" if the peak condition is met within the file.
Calibration: threshold from the subject's first two repetitions of that gesture (25% margin), tested on the rest.
Note: the IRDS skeleton stream is mirrored (gesture "Left" moves the joints named Right); handled in the script.

## Result: partial replication of the squat finding

| Exercise, view | Cohort | Test reps | Fixed registers (all attempts) | Calibrated | Fixed registers **annotator-correct** reps | Calibrated |
|---|---|---|---|---|---|---|
| Lateral raise, Kinect image plane (frontal camera) | Patients (14) | 164 | 65% [51, 83] | 88% [74, 99] | **77% [55, 98]** of 103 | **98% [95, 100]** |
| | Healthy (14) | 261 | 100% | 98% | 100% | 100% |
| Arm curl, true 3D elbow angle | Patients (14) | 172 | 51% [33, 75] | 85% [73, 98] | **72% [56, 87]** of 102 | **99% [97, 100]** |
| | Healthy (14) | 255 | 91% [78, 99] | 100% | 91% | 100% |
| Arm curl, Kinect image plane (frontal camera) | Patients (14) | 172 | 87% [77, 97] | 92% | 100% | 100% |
| | Healthy (14) | 255 | 100% | 100% | 100% | 100% |

Subject-level bootstrap 95% CIs (2,000 resamples).

- **Lateral raise is the clean replication**: a real frontal camera view, which is the natural view for abduction.
  The fixed rule missed about a quarter of the patient repetitions that annotators judged correct; calibration
  recovered them. Healthy controls were unaffected.
- **Arm curl depends on the view.** With the true 3D elbow angle the fixed 45° threshold missed 28% of
  annotator-correct patient repetitions. In the frontal image plane the forearm is foreshortened, the 2D angle
  collapses (median minimum 15°) and everything passes the threshold, so there is no under-counting but the angle
  carries little information (AUC vs correctness 0.60, against 0.84 in 3D). Camera placement matters for the curl.
- Smaller effect than the KIMORE squat: no patient was logged as zero overall (KIMORE squat: 58% of patients).
- Range of motion vs annotator label, patients: AUC 0.84 (curl, 3D), 0.70 (raise, 2D).

## Limits
- "Registered" is the peak half of the state machine only; files are pre-cut per repetition.
- Calibration and test are the same session; patients repeated gestures across several days in some cases.
- Annotators' professional background is not stated in the data descriptor; labels are binary.
- Patients performed seated, in wheelchairs or standing; not analysed separately.
- No prescription is defined for these gestures, so there is no "fidelity" number here; the annotator label stands in.
