# Branch `feature/calibration-isvr2027` — what changed and how to test it

Default behaviour is unchanged: `python main_v2.py` with no flags, and the unmodified curl / lateral-raise
entries in `exercises_v2.json`, run exactly as before (1000x780, fixed thresholds, old LED colours, no log).

## New
| File | Change |
|---|---|
| `exercise_v2.py` | Optional `prescription` and `calibration` config keys. `mode` = `calibrating` → `counting`. Calibration: buffer the per-frame angle, median-filter (5), find squat minima (prominence 15°, min spacing 15 frames, pure numpy), take the first `cycles` (2) minima, depth = their mean, stand = 90th percentile of the buffer up to the 2nd minimum; once the user is back up, `peak = depth + margin·ROM`, `rest = stand − margin·ROM` (margin 0.25). Depth is clamped into the physio band before the margin is applied (`clamp_to_prescription`), so a user cannot calibrate to a non-prescribed depth. If no two cycles appear within `max_frames` (900) it falls back to the fixed thresholds. `band_status(angle)` → `above` / `inside` / `deeper`. Also exposes `last_side_angles` and `last_mean_visibility` for the logger. Rep-counting state machine, `max` over sides and the 0.6 visibility gate are untouched. |
| `exercises_v2.json` | Squat gains `"prescription": {"min_flexion": 60, "max_flexion": 70}` (joint angle 110–120°, matching the KIMORE prescription; the physio sets this per exercise in clinical use) and `"calibration": {"enabled": false, ...}`. Enable per exercise here, or for all with `--calibrate`. |
| `main_v2.py` | Flags: `--tested` (640x480, the thesis test configuration), `--res WxH`, `--calibrate`, `--log PATH`, `--no-window`. Calibration prompt (TTS file if present + on-screen text + dim white bar strip), then a beep and a console line with the learned thresholds. Progress bar: green inside the band, amber deeper; white ticks mark the band on the bar. Bar x-position now follows the resolution. |
| `led_manager.py` | `update_bar(per, band_status=None)`: blue → green inside band → amber deeper; unchanged when `band_status` is None. `calibration_blink(on)`. |
| `sound_manager.py` | Loads `calibration instructions p10.wav` if it exists ("Perform two squats at your prescribed depth"). Generate it with the same TTS voice as the other prompts. |
| `frame_logger.py` | Per-frame CSV: timestamp, exercise, mode, angle, left/right knee angle, state, reps, mean visibility, band status, thresholds, fps. For the goniometer, lux and latency tests. |
| `requirements.txt` | Pinned (`mediapipe==0.10.14`). |
| `test_calibration_offline.py` | Runs without camera/LEDs. `python test_calibration_offline.py [KIMORE JointPosition.csv ...]`. |

## Test on the Pi
```
python main_v2.py                                   # must behave exactly as before
python main_v2.py --tested --log logs/t1.csv        # 640x480 + CSV
python main_v2.py --tested --calibrate --log logs/c1.csv
```
With `--calibrate`: do two squats at your prescribed depth, wait for the beep, then count. Check the console
line (depth / stand / thresholds, "clamped" if the band pulled it). Then try: shallow squats that the fixed
90° peak would miss (should now count); squats deeper than the band (should count, bar/LEDs amber).

## Known differences from the offline evaluation
The on-device detector runs on the buffer available at the moment the user returns to standing after the
2nd rep; the offline script (`calib_eval.py`) runs `scipy.find_peaks` on the whole recording. On three KIMORE
recordings two gave identical thresholds and one (a stroke patient) differed by ~10° in learned depth. Treat the offline
numbers as the evaluation; on-device agreement is to be measured in P4.

Then tag: `git tag v2.1-isvr2027 && git push --tags`.
