# P4 bench protocol — on-device measurements (Aaron only, evenings 11–13 Nov)

Setup for all tests unless stated: Pi 5, Camera Module 3 Wide, `python main_v2.py --tested --log logs/<test>.csv`
(640×480, the thesis configuration), subject side-on at 3 m, same room and camera height (~1 m) each time.
Record for every test: date/time, Pi model and RAM, mediapipe version (`pip show mediapipe`), camera module,
distance (tape), lux at the subject (phone lux app at chest height, facing the camera), clothing (fitted).
Keep every CSV; the per-frame log has angle, both knee angles, state, reps, visibility, thresholds, fps.

## 1. Peak RAM and fps (5-minute session)
1. Start `vmstat 1 > logs/vmstat_<run>.txt &` and `python main_v2.py --tested --log logs/ram_<run>.csv`.
2. Squat continuously at a comfortable pace for 5 min (rest in frame between sets of 10).
3. In a second terminal every 30 s: `ps -o rss= -p $(pgrep -f main_v2)` → note the max (kB).
4. Repeat once with Flask on (`python app.py &` first) and once without. Also once with `--calibrate`.
Report: peak RSS (MB), fps median/IQR from the CSV `fps` column, CPU temperature (`vcgencmd measure_temp`)
at start and end. Minimum board = smallest Pi 5 RAM tier whose capacity exceeds peak RSS with ≥30% headroom.

## 2. Camera-to-LED latency (≥50 reps)
1. Phone at 240 fps (slow-motion) framing BOTH you and the LED strips in one shot; the strips must be visible
   through the whole squat.
2. Do 5 sets of 10–12 squats, standard depth, `--tested`. Vary the pace: slow set, normal, fast.
3. Afterwards, in the slow-mo clip, for each rep find (a) the first frame in which your knees stop rising
   (standing, rep complete) and (b) the first frame in which the rep strip changes. Latency = (b − a)/240 s.
   If a is hard to judge, use the frame in which the progress bar strip first drops to 0 % instead and say so.
Report: n, median, IQR, p95 (ms). Also note the frame rate from the CSV during the clip.

## 3. Goniometer static holds
Helper holds a goniometer on the lateral knee (axis on the lateral femoral epicondyle, arms along femur and
fibula). You hold a squat at the goniometer readings ~170, 150, 130, 110, 90° — 3 holds each, 5 s each —
first side-on at 3 m, then at 45° to the camera, then repeat side-on with the standard (non-Wide) camera
if available. For each hold: helper reads the goniometer aloud, you mark the time (say it, or press a key you
note), and the CSV `angle`, `angle_left`, `angle_right` over the middle 3 s of the hold are the measurement.
Report per orientation: MAE, bias and Bland–Altman limits (system − goniometer), n = 15 holds each.
State: goniometer model and resolution, who read it, that one subject (you) was measured.

## 4. Lux sweep
Side-on at 3 m, `--tested`. Target lux at the subject: ~500, 300, 150, 75, 30, 10 (dim the room lights,
curtains, lamp with dimmer; measure before each set). At each level do 8 squats to the same depth
(use the audio beep; pause 2 s between reps). Then repeat the 30 and 10 lux levels with `--calibrate`.
Report per level: measured lux, reps counted / 8, mean `mean_visibility`, fraction of frames with a usable
angle, fps. Reliability threshold = lowest lux with 8/8 and ≥90 % usable frames.

## 5. (If time) calibration on-device check
`--tested --calibrate`: 5 sessions, each: 2 calibration squats at a self-chosen depth (vary: shallow,
normal, deep), beep, then 10 squats at the same depth. Report counted/10 per session and the console
threshold line. This is the only check of the online calibration code against the offline numbers.

Order: 1 → 4 → 3 → 2 → 5. Items 1 and 4 need only you; 3 needs the helper; 2 needs a phone stand.
Whether 3 and 4 go into the abstract depends on fecrec@dcu.ie clearing self-experimentation by Fri 13 Nov.
