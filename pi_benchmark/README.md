# Pi 5 replay benchmark (no camera, no person, no ethics question)

Runs the REHAB24-6 squat videos through the deployed pose pipeline **on the Pi** and records what the device
computes and how fast. Claude then re-runs the rep-counting analysis on the Pi's own output.

## One-time setup on the Pi (about 15 min, mostly download)
```bash
mkdir -p ~/isvr_bench && cd ~/isvr_bench
# 1. the videos (2.5 GB zip; only the Ex6 squat folder is used, 440 MB)
wget -c "https://zenodo.org/records/13305826/files/videos.zip?download=1" -O videos.zip
unzip videos.zip "Ex6/*"                 # creates ./Ex6 with 18 .mp4 files
# 2. put pi_replay_bench.py in this folder
```
Use the same Python environment that runs main_v2.py (mediapipe 0.10.14, opencv, numpy). Close the browser and
anything else heavy; plug in the official power supply; fan/cooler as you normally run it.

## Run (about 35-45 min each at ~20 fps)
```bash
python pi_replay_bench.py Ex6 out --mode all        # every frame: reproduces the offline numbers on ARM
python pi_replay_bench.py Ex6 out --mode realtime   # drops frames like a live 30 fps camera would
zip -r pi_bench_out.zip out
```
Quick smoke test first (1 min): `python pi_replay_bench.py Ex6 out_test --mode realtime --videos PM_022 --max-frames 300`

Interrupted? Just run the same command again: finished videos are skipped.

## Send back
`pi_bench_out.zip` (attach it in the chat, or commit it to a branch of CV-Fitness-Tracker-Upgrade).

## What goes in the abstract from this
- on-device pipeline fps, p95 per-frame time, peak RAM, CPU temperature for the submitted configuration
- whether the Pi reproduces the offline knee angles (max difference in degrees)
- rep-counting recall at the device's real frame rate (fixed vs calibrated)
Label: "replayed dataset video on the device; camera, lens and feedback loop not exercised".
