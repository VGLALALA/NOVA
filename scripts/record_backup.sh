#!/usr/bin/env bash
# Backup recording notes for the live dashboard demo.
# The mesh will fail for reasons that are not our code. Always have a take.
set -euo pipefail

cat <<'EOF'
NOVA dashboard backup recording
================================

1. On the projector machine:
     nova start --dummy
     # or the real coordinator once workers are up
     open http://<lan-ip>:8080   (printed on start)

2. Start recording BEFORE `nova run`.

macOS (simplest):
  Cmd+Shift+5  →  record selected window (the browser, full-screen)

macOS QuickTime:
  osascript -e 'tell application "QuickTime Player" to activate'
  osascript -e 'tell application "QuickTime Player" to start new screen recording'
  # stop from the menu bar extra, then export

ffmpeg, macOS display (avfoundation index varies — list with ffmpeg -f avfoundation -list_devices true -i ""):
  ffmpeg -f avfoundation -framerate 30 -i "1:none" -t 180 -c:v h264 -pix_fmt yuv420p backup-dashboard.mp4

ffmpeg, Linux:
  ffmpeg -f x11grab -framerate 30 -i "${DISPLAY:-:0.0}" -t 180 -c:v libx264 -pix_fmt yuv420p backup-dashboard.mp4

3. Demo flow to capture (~2 minutes talking, generation during the talk):
     nova run demo/gallery.yaml
     wait until ~40% of tiles are filled
     Ctrl+C one worker (do NOT close the lid)
     watch NODE_DISCONNECTED + requeue in the event feed
     job finishes 24/24

4. If live GPUs are dead, drop pre-generated PNGs in demo/tiles/ and
   keep a screen recording of a successful dummy or mixed-vendor run.

Output: backup-dashboard.mp4 next to this script, or wherever you save the OS recorder.
EOF
