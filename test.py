"""Console webcam streaming application (no Tkinter).

Features:
 - Streams frames from a webcam (default index 0)
 - Optionally displays an OpenCV window (default) OR ASCII video in console (--ascii)
 - Shows realtime FPS in console
 - Press 'q' in window mode or Ctrl+C in ASCII mode to quit
 - Optional frame capture on key 'c' (window mode) or automatic periodic capture (--capture-every N)
 - Optional recording to a video file (--record output.mp4)

Usage examples:
  python test.py                   # normal window stream
  python test.py --device 1        # use webcam 1
  python test.py --ascii --width 80 --height 45  # ASCII console stream
  python test.py --record out.mp4  # save MP4 while viewing
  python test.py --capture-every 5 # save a frame every 5 seconds
"""

from __future__ import annotations
import cv2
import time
import argparse
import sys
import os
from datetime import datetime
from typing import Optional

ASCII_CHARS = " .:-=+*#%@"  # from light to dark

def to_ascii(frame, width: int, height: int) -> str:
    """Convert BGR frame to an ASCII art string of given width/height.
    Downscales, converts to grayscale, maps intensities to ASCII chars.
    """
    import numpy as np
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
    # Normalize to indices
    idx = (resized.astype(float) / 255.0) * (len(ASCII_CHARS) - 1)
    idx = idx.astype(int)
    lines = ["".join(ASCII_CHARS[i] for i in row) for row in idx]
    return "\n".join(lines)

def open_capture(index: int):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam index {index}")
    return cap

def build_writer(cap, path: str, fps: float):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cv2.VideoWriter(path, fourcc, fps, (w, h))

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Console webcam streamer")
    p.add_argument('--device', type=int, default=0, help='Webcam index')
    p.add_argument('--ascii', action='store_true', help='Render as ASCII in console instead of window')
    p.add_argument('--width', type=int, default=96, help='ASCII width (chars)')
    p.add_argument('--height', type=int, default=54, help='ASCII height (rows)')
    p.add_argument('--record', type=str, default='', help='Optional output video filename (mp4)')
    p.add_argument('--target-fps', type=int, default=30, help='Target processing fps (caps sleep)')
    p.add_argument('--capture-every', type=int, default=0, help='Auto-save a frame every N seconds (0=disabled)')
    p.add_argument('--no-window', action='store_true', help='Disable cv2 window even in non-ascii mode (headless)')
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    cap = open_capture(args.device)
    writer = None
    if args.record:
        writer = build_writer(cap, args.record, args.target_fps)
        print(f"[record] Writing to {args.record}")
    last_capture_time = 0.0
    frame_counter = 0
    fps_avg_window = 30
    fps_hist = []
    print("Press 'q' to quit (window mode) or Ctrl+C (ASCII/headless). 'c' to capture a frame.")

    try:
        while True:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                print("[warn] Failed to read frame; retrying...")
                time.sleep(0.05)
                continue
            frame_counter += 1

            if writer is not None:
                writer.write(frame)

            # Auto capture
            if args.capture_every > 0 and (t0 - last_capture_time) >= args.capture_every:
                last_capture_time = t0
                fname = datetime.now().strftime("frame_%Y%m%d_%H%M%S.jpg")
                cv2.imwrite(fname, frame)
                print(f"[auto-capture] Saved {fname}")

            if args.ascii:
                art = to_ascii(frame, args.width, args.height)
                # Clear console (ANSI) then print
                sys.stdout.write("\x1b[H\x1b[J")
                sys.stdout.write(art + "\n")
                # Show FPS line
                elapsed = time.time() - t0
                fps = 1.0 / elapsed if elapsed > 0 else 0.0
                fps_hist.append(fps)
                if len(fps_hist) > fps_avg_window:
                    fps_hist.pop(0)
                sys.stdout.write(f"FPS: {fps:.1f} (avg {sum(fps_hist)/len(fps_hist):.1f})\n")
                sys.stdout.flush()
                # Sleep to regulate roughly target FPS
                frame_period = 1.0 / max(1, args.target_fps)
                spend = time.time() - t0
                if spend < frame_period:
                    time.sleep(frame_period - spend)
            else:
                if not args.no_window:
                    disp = frame.copy()
                    cv2.putText(disp, f"Frame {frame_counter}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0),2)
                    cv2.imshow('Webcam', disp)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('c'):
                        fname = datetime.now().strftime("capture_%Y%m%d_%H%M%S.jpg")
                        cv2.imwrite(fname, frame)
                        print(f"[capture] Saved {fname}")
                else:
                    # Headless non-ascii: just log FPS periodically
                    if frame_counter % args.target_fps == 0:
                        print(f"[info] Frames: {frame_counter}")
                    frame_period = 1.0 / max(1, args.target_fps)
                    spend = time.time() - t0
                    if spend < frame_period:
                        time.sleep(frame_period - spend)
    except KeyboardInterrupt:
        print("\n[exit] KeyboardInterrupt")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.ascii and not args.no_window:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        print("[done] Stream ended")

if __name__ == '__main__':
    main()