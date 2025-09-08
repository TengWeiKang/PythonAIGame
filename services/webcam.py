"""Webcam abstraction."""
from __future__ import annotations
import cv2, subprocess, os
from typing import List, Tuple

class WebcamManager:
    def __init__(self):
        self.cap = None
        self.index = None

    def open(self, index:int, width=None, height=None, fps=None) -> bool:
        self.close()
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap.release(); self.cap=None
            return False
        self.index = index
        if width: self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height: self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps: self.cap.set(cv2.CAP_PROP_FPS, fps)
        return True

    def read(self):
        if not self.cap: return False, None
        return self.cap.read()

    def close(self):
        if self.cap:
            try: self.cap.release()
            except Exception: pass
        self.cap=None; self.index=None

    @staticmethod
    def list_devices(max_test=5) -> List[Tuple[int,str]]:
        names=[]
        if os.name=='nt':
            queries=[["wmic","path","Win32_PnPEntity","where","Service='usbvideo'","get","Name"]]
            seen=set()
            for cmd in queries:
                try:
                    r=subprocess.run(cmd,capture_output=True,text=True,timeout=2)
                    if r.returncode!=0: continue
                    lines=[l.strip() for l in r.stdout.splitlines() if l.strip()]
                    if lines and lines[0].lower().startswith('name'): lines=lines[1:]
                    for l in lines:
                        if l and l not in seen:
                            names.append(l); seen.add(l)
                except Exception: continue
        devices=[]
        for i in range(max_test):
            cap=cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                label = names[i] if i < len(names) else f"Device {i}"
                devices.append((i,label))
                cap.release()
        return devices
