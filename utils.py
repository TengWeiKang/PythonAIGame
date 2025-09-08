# utils.py
import math
import cv2
import numpy as np
import json
import os
import base64, hashlib, hmac, secrets

_GEMINI_SALT_FILE = 'gemini_salt.bin'

def _get_local_salt():
    """Return a stable per-installation random salt (creates file if missing)."""
    try:
        if os.path.exists(_GEMINI_SALT_FILE):
            with open(_GEMINI_SALT_FILE,'rb') as f:
                data = f.read()
                if len(data) >= 16:
                    return data[:16]
        salt = secrets.token_bytes(16)
        with open(_GEMINI_SALT_FILE,'wb') as f:
            f.write(salt)
        return salt
    except Exception:
        # Fallback deterministic salt (weak) if filesystem blocked
        return b'fallback-fixed!!'

def _derive_key(salt: bytes) -> bytes:
    user = os.environ.get('USERNAME') or os.environ.get('USER') or 'user'
    base = (salt + user.encode('utf-8'))
    # PBKDF2 for key stretching
    return hashlib.pbkdf2_hmac('sha256', base, salt, 100_000, dklen=32)

def _xor_stream(data: bytes, key: bytes) -> bytes:
    # Generate keystream blocks via sha256(key + counter)
    out = bytearray(len(data))
    pos = 0; counter = 0
    while pos < len(data):
        block = hashlib.sha256(key + counter.to_bytes(4,'big')).digest()
        for b in block:
            if pos >= len(data):
                break
            out[pos] = data[pos] ^ b
            pos += 1
        counter += 1
    return bytes(out)

def encrypt_api_key(plain: str) -> str:
    """Encrypt API key locally (not bulletproof, layered obfuscation + integrity).
    Structure (base64): version(1 byte='1') | cipher | hmac(32)
    Salt stored separately per installation. Returns base64 string.
    """
    if not plain:
        return ''
    salt = _get_local_salt()
    key = _derive_key(salt)
    data = plain.encode('utf-8')
    cipher = _xor_stream(data, key)
    mac = hmac.new(key, cipher, hashlib.sha256).digest()
    blob = b'1' + cipher + mac
    return base64.urlsafe_b64encode(blob).decode('ascii')

def decrypt_api_key(enc: str) -> str:
    if not enc:
        return ''
    try:
        blob = base64.urlsafe_b64decode(enc.encode('ascii'))
        if len(blob) < 1+32:  # need at least version + hmac
            return ''
        version = blob[0:1]
        if version != b'1':
            return ''
        salt = _get_local_salt()
        key = _derive_key(salt)
        cipher = blob[1:-32]
        mac = blob[-32:]
        if hmac.compare_digest(mac, hmac.new(key, cipher, hashlib.sha256).digest()):
            plain = _xor_stream(cipher, key).decode('utf-8', errors='ignore')
            return plain
        return ''
    except Exception:
        return ''

def get_gemini_api_key(cfg: dict) -> str:
    """Return decrypted Gemini API key (handles legacy plaintext)."""
    enc = cfg.get('gemini_api_key_enc')
    if enc:
        return decrypt_api_key(enc)
    return cfg.get('gemini_api_key','')

def set_gemini_api_key(cfg: dict, key: str):
    """Set (encrypt) Gemini API key in cfg; blank key clears value."""
    if key:
        cfg['gemini_api_key_enc'] = encrypt_api_key(key)
    else:
        cfg['gemini_api_key_enc'] = ''
    cfg.pop('gemini_api_key', None)  # remove legacy

def load_config(path="config.json"):
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_config(cfg, path="config.json"):
    """Persist updated config dict to disk (pretty-printed)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

def xywh_to_xyxy(box, w=1, h=1, normalized=True):
    # box: (x_center, y_center, w, h) normalized 0..1 or absolute
    if normalized:
        cx, cy, bw, bh = box
        x1 = (cx - bw/2) * w
        y1 = (cy - bh/2) * h
        x2 = (cx + bw/2) * w
        y2 = (cy + bh/2) * h
    else:
        cx, cy, bw, bh = box
        x1 = cx - bw/2
        y1 = cy - bh/2
        x2 = cx + bw/2
        y2 = cy + bh/2
    return [int(x1), int(y1), int(x2), int(y2)]

def xyxy_to_xywh_norm(xyxy, img_w, img_h):
    x1, y1, x2, y2 = xyxy
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return [cx, cy, bw, bh]

def iou_xyxy(boxA, boxB):
    # boxA/B: [x1,y1,x2,y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
    boxBArea = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
    denom = float(boxAArea + boxBArea - interArea)
    if denom == 0:
        return 0.0
    return interArea / denom

def centroid(box):
    x1,y1,x2,y2 = box
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def centroid_distance(boxA, boxB):
    (ax,ay) = centroid(boxA)
    (bx,by) = centroid(boxB)
    return math.hypot(ax-bx, ay-by)

def estimate_orientation(img, bbox):
    """
    Estimate object orientation (degrees) within bbox in image.
    bbox: [x1,y1,x2,y2]
    Returns angle in degrees (-90..90). 0 means horizontal.
    """
    x1,y1,x2,y2 = bbox
    crop = img[max(0,y1):y2, max(0,x1):x2].copy()
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # use Canny and contours
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    # choose largest contour
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 10:
        return 0.0
    rect = cv2.minAreaRect(c)
    angle = rect[-1]
    # rect angle parsing: return normalized angle
    # OpenCV angle semantics: angle of the rotated rectangle.
    return float(angle)

def read_yolo_labels(path):
    """
    Read YOLO labels file, return list of [class_idx, x_center, y_center, w, h] (float)
    """
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            vals = [float(x) if i>0 else int(x) for i,x in enumerate(parts)]
            out.append(vals)
    return out

def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)
