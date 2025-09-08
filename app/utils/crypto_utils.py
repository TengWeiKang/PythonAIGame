"""Cryptographic utilities for API key encryption."""

import os
import base64
import hashlib
import hmac
import secrets

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