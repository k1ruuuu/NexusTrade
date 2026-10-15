from __future__ import annotations

import base64
import hashlib
import hmac
import secrets


def mask_api_key(api_key: str) -> str:
    """Mask an API key for safe Telegram display (e.g., 'ABCD****WXYZ')."""
    cleaned = api_key.strip()
    if len(cleaned) <= 8:
        return "****"
    return f"{cleaned[:4]}****{cleaned[-4:]}"


def derive_user_keys(master_secret: str, user_id: int, nonce: bytes) -> tuple[bytes, bytes]:
    """Derive 32-byte encryption key and 32-byte MAC key via PBKDF2-HMAC-SHA256."""
    master = master_secret.encode("utf-8")
    salt = f"nexustrade_uid_{int(user_id)}_".encode("utf-8") + nonce
    dk = hashlib.pbkdf2_hmac("sha256", master, salt, 60_000, dklen=64)
    return dk[:32], dk[32:]


def xor_keystream(data: bytes, enc_key: bytes, nonce: bytes) -> bytes:
    """Apply counter-mode HMAC-SHA256 keystream XOR cipher."""
    out = bytearray(len(data))
    block_count = (len(data) + 31) // 32
    for counter in range(block_count):
        ctr_bytes = counter.to_bytes(4, "big")
        keystream_block = hmac.new(enc_key, nonce + ctr_bytes, hashlib.sha256).digest()
        start = counter * 32
        end = min(start + 32, len(data))
        for i in range(start, end):
            out[i] = data[i] ^ keystream_block[i - start]
    return bytes(out)


def encrypt_secret_for_user(master_secret: str, user_id: int, plaintext: str) -> str:
    """Encrypt a secret string with per-user PBKDF2-HMAC-SHA256 + authenticated tag."""
    nonce = secrets.token_bytes(16)
    enc_key, mac_key = derive_user_keys(master_secret, user_id, nonce)
    raw_bytes = plaintext.encode("utf-8")
    ciphertext = xor_keystream(raw_bytes, enc_key, nonce)
    tag = hmac.new(mac_key, nonce + ciphertext, hashlib.sha256).digest()
    payload = nonce + tag + ciphertext
    return base64.urlsafe_b64encode(payload).decode("ascii")


def decrypt_secret_for_user(master_secret: str, user_id: int, token: str) -> str:
    """Verify HMAC tag and decrypt a user-bound secret string."""
    raw = base64.urlsafe_b64decode(token.encode("ascii"))
    if len(raw) < 48:
        raise ValueError("Invalid encrypted vault token length")
    nonce = raw[:16]
    tag = raw[16:48]
    ciphertext = raw[48:]
    enc_key, mac_key = derive_user_keys(master_secret, user_id, nonce)
    expected_tag = hmac.new(mac_key, nonce + ciphertext, hashlib.sha256).digest()
    if not hmac.compare_digest(tag, expected_tag):
        raise ValueError("Vault token integrity check failed (wrong user_id or tampered)")
    plaintext_bytes = xor_keystream(ciphertext, enc_key, nonce)
    return plaintext_bytes.decode("utf-8")
