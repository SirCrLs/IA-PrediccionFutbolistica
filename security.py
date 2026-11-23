import hashlib
import hmac
import json
import logging
import os
import time
from pathlib import Path

MANIFEST = Path("manifest.json")
USERS = Path("users.json")
LOCKOUT_MAX_ATTEMPTS = 5
LOCKOUT_SECONDS = 5 * 60
SCRYPT_N = 2**15
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_MAXMEM = 64 * 1024 * 1024

ROLES = {
    "viewer": {"predict", "read_data"},
    "trainer": {"predict", "train", "read_data", "write_processed", "write_models"},
    "admin": {"*"},
}

logging.basicConfig(filename="audit.log", level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("audit")


def _read_users():
    try:
        data = json.loads(USERS.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError
        return data
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError) as exc:
        log.error("Authentication data unavailable")
        raise PermissionError("Access denied") from exc


def _write_users(users):
    USERS.write_text(json.dumps(users, sort_keys=True), encoding="utf-8")
    try:
        os.chmod(USERS, 0o600)
    except (AttributeError, OSError):
        pass


def authenticate(user, token):
    users = _read_users()
    entry = users.get(user)
    if not entry or not isinstance(entry, dict):
        log.warning("Authentication failed for unknown user")
        return None

    now = time.time()
    locked_until = float(entry.get("locked_until", 0))
    if locked_until > now:
        log.warning("Authentication denied: account temporarily locked")
        return None
    if locked_until and locked_until <= now:
        entry["failed_attempts"] = 0
        entry["locked_until"] = 0

    try:
        salt = bytes.fromhex(entry["salt"])
        stored_hash = entry["hash"]
        candidate = hashlib.scrypt(
            token.encode("utf-8"),
            salt=salt,
            n=SCRYPT_N,
            r=SCRYPT_R,
            p=SCRYPT_P,
            maxmem=SCRYPT_MAXMEM,
        ).hex()
    except (KeyError, TypeError, ValueError, OverflowError):
        log.warning("Authentication denied: invalid account record")
        return None

    if not hmac.compare_digest(candidate, stored_hash):
        attempts = int(entry.get("failed_attempts", 0)) + 1
        entry["failed_attempts"] = attempts
        if attempts >= LOCKOUT_MAX_ATTEMPTS:
            entry["locked_until"] = now + LOCKOUT_SECONDS
            log.warning("Authentication lockout activated")
        else:
            log.warning("Authentication failed")
        _write_users(users)
        return None

    entry["failed_attempts"] = 0
    entry["locked_until"] = 0
    _write_users(users)
    role = entry.get("role")
    if role not in ROLES:
        log.warning("Authentication denied: invalid role")
        return None
    return role


def require(action):
    user = os.environ.get("PIPELINE_USER", "")
    token = os.environ.get("PIPELINE_TOKEN", "")
    role = authenticate(user, token)
    allowed = role in ROLES and (action in ROLES[role] or "*" in ROLES[role])
    if not allowed:
        log.warning("DENIED action=%s", action)
        raise PermissionError("Access denied")
    log.info("GRANTED action=%s role=%s", action, role)
    return role


def secret(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing env var {name}")
    key = value.encode("utf-8")
    if len(key) < 32:
        raise RuntimeError(f"{name} must be at least 32 bytes")
    return key


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(path):
    return Path(path).resolve().as_posix()


def _signature(hashes):
    payload = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hmac.new(secret("PIPELINE_HMAC_KEY"), payload, hashlib.sha256).hexdigest()


def build_manifest(paths):
    hashes = {_canonical(path): sha256(path) for path in paths}
    data = {"hashes": hashes, "sig": _signature(hashes)}
    MANIFEST.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")


def verify(path):
    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        hashes = manifest["hashes"]
        signature = manifest["sig"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise PermissionError("Invalid or missing manifest") from exc
    if not hmac.compare_digest(signature, _signature(hashes)):
        raise PermissionError("Manifest tampered")
    key = _canonical(path)
    if hashes.get(key) != sha256(path):
        raise PermissionError(f"Hash mismatch: {key}")
    return key


def load_pkl(path):
    path = Path(path)
    if path.suffix.lower() != ".pkl":
        raise ValueError("Only .pkl files can be loaded")
    verify(path)
    import joblib
    return joblib.load(path)


def load_model(path):
    path = Path(path)
    if path.suffix.lower() != ".keras":
        raise ValueError("Only .keras models can be loaded")
    verify(path)
    import tensorflow as tf
    return tf.keras.models.load_model(path, safe_mode=True)
