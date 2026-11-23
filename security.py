import hashlib
import hmac
import json
import os
from pathlib import Path

MANIFEST = Path("manifest.json")


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
