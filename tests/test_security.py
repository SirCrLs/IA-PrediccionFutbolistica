import hashlib
import json
import os
import pickle

import pandas as pd
import pytest

import security


@pytest.fixture
def isolated_security(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PIPELINE_HMAC_KEY", "A" * 32)
    security.MANIFEST = tmp_path / "manifest.json"
    security.USERS = tmp_path / "users.json"
    yield tmp_path


def create_user(path, user, token, role="viewer"):
    salt = os.urandom(16)
    password_hash = hashlib.scrypt(
        token.encode("utf-8"),
        salt=salt,
        n=security.SCRYPT_N,
        r=security.SCRYPT_R,
        p=security.SCRYPT_P,
        maxmem=security.SCRYPT_MAXMEM,
    ).hex()
    path.write_text(json.dumps({
        user: {
            "salt": salt.hex(),
            "hash": password_hash,
            "role": role,
            "failed_attempts": 0,
            "locked_until": 0,
        }
    }), encoding="utf-8")


def test_valid_manifest_and_sha256(isolated_security):
    artifact = isolated_security / "artifact.pkl"
    artifact.write_bytes(b"trusted")
    security.build_manifest([artifact])
    assert security.sha256(artifact) == hashlib.sha256(b"trusted").hexdigest()
    assert security.verify(artifact)


@pytest.mark.parametrize("suffix", ["csv", "pkl", "keras"])
def test_tampered_artifact_rejected(isolated_security, suffix):
    artifact = isolated_security / f"artifact.{suffix}"
    artifact.write_bytes(b"trusted")
    security.build_manifest([artifact])
    artifact.write_bytes(b"tampered")
    with pytest.raises(PermissionError):
        security.verify(artifact)


def test_tampered_manifest_rejected(isolated_security):
    artifact = isolated_security / "artifact.csv"
    artifact.write_bytes(b"data")
    security.build_manifest([artifact])
    manifest = json.loads(security.MANIFEST.read_text(encoding="utf-8"))
    manifest["sig"] = "0" * 64
    security.MANIFEST.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PermissionError):
        security.verify(artifact)


@pytest.mark.parametrize("value", [
    "' OR 1=1 --",
    "../../etc/passwd",
    "x" * 500,
    "",
    "A\x00BC",
])
def test_malformed_match_rejected(isolated_security, value):
    with pytest.raises(ValueError):
        security.validate_match(value, "Chelsea", "E0", {"Arsenal", "Chelsea"}, {"E0"})


def test_same_team_and_unknown_league_rejected(isolated_security):
    with pytest.raises(ValueError):
        security.validate_match("Arsenal", "Arsenal", "E0", {"Arsenal"}, {"E0"})
    with pytest.raises(ValueError):
        security.validate_match("Arsenal", "Chelsea", "XX", {"Arsenal", "Chelsea"}, {"E0"})


def test_csv_schema_range_and_nan_rejected(isolated_security):
    valid = pd.DataFrame({"HomeTeam": ["Arsenal"], "AwayTeam": ["Chelsea"], "FTHG": [1], "FTAG": [2]})
    assert security.validate_csv(valid)

    missing = pd.DataFrame({"HomeTeam": ["Arsenal"], "AwayTeam": ["Chelsea"], "FTHG": [1]})
    with pytest.raises(ValueError):
        security.validate_csv(missing)
    out_of_range = pd.DataFrame({"HomeTeam": ["Arsenal"], "AwayTeam": ["Chelsea"], "FTHG": [21], "FTAG": [2]})
    with pytest.raises(ValueError):
        security.validate_csv(out_of_range)
    nan = pd.DataFrame({"HomeTeam": ["Arsenal"], "AwayTeam": ["Chelsea"], "FTHG": [1], "FTAG": [None]})
    with pytest.raises(ValueError):
        security.validate_csv(nan)


def test_viewer_cannot_train_and_unknown_user_denied(isolated_security, monkeypatch):
    create_user(security.USERS, "viewer", "viewer-password", "viewer")
    monkeypatch.setenv("PIPELINE_USER", "viewer")
    monkeypatch.setenv("PIPELINE_TOKEN", "viewer-password")
    assert security.require("predict") == "viewer"
    with pytest.raises(PermissionError):
        security.require("train")

    monkeypatch.setenv("PIPELINE_USER", "unknown")
    monkeypatch.setenv("PIPELINE_TOKEN", "wrong")
    with pytest.raises(PermissionError):
        security.require("predict")


def test_failed_attempt_lockout(isolated_security, monkeypatch):
    create_user(security.USERS, "locked", "correct-password", "trainer")
    monkeypatch.setenv("PIPELINE_USER", "locked")
    monkeypatch.setenv("PIPELINE_TOKEN", "wrong-password")
    for _ in range(security.LOCKOUT_MAX_ATTEMPTS):
        assert security.authenticate("locked", "wrong-password") is None
    assert security.authenticate("locked", "correct-password") is None
    with pytest.raises(PermissionError):
        security.require("train")


def test_short_hmac_key_rejected(isolated_security, monkeypatch):
    monkeypatch.setenv("PIPELINE_HMAC_KEY", "short")
    with pytest.raises(RuntimeError):
        security.secret("PIPELINE_HMAC_KEY")


def test_path_traversal_rejected(isolated_security):
    with pytest.raises(ValueError):
        security.validate_path("../../etc/passwd")


def test_oversized_file_rejected(isolated_security, monkeypatch):
    artifact = isolated_security / "oversized.csv"
    artifact.write_bytes(b"x" * 11)
    monkeypatch.setattr(security, "MAX_FILE_SIZE", 10)
    with pytest.raises(ValueError):
        security.validate_path(
            artifact,
            allowed_dirs=(isolated_security,),
            allowed_extensions={".csv"},
        )


class MaliciousPickle:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return (os.system, (f'python -c "open(r\'{self.marker}\',\'w\').write(\'executed\')"',))


def test_malicious_pickle_does_not_execute(isolated_security):
    artifact = isolated_security / "malicious.pkl"
    artifact.write_bytes(pickle.dumps(MaliciousPickle(str(isolated_security / "marker.txt"))))
    marker = isolated_security / "marker.txt"
    with pytest.raises(PermissionError):
        security.load_pkl(artifact)
    assert not marker.exists()


def test_secret_not_written_to_log(isolated_security, monkeypatch):
    secret_marker = "TOP-SECRET-MARKER-DO-NOT-LOG"
    monkeypatch.setenv("PIPELINE_TOKEN", secret_marker)
    security.log.info("Authentication attempt")
    log_path = isolated_security / "audit.log"
    assert not log_path.exists() or secret_marker not in log_path.read_text(encoding="utf-8")
