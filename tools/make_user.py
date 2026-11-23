import argparse
import hashlib
import json
import os
from pathlib import Path

SCRYPT_N = 2**15
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_MAXMEM = 64 * 1024 * 1024
ROLES = {"viewer", "trainer", "admin"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("user")
    parser.add_argument("--role", choices=sorted(ROLES), default="viewer")
    args = parser.parse_args()

    token = os.environ.get("PIPELINE_TOKEN", "")
    if len(token) < 12:
        parser.error("PIPELINE_TOKEN must contain at least 12 characters")

    users_path = Path("users.json")
    try:
        users = json.loads(users_path.read_text(encoding="utf-8")) if users_path.exists() else {}
    except json.JSONDecodeError as exc:
        parser.error(f"Invalid users.json: {exc}")
    if not isinstance(users, dict):
        parser.error("Invalid users.json")

    salt = os.urandom(16)
    token_hash = hashlib.scrypt(
        token.encode("utf-8"),
        salt=salt,
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
        maxmem=SCRYPT_MAXMEM,
    ).hex()
    users[args.user] = {
        "salt": salt.hex(),
        "hash": token_hash,
        "role": args.role,
        "failed_attempts": 0,
        "locked_until": 0,
    }
    users_path.write_text(json.dumps(users, sort_keys=True), encoding="utf-8")
    os.chmod(users_path, 0o600)
    print(f"Created/updated user: {args.user}")


if __name__ == "__main__":
    main()
