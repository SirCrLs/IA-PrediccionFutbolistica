# Security Findings

## Real findings

| ID | Finding | Evidence (file:line / tool output) | Root cause | Impact | Risk | CWE | CVSS v3.1 | Fix | Revalidation |
|---|---|---|---|---|---|---|---|---|---|
| FIND-01 | Subprocess execution used to read the Git commit | Fixed: `Fase2_entrenamiento.py` no longer imports or calls `subprocess`; `reports/bandit_after.txt` | The training log imported `subprocess` and invoked `git` from a partial executable name, even though arguments were fixed and `shell=True` was not used | A compromised executable in `PATH` or future changes could execute unintended code in the training account | Low | CWE-78 | 3.3 — `CVSS:3.1/AV:L/AC:L/PR:L/UI:N/S:U/C:L/I:N/A:N` | Removed `subprocess`; the commit is now read from `.git/HEAD`/the referenced ref, with `unknown` as fallback | Revalidated: Bandit exit 0, 0 issues; focused pytest suite 19 passed |
| FIND-02 | Unsafe direct deserialization in the original code | Original commit `6cf08b5033378798553bc76e711d818db561b44c`; `Fase2_entrenamiento.py` contained `joblib.load("processed_data/le_home.pkl")` and `joblib.load("processed_data/le_away.pkl")`; dynamic probe `reports/malicious_pickle_before_after.txt` | The original pipeline loaded `.pkl` artifacts without integrity verification or any safe deserialization boundary | A malicious or swapped preprocessor can execute arbitrary code while Fase 2 loads it | High | CWE-502 | 7.2 — `CVSS:3.1/AV:L/AC:L/PR:L/UI:R/S:U/C:H/I:H/A:H` | Replaced direct loads with `load_pkl()`; verify the HMAC manifest and SHA-256 hash before deserializing | Dynamic probe: original created `original-marker.txt`; current returned `PermissionError` and did not deserialize |

## No findings

| Source | Result | Evidence |
|---|---|---|
| pip-audit 2.10.1 | No known vulnerabilities | `reports/pip_audit_after.txt`; exit code 0 |
| Bandit original commit | 0 issues | `reports/bandit_original.txt`; commit `6cf08b5033378798553bc76e711d818db561b44c`, 364 lines |
| Dynamic tests | No attack test failures | `reports/pytest.xml`; 19 tests, 0 failures, 0 errors |

## Before/after summary

| Tool | Before | After |
|---|---:|---:|
| Bandit original commit | 0 issues | No subprocess implementation existed in that baseline |
| Bandit current | FIND-01 found before fix | FIND-01 fixed; 0 issues |
| pip-audit | 0 known vulnerabilities | 0 known vulnerabilities |
