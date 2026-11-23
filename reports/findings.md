# Security Findings

## Real findings

| ID | Finding | Evidence (file:line / tool output) | Root cause | Impact | Risk | CWE | CVSS v3.1 | Fix | Revalidation |
|---|---|---|---|---|---|---|---|---|---|
| FIND-01 | Subprocess execution used to read the Git commit | Fixed: `Fase2_entrenamiento.py` no longer imports or calls `subprocess`; `reports/bandit_after.txt` | The training log previously imported `subprocess` and invoked `git` from a partial executable name, even though arguments were fixed and `shell=True` was not used | A compromised executable in `PATH` or future changes to arguments could execute unintended code in the training account | Low | CWE-78 | 3.3 — `CVSS:3.1/AV:L/AC:L/PR:L/UI:R/S:U/C:L/I:N/A:N`; local access and user interaction are required; current fixed arguments and no shell reduce impact | Removed `subprocess`; the commit is now read from `.git/HEAD`/the referenced ref, with `unknown` as fallback | Revalidated: Bandit exit 0, 0 issues; focused pytest suite 19 passed |

## No findings

| Source | Result | Evidence |
|---|---|---|
| pip-audit 2.10.1 | No known vulnerabilities | `reports/pip_audit_after.txt`; exit code 0 |
| Dynamic tests | No attack test failures | `reports/pytest.xml`; 19 tests, 0 failures, 0 errors |

## Before/after summary

| Tool | Before | After |
|---|---:|---:|
| Bandit | 0 issues | FIND-01 fixed; 0 issues |
| pip-audit | 0 known vulnerabilities | 0 known vulnerabilities |
