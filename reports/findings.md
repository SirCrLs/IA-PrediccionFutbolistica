# Security Findings

## Real findings

| ID | Finding | Evidence (file:line / tool output) | Root cause | Impact | Risk | CWE | CVSS v3.1 | Fix | Revalidation |
|---|---|---|---|---|---|---|---|---|---|
| FIND-01 | Subprocess execution used to read the Git commit | Bandit B404/B603/B607; `Fase2_entrenamiento.py:9`, `Fase2_entrenamiento.py:324-329`; `reports/bandit_after.txt` | The training log imports `subprocess` and invokes `git` from a partial executable name, even though arguments are fixed and `shell=True` is not used | A compromised executable in `PATH` or future changes to arguments could execute unintended code in the training account | Low | CWE-78 | 3.3 — `CVSS:3.1/AV:L/AC:L/PR:L/UI:R/S:U/C:L/I:N/A:N`; local access and user interaction are required; current fixed arguments and no shell reduce impact | Remove `subprocess`; read the checked-out commit through Git metadata or record `git_commit=unknown`/a repository-independent value | Rerun Bandit and the focused Fase2 tests; require no subprocess finding |

## No findings

| Source | Result | Evidence |
|---|---|---|
| pip-audit 2.10.1 | No known vulnerabilities | `reports/pip_audit_after.txt`; exit code 0 |
| Dynamic tests | No attack test failures | `reports/pytest.xml`; 19 tests, 0 failures, 0 errors |

## Before/after summary

| Tool | Before | After |
|---|---:|---:|
| Bandit | 0 issues | 3 Low-severity findings from one subprocess pattern |
| pip-audit | 0 known vulnerabilities | 0 known vulnerabilities |
