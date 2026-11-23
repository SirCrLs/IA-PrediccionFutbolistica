# Security Test Reproduce Commands

Windows PowerShell, from the repository root:

```powershell
.venv\Scripts\pytest.exe -q tests/test_security.py
.venv\Scripts\pytest.exe -q --junitxml=reports\pytest.xml tests\test_security.py
```

The JUnit report is written to `reports/pytest.xml`.
