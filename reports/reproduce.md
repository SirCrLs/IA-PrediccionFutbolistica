# Comandos reproducibles de pruebas de seguridad

Desde PowerShell en la raíz del repositorio:

```powershell
.venv\Scripts\pytest.exe -q tests/test_security.py
.venv\Scripts\pytest.exe -q --junitxml=reports\pytest.xml tests\test_security.py
```

El informe JUnit se escribe en `reports/pytest.xml`.
