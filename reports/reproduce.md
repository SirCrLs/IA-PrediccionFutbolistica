# Comandos reproducibles de pruebas de seguridad

Desde PowerShell en la raíz del repositorio:

```powershell
.venv\Scripts\pytest.exe -q tests/test_security.py
.venv\Scripts\pytest.exe -q --junitxml=reports\pytest.xml tests\test_security.py
```

El informe JUnit se escribe en `reports/pytest.xml`.

## Bandit

Estado actual:

```powershell
.venv\Scripts\bandit.exe -r . -x ./tests,./.venv -f txt
```

Resultado esperado: 0 hallazgos después de retirar `subprocess`.

Estado original (`6cf08b5033378798553bc76e711d818db561b44c`):

```powershell
git worktree add --detach C:\Users\User\AppData\Local\Temp\baseline-6cf08b5 6cf08b5
Set-Location C:\Users\User\AppData\Local\Temp\baseline-6cf08b5
C:\ruta\al\repo\.venv\Scripts\bandit.exe -r . -f txt
git worktree remove --force C:\Users\User\AppData\Local\Temp\baseline-6cf08b5
```

Resultado real del baseline: 364 líneas analizadas y 0 hallazgos. Guardado en `reports/bandit_original.txt`.

## pip-audit

```powershell
.venv\Scripts\pip-audit.exe -r requirements.txt --progress-spinner off
.venv\Scripts\python.exe -m pip list --outdated
```

Resultado esperado actual: `No known vulnerabilities found`. La salida de paquetes desactualizados se guarda en `reports/pip_outdated.txt`.

El commit original no contiene `requirements.txt`, por lo que no existe un baseline pip-audit reproducible para ese estado.

## Gitleaks

Escaneo del historial:

```powershell
gitleaks detect --source . --redact --report-format json --report-path reports\gitleaks_history.json
```

Escaneo del árbol excluyendo `.venv` y artefactos generados:

```powershell
gitleaks detect --source . --no-git --redact --report-format json --report-path reports\gitleaks_worktree.json
```

Los resultados locales se resumen en `reports/gitleaks_history.txt` y `reports/gitleaks_worktree.txt`.

## Prueba dinámica de pickle

La prueba compara el commit original con el código actual:

1. Crear un objeto joblib con `__reduce__` que intenta escribir un marcador.
2. Cargarlo con `joblib.load()` en `6cf08b5033378798553bc76e711d818db561b44c`.
3. Confirmar que el marcador fue creado.
4. Cargar el mismo archivo con `security.load_pkl()` en el estado actual.
5. Confirmar `PermissionError` y ausencia de deserialización.

Resultado real: original `original_load_executed True`; actual `current_load_rejected True`. Evidencia: `reports/malicious_pickle_before_after.txt`.
