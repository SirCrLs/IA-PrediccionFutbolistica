# Contexto para Claude

## Proyecto

Repositorio: `SirCrLs/IA-PrediccionFutbolistica`. Python/TensorFlow para predicción futbolística con datos de las cinco grandes ligas europeas.

Archivo fuente de requisitos y convenciones: `COPILOT.md`. No modificar `README.md`. Los informes finales están en español; código, comentarios y mensajes en inglés.

## Cambios implementados

- `security.py`: SHA-256, manifiesto HMAC-SHA256, carga segura de `.pkl`/`.keras`, scrypt, lockout, roles, validación de partidos/CSV/rutas/tamaños y carga de `.env` mediante `python-dotenv`.
- `tools/make_user.py`: creación de usuarios con salt aleatorio y scrypt.
- Fase 1: verifica CSV, valida datos, fija semillas y registra hashes.
- Fase 2: exige roles, verifica `.npy`/`.pkl`, solo guarda `.keras`, elimina `subprocess` y registra ejecución/hashes.
- `main.py`: argparse, autorización, validación y errores genéricos.
- CI: Bandit, pip-audit, Gitleaks, pytest y carga de reports como artefacto.
- Dependencias: pins directos y transitivos con hashes en `requirements.txt`.

## Evidencia actual

- pytest: 19 aprobadas.
- Bandit actual: 0 hallazgos.
- pip-audit: sin vulnerabilidades conocidas.
- Gitleaks historial: sin secretos encontrados.
- Gitleaks copia limpia rastreada: sin secretos encontrados.
- RS-01 a RS-10 y T-01 a T-12 trazados.
- EA-01 a EA-08 trazados.
- `manifest.json`, `.env`, `users.json` y `audit.log` no deben versionarse.

## Hallazgos corregidos

- FIND-01: uso de `subprocess` para leer el commit. Corregido leyendo `.git/HEAD`; CVSS UI:N 3.3.
- FIND-02: carga directa de `.pkl` en el commit original `6cf08b5033378798553bc76e711d818db561b44c`. Demostración dinámica: el objeto malicioso creó el marcador original y `load_pkl()` actual rechazó antes de deserializar; CVSS 7.2.

El baseline original tuvo 364 líneas analizadas por Bandit y 0 hallazgos. No atribuirle hallazgos Bandit que no existen.

## Dependencias

`python-dotenv==1.2.4` permite cargar `.env` mediante `load_dotenv()`.

`pip list --outdated` mostró únicamente `h5py 3.14.0 → 3.16.0`. No se actualizó por la restricción de TensorFlow `h5py<3.15.0`.

## Pendientes

- SSH individual y deshabilitación de contraseña.
- Usuario de servidor sin privilegios.
- Backup externo/restauración.
- Cifrado de disco/cuotas.
- Configuración manual de GitHub.
- Ejecución real del workflow en GitHub.

## Comandos

```powershell
.venv\Scripts\python.exe -m pip install --require-hashes -r requirements.txt
.venv\Scripts\pytest.exe -q --junitxml=reports\pytest.xml
.venv\Scripts\bandit.exe -r . -x ./tests,./.venv -f txt
.venv\Scripts\pip-audit.exe -r requirements.txt
```

## Reglas

- Leer `COPILOT.md` antes de cambiar código.
- No reescribir Fase 1/Fase 2.
- No considerar `.pkl` seguro sin `verify()`.
- Ejecutar pytest, Bandit y pip-audit después de cambios.
- No modificar `README.md`.
