# Matriz de trazabilidad

## Requisitos RS-01 a RS-10

| RS | Amenaza | Riesgo | Mitigación | Implementación | Prueba | Resultado |
|---|---|---|---|---|---|---|
| RS-01 | CSV alterado | Entrenamiento contaminado | SHA-256 y validación antes de combinar datos | `verify()` y `validate_csv()` en Fase 1 | `test_tampered_artifact_rejected`, `test_csv_schema_range_and_nan_rejected` | Aprobadas |
| RS-02 | Preprocesador/modelo cambiado | Uso de artefacto no confiable | Firma HMAC-SHA256 y hash exacto | `build_manifest()` y `verify()` | `test_valid_manifest_and_sha256`, `test_tampered_manifest_rejected` | Aprobadas |
| RS-03 | `.pkl` o modelo de origen desconocido | Ejecución o intercambio | Allowlist `.pkl`/`.keras` y carga posterior a verificación | `load_pkl()`, `load_model()` | `test_malicious_pickle_does_not_execute`, `test_tampered_artifact_rejected` | Aprobadas |
| RS-04 | Credenciales en el repositorio | Compromiso de cuentas o claves | Variables de entorno, `.gitignore`, escaneo Gitleaks | `secret()`, `.env.example`, `.gitignore` | `test_secret_not_written_to_log`, escaneo histórico y árbol rastreado | Sin secretos en historial; prueba aprobada |
| RS-05 | Acceso SSH o usuario no individual | Acceso indebido | Clave individual y autenticación del usuario del pipeline | `authenticate()`/`tools/make_user.py`; SSH pendiente | `test_viewer_cannot_train_and_unknown_user_denied`, lockout | Parcial: autenticación probada, SSH externo pendiente |
| RS-06 | Cuenta operativa con privilegios altos | Escalamiento | Permisos por recurso y separación de roles | `ROLES`, `require()`, entrada `main.py` | `test_viewer_cannot_train_and_unknown_user_denied` | Parcial: RBAC probado, usuario de servidor pendiente |
| RS-07 | Dependencia typosquatted o vulnerable | Supply chain | Entorno venv, pins y hashes | `requirements.in`, `requirements.txt` | `pip install --require-hashes`, `pip-audit` | Aprobados |
| RS-08 | Ejecución no reproducible o repudiable | Falta de trazabilidad | Fecha, commit y hashes en el log | Fase 1/Fase 2 y `audit.log` | Prueba de salida de Fase 1/Fase 2 y escaneo de log | Implementado parcialmente; el log no se versiona |
| RS-09 | Pérdida de datos/modelos | Recuperación imposible | Integridad Git y manifiesto; backup externo pendiente | Manifesto e integridad local | Inspección de Git y manifiesto | Parcial |
| RS-10 | Entrada inválida o exhaustiva | Error inesperado o DoS | Validación de equipos, CSV, rutas y tamaños | `validate_match()`, `validate_csv()`, `validate_path()`, `argparse` | `test_malformed_match_rejected`, `test_path_traversal_rejected`, `test_oversized_file_rejected` | Aprobadas |

## Amenazas T-01 a T-12

| T | Amenaza | Riesgo | Mitigación | Implementación | Prueba | Resultado |
|---|---|---|---|---|---|---|
| T-01 | Spoofing por robo de credenciales | Acceso como usuario legítimo | scrypt, lockout, llaves SSH individuales | `authenticate()`, lockout | `test_failed_attempt_lockout` | Parcial |
| T-02 | Commits atribuidos a otra identidad | Integridad repudiable | 2FA, rama protegida y commits firmados | Workflow y guía GitHub | No existe prueba local de configuración | Parcial |
| T-03 | Envenenamiento de CSV | Modelo entrenado con datos incorrectos | Hash y validación de esquema/rango | `build_manifest()`/`verify()` y Fase 1 | `test_tampered_artifact_rejected`, `test_csv_schema_range_and_nan_rejected` | Mitigado |
| T-04 | Intercambio de `.pkl` o modelo | Ejecución/cambios arbitrarios | Verificación previa y formatos permitidos | `load_pkl()`, `load_model()` | `test_malicious_pickle_does_not_execute`, `test_tampered_artifact_rejected` | Mitigado |
| T-05 | Métricas o gráficos falsificados | Evidencia engañosa | Generación del pipeline y registro de modelo | Fase 1/Fase 2 | Inspección de código; falta hash por PNG | Parcial |
| T-06 | Ejecución sin trazabilidad | Repudio | Registro de fecha, commit y hashes | `audit.log` | Muestra sin secretos en `reports/audit_sample.txt` | Mitigado en código |
| T-07 | Filtración de tokens o claves | Acceso externo | Entorno, `.env.example`, `.gitignore`, Gitleaks | `secret()` y reglas de Git | Escaneos Gitleaks | Mitigado |
| T-08 | Copia no autorizada de modelos/datos | Divulgación de activos | RBAC, permisos de archivos y servidor | `ROLES`, `require()`, archivos restringidos | Role tests; cifrado/ACL externos pendientes | Parcial |
| T-09 | Agotamiento de recursos | Denegación de servicio | Límites de CSV/archivo y rutas; cuotas pendientes | `validate_csv()`, `validate_path()` | `test_oversized_file_rejected` | Parcial |
| T-10 | Pérdida de datos/modelos | Indisponibilidad o reversión imposible | Git, manifiesto y backup externo pendiente | Manifesto local y repositorio | Inspección de artefactos; restauración no probada | Parcial |
| T-11 | Elevación de permisos | Acciones no autorizadas | RBAC por recurso | `ROLES`, `require()` | `test_viewer_cannot_train_and_unknown_user_denied` | Mitigado en código |
| T-12 | Dependencia comprometida | Compromiso de supply chain | Pines, hashes y auditoría | `requirements.txt`, CI | `pip-audit`, instalación con hashes | Mitigado |

## Escenarios de abuso EA-01 a EA-08

| EA | Amenaza | RS/T relacionados | Evidencia | Prueba | Resultado |
|---|---|---|---|---|---|
| EA-01 | Poisoning de CSV | RS-01, T-03 | CSV leído e integrado solo después de verificar/validar | `test_tampered_artifact_rejected`, `test_csv_schema_range_and_nan_rejected` | Aprobadas |
| EA-02 | `scaler.pkl` malicioso | RS-02, RS-03, T-04 | Carga directa retirada y reemplazada por `load_pkl()` | `test_malicious_pickle_does_not_execute` | Aprobada |
| EA-03 | Modelo `.keras` intercambiado | RS-02, RS-03, T-04 | Hash registrado y allowlist | `test_tampered_artifact_rejected` | Aprobada |
| EA-04 | GitHub token robado | RS-04, T-07 | `.gitignore`, secretos por entorno y Gitleaks | Escaneo histórico y `test_secret_not_written_to_log` | Sin secreto probado |
| EA-05 | Paquete typosquatted | RS-07, T-12 | Pins/hashes y pip-audit | Instalación hash-only y pip-audit | Aprobados |
| EA-06 | Input malformado en `main.py` | RS-10, T-09 | `argparse` y `validate_match()` | `test_malformed_match_rejected`, prueba traversal | Aprobadas |
| EA-07 | Métricas/gráficos manipulados | T-05 | Generación desde pipeline; falta hash por imagen | Inspección de Fase 1/Fase 2 | Parcial |
| EA-08 | Exhaustión de disco o memoria | RS-10, T-09 | Límites de tamaño/filas | `test_oversized_file_rejected` | Mitigado en aplicación; cuotas externas pendientes |
