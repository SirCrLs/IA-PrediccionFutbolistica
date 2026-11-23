# Matriz de trazabilidad

## Requisitos de seguridad RS-01 a RS-10

| ID | Estado | Implementación | Evidencia |
|---|---|---|---|
| RS-01 | Mitigado | Verificación SHA-256 y validación de esquema/rango antes de usar cada CSV | `security.py`, `Fase1_preparacion.py`, `tests/test_security.py` |
| RS-02 | Mitigado | Manifiesto HMAC-SHA256 y comparación de hashes antes de cargar archivos | `security.py`, `tests/test_security.py`, `manifest.json` |
| RS-03 | Mitigado | Solo `.pkl` y `.keras`; carga mediante `load_pkl()`/`load_model()` después de `verify()` | `security.py`, `Fase2_entrenamiento.py`, `tests/test_security.py` |
| RS-04 | Mitigado | Secretos únicamente por variables de entorno; `.gitignore`, `.env.example`, escaneo local e histórico | `security.py`, `.env.example`, `.gitignore`, `reports/gitleaks_history.txt`, `reports/gitleaks_worktree.txt` |
| RS-05 | Parcial | Autenticación individual por usuario/token implementada en código; SSH, claves individuales y deshabilitar contraseña siguen siendo configuración de servidor | `security.py`, `tools/make_user.py`, guía GitHub |
| RS-06 | Parcial | Roles de acceso y permisos por recurso implementados; la ejecución del pipeline con un usuario de sistema no administrador no está configurada en este repositorio | `security.py`, `main.py`, `Fase2_entrenamiento.py` |
| RS-07 | Mitigado | Entorno virtual, versiones fijadas y archivo con hashes de pip compile | `requirements.in`, `requirements.txt`, `reports/pip_audit_after.txt` |
| RS-08 | Mitigado | Registro de fecha, commit disponible, hashes de entrada y modelo después del entrenamiento | `Fase1_preparacion.py`, `Fase2_entrenamiento.py`, `audit.log` |
| RS-09 | Parcial | Existe integridad local y Git, pero no hay backup ni ubicación externa demostrada | `manifest.json`, repositorio Git; falta automatización de respaldo |
| RS-10 | Mitigado | Validación de partidos, CSV, rutas, extensiones y tamaños | `security.py`, `main.py`, `tests/test_security.py`, `reports/pytest.xml` |

## Amenazas STRIDE T-01 a T-12

| ID | Riesgo asociado | Estado | Evidencia y mitigación |
|---|---|---|---|
| T-01 | Spoofing por robo de credenciales | Parcial | scrypt y lockout en `security.py`; llaves SSH, bloqueo de contraseña y 2FA son controles de infraestructura pendientes |
| T-02 | Spoofing mediante commits falsificados | Parcial | 2FA, commits firmados y verificación quedan como configuración manual en `SECURITY_GUIDE.md`; no están demostrados en el repositorio local |
| T-03 | Envenenamiento de CSV | Mitigado | Manifestación SHA-256/HMAC y `validate_csv()` antes de leer los datos en Fase 1; prueba de tampering |
| T-04 | Intercambio de `.pkl` o modelo | Mitigado | `verify()` antes de `load_pkl()`/`load_model()` y allowlist de formatos; pruebas con artefactos alterados |
| T-05 | Métricas o gráficas falsificadas | Parcial | Gráficas generadas por el pipeline y modelo registrado con hash; no existe hash individual ni firma para cada PNG |
| T-06 | Ejecución sin trazabilidad | Mitigado | `audit.log` registra verificaciones, denegaciones y ejecución Fase 1/Fase 2 con hashes |
| T-07 | Revelación de tokens o claves | Mitigado | Secretos por entorno, `.gitignore`, `.env.example` y escaneos Gitleaks del árbol/historial |
| T-08 | Copia no autorizada de modelos/datos | Parcial | Permisos de archivos sensibles mediante `.gitignore`/ACL y roles de aplicación; cifrado de disco y ACL de servidor no demostrados |
| T-09 | Agotamiento de recursos | Mitigado | Límites de filas, tamaño de archivo y extensiones; monitoreo/quotas del servidor siguen como infraestructura |
| T-10 | Pérdida de datos o modelos | Parcial | Existe integridad y control de versiones, pero falta backup separado y restauración probada |
| T-11 | Elevación por cuenta o permiso excesivo | Parcial | RBAC en `security.py`, `main.py` y Fase 2; no está demostrado un usuario operativo no administrador |
| T-12 | Compromiso mediante dependencia | Mitigado | Entorno virtual, pins con hashes, pip-audit y controles CI |

## Resumen

- Mitigado: implementado en código y demostrado por pruebas o escaneos.
- Parcial: requiere infraestructura, operación externa o evidencia adicional no disponible en el repositorio.
- Aceptado: no aplica a ninguna entrada como estado final; las limitaciones están documentadas como parciales o se pueden justificar fuera de código por el usuario.
