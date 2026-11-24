# Guía de seguridad

![Comprobaciones de seguridad](https://github.com/SirCrLs/IA-PrediccionFutbolistica/actions/workflows/security.yml/badge.svg)

## Instalación

```text
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --require-hashes -r requirements.txt
```

## Variables de entorno

Copia `.env.example` como `.env` y configura:

- `PIPELINE_HMAC_KEY`: clave HMAC de al menos 32 bytes.
- `PIPELINE_USER`: nombre del usuario registrado.
- `PIPELINE_TOKEN`: token privado del usuario.

`security.py` carga `.env` automáticamente mediante `python-dotenv`; las variables ya existentes en el proceso tienen prioridad.

Nunca hagas commit de `.env`, `users.json`, `audit.log` ni `manifest.json`.

## Crear un usuario

```text
$env:PIPELINE_TOKEN="a-private-token-with-at-least-12-characters"
.venv\Scripts\python.exe tools\make_user.py analyst --role trainer
```

## Ejecutar el pipeline

```text
.venv\Scripts\python.exe main.py phase1
.venv\Scripts\python.exe main.py phase2
.venv\Scripts\python.exe main.py predict --home Arsenal --away Chelsea --league E0
```

Las acciones `phase1` y `phase2` requieren el rol correspondiente. La acción `predict` valida equipos y liga antes de cargar los artefactos de preprocesamiento verificados.

En un clon limpio, ejecuta primero `phase1` para crear `processed_data/` y `manifest.json`. `manifest.json` no se versiona y debe reconstruirse antes de verificar o cargar artefactos.

## Ejecutar las comprobaciones de seguridad

```text
.venv\Scripts\python.exe -m pytest -q --junitxml=reports/pytest.xml
.venv\Scripts\bandit.exe -r . -x ./tests,./.venv -ll
.venv\Scripts\pip-audit.exe -r requirements.txt
```

Los informes se guardan en `reports/` y el workflow de CI los publica como artefactos.

La revisión técnica de criptografía, formato seguro de artefactos y dependencias fijadas está en `reports/crypto_dependencies.md`.

## Configuración manual requerida en GitHub

- Habilita la protección de la rama principal y exige la comprobación de estado del workflow `security`.
- Habilita las alertas de Dependabot y las actualizaciones automáticas de seguridad.
- Habilita el escaneo de secretos y la protección contra pushes con secretos.
- Exige autenticación de dos factores (2FA) en las cuentas relevantes.
- Exige commits firmados o verificados según la política del equipo.
