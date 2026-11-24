# Revisión criptográfica y de dependencias

## Criptografía

| Componente | Configuración | Justificación |
|---|---|---|
| Integridad | SHA-256 sobre CSV y artefactos registrados | Detecta cambios de contenido sin depender de cifras obsoletas como MD5 o SHA-1 |
| Manifiesto | HMAC-SHA256 con `PIPELINE_HMAC_KEY` de al menos 32 bytes | Un hash simple no autentica quién creó el manifiesto; HMAC demuestra la integridad y autenticidad mediante clave secreta |
| Contraseñas/tokens | scrypt con `n=2^15`, `r=8`, `p=1`, `maxmem=64 MiB` y salt aleatorio de 16 bytes | Aumenta el costo de fuerza bruta; la sal evita hashes idénticos para tokens iguales |
| Comparación | `hmac.compare_digest()` | Reduce el riesgo de ataques de comparación por tiempo |
| Claves | Variables de entorno cargadas desde `.env` por `python-dotenv`; `.env` ignorado por Git | Evita almacenar credenciales en el código o el historial y permite configurar procesos locales |

No se implementó criptografía personalizada. Los algoritmos usados son primitivas estándar de Python. Una clave HMAC débil o reutilizada puede invalidar la autenticidad del manifiesto; debe generarse aleatoriamente y administrarse fuera del repositorio.

Limitación conocida: `pickle` y `joblib.load()` pueden ejecutar código durante la deserialización. El proyecto reduce el riesgo verificando el hash antes de cargar `.pkl`, aceptando solo entradas registradas y usando pruebas con objetos maliciosos. La hardening adicional sería sustituir los encoders por formatos de datos no ejecutables como JSON.

## Dependencias

El archivo `requirements.in` declara dependencias directas. `requirements.txt` fue generado con `pip-compile --generate-hashes` y contiene el cierre transitivo con hashes SHA-256.

Dependencias principales:

| Paquete | Versión | Uso |
|---|---|---|
| TensorFlow/Keras | 2.21.0 | Arquitectura y entrenamiento del modelo |
| NumPy | 2.4.6 | Tensores y arrays `.npy` |
| pandas | 3.1.1 | Lectura y limpieza de CSV |
| scikit-learn | 1.9.1 | División, encoders y escalado |
| joblib | 1.6.0 | Persistencia controlada de preprocesadores |
| Matplotlib/Seaborn | 3.11.2/0.13.2 | Gráficas y evidencia visual |

Los paquetes se consideran confiables por procedencia en PyPI, versión fijada, hashes reproducibles y auditoría posterior con pip-audit.

## Estado de auditoría

- `pip-audit 2.10.1`: no encontró vulnerabilidades conocidas.
- `requirements.txt`: instalación verificada con `pip install --require-hashes -r requirements.txt`.
- No se hicieron upgrades de versiones sin una vulnerabilidad demostrada.

Los hashes protegen integridad y reproducibilidad, pero no prueban por sí solos que una versión publicada sea segura; pip-audit complementa esa verificación.

## Estado de carga de `.env`

Se añadió `python-dotenv==1.2.4`. `security.py` llama `load_dotenv()` al importar el módulo. Una prueba real creó temporalmente un `.env`, eliminó la variable del proceso y confirmó `dotenv_loaded True` y `secret_ok True`. El `.env` temporal fue eliminado al finalizar.

Revisa `reports/pip_outdated.txt` para los paquetes con una versión posterior disponible. El estado actual debe actualizarse solo después de revisar compatibilidad y ejecutar pruebas.

Resultado real de `pip list --outdated`: solo `h5py` aparece desactualizado (`3.14.0` → `3.16.0`). No se actualizó porque TensorFlow `2.21.0` declara `h5py<3.15.0`; conservar `3.14.0` evita romper compatibilidad sin vulnerabilidad conocida.
