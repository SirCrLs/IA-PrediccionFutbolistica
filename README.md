# IA- Predicción Futbolistica

Un sistema integral de **Machine Learning** y **Deep Learning** diseñado para procesar datos históricos del fútbol europeo de primer nivel (*Big 5 Leagues*), realizar análisis exploratorio (EDA) y entrenar redes neuronales secuenciales y multisalida para predecir resultados de partidos, goles y estadísticas disciplinarias.

---

## Características del Proyecto

- **Pipeline de Datos Automatizado:** Carga, limpieza y normalización de datasets históricos en formato CSV.
- **Análisis Exploratorio de Datos (EDA):** Generación automática de métricas visuales sobre distribución de victorias, correlaciones, promedios de goles y análisis disciplinario (tarjetas por equipo y condición).
- **División Estratificada (70 / 10 / 20):** Separación adecuada para evitar la fuga de información (*data leakage*):
  - **70%** Entrenamiento (*Training*)
  - **10%** Validación (*Validation*)
  - **20%** Evaluación Final (*Test*)
- **Modelos de Redes Neuronales (TensorFlow / Keras):**
  - **Modelo de Clasificación:** Predicción del resultado final (Local / Empate / Visitante).
  - **Modelo Multisalida (*Multi-output*):** Predicción simultánea de resultado del partido, tarjetas y métricas de goles.

---

## Estructura del Repositorio

```text
├── footballdata/
│   └── Big 5 Leagues (05-06 to 18-19)/  # Datasets raw por liga y temporada
├── processed_data/                      # Archivos binarios (.npy) procesados
│   ├── X_train.npy / y_train.npy
│   ├── X_val.npy / y_val.npy
│   └── X_test.npy / y_test.npy
├── Resultados/                          # Artefactos visuales y modelos entrenados
│   ├── *.png                            # Gráficos de EDA, pérdida, MAE y matrices de confusión
│   └── *.keras / *.h5                   # Modelos guardados en Keras/HDF5
├── .gitignore
├── Fase1_preparacion.py                 # ETL, EDA, codificación y partición de datos
├── Fase2_entrenamiento.py              # Arquitectura, entrenamiento y evaluación de la Red Neuronal
└── Main.py                              # Script principal que orquesta el pipeline completo
```

# Resultados y Visualizaciones Generadas
El pipeline guarda automáticamente los reportes gráficos y modelos entrenados dentro de la carpeta Resultados/:

## Análisis Exploratorio de Datos (EDA)
- distribucion_resultados.png: Balance de clases para resultados (Local / Empate / Visitante).

- mapa_correlaciones.png: Matriz de correlación entre las características del dataset.

- goles_promedio.png & promedio_tarjetas_global.png: Análisis comparativo de goles y conducta disciplinaria.

- top10_indisciplinados.png & top10_disciplinados.png: Rankings de equipos por acumulación de tarjetas.

# Métricas del Modelo
- precision_entrenamiento.png & perdida_total.png: Curvas de aprendizaje por época.

- matriz_confusion.png & matriz_confusion_multi.png: Evaluación de la capacidad predictiva.

- mae_goles.png & mae_tarjetas.png: Error absoluto medio (Mean Absolute Error) en la regresión de métricas continuas.

# Requisitos e Instalación
## Prerrequisitos
Asegúrate de contar con Python 3.8+ instalado en tu sistema.

## Dependencias
Puedes instalar los paquetes necesarios ejecutando:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

# 🛠️ Tecnologías Utilizadas
- Lenguaje: Python
- Manipulación de Datos: pandas, numpy
- Visualización: matplotlib, seaborn
- Machine Learning & Preprocesamiento: scikit-learn (StandardScaler, LabelEncoder, train_test_split)
- Deep Learning: tensorflow / keras (Sequential, Dense, Dropout)
EOF
