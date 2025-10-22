def ejecutar_fase2():
    # FASE 2 - RED NEURONAL PARA PREDICCIÓN DE RESULTADOS DE FÚTBOL
    import numpy as np
    import matplotlib.pyplot as plt
    from tensorflow.keras import Input
    import os
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.model_selection import train_test_split
    import joblib
    import seaborn as sns
    import pandas as pd

    # CARGA DE DATOS
    print("Cargando los conjuntos de datos preprocesados desde la Fase 1...\n")

    X_train = np.load("processed_data/X_train.npy")
    X_val = np.load("processed_data/X_val.npy")
    X_test = np.load("processed_data/X_test.npy")
    y_train = np.load("processed_data/y_train.npy")
    y_val = np.load("processed_data/y_val.npy")
    y_test = np.load("processed_data/y_test.npy")

    print("Datos cargados correctamente:")
    print(f"- Conjunto de entrenamiento: {X_train.shape[0]} muestras")
    print(f"- Conjunto de validación:    {X_val.shape[0]} muestras")
    print(f"- Conjunto de prueba:        {X_test.shape[0]} muestras\n")

    #  DEFINICIÓN DE LA ARQUITECTURA DEL MODELO
    num_features = X_train.shape[1]  # Número de columnas (atributos)
    num_classes = 3  # 0=Local, 1=Empate, 2=Visitante

    model = Sequential([
        Input(shape=(num_features,)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Resumen del modelo:")
    model.summary()


    # ENTRENAMIENTO DEL MODELO
    print("\n Iniciando el entrenamiento del modelo...")
    print("   Se usarán 70% de datos para entrenamiento y 10% para validación.\n")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=64,
        verbose=1
    )


    #  EVALUACIÓN EN CONJUNTO DE PRUEBA
    print("\n Evaluando el modelo con el conjunto de prueba (20%)...")

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n Resultados de evaluación final:")
    print(f"   - Pérdida (Loss): {loss:.4f}")
    print(f"   - Exactitud (Accuracy): {accuracy * 100:.2f}%\n")


    # VISUALIZACIÓN DEL ENTRENAMIENTO
    output_dir = "Resultados"
    os.makedirs(output_dir, exist_ok=True)

    # Gráfica de precisión
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Validación')
    plt.title('Precisión del Modelo por Época')
    plt.xlabel('Época')
    plt.ylabel('Exactitud')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "precisión_entrenamiento.png"))
    plt.show()

    # Gráfica de pérdida
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Entrenamiento')
    plt.plot(history.history['val_loss'], label='Validación')
    plt.title('Pérdida (Loss) del Modelo por Época')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "pérdida_entrenamiento.png"))
    plt.show()


    #  VALIDACIÓN DETALLADA DEL MODELO
    print(" Validando el modelo con métricas de clasificación\n")

    # Predicciones sobre el conjunto de prueba
    y_pred = np.argmax(model.predict(X_test), axis=1)

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    labels = ["Local", "Empate", "Visitante"]

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title("Matriz de Confusión - Conjunto de Prueba")
    plt.xlabel("Predicción")
    plt.ylabel("Valor Real")
    plt.savefig(os.path.join(output_dir, "matriz_confusion.png"))
    plt.show()

    # Reporte de clasificación
    print("Reporte de Clasificación:")
    print(classification_report(y_test, y_pred, target_names=labels))

    # PREDICCIONES DE EJEMPLO

    print("\nEjemplos de predicciones (primeros 10 partidos del conjunto de prueba):\n")

    # Cargar codificadores y el CSV de referencia
    le_home = joblib.load("processed_data/le_home.pkl")
    le_away = joblib.load("processed_data/le_away.pkl")
    df_ref = pd.read_csv("processed_data/datos_referencia.csv")

    # Recrear el mismo orden de división de datos
    from sklearn.model_selection import train_test_split
    _, X_temp, _, y_temp = train_test_split(
        df_ref, df_ref["FTR"], test_size=0.3, random_state=42, stratify=df_ref["FTR"]
    )
    df_val = train_test_split(
        X_temp, test_size=2 / 3, random_state=42, stratify=y_temp
    )
    df_test = pd.read_csv("processed_data/df_test.csv")

    # Obtener las primeras 10 muestras del conjunto de prueba
    ejemplos = df_test.head(10).copy()
    predicciones = np.argmax(model.predict(X_test[:10]), axis=1)
    labels = ["Local", "Empate", "Visitante"]

    # Decodificar los nombres de equipos
    for i in range(10):
        equipo_local = le_home.inverse_transform([int(df_test.iloc[i]["HomeTeam"])])[0]
        equipo_visitante = le_away.inverse_transform([int(df_test.iloc[i]["AwayTeam"])])[0]
        real = ["Local", "Empate", "Visitante"][int(df_test.iloc[i]["FTR"])]
        predicho = ["Local", "Empate", "Visitante"][int(predicciones[i])]
        print(f" --Partido {i + 1}: {equipo_local} vs {equipo_visitante}")
        print(f"    Real={real} | Predicho={predicho}")


    # PRUEBA DE GENERALIZACIÓN (ROBUSTEZ)
    print("\n Evaluando la generalización del modelo con datos perturbados...\n")

    # Agregar ruido gaussiano leve al conjunto de prueba (simula incertidumbre)
    ruido = np.random.normal(0, 0.10, X_test.shape)
    X_noisy = X_test + ruido

    loss_noisy, acc_noisy = model.evaluate(X_noisy, y_test, verbose=0)
    print(f"Resultados con datos perturbados:")
    print(f"   - Pérdida con ruido: {loss_noisy:.6f}")
    print(f"   - Exactitud con ruido: {acc_noisy * 100:.2f}%")

    # GUARDADO DEL MODELO
    model_path = os.path.join(output_dir, "modelo_prediccion_futbol.keras")
    model.save(model_path)
    print(f"\n Modelo guardado en: {model_path}")

    print("\n Fase 2 completada exitosamente. El modelo está entrenado, validado y listo para usar.")


