def ejecutar_fase2():
    # ============================================
    # FASE 2 - RED NEURONAL MULTISALIDA
    # Predicción de resultados, goles y tarjetas
    # ============================================

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import os
    import joblib
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from sklearn.metrics import classification_report, confusion_matrix

    # 1. CARGA DE DATOS
    print("Cargando los conjuntos de datos preprocesados...\n")
    X_train = np.load("processed_data/X_train.npy")
    X_val = np.load("processed_data/X_val.npy")
    X_test = np.load("processed_data/X_test.npy")
    y_train = np.load("processed_data/y_train.npy")
    y_val = np.load("processed_data/y_val.npy")
    y_test = np.load("processed_data/y_test.npy")

    print(f"Datos cargados correctamente:")
    print(f"   - Entrenamiento: {X_train.shape[0]} muestras")
    print(f"   - Validación:    {X_val.shape[0]} muestras")
    print(f"   - Prueba:        {X_test.shape[0]} muestras\n")

    # Separar las tres salidas desde el arreglo y_train, y_val, y_test

    yres_train, yg_train, yt_train = y_train[:,
                                             0], y_train[:, 1:3], y_train[:, 3:5]
    yres_val, yg_val, yt_val = y_val[:, 0], y_val[:, 1:3], y_val[:, 3:5]
    yres_test, yg_test, yt_test = y_test[:, 0], y_test[:, 1:3], y_test[:, 3:5]

    # 2. DEFINICIÓN DEL MODELO MULTISALIDA

    print(" Construyendo el modelo neuronal multisalida...\n")

    entrada = Input(shape=(X_train.shape[1],), name="entrada_principal")
    
    """
    # ==== PRIMER MODELO DE 256 ====
    # Capa base compartida

    x = Dense(256, activation="relu")(entrada)
    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)

    # Salida 1: resultado del partido (clasificación)
    out_resultado = Dense(3, activation="softmax", name="resultado")(x)

    # Salida 2: predicción de goles locales y visitantes
    out_goles = Dense(2, activation="relu", name="goles")(x)

    # Salida 3: predicción de tarjetas amarillas/rojas
    out_tarjetas = Dense(2, activation="relu", name="tarjetas")(x)
    

    # ==== SEGUNDO MODELO DE RAMAS ====
    
    # Base compartida
    x = Dense(512, activation="relu")(entrada)
    x = Dropout(0.3)(x)
    x = Dense(256, activation="relu")(x)

    # resultado
    r1 = Dense(128, activation="relu")(x)
    out_resultado = Dense(3, activation="softmax", name="resultado")(r1)

    # goles
    r2 = Dense(128, activation="relu")(x)
    out_goles = Dense(2, activation="relu", name="goles")(r2)

    # tarjetas
    r3 = Dense(128, activation="relu")(x)
    out_tarjetas = Dense(2, activation="relu", name="tarjetas")(r3)
    
    """
    
    # ==== NUEVA COMBINACIÓN: Capa Única (Simplificación) ====
    x = Dense(128, activation="relu")(entrada) # Una sola capa oculta
    x = Dropout(0.2)(x) # Reducimos el Dropout por si la red es muy pequeña
    
    # Salida 1: resultado del partido (clasificación)
    out_resultado = Dense(3, activation="softmax", name="resultado")(x)

    # Salida 2: predicción de goles
    out_goles = Dense(2, activation="relu", name="goles")(x)

    # Salida 3: predicción de tarjetas
    out_tarjetas = Dense(2, activation="relu", name="tarjetas")(x)
    
    # Crear modelo multisalida
    model = Model(inputs=entrada, outputs=[
                  out_resultado, out_goles, out_tarjetas])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={
            "resultado": "sparse_categorical_crossentropy",
            "goles": "mse",
            "tarjetas": "mse"
        },
        metrics={
            "resultado": "accuracy",
            "goles": "mae",
            "tarjetas": "mae"
        }
    )

    print("Resumen del modelo:")
    model.summary()

    # 3. ENTRENAMIENTO DEL MODELO
    print("\n Iniciando entrenamiento del modelo ")
    history = model.fit(
        X_train,
        {"resultado": yres_train, "goles": yg_train, "tarjetas": yt_train},
        validation_data=(X_val, {"resultado": yres_val,
                         "goles": yg_val, "tarjetas": yt_val}),
        epochs=100,
        batch_size=128,
        verbose=1
    )

    # 4. EVALUACIÓN FINAL
    print("\n Evaluando el modelo en el conjunto de prueba...")

    resultados = model.evaluate(
        X_test,
        {"resultado": yres_test, "goles": yg_test, "tarjetas": yt_test},
        verbose=1
    )

    loss_total = resultados[0]
    acc_resultado = resultados[4]
    mae_goles = resultados[5]
    mae_tarjetas = resultados[6]

    print(f"\nPérdida total: {loss_total:.4f}")
    print(f"Exactitud (resultado): {acc_resultado * 100:.2f}%")
    print(f" MAE Goles: {mae_goles:.3f}")
    print(f"MAE Tarjetas: {mae_tarjetas:.3f}\n")

    # === GRÁFICAS DE ENTRENAMIENTO ===
    print("\nGenerando gráficas de entrenamiento...\n")
    os.makedirs("Resultados", exist_ok=True)

    # Gráfica de precisión (accuracy) para la salida "resultado"
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['resultado_accuracy'], label='Entrenamiento')
    plt.plot(history.history['val_resultado_accuracy'], label='Validación')
    plt.title('Precisión del Modelo - Resultado del Partido')
    plt.xlabel('Épocas')
    plt.ylabel('Exactitud')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Resultados/precision_entrenamiento.png")
    plt.close()

    # Gráfica de pérdida total
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Entrenamiento')
    plt.plot(history.history['val_loss'], label='Validación')
    plt.title('Pérdida Total del Modelo')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Resultados/perdida_total.png")
    plt.close()

    # Gráfica MAE goles
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['goles_mae'], label='Entrenamiento')
    plt.plot(history.history['val_goles_mae'], label='Validación')
    plt.title('Error Medio Absoluto (MAE) - Predicción de Goles')
    plt.xlabel('Épocas')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Resultados/mae_goles.png")
    plt.close()

    # Gráfica MAE tarjetas
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['tarjetas_mae'], label='Entrenamiento')
    plt.plot(history.history['val_tarjetas_mae'], label='Validación')
    plt.title('Error Medio Absoluto (MAE) - Predicción de Tarjetas')
    plt.xlabel('Épocas')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Resultados/mae_tarjetas.png")
    plt.close()

    # 5. VALIDACIÓN DETALLADA (MATRIZ Y REPORTE)
    print("Generando matriz de confusión y reporte de clasificación...\n")

    yres_pred = np.argmax(model.predict(X_test)[0], axis=1)

    labels = ["Local", "Empate", "Visitante"]
    cm = confusion_matrix(yres_test, yres_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm",
                xticklabels=labels, yticklabels=labels)
    plt.title("Matriz de Confusión - Resultado del Partido")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig("Resultados/matriz_confusion.png")
    plt.close()

    print(classification_report(yres_test, yres_pred, target_names=labels))

    # 6. PREDICCIONES DE EJEMPLO (10 PARTIDOS)
    print("\nPredicciones de ejemplo (primeros 10 partidos del conjunto de prueba):\n")

    le_home = joblib.load("processed_data/le_home.pkl")
    le_away = joblib.load("processed_data/le_away.pkl")
    df_test_ref = pd.read_csv("processed_data/df_test.csv")

    pred_resultado, pred_goles, pred_tarjetas = model.predict(X_test[:10])

    for i in range(10):
        equipo_local = le_home.inverse_transform(
            [int(df_test_ref.iloc[i]["HomeTeam"])])[0]
        equipo_visitante = le_away.inverse_transform(
            [int(df_test_ref.iloc[i]["AwayTeam"])])[0]

        real_res = ["Local", "Empate", "Visitante"][int(yres_test[i])]
        pred_res = ["Local", "Empate", "Visitante"][int(
            np.argmax(pred_resultado[i]))]

        print(f"Partido {i + 1}: {equipo_local} vs {equipo_visitante}")
        print(f"  Resultado real: {real_res} | Predicho: {pred_res}")
        print(
            f"  Goles reales: {yg_test[i]} | Predichos: {pred_goles[i].round(1)}")
        print(
            f"  Tarjetas reales: {yt_test[i]} | Predichas: {pred_tarjetas[i].round(1)}\n")

    # 7. PRUEBA DE GENERALIZACIÓN / ROBUSTEZ
    print("Evaluando robustez con datos perturbados\n")

    ruido = np.random.normal(0, 0.1, X_test.shape)
    X_noisy = X_test + ruido

    resultados_ruido = model.evaluate(
        X_noisy,
        {"resultado": yres_test, "goles": yg_test, "tarjetas": yt_test},
        verbose=0
    )

    loss_r, acc_r, loss_g, mae_g, loss_t, mae_t = resultados_ruido[0:6]
    print(f"- Pérdida con ruido: {loss_r:.6f}")
    print(f"- Exactitud con ruido: {acc_r * 100:.2f}%")
    print(f"- MAE Goles con ruido: {mae_g:.3f}")
    print(f"- MAE Tarjetas con ruido: {mae_t:.3f}\n")

    # 8. GUARDADO DEL MODELO
    os.makedirs("Resultados", exist_ok=True)
    model.save("Resultados/modelo_multisalida.keras")
    print("Modelo multisalida guardado correctamente en 'Resultados/'\n")

    print("Fase 2 completada con éxito.")
