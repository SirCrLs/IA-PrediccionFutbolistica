def ejecutar_fase2():
    # FASE 2 - ENTRENAMIENTO DE RED NEURONAL CON VALIDACIÓN (70/20/10)

    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    import os


    # CARGA DE DATOS PROCESADOS
    X_train = np.load("processed_data/X_train.npy")
    X_val = np.load("processed_data/X_val.npy")
    X_test = np.load("processed_data/X_test.npy")
    y_train = np.load("processed_data/y_train.npy")
    y_val = np.load("processed_data/y_val.npy")
    y_test = np.load("processed_data/y_test.npy")

    print(" Datos cargados correctamente:")
    print(f"Entrenamiento: {X_train.shape[0]} muestras")
    print(f"Validación:    {X_val.shape[0]} muestras")
    print(f"Prueba:        {X_test.shape[0]} muestras")


    # CONFIGURACIÓN DEL MODELO
    num_features = X_train.shape[1]
    num_classes = 3  # 0=Local, 1=Empate, 2=Visitante

    model = Sequential([
        Dense(128, activation='relu', input_shape=(num_features,)),
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

    model.summary()


    # ENTRENAMIENTO DEL MODELO
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=64,
        verbose=1
    )


    # EVALUACIÓN FINAL
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n Exactitud final en conjunto de prueba: {accuracy * 100:.2f}%")


    # VISUALIZACIÓN DE RESULTADOS
    output_dir = "Resultados"
    os.makedirs(output_dir, exist_ok=True)

    # Precisión
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

    # Pérdida
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


    # PREDICCIONES DE EJEMPLO
    predicciones = np.argmax(model.predict(X_test[:15]), axis=1)
    print("\n Predicciones ejemplo (0=Local,1=Empate,2=Visitante):")
    print("Predicho:", predicciones)
    print("Real:    ", y_test[:15])


    # GUARDADO DEL MODELO
    model_path = os.path.join(output_dir, "modelo_prediccion_futbol.h5")
    model.save(model_path)
    print(f"\n Modelo guardado en '{model_path}'")

