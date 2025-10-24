def ejecutar_fase1():

    # FASE 1 - CARGA, LIMPIEZA, ANÁLISIS Y DIVISIÓN DE DATOS

    import os
    import glob
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    import joblib


    # CONFIGURACIÓN INICIAL
    output_dir = "Resultados"
    os.makedirs(output_dir, exist_ok=True)
    leagues = ["E0", "E1", "E2", "E3", "D1", "F1", "I1", "SP1"]
    base_path = "footballdata/Big 5 Leagues (05-06 to 18-19)"

    dataframes = []


    # CARGA DE TODOS LOS CSV
    for league in leagues:
        league_folder = os.path.join(base_path, league)
        csv_files = glob.glob(os.path.join(league_folder, "*.csv"))
        for file in csv_files:
            try:
                df_temp = pd.read_csv(file)
                df_temp["League"] = league
                df_temp["Season"] = os.path.basename(file).split(".")[0]
                dataframes.append(df_temp)
            except Exception as e:
                print(f"Error al leer {file}: {e}")

    df_raw = pd.concat(dataframes, ignore_index=True)
    print(f"\nDatos combinados: {df_raw.shape[0]} registros totales, {df_raw.shape[1]} columnas")


    # SELECCIÓN Y LIMPIEZA DE VARIABLES
    cols = [
        "League", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
        "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC",
        "HY", "AY", "HR", "AR", "B365H", "B365D", "B365A"
    ]

    df = df_raw[cols].dropna()
    print(f"Datos limpios: {df.shape[0]} registros válidos")


    # TRANSFORMACIÓN DE VARIABLES
    label_map = {"H": 0, "D": 1, "A": 2}
    df["FTR"] = df["FTR"].map(label_map)

    le_home, le_away, le_league = LabelEncoder(), LabelEncoder(), LabelEncoder()
    df["HomeTeam"] = le_home.fit_transform(df["HomeTeam"])
    df["AwayTeam"] = le_away.fit_transform(df["AwayTeam"])
    df["League"] = le_league.fit_transform(df["League"])


    # VARIABLES ADICIONALES DE DISCIPLINA
    df["Total_Yellow"] = df["HY"] + df["AY"]
    df["Total_Red"] = df["HR"] + df["AR"]


    # NORMALIZACIÓN Y DIVISIÓN DE CONJUNTOS (70 / 20 / 10)
    # La red neuronal predecirá:
    #  - Resultado (FTR)
    #  - Goles locales y visitantes (FTHG, FTAG)
    #  - Tarjetas amarillas locales y visitantes (HY, AY)

    X = df.drop(["FTR", "FTHG", "FTAG", "HY", "AY"], axis=1)
    y = df[["FTR", "FTHG", "FTAG", "HY", "AY"]].values  # <-- y con 5 columnas

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Guardar el escalador
    os.makedirs("processed_data", exist_ok=True)
    joblib.dump(scaler, "processed_data/scaler.pkl")

    # Dividir conjuntos
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=df["FTR"]
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp[:, 0]
    )

    # Guardar datos
    np.save("processed_data/X_train.npy", X_train)
    np.save("processed_data/X_val.npy", X_val)
    np.save("processed_data/X_test.npy", X_test)
    np.save("processed_data/y_train.npy", y_train)
    np.save("processed_data/y_val.npy", y_val)
    np.save("processed_data/y_test.npy", y_test)

    print("\nArchivos guardados correctamente con las nuevas salidas:")
    print("\nFase 1 completada correctamente.")
    print(f" - Datos de entrenamiento: {len(X_train)}")
    print(f" - Datos de validación: {len(X_val)}")
    print(f" - Datos de prueba: {len(X_test)}")

