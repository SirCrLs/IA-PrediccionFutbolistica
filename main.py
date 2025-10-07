import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# CONFIGURACIÓN INICIAL
output_dir = "Resultados"
os.makedirs(output_dir, exist_ok=True)
leagues = ["E0", "E1", "E2", "E3", "D1", "F1", "I1", "SP1"]
base_path = "footballdata/Big 5 Leagues (05-06 to 18-19)"

dataframes = []

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


cols = [
    "League", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
    "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC",
    "HY", "AY", "HR", "AR", "B365H", "B365D", "B365A"
]

df = df_raw[cols].dropna()
print(f"Datos limpios: {df.shape[0]} registros válidos")

label_map = {"H": 0, "D": 1, "A": 2}
df["FTR"] = df["FTR"].map(label_map)

le_home, le_away, le_league = LabelEncoder(), LabelEncoder(), LabelEncoder()
df["HomeTeam"] = le_home.fit_transform(df["HomeTeam"])
df["AwayTeam"] = le_away.fit_transform(df["AwayTeam"])
df["League"] = le_league.fit_transform(df["League"])


# ANÁLISIS EXPLORATORIO GENERAL
print("\nEstadísticas generales:")
print(df.describe())

# --- Gráfico 1: distribución de resultados ---
plt.figure(figsize=(6,4))
ax = sns.countplot(x=df["FTR"])
plt.title("Distribución de Resultados (0=Local,1=Empate,2=Visitante)")
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2, p.get_height() + 200,
            f"{int(p.get_height())}", ha='center', va='bottom', fontsize=9)
plt.savefig(os.path.join(output_dir, "distribucion_resultados.png"))
plt.close()

# --- Gráfico 2: mapa de correlaciones ---
plt.figure(figsize=(10,8))
corr_matrix = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr_matrix, cmap="coolwarm", annot=False)
plt.title("Mapa de Correlaciones (solo variables numéricas)")
plt.savefig(os.path.join(output_dir, "mapa_correlaciones.png"))
plt.close()

# --- Gráfico 3: promedio de goles por condición ---
plt.figure(figsize=(8,5))
ax = sns.barplot(x=["Local", "Visitante"], y=[df["FTHG"].mean(), df["FTAG"].mean()])
plt.title("Promedio de Goles por Condición")
plt.ylabel("Goles promedio")
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2, p.get_height() + 0.02,
            f"{p.get_height():.2f}", ha='center', va='bottom', fontsize=10)
plt.savefig(os.path.join(output_dir, "goles_promedio.png"))
plt.close()


# ANÁLISIS DE TARJETAS (DISCIPLINA)
df["Total_Yellow"] = df["HY"] + df["AY"]
df["Total_Red"] = df["HR"] + df["AR"]

# --- Gráfico 4: promedio global de tarjetas ---
avg_yellow = df["Total_Yellow"].mean()
avg_red = df["Total_Red"].mean()

df_tarjetas = pd.DataFrame({
    "Tipo": ["Amarillas", "Rojas"],
    "Promedio": [avg_yellow, avg_red]
})

plt.figure(figsize=(7,5))
ax = sns.barplot(data=df_tarjetas, x="Tipo", y="Promedio", hue="Tipo",
                 palette={"Amarillas": "gold", "Rojas": "red"}, legend=False)
plt.title("Promedio de Tarjetas por Partido (Todas las ligas)")
plt.ylabel("Promedio de tarjetas")
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2, p.get_height() + 0.02,
            f"{p.get_height():.2f}", ha='center', va='bottom', fontsize=10)
plt.savefig(os.path.join(output_dir, "promedio_tarjetas_global.png"))
plt.close()

# --- Gráfico 5: tarjetas por condición ---
yellow_local = df["HY"].mean()
yellow_away = df["AY"].mean()
red_local = df["HR"].mean()
red_away = df["AR"].mean()

tarjetas_condicion = pd.DataFrame({
    "Condición": ["Local", "Visitante"],
    "Amarillas": [yellow_local, yellow_away],
    "Rojas": [red_local, red_away]
})

fig, ax = plt.subplots(figsize=(8,5))
tarjetas_condicion.set_index("Condición")[["Amarillas", "Rojas"]].plot(
    kind="bar", color=["gold", "red"], ax=ax
)
plt.title("Promedio de Tarjetas por Condición")
plt.ylabel("Promedio de tarjetas")
plt.xticks(rotation=0)
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", label_type="edge", fontsize=9)
plt.savefig(os.path.join(output_dir, "tarjetas_por_condicion.png"))
plt.close()


# TOP 10 EQUIPOS MÁS / MENOS INDISCIPLINADOS
df_equipo_disciplina = df_raw.groupby("HomeTeam")[["HY", "HR"]].mean().reset_index()
df_equipo_disciplina.columns = ["Equipo", "Promedio_HY", "Promedio_HR"]

df_away = df_raw.groupby("AwayTeam")[["AY", "AR"]].mean().reset_index()
df_away.columns = ["Equipo", "Promedio_AY", "Promedio_AR"]

df_equipo = pd.merge(df_equipo_disciplina, df_away, on="Equipo", how="outer").fillna(0)
df_equipo["Promedio_Total_Amarillas"] = (df_equipo["Promedio_HY"] + df_equipo["Promedio_AY"]) / 2
df_equipo["Promedio_Total_Rojas"] = (df_equipo["Promedio_HR"] + df_equipo["Promedio_AR"]) / 2
df_equipo["Total_Promedio_Tarjetas"] = df_equipo["Promedio_Total_Amarillas"] + df_equipo["Promedio_Total_Rojas"]

top10_indisciplinados = df_equipo.sort_values("Total_Promedio_Tarjetas", ascending=False).head(10)
top10_disciplinados = df_equipo.sort_values("Total_Promedio_Tarjetas", ascending=True).head(10)

# --- Gráfico 6: Indisciplinados ---
plt.figure(figsize=(10,6))
ax = sns.barplot(
    data=top10_indisciplinados,
    x="Total_Promedio_Tarjetas",
    y="Equipo",
    color="red"
)
plt.title("Top 10 Equipos Más Indisciplinados (Promedio de Tarjetas por Partido)")
plt.xlabel("Promedio de Tarjetas Totales")
plt.ylabel("Equipo")
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", label_type="edge", fontsize=9)
plt.savefig(os.path.join(output_dir, "top10_indisciplinados.png"))
plt.close()

# --- Gráfico 7: Disciplinados ---
plt.figure(figsize=(10,6))
ax = sns.barplot(
    data=top10_disciplinados,
    x="Total_Promedio_Tarjetas",
    y="Equipo",
    color="green"
)
plt.title("Top 10 Equipos Más Disciplinados (Promedio de Tarjetas por Partido)")
plt.xlabel("Promedio de Tarjetas Totales")
plt.ylabel("Equipo")
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", label_type="edge", fontsize=9)
plt.savefig(os.path.join(output_dir, "top10_disciplinados.png"))
plt.close()

# NORMALIZACIÓN Y DIVISIÓN DE CONJUNTOS
X = df.drop("FTR", axis=1)
y = df["FTR"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)


# GUARDADO DE ARCHIVOS
os.makedirs("processed_data", exist_ok=True)
np.save("processed_data/X_train.npy", X_train)
np.save("processed_data/X_test.npy", X_test)
np.save("processed_data/y_train.npy", y_train)
np.save("processed_data/y_test.npy", y_test)


print("\nFase 1 completada correctamente")
print("\nResumen de conjuntos de datos:")
print(f"Total de registros originales: {len(X)}")
print(f"Datos de entrenamiento: {len(X_train)} ({len(X_train)/len(X)*100:.2f}%)")
print(f"Datos de prueba: {len(X_test)} ({len(X_test)/len(X)*100:.2f}%)")
