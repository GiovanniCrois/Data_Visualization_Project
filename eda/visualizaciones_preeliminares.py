from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")
CSV_LOCAL = Path("../water_potability.csv")
CSV_URL = (
    "https://huggingface.co/datasets/jason1966/"
    "adityakadiwal_water-potability/resolve/main/water_potability.csv"
)
OUTPUT_DIR = Path("graficos_water_potability")
OUTPUT_DIR.mkdir(exist_ok=True)

#Cargar los datos: primero intenta el archivo local, después la URL.
if CSV_LOCAL.exists():
    df = pd.read_csv(CSV_LOCAL)
else:
    try:
        df = pd.read_csv(CSV_URL)
    except Exception as exc:
        raise RuntimeError(
            "No se pudo descargar el dataset. Descarga water_potability.csv "
            "desde Kaggle y colócalo junto a este script."
        ) from exc

required = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate", "Conductivity",
    "Organic_carbon", "Trihalomethanes", "Turbidity", "Potability"
]
missing_columns = [c for c in required if c not in df.columns]
if missing_columns:
    raise ValueError(f"Faltan columnas esperadas: {missing_columns}")

#Eliminar datos nulos y Normalizar tipos numéricos .
df = df.dropna(how="all")
for col in required:
    df[col] = pd.to_numeric(df[col], errors="coerce")



#VISUALIZACION UNIVARIADA: histograma de pH
fig, ax = plt.subplots(figsize=(9, 5.5))
sns.histplot(data=df, x="ph", bins=30, kde=True, ax=ax, color="steelblue")
ax.set(
    title="Distribución del pH de las muestras de agua",
    xlabel="pH",
    ylabel="Número de muestras"
)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "01_univariada_distribucion_ph.png", dpi=180)
plt.show()

#VISUALIZACION BIVARIADA dureza vs sulfatos, con color por clase de potabilidad.
# La relación principal es entre dos mediciones; el color distingue la clase.
g = df.dropna(subset=["Hardness", "Sulfate", "Potability"]).copy()
fig, ax = plt.subplots(figsize=(9, 6))
sns.scatterplot(
    data=g, x="Hardness", y="Sulfate", hue="Potability",
    hue_order=[0, 1], alpha=0.55, s=35, ax=ax,
    palette={0: "darkorange", 1: "teal"}
)
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles, ["No potable (0)", "Potable (1)"], title="Potabilidad")
ax.set(
    title="Relación entre dureza y concentración de sulfatos",
    xlabel="Dureza del agua",
    ylabel="Sulfatos"
)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "02_bivariada_dureza_sulfatos.png", dpi=180)
plt.show()

#VISUALIZACION MULTIVARIADA: pH (x), sólidos (y), color=potabilidad,
# tamaño=concentración de cloraminas.
g = df.dropna(subset=["ph", "Solids", "Chloramines", "Potability"]).copy()
# Escalar el tamaño de las burbujas para que sea legible.
span = g["Chloramines"].max() - g["Chloramines"].min()
g["bubble_size"] = 25 + 150 * (g["Chloramines"] - g["Chloramines"].min()) / span if span else 70
fig, ax = plt.subplots(figsize=(10, 6.5))
sns.scatterplot(
    data=g, x="ph", y="Solids", hue="Potability", size="bubble_size",
    sizes=(25, 175), size_norm=(25, 175), alpha=0.45,
    palette={0: "darkorange", 1: "teal"}, ax=ax, legend="brief"
)
ax.set(
    title="Visualización multivariada: pH, sólidos, cloraminas y potabilidad",
    xlabel="pH",
    ylabel="Sólidos disueltos"
)
ax.legend(title="Potabilidad / tamaño", bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "03_multivariada_ph_solidos_cloraminas.png", dpi=180)
plt.show()


#VISUALIZACION DE COMPARACIÓN DE GRUPOS / DISPERSIÓN: boxplot de pH por potabilidad.
# Se muestran mediana, rango intercuartílico y valores atípicos.
g = df.dropna(subset=["ph", "Potability"]).copy()
g["Clase de potabilidad"] = g["Potability"].map(
    {0: "No potable (0)", 1: "Potable (1)"}
)
fig, ax = plt.subplots(figsize=(8.5, 5.5))
sns.boxplot(
    data=g, x="Clase de potabilidad", y="ph",
    order=["No potable (0)", "Potable (1)"], ax=ax,
    showfliers=True
)
ax.set(
    title="Comparación de la distribución del pH por potabilidad",
    xlabel="Clasificación de la muestra",
    ylabel="pH"
)

#MOSTRAR VISUALIZACIONES Y GUARDAR IMÁGENES EN ARCHIVO LOCAL
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "04_boxplot_ph_por_potabilidad.png", dpi=180)
plt.show()
