import dataclasses

import numpy as np
import sys
sys.modules["numpy.linalg.linalg"] = np.linalg


if not hasattr(dataclasses, "_replace"):

    def _replace(self, /, **changes):
        for f in dataclasses.fields(self):
            if not f.init:
                if f.name in changes:
                    raise TypeError(
                        f"field {f.name!r} is declared with init=False, "
                        "it cannot be specified with replace()"
                    )
                continue

            if f.name not in changes:
                changes[f.name] = getattr(self, f.name)

        return self.__class__(**changes)

    dataclasses._replace = _replace



from pathlib import Path
import cloudpickle
import pandas as pd
import streamlit as st
import folium
from pyproj import Transformer
from streamlit_folium import st_folium

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
MODELS = BASE / "models"
RESULTS = BASE / "results"

FEATURES = [
    "latitude", "longitude",
    "band1", "band2", "band3", "band4",
    "band5", "band6", "band7"
]

# El artículo de referencia indica que las coordenadas están proyectadas
# en Web Mercator EPSG:3857. En los CSV, latitude contiene X (Este-Oeste)
# y longitude contiene Y (Norte-Sur), por lo que se intercambian antes
# de transformar a WGS84 para Folium.
PROJECTED_TO_WGS84 = Transformer.from_crs(
    "EPSG:3857", "EPSG:4326", always_xy=True
)
WGS84_TO_PROJECTED = Transformer.from_crs(
    "EPSG:4326", "EPSG:3857", always_xy=True
)

st.set_page_config(
    page_title="Clasificación de irradiancia",
    page_icon="☀️",
    layout="wide"
)

st.title("☀️ Clasificación de irradiancia con funciones Kernel")
st.caption(
    "Comparación de configuraciones seleccionadas para Landsat y MODIS."
)

def available_datasets():
    result = []
    for name in ("landsat", "modis"):
        csv_ok = (DATA / f"{name}_model.csv").exists()
        pkl_ok = any((MODELS / f"{name}_model_{i}.pkl").exists() for i in (1, 2, 3))
        if csv_ok and pkl_ok:
            result.append(name)
    return result

@st.cache_resource
def load_model(path_str):
    with open(path_str, "rb") as f:
        return cloudpickle.load(f)

@st.cache_data
def load_data(path_str):
    return pd.read_csv(path_str)

def get_models(dataset):
    models = {}
    for i in (1, 2, 3):
        path = MODELS / f"{dataset}_model_{i}.pkl"
        if path.exists():
            artifact = load_model(str(path))
            models[artifact["configuration"]] = artifact
    return models

def add_geographic_coordinates(df):
    df = df.copy()

    # CSV original:
    # latitude  = X proyectada (metros), normalmente negativa
    # longitude = Y proyectada (metros)
    x = df["latitude"].astype(float).to_numpy()
    y = df["longitude"].astype(float).to_numpy()

    lon, lat = PROJECTED_TO_WGS84.transform(x, y)

    df["map_latitude"] = lat
    df["map_longitude"] = lon
    return df

def predict_dataset(dataset, artifact):
    df = load_data(str(DATA / f"{dataset}_model.csv")).copy()
    X = df[artifact["features"]].to_numpy(dtype=float)

    model = artifact["model"]
    df["predicted_class"] = model.predict(X)

    try:
        scores = model.decision_function(X)
        scores = np.asarray(scores)
        if scores.ndim == 1:
            df["score"] = np.abs(scores)
        else:
            df["score"] = np.max(scores, axis=1)
    except Exception:
        df["score"] = np.nan

    return add_geographic_coordinates(df)

def metrics_for_models(dataset, names):
    path = RESULTS / f"{dataset}_results.csv"

    if not path.exists():
        return pd.DataFrame()

    results = pd.read_csv(path)
    return results[results["configuration"].isin(names)].copy()

def metric_table(results):
    desired = [
        "configuration",
        "holdout_accuracy",
        "holdout_f1_macro",
        "holdout_auc_ovr",
        "holdout_mcc"
    ]

    available = [c for c in desired if c in results.columns]
    table = results[available].copy()

    rename = {
        "configuration": "Configuración",
        "holdout_accuracy": "Accuracy",
        "holdout_f1_macro": "F1 macro",
        "holdout_auc_ovr": "AUC",
        "holdout_mcc": "MCC"
    }

    return table.rename(columns=rename)

def make_map(df, title, key):
    center = [
        float(df["map_latitude"].mean()),
        float(df["map_longitude"].mean())
    ]

    mp = folium.Map(
        location=center,
        zoom_start=8,
        control_scale=True
    )

    colors = [
        "blue", "green", "orange", "red",
        "purple", "darkred", "cadetblue", "darkgreen"
    ]

    for _, row in df.iterrows():
        cls = int(row["predicted_class"])

        popup = folium.Popup(
            f"""
            <b>{title}</b><br>
            Clase: {cls}<br>
            Latitud: {row["map_latitude"]:.6f}<br>
            Longitud: {row["map_longitude"]:.6f}<br>
            Irradiancia observada: {row["value"]:.4f} W/m²
            """,
            max_width=300
        )

        folium.CircleMarker(
            location=[
                float(row["map_latitude"]),
                float(row["map_longitude"])
            ],
            radius=5,
            color=colors[cls % len(colors)],
            fill=True,
            fill_opacity=0.75,
            popup=popup
        ).add_to(mp)

    return mp

def nearest_point(df, click):
    if not click:
        return None

    lat = float(click["lat"])
    lon = float(click["lng"])

    # Convertimos el clic del mapa (WGS84) al sistema proyectado original.
    x_click, y_click = WGS84_TO_PROJECTED.transform(lon, lat)

    # En el CSV: latitude = X, longitude = Y.
    distance = (
        (df["latitude"].astype(float) - x_click) ** 2
        + (df["longitude"].astype(float) - y_click) ** 2
    )

    return df.loc[distance.idxmin()]

datasets = available_datasets()

if not datasets:
    st.error(
        "No se encontraron los CSV y modelos. "
        "Verifica que data/ y models/ estén dentro del proyecto."
    )
    st.stop()

with st.sidebar:
    st.header("Configuración")

    dataset = st.selectbox(
        "Dataset",
        datasets,
        format_func=lambda x: x.upper()
    )

models = get_models(dataset)
model_names = list(models.keys())

if len(model_names) < 2:
    st.error("Se necesitan al menos dos modelos exportados para comparar.")
    st.stop()

with st.sidebar:
    model_a_name = st.selectbox(
        "Modelo A",
        model_names,
        index=0
    )

    model_b_name = st.selectbox(
        "Modelo B",
        model_names,
        index=1 if len(model_names) > 1 else 0
    )

artifact_a = models[model_a_name]
artifact_b = models[model_b_name]

df_a = predict_dataset(dataset, artifact_a)
df_b = predict_dataset(dataset, artifact_b)

st.subheader("1. Modelos seleccionados")

c1, c2 = st.columns(2)

with c1:
    st.info(f"**Modelo A:** `{model_a_name}`")

with c2:
    st.info(f"**Modelo B:** `{model_b_name}`")

st.subheader("2. Comparación de métricas")

metrics = metrics_for_models(
    dataset,
    [model_a_name, model_b_name]
)

if metrics.empty:
    st.warning("No se encontró el archivo de resultados para este dataset.")
else:
    st.dataframe(
        metric_table(metrics),
        use_container_width=True,
        hide_index=True
    )

st.subheader("3. Comparación geográfica")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown(f"**Modelo A — {model_a_name}**")
    event_a = st_folium(
        make_map(df_a, "Modelo A", "map_a"),
        height=520,
        width=None,
        key="map_a"
    )

with col_b:
    st.markdown(f"**Modelo B — {model_b_name}**")
    event_b = st_folium(
        make_map(df_b, "Modelo B", "map_b"),
        height=520,
        width=None,
        key="map_b"
    )

st.subheader("4. Consulta de clasificación por punto")

left, right = st.columns(2)

with left:
    st.markdown("### Modelo A")

    if event_a and event_a.get("last_object_clicked"):
        row = nearest_point(df_a, event_a["last_object_clicked"])

        st.success(
            f"Clase predicha: **{int(row['predicted_class'])}**"
        )
        st.write(
            f"Latitud: `{row['map_latitude']:.6f}`"
        )
        st.write(
            f"Longitud: `{row['map_longitude']:.6f}`"
        )
        st.write(
            f"Irradiancia: `{row['value']:.4f} W/m²`"
        )
    else:
        st.info("Haz clic sobre un punto del mapa A.")

with right:
    st.markdown("### Modelo B")

    if event_b and event_b.get("last_object_clicked"):
        row = nearest_point(df_b, event_b["last_object_clicked"])

        st.success(
            f"Clase predicha: **{int(row['predicted_class'])}**"
        )
        st.write(
            f"Latitud: `{row['map_latitude']:.6f}`"
        )
        st.write(
            f"Longitud: `{row['map_longitude']:.6f}`"
        )
        st.write(
            f"Irradiancia: `{row['value']:.4f} W/m²`"
        )
    else:
        st.info("Haz clic sobre un punto del mapa B.")

st.divider()

st.caption(
    "Las coordenadas del CSV están en Web Mercator (EPSG:3857) y "
    "se transforman a WGS84 para la visualización interactiva."
)
