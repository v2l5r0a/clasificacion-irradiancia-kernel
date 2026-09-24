# Clasificación de irradiancia con funciones Kernel

Proyecto académico para comparar pipelines de clasificación de irradiancia usando los datasets Landsat y MODIS.

## Estructura

```text
.
├── app/
│   └── app.py
├── data/
│   ├── landsat_model.csv
│   └── modis_model.csv
├── models/
│   ├── landsat_model_1.pkl
│   ├── landsat_model_2.pkl
│   ├── landsat_model_3.pkl
│   ├── modis_model_1.pkl
│   ├── modis_model_2.pkl
│   └── modis_model_3.pkl
├── results/
│   ├── landsat_results.csv
│   └── modis_results.csv
├── notebooks/
│   └── Clasificacion_Irradiancia_Valeria_CORREGIDO.ipynb
└── requirements.txt
```

## Ejecutar la aplicación

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar:

```bash
streamlit run app/app.py
```

## Funcionalidades

- Selección de dataset Landsat o MODIS.
- Carga de modelos almacenados.
- Selección de dos configuraciones.
- Comparación de Accuracy, F1 macro, AUC y MCC.
- Visualización de las predicciones sobre un mapa.
- Consulta de la clase predicha al hacer clic en un punto.

## Coordenadas

El artículo de referencia describe las coordenadas del dataset como proyectadas en Web Mercator EPSG:3857. En los CSV utilizados en este proyecto, las columnas llamadas `latitude` y `longitude` contienen las componentes proyectadas X e Y respectivamente, por lo que la aplicación las transforma a WGS84 antes de enviarlas a Folium.


## Aplicación web

La aplicación web fue desarrollada con Streamlit y permite visualizar y comparar modelos de clasificación de irradiancia para los datasets Landsat y MODIS.

### Acceso a la aplicación

[🚀 Abrir aplicación web](https://v2l5r0a-clasificacion-irradiancia-kernel-app-cucuzbh.streamlit.app/)

La aplicación permite:

- Seleccionar el dataset Landsat o MODIS.
- Seleccionar dos configuraciones de modelos.
- Comparar Accuracy, F1 macro, AUC y MCC.
- Visualizar las clasificaciones de los modelos sobre mapas interactivos.
- Consultar la clase predicha al seleccionar un punto del mapa.

Referencia:
https://doi.org/10.19053/01211129.v30.n58.2021.13845

## Nota

Los modelos `.pkl` generados por el Notebook se cargan con `cloudpickle`. No se deben publicar credenciales ni datos privados.


## Despliegue

La aplicación se encuentra desplegada en Streamlit Community Cloud y está conectada directamente con el repositorio de GitHub.

- Repositorio: https://github.com/v2l5r0a/clasificacion-irradiancia-kernel
- Aplicación: https://v2l5r0a-clasificacion-irradiancia-kernel-app-cucuzbh.streamlit.app/
