# Portafolio de Data Science — TripleTen

Colección completa de los proyectos del bootcamp de **Data Science de TripleTen**, desde el
análisis exploratorio inicial hasta visión por computadora y el proyecto final de negocio.
Cada sprint se entrega como un notebook autocontenido en `notebooks/`, y el proyecto final
(predicción de churn para Interconnect) vive en `proyectoFinal/` con sus tres etapas.

**Autor:** Gustavo Mandujano Rojas

---

## Entorno y reproducción

- **Python 3.13**, gestionado con [`uv`](https://docs.astral.sh/uv/).

```bash
# Instalar dependencias (crea/actualiza el .venv a partir de uv.lock)
uv sync

# Abrir JupyterLab
uv run jupyter lab

# o el Notebook clásico
uv run jupyter notebook
```

`main.py` es solo el stub generado por `uv init`; todo el trabajo está en los notebooks.

## Datos

Los directorios `data/` y `datasets/` están en `.gitignore`, por lo que **los CSV no se
versionan en este repositorio**. Los conjuntos de datos se descargan desde la plataforma de
TripleTen y se colocan en esas carpetas; cada notebook indica el archivo de origen que espera
(p. ej. `data/insurance_us.csv`, `datasets/final_provider/`).

## Estructura del repositorio

```
.
├── notebooks/        # Un notebook por sprint (1–18) + ejercicios de visión por computadora
├── proyectoFinal/    # Proyecto final: plan, código e informe (caso Interconnect)
├── data/             # Datasets (ignorado por git)
├── datasets/         # Datasets adicionales (ignorado por git)
├── pyproject.toml    # Dependencias del proyecto
└── uv.lock           # Lockfile reproducible
```

## Proyectos por sprint

| Sprint | Proyecto | Tema / Empresa | Técnicas principales |
|--------|----------|----------------|----------------------|
| 1  | [Análisis de comportamiento de usuarios](notebooks/Sprint1_Analisis_Comportamiento_Usuarios.ipynb) | Megaline | EDA, pandas |
| 2  | [Análisis de clientes](notebooks/Sprint2_Analisis_Clientes.ipynb) | — | Preprocesamiento y limpieza de datos |
| 3  | [Proyecto de música](notebooks/Sprint3_Musica_Project.ipynb) | Hábitos musicales por ciudad | Análisis exploratorio, prueba de hipótesis |
| 4  | [Instacart](notebooks/Sprint4_Instacart.ipynb) | Pedidos de supermercado | EDA, visualización |
| 5  | [La mejor tarifa](notebooks/Sprint5_Megaline_Mejor_Tarifa.ipynb) | Megaline | Análisis estadístico, pruebas de hipótesis |
| 6  | [Videojuegos](notebooks/Sprint6_Videojuegos_ICE.ipynb) | Tienda ICE | Análisis de ventas, pruebas de hipótesis |
| 8  | [Taxis de Chicago](notebooks/Sprint8_Taxis_Chicago.ipynb) | Zuber | SQL + análisis de datos |
| 10 | [Recomendación de plan](notebooks/Sprint10_ML_Usuarios_Megaline.ipynb) | Megaline | Clasificación supervisada |
| 11 | [Modelos predictivos de churn](notebooks/Sprint11_Modelos_Predictivos_Churn.ipynb) | Beta Bank | Clasificación con clases desbalanceadas |
| 12 | [OilyGiant](notebooks/Sprint12_Negocios_OilyGiant.ipynb) | Petrolera | Regresión + bootstrapping para elegir región |
| 13 | [Recuperación de oro](notebooks/Sprint13_Recuperacion_De_Oro.ipynb) | Zyfra | Regresión, métrica sMAPE |
| 14 | [Seguros Sure Tomorrow](notebooks/Sprint14_Seguros_Sure_Tomorrow.ipynb) | Sure Tomorrow | kNN, regresión lineal desde cero, ofuscación de datos |
| 15 | [Rusty Bargain](notebooks/Sprint15_Rusty_Bargain_Autos_Usados.ipynb) | Autos usados | Gradient boosting (LightGBM / XGBoost / CatBoost) |
| 16 | [Sweet Lift Taxi](notebooks/Sprint16_Sweet_Lift_Taxi.ipynb) | Pedidos de taxis | Series temporales, pronóstico |
| 17 | [Film Junky Union](notebooks/Sprint17_Film_Junky_Union_IMDB.ipynb) | Reseñas IMDB | NLP, análisis de sentimiento (incl. BERT) |
| 18 | [Verificación de edad](notebooks/Sprint18_Verificacion_Edad_Proyecto.ipynb) | Good Seed | Visión por computadora, ResNet50 (TensorFlow / Keras) |

El Sprint 18 incluye además varios notebooks de apoyo: ejercicios de regresión sobre edad,
una variante en **PyTorch** (`Sprint18_Verificacion_Edad_PyTorch.ipynb`), el notebook de
entrenamiento, la implementación con ResNet y el script `notebooks/run_model_on_gpu.py`.

## Proyecto final — Predicción de churn para Interconnect

Caso de negocio completo para el operador de telecomunicaciones **Interconnect**: anticipar
qué clientes están por cancelar el servicio para intervenir con promociones antes de que se
vayan. Se aborda como **clasificación binaria supervisada** evaluada con **AUC-ROC**.

Entregables, en `proyectoFinal/`:

1. [Plan de trabajo](proyectoFinal/etapa1_plan_de_trabajo.md) — definición del problema,
   análisis de los datos, plan de preparación, modelado y riesgos.
2. [Código de solución](proyectoFinal/etapa2_codigo_solucion.ipynb) — limpieza, merge de las
   cuatro tablas, ingeniería de features y comparación de cinco modelos por validación cruzada.
3. [Informe de solución](proyectoFinal/etapa3_informe_solucion.md) — resultados, hallazgos,
   limitaciones y recomendaciones de negocio.

**Resultado:** el modelo final (**LightGBM** afinado por CV estratificada de 5 pliegues) alcanza
**AUC-ROC = 0.9696** en el holdout del 25 % (exactitud 0.9256), correspondiente a la banda
máxima de evaluación (6 SP). El informe documenta además una posible fuga de información en las
features de antigüedad y propone una versión conservadora del modelo para producción.

## Stack tecnológico

`pandas` · `numpy` · `scikit-learn` · `scipy` · `lightgbm` · `xgboost` · `catboost` ·
`tensorflow` (Keras) · `transformers` · `spacy` · `nltk` · `seaborn` · `plotly` · `jupyterlab`

## Autor

**Gustavo Mandujano Rojas** — Bootcamp de Data Science, TripleTen.
