# Plan de trabajo — Predicción de churn para Interconnect

**Autor:** Gustavo Mandujano Rojas
**Sprint:** Proyecto final TripleTen — Data Science
**Fecha de corte de los datos:** 2020-02-01

---

## 1. Descripción del problema y objetivo de negocio

Interconnect, operador de telecomunicaciones, necesita anticipar qué clientes están por cancelar el servicio para ofrecerles códigos promocionales y planes especiales antes de que se vayan. El problema es **clasificación binaria supervisada**: a partir de la información contractual, personal y de servicios contratados de cada cliente, predecir si seguirá activo.

- **Característica objetivo (definición oficial):** `target = (EndDate == 'No')` derivada de `contract.csv`.
  Clase **1 = cliente activo** (no canceló); clase **0 = cliente cancelado**. Distribución observada: 73.46 % / 26.54 %.
  *Nota práctica:* internamente el modelo aprende a separar ambas clases; como AUC-ROC es invariante a la dirección, el ranking de "riesgo de cancelar" se obtiene ordenando por `P(target = 0)` (o, equivalentemente, `1 − P(target = 1)`).
- **Métrica primaria:** **AUC-ROC**. Tabla oficial de evaluación en puntos SP:

  | AUC-ROC                  | SP  |
  |--------------------------|-----|
  | < 0.75                   | 0   |
  | 0.75 ≤ AUC-ROC < 0.81    | 4   |
  | 0.81 ≤ AUC-ROC < 0.85    | 4.5 |
  | 0.85 ≤ AUC-ROC < 0.87    | 5   |
  | 0.87 ≤ AUC-ROC < 0.88    | 5.5 |
  | ≥ 0.88                   | 6   |

  **Meta de trabajo:** apuntar a **AUC-ROC ≥ 0.88** (6 SP), con piso aceptable en 0.85 (5 SP).
- **Métrica secundaria:** exactitud (`accuracy`) y matriz de confusión para interpretar el costo de falsos negativos (clientes que se van sin detectar) vs falsos positivos (cupones desperdiciados).
- **Decisión de negocio asociada:** una vez en producción, el equipo de marketing recibirá la lista de clientes con mayor probabilidad de cancelación (`1 − P(target = 1)` > umbral) para campaña proactiva de retención.

---

## 2. Descripción de los datos

Los datos se encuentran en `datasets/final_provider/` y constan de cuatro archivos CSV unidos por `customerID`:

| Archivo         | Filas | Cobertura | Contenido                                                                                  |
|-----------------|-------|-----------|--------------------------------------------------------------------------------------------|
| `contract.csv`  | 7 043 | 100 %     | Tipo de contrato, fechas, facturación, método de pago, cargos mensuales y totales.         |
| `personal.csv`  | 7 043 | 100 %     | Género, ciudadano senior, pareja, dependientes.                                            |
| `internet.csv`  | 5 517 | 78.3 %    | Tipo de servicio de internet y add-ons (seguridad, backup, soporte, streaming).            |
| `phone.csv`     | 6 361 | 90.3 %    | Indicador de múltiples líneas.                                                             |

**Distribución del target observada:** 73.46 % activos (`target = 1`) y 26.54 % cancelados (`target = 0`) sobre 7 043 clientes. Desbalance moderado, manejable sin sobremuestreo agresivo.

### Hallazgos preliminares relevantes

- `EndDate` toma valor `'No'` para clientes activos y, para cancelados, **solo cuatro fechas distintas** (2019-10-01, 2019-11-01, 2019-12-01, 2020-01-01). No hay granularidad temporal suficiente para validación por cohorte de salida.
- `TotalCharges` viene como **cadena de texto** y contiene **11 registros con espacio en blanco** (clientes recién dados de alta). Requiere conversión a numérico e imputación dirigida.
- **1 526 clientes (21.7 %)** no tienen registro en `internet.csv` y **682 (9.7 %)** no aparecen en `phone.csv`. El merge generará NaN en esas columnas, que deben tratarse como categoría `"No service"`, no como datos faltantes.
- Variables fuertemente discriminantes detectadas:
  - `Type`: churn de **42.7 %** en Month-to-month frente a **2.8 %** en Two year.
  - `PaymentMethod`: churn de **45.3 %** con Electronic check frente a 15–17 % con pagos automáticos.
- Clientes sin internet muestran churn de **7.4 %** frente a **32.8 %** en quienes sí lo tienen — sesgo a vigilar para evitar conclusiones causales prematuras.

---

## 3. Plan de preparación de datos

Pasos secuenciales para construir el dataset modelable:

1. **Carga e inspección** de los cuatro CSVs; validar que `customerID` es único en `contract` y `personal`.
2. **Conversión de tipos**
   - `BeginDate` y `EndDate` a `datetime` (con manejo del literal `'No'` antes del parse).
   - `TotalCharges` a numérico vía `pd.to_numeric(errors='coerce')`; imputar los 11 NaN con `MonthlyCharges` (la lectura natural es "el primer mes facturado").
   - `SeniorCitizen` ya viene como 0/1.
3. **Construcción del target**: `target = (EndDate == 'No').astype(int)` (1 = activo, 0 = cancelado). Eliminar `EndDate` del set de features para evitar fuga de datos trivial.
4. **Merge**: `contract LEFT JOIN personal LEFT JOIN internet LEFT JOIN phone` sobre `customerID`. Confirmar que `contract` (7 043) gobierna el cardinal final.
5. **Imputación de servicios faltantes**: las columnas provenientes de `internet.csv` y `phone.csv` para clientes sin registro reciben la categoría literal `"No service"` antes de codificar.
6. **Ingeniería de features**
   - `tenure_days = fecha_corte − BeginDate` (fecha de corte = 2020-02-01).
   - `tenure_months = TotalCharges / MonthlyCharges` (referencia cruzada de antigüedad).
   - `num_services` = conteo de add-ons activos (suma de indicadores en `internet` + `phone`).
   - `is_auto_payment` = 1 si `PaymentMethod` ∈ {Bank transfer (automatic), Credit card (automatic)}.
   - `has_internet`, `has_phone` (booleanas a partir del merge).
7. **Split**: train/test 75/25 estratificado por `target`, `random_state=12345`. Para validación durante tuning, 5-fold estratificado dentro del train.
8. **Codificación**
   - Variables binarias → 0/1.
   - Variables categóricas nominales (`PaymentMethod`, `InternetService`, `Type`) → **One-Hot** para modelos lineales y árboles clásicos; **categórica nativa** para CatBoost.
   - Variables numéricas (`MonthlyCharges`, `TotalCharges`, `tenure_*`) → `StandardScaler` para LogisticRegression; sin escalar para árboles.

---

## 4. Plan de modelado

Estrategia de comparación de candidatos crecientes en complejidad:

1. **Baseline 1 — DummyClassifier (`strategy='stratified'`)**: piso de referencia; se espera AUC ≈ 0.5.
2. **Baseline 2 — LogisticRegression con escalado y `class_weight='balanced'`**: techo del lado lineal; rápido y diagnóstico.
3. **RandomForest** con `class_weight='balanced'`: candidato no lineal robusto, sin tuning agresivo.
4. **LightGBM** (objetivo `binary`, `scale_pos_weight` ajustado al ratio 73/27; recordar que la clase positiva es "activo", por lo que el peso compensa a la clase minoritaria "cancelado").
5. **CatBoost** con `cat_features` nativo: principal candidato esperado por su buen desempeño en tabular con muchas categóricas y la disponibilidad de la librería en `pyproject.toml`.

**Selección de hiperparámetros**: `GridSearchCV` (o `RandomizedSearchCV` si el espacio crece) con 5-fold estratificado y `scoring='roc_auc'`. Para LightGBM/CatBoost se afinará `max_depth`/`num_leaves`, `learning_rate`, `n_estimators` con early stopping, y `min_child_samples`/`l2_leaf_reg`.

**Manejo del desbalance**: probar con y sin `class_weight`/`scale_pos_weight`; decidir por AUC-ROC en validación, no por accuracy.

**Selección del modelo final**: mejor AUC-ROC promedio en validación cruzada. Confirmar con AUC-ROC en el holdout 25 %. Reportar también exactitud, matriz de confusión y curva ROC.

**Interpretabilidad**: feature importance del modelo elegido + permutation importance para validar que el peso de `Type` y `tenure_*` es razonable y no se debe a fuga.

---

## 5. Riesgos identificados y mitigación

| Riesgo                                                                                          | Mitigación                                                                                                              |
|-------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| Fuga de datos vía `EndDate` o features derivadas de él (p. ej. tenure calculada hasta la salida). | Usar `tenure_days` con `BeginDate` y la fecha de corte fija 2020-02-01, **nunca** `EndDate`. Excluir `EndDate` del set X. |
| `Type = Month-to-month` casi separable del churn — riesgo de modelo dominado por una sola variable. | Validar con permutation importance; verificar que el modelo se mantenga por encima del baseline lineal aun retirando `Type`. |
| Clientes sin internet sesgan la tasa global de churn hacia abajo.                                | Mantenerlos en el dataset, pero documentar el comportamiento bivariado e inspeccionar el modelo con/sin ese subconjunto.   |
| 11 `TotalCharges` blancos en clientes nuevos.                                                    | Imputación dirigida con `MonthlyCharges` (interpretación contable), no con la media.                                       |
| Categorías raras en `PaymentMethod` o combinaciones poco frecuentes.                              | Validar con `value_counts`; consolidar si alguna queda con <30 observaciones.                                              |
| Sobreajuste a la métrica de validación durante tuning.                                           | Holdout 25 % no se toca hasta evaluación final; CV interno dentro del train.                                              |

---

## 6. Cronograma estimado

| Día | Actividad                                                                                          |
|-----|----------------------------------------------------------------------------------------------------|
| 1   | EDA exhaustiva en notebook, limpieza, merge, target, validación de hallazgos preliminares.         |
| 2   | Ingeniería de features, split, codificación, baselines (Dummy + LogReg) y primer árbol simple.     |
| 3   | LightGBM + CatBoost con tuning de hiperparámetros, validación cruzada.                              |
| 4   | Evaluación en holdout, interpretabilidad, gráficas finales.                                        |
| 5   | Redacción del informe de solución y consolidación del notebook.                                    |

---

## 7. Preguntas y supuestos para revisar con el tutor

**Supuestos asumidos:**

1. La fecha de corte para `tenure` y para considerar "activos" es **2020-02-01** (un día después de la última cancelación observada).
2. Los clientes sin registro en `internet.csv`/`phone.csv` simplemente **no contrataron** ese servicio (no es dato faltante).
3. La métrica de aprobación es AUC-ROC en el holdout estratificado. La calificación por SP se otorga según la tabla oficial (sección 1): se trabajará para superar 0.88 (6 SP) y se considera aprobado a partir de 0.85 (5 SP).
4. No se requiere validación temporal estricta porque `EndDate` carece de granularidad suficiente.

**Preguntas abiertas:**

1. ¿Hay un costo asimétrico definido entre falso positivo y falso negativo que justifique optimizar un umbral distinto al 0.5?
2. ¿El modelo final debe entregarse serializado (`joblib`) o basta con el notebook reproducible?
3. ¿Se espera reporte de calibración de probabilidades (curva de calibración / Brier score) además de AUC-ROC?
