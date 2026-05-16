# Informe de solución — Predicción de churn para Interconnect

**Autor:** Gustavo Mandujano Rojas
**Sprint:** Proyecto final TripleTen — Data Science
**Fecha:** 2026-05-16
**Entregables relacionados:** `etapa1_plan_de_trabajo.md`, `etapa2_codigo_solucion.ipynb`

---

## 1. Resumen ejecutivo

Se entrenó un modelo predictivo de cancelación de clientes (churn) para el operador Interconnect, con el objetivo de identificar a los suscriptores con mayor probabilidad de irse y permitir al equipo de marketing intervenir con códigos promocionales y planes especiales antes de que la baja ocurra.

El modelo final, **LightGBM** afinado por validación cruzada estratificada de 5 pliegues, alcanza una **AUC-ROC de 0.9696 en el set de prueba (holdout 25 %)**, equivalente a la **banda 6 SP** (la máxima de la tabla oficial de evaluación). Su exactitud asociada es de **0.9256**.

A pesar del excelente rendimiento estadístico, el análisis de importancia de variables revela que dos features dominan la predicción: la antigüedad calendario (`tenure_days`) y la antigüedad facturada (`tenure_months`). La combinación de ambas permite al modelo inferir indirectamente la fecha de cancelación, lo que explica el AUC tan alto. Para uso en producción se recomienda una versión más conservadora del modelo y un proceso de validación periódica con datos nuevos.

---

## 2. Contexto y problema de negocio

Interconnect ofrece servicios de telefonía fija e internet (DSL y fibra óptica) junto con add-ons como streaming, antivirus y soporte técnico. Los clientes pueden contratar planes mensuales o anuales (1 ó 2 años) y pagar por distintos métodos. La tasa observada de cancelación en el período analizado (octubre 2019 – enero 2020) es del **26.5 %**, lo que significa que aproximadamente uno de cada cuatro clientes deja la empresa.

El problema se formuló como **clasificación binaria supervisada**: dada la información contractual, personal y de servicios contratados de cada cliente, predecir si seguirá activo a la fecha de corte (2020-02-01).

- **Característica objetivo:** `target = (EndDate == 'No')` ⇒ clase 1 = activo, clase 0 = cancelado.
- **Métrica primaria:** AUC-ROC, con escala de puntos SP que va de 0 (AUC < 0.75) a 6 (AUC ≥ 0.88).
- **Métrica secundaria:** exactitud y matriz de confusión.

---

## 3. Datos utilizados

| Archivo | Filas | Cobertura | Contenido |
|---|---|---|---|
| `contract.csv` | 7 043 | 100 % | Tipo de contrato, fechas, facturación, método de pago. |
| `personal.csv` | 7 043 | 100 % | Género, ciudadano senior, pareja, dependientes. |
| `internet.csv` | 5 517 | 78.3 % | Servicio de internet y add-ons (seguridad, backup, soporte, streaming). |
| `phone.csv` | 6 361 | 90.3 % | Indicador de múltiples líneas. |

**Hallazgos durante la limpieza:**

- `TotalCharges` venía como cadena de texto con 11 valores en blanco (clientes recién dados de alta), imputados con `MonthlyCharges`.
- 1 526 clientes (21.7 %) no tienen registro en `internet.csv` y 682 (9.7 %) no aparecen en `phone.csv`. Estos vacíos no son datos perdidos: el cliente no contrató ese servicio. Se trataron con la categoría literal `"No service"`.
- `EndDate` toma solo cuatro fechas distintas (2019-10-01, 2019-11-01, 2019-12-01, 2020-01-01), insuficiente para validación temporal por cohortes.

---

## 4. Metodología

El flujo completo está implementado en `etapa2_codigo_solucion.ipynb`. Pasos principales:

1. **Carga e inspección** de los cuatro CSVs y validación de unicidad de `customerID`.
2. **Limpieza:** conversión de tipos, imputación de `TotalCharges`, construcción del target sin fuga (eliminando `EndDate` del set de features).
3. **Merge:** `contract LEFT JOIN personal LEFT JOIN internet LEFT JOIN phone` con imputación `"No service"` para servicios no contratados.
4. **Ingeniería de features:**
   - `tenure_days` = días entre `BeginDate` y la fecha de corte 2020-02-01.
   - `tenure_months` = `TotalCharges / MonthlyCharges`.
   - `num_services` = número de add-ons contratados.
   - `is_auto_payment` = indicador de pago automático.
   - `has_internet`, `has_phone` = indicadores de servicio contratado.
5. **Split** train/test 75/25 estratificado por `target` (`random_state=12345`).
6. **Comparación de cinco modelos** con validación cruzada estratificada de 5 pliegues:
   - **DummyClassifier** (`stratified`) como piso de referencia.
   - **LogisticRegression** con escalado y `class_weight='balanced'` (GridSearch sobre `C`).
   - **RandomForest** con `class_weight='balanced'` (GridSearch sobre profundidad, hojas y árboles).
   - **LightGBM** con `scale_pos_weight` ajustado al desbalance (RandomizedSearch sobre 5 hiperparámetros, 15 iteraciones).
   - **CatBoost** con manejo nativo de categóricas y `auto_class_weights='Balanced'` (búsqueda manual sobre 8 combinaciones con early stopping).
7. **Selección** del modelo con mayor AUC-ROC promedio en CV.
8. **Evaluación final** en el holdout 25 %: AUC-ROC, exactitud, matriz de confusión y curva ROC.
9. **Interpretabilidad** mediante permutation importance sobre el modelo elegido.

---

## 5. Resultados principales

### 5.1 Comparación de modelos (AUC-ROC en validación cruzada)

| Modelo | AUC-ROC CV |
|---|---|
| **LightGBM** ← elegido | **0.9646** |
| LogisticRegression | 0.9614 |
| CatBoost | 0.9597 |
| RandomForest | 0.8742 |
| Dummy | 0.5082 |

LightGBM resulta el mejor por un margen pequeño sobre LogisticRegression y CatBoost; RandomForest queda claramente atrás. El Dummy confirma que las clases no son trivialmente predecibles a partir de la distribución sola.

### 5.2 Hiperparámetros del modelo final

```
LightGBM:
  n_estimators       = 400
  learning_rate      = 0.03
  num_leaves         = 31
  min_child_samples  = 50
  reg_lambda         = 0.0
  scale_pos_weight   ≈ 0.36 (calculado de la distribución del train)
```

### 5.3 Evaluación en el holdout (1 761 clientes)

| Métrica | Valor |
|---|---|
| AUC-ROC | **0.9696** |
| Banda SP (tabla oficial) | **6 SP** (≥ 0.88) |
| Exactitud | 0.9256 |

**Matriz de confusión:**

|  | pred cancelado (0) | pred activo (1) |
|---|---|---|
| **real cancelado (0)** | 402 | 65 |
| **real activo (1)** | 66 | 1 228 |

- **Recall sobre cancelados:** 402 / 467 ≈ **86 %** — el modelo detecta a la gran mayoría de clientes en riesgo.
- **Precisión sobre cancelados:** 402 / 468 ≈ **86 %** — cuando dice "este cliente va a cancelar", acierta el 86 % de las veces.

### 5.4 Variables más informativas (permutation importance)

| # | Feature | Importancia (caída de AUC al permutar) |
|---|---|---|
| 1 | `tenure_months` | 0.447 |
| 2 | `tenure_days` | 0.233 |
| 3 | `Type` | 0.015 |
| 4 | `TotalCharges` | 0.011 |
| 5 | `InternetService` | 0.010 |
| 6 | `MonthlyCharges` | 0.010 |

Las dos variables de antigüedad concentran prácticamente toda la señal predictiva.

---

## 6. Hallazgos clave

### 6.1 La señal está en la antigüedad — y en su contraste

La importancia de `tenure_months` (0.45) y `tenure_days` (0.23) no es un efecto trivial. **Es la diferencia entre ambas la que carga la señal**:

- `tenure_days` mide la antigüedad calendario desde la fecha de alta hasta el corte fijo (2020-02-01).
- `tenure_months = TotalCharges / MonthlyCharges` mide la antigüedad **realmente facturada**.
- Para clientes activos, ambas concuerdan (con un mes de diferencia a lo sumo).
- Para clientes cancelados, `tenure_months` es estrictamente menor que `tenure_days/30`, porque dejaron de pagar antes del corte.

El modelo aprovecha esa discrepancia como un proxy casi perfecto de la cancelación. Por eso el AUC es excepcionalmente alto.

### 6.2 Tras la antigüedad, los predictores son los esperados

Quitando el efecto de `tenure_*`, las variables más informativas son las que ya se anticipaban en el plan de trabajo: el **tipo de contrato** (`Month-to-month` concentra la mayor parte de las cancelaciones), el **método de pago** (Electronic check duplica la tasa de churn de los métodos automáticos) y el **tipo de servicio de internet** (los clientes de fibra óptica cancelan más que los de DSL).

### 6.3 El género y los datos demográficos no aportan

`gender`, `Dependents`, `SeniorCitizen` y `Partner` figuran al fondo del ranking de importancia. La decisión de irse no se explica por el perfil personal del cliente, sino por la calidad y el contrato del servicio.

---

## 7. Limitaciones

1. **Posible fuga de información en las features de antigüedad.** Como se discute en §6.1, el modelo se apoya en una diferencia que solo existe en el dataset porque la fecha de cancelación quedó "registrada" indirectamente en `TotalCharges`. En un escenario productivo donde se predice churn antes de que ocurra, esa señal no estaría disponible con la misma claridad.
2. **Ventana temporal estrecha.** Las cancelaciones observadas se concentran en cuatro meses (oct-2019 a ene-2020). No es posible validar el modelo sobre cohortes anteriores ni sobre comportamiento estacional.
3. **Snapshot único.** No hay datos longitudinales (uso mensual, cantidad de llamadas a soporte, quejas). Modelar la trayectoria del cliente, no solo su estado actual, daría señales más prospectivas.
4. **Distribución por servicios:** los clientes sin internet (n = 1 526) muestran un churn de 7.4 % vs 32.8 % en quienes lo tienen. El modelo trata bien a ambos subgrupos, pero la decisión de retención debe segmentarse: no tiene sentido enviar la misma promoción a un cliente solo de teléfono que a uno de fibra óptica.
5. **Costo asimétrico no especificado.** El umbral de decisión se fijó en 0.5; si el costo de un falso negativo (perder un cliente sin avisar) es mayor que el de un falso positivo (regalar un cupón), conviene mover el umbral hacia la izquierda.

---

## 8. Recomendaciones

### Para el equipo de negocio

- **Priorizar a los clientes mensuales con pago manual.** El cruce `Type = Month-to-month` × `PaymentMethod ∈ {Electronic check, Mailed check}` concentra la mayor tasa de churn observada y es accionable: ofrecer migración a pago automático y a contratos de 1 ó 2 años con un descuento puede capturar buena parte del riesgo.
- **Revisar el producto de fibra óptica.** Los clientes de fibra cancelan más que los de DSL pese al ARPU mayor. Vale la pena analizar si la diferencia se debe a calidad de servicio, expectativas, o competencia local.
- **Definir un umbral operativo.** Hoy el modelo entrega una probabilidad continua. Para usarlo en campañas conviene fijar un umbral (p. ej. predecir como "en riesgo" al 20 % de clientes con menor `P(target=1)`) y medir el lift de la retención sobre ese segmento vs un grupo de control.

### Para el equipo de datos

- **Construir una versión "conservadora" del modelo** que excluya `tenure_months` (o `TotalCharges`) y se base en features estrictamente prospectivas. Aunque su AUC esperado caiga a ~0.85, ese modelo será más confiable para predecir bajas futuras.
- **Recolectar señales de uso.** Variables como número de tickets de soporte, MB consumidos, cambios recientes de plan o quejas registradas suelen anticipar la cancelación con mejor calibración.
- **Re-entrenamiento periódico.** Dado que solo se observan cuatro meses de cancelaciones, el modelo debería re-entrenarse al menos trimestralmente con datos nuevos para evitar que las distribuciones cambien sin que se note.

---

## 9. Próximos pasos sugeridos

1. **Producto mínimo viable:** desplegar el modelo conservador (sin `tenure_months`) tras una semana de A/B testing en sombra contra la heurística actual.
2. **Pipeline de monitoreo:** registrar diariamente la distribución de scores y la tasa real de churn por segmento, para detectar drift.
3. **Iteración 2 del modelo:** incorporar señales de uso/CRM y volver a comparar boosting vs. modelos lineales sobre el set enriquecido.
4. **Definir KPIs de negocio:** medir el impacto del programa de retención no por AUC sino por *clientes retenidos × ARPU* y *costo de promoción / cliente retenido*.

---

## 10. Conclusión

El proyecto cumple holgadamente la métrica de aprobación del sprint (AUC-ROC ≥ 0.85, banda 6 SP). El notebook entregado es reproducible y deja la puerta abierta tanto para ajustes técnicos (versión conservadora, recalibración de umbral) como para acciones de negocio inmediatas (segmentación por contrato y método de pago).

El verdadero valor del trabajo no está en haber alcanzado un AUC alto, sino en haber identificado **por qué** ese AUC es alto y, con esa lectura, haber distinguido entre la señal estructural del problema (tipo de contrato, método de pago, tipo de servicio) y un artefacto del propio dataset. Esa distinción es la que permite hacer recomendaciones útiles más allá del número.
