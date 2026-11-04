# Mi restaurante — análisis y predicción con datos de sushi

Este proyecto sustituye los archivos y cálculos anteriores de pizzería, catálogo de productos y materias primas por **dos CSV del ZIP de sushi**. Lo he preparado como proyecto educativo: quiero aprender, visualizar y explicar tres métricas en Python y Streamlit.

## 1. ¿Qué quiero predecir?

**Primero, la demanda** (`pedidos_diarios`). Agrupo los pedidos por día, creo variables de calendario y de días anteriores y entreno nueve modelos de regresión. **Después, la facturación** (`facturacion_diaria`). Comparo una previsión directa mediante nueve regresores con el cálculo **pedidos previstos × ticket medio**. **Finalmente, la rentabilidad**, que obtengo mediante una fórmula transparente con costes editables:

```text
Facturación prevista  = Pedidos previstos × Ticket medio
Costes variables      = Facturación prevista × porcentaje de costes variables
Costes diarios        = Personal mensual / días abiertos + Fijos mensuales / días abiertos
Beneficio operativo   = Facturación prevista − Costes variables − Costes diarios
Punto de equilibrio  = redondeo hacia arriba(Costes diarios / margen por pedido)
Margen por pedido     = Ticket medio × (1 − porcentaje de costes variables)
```

También se calcula la facturación diaria directamente por machine learning. Puedo elegir el método en Streamlit.

**Importante:** no entreno un modelo de predicción de rentabilidad independiente: el beneficio operativo se calcula a partir de ingresos previstos y costes supuestos. Esto es más explicable y coherente con mi notebook original.

## 2. ¿Qué archivos contiene la carpeta?

```text
Proyecto_Sushi_ML/
├── app.py                      # Interfaz con gráficos y parámetros en Streamlit
├── backend.py                  # Lectura, limpieza, variables, predicción y fórmulas
├── train_models.py             # Entrenamiento y comparación de los 9 regresores (dos objetivos)
├── Analisis_Sushi_ML.ipynb     # Notebook paso a paso, en primera persona y con gráficos
├── requirements.txt            # Librerías que necesito instalar
├── README.md                   # Esta guía
├── data/
│   ├── sushi_orders.csv        # 83.890 pedidos, datos por transacción
│   └── sushi_daily_pnl.csv     # 3.653 días, ingresos/costes/beneficio operativo
├── modelos/
│   ├── demanda_*.joblib         # Los nueve modelos de pedidos ya entrenados
│   ├── facturacion_*.joblib     # Los nueve modelos de facturación ya entrenados
│   ├── comparacion_*.csv       # MAE, RMSE y R² de ambos objetivos
│   ├── predicciones_test_*.csv # Predicciones sobre el conjunto de prueba
│   └── metadata.json           # Fechas de entrenamiento y mejores modelos
└── graficos/                  # Gráficos de referencia generados al preparar el proyecto
```

El ZIP original contenía además `sushi_order_items.csv`, `sushi_customers.csv`, `sushi_reviews.csv` y `sushi_staff.csv`. No los necesito para los tres objetivos elegidos. No incluyo el CSV casi millonario de productos para evitar peso y complejidad que no aportan valor al modelo diario.

## 3. Pasos para descargarlo y ejecutarlo en Windows

1. Descargar el ZIP, hacer clic con el botón derecho y elegir **Extraer todo**.
2. Abrir la carpeta extraída `Proyecto_Sushi_ML` en Visual Studio Code o Anaconda.
3. Instalar las librerías desde una terminal dentro de esa carpeta:

```powershell
python -m pip install -r requirements.txt
```

4. Abrir Streamlit, también desde esa carpeta:

```powershell
python -m streamlit run app.py
```

5. Streamlit mostrará una dirección local, normalmente `http://localhost:8501`. Se abrirá en el navegador. Los archivos `.joblib` ya están incluidos, por lo que **no es obligatorio entrenar** antes de abrirla.

**Si quiero volver a entrenar con los mismos datos o tras actualizarlos:**

```powershell
python train_models.py
```

Esto entrena los **nueve modelos de demanda y los nueve de facturación**, vuelve a guardar los archivos en `modelos/` y actualiza las tablas de resultados. El proceso puede tardar dependiendo del ordenador.

**Si quiero estudiar el código con explicaciones y ejecutar los gráficos:** abrir `Analisis_Sushi_ML.ipynb` en Jupyter Notebook o Visual Studio Code y ejecutar las celdas de arriba abajo. La celda de entrenamiento vuelve a entrenar los modelos, por tanto no es necesario ejecutarla cada vez que se leen los resultados ya guardados.

**Si en Windows el comando `python` apunta a otro entorno:** puedo seleccionar el intérprete Python de Anaconda en Visual Studio Code y usar ese mismo intérprete para instalar requisitos y ejecutar Streamlit. No necesito cambiar rutas absolutas: `backend.py` localiza los CSV respecto de su propia carpeta.

## 4. Mis nueve modelos (igual que en mi notebook anterior)

1. Linear Regression.
2. Ridge.
3. Random Forest Regressor.
4. XGBoost Regressor.
5. Gradient Boosting Regressor.
6. HistGradientBoosting Regressor.
7. Decision Tree Regressor.
8. KNeighborsRegressor con StandardScaler.
9. SVR con StandardScaler.

Uso **80 % de fechas antiguas para entrenamiento** y **20 % de fechas recientes para prueba**. Selecciono el mejor modelo por **MAE menor**. Los resultados reales del entrenamiento se guardan en los archivos `comparacion_demanda.csv` y `comparacion_facturacion.csv`. Las métricas del test corresponden a **predicciones de un día usando días reales anteriores**: no significan que una previsión recursiva a 30 días tenga el mismo error.

## 5. ¿Qué variables uso?

| Objetivo (`y`) | Variables predictoras (`X`) |
|---|---|
| Pedidos diarios | `dia_semana`, `mes`, `fin_semana`, `pedidos_ayer`, `pedidos_hace_7_dias`, `media_7_dias` |
| Facturación diaria | `dia_semana`, `mes`, `fin_semana`, `ventas_ayer`, `ventas_hace_7_dias`, `media_ventas_7_dias` |

`.shift(1)` se usa para que **los datos del propio día que quiero predecir no entren en las variables de ese mismo día**. La predicción futura avanza día a día y rellena los datos atrasados con predicciones propias cuando ya no existen ventas reales. Así evito introducir sin querer resultados futuros.

## 6. ¿Qué puedo hacer desde Streamlit?

La aplicación contiene cinco pestañas: **Exploración, Demanda, Facturación, Rentabilidad y Modelos**. Tiene gráficos de ventas y pedidos históricos, comparación de nueve regresores, gráfico de predicción frente a realidad, previsión de 1 a 30 días, elección de método para ingresos, costes editables, beneficio operativo, punto de equilibrio y exportación a CSV.

En la barra lateral puedo ajustar el **ticket medio**, los **costes variables**, el **coste del personal**, los **otros costes fijos**, los **días de apertura usados para repartir costes** y los **modelos de predicción**. Los valores iniciales de costes y ticket proceden de los últimos 365 días del dataset, pero puedo sustituirlos por mis propios supuestos.

## 7. Límites que debo explicar en la entrega

- **Datos históricos:** comienzan el 01/01/2015 y terminan el 31/12/2024. La previsión de la aplicación empieza el 01/01/2025, un día después de la última fecha disponible, y permite 1–30 días. No es una previsión actual fiable de 2026.
- **Negocio diferente:** los datos describen un caso de sushi, no un restaurante africano real. Lo que aprende el modelo no debe transferirse automáticamente a ese negocio.
- **Moneda:** el CSV no declara explícitamente una divisa. En la app uso el símbolo **€ como supuesto visual** para el ejercicio, **sin conversión de moneda**. Para estimar costes reales de un restaurante español tengo que sustituir el ticket y los costes por importes propios en euros, y no tomar la predicción ML monetaria como euros comprobados.
- **Ventas frente a cobros:** uso `revenue` porque coincide exactamente con `gross_revenue` del CSV de rentabilidad. **No uso `total_collected`** para no confundir ingresos de ventas con propinas y otros conceptos.
- **Sin horas:** `sushi_orders.csv` tiene fecha pero **no hora del pedido**. Por ese motivo no genero un modelo horario. Inventar horas o reutilizar horas de pizzería sería incorrecto.
- **Escenario operativo:** el resultado de rentabilidad no incluye IVA, impuestos sobre beneficios, inversión inicial, amortizaciones ni financiación; tampoco modela un calendario de cierres. Los días de apertura solo se usan para repartir costes mensuales.
- **Precisión:** debo considerar el MAE, RMSE, R² y la distancia temporal respecto de 2024 antes de interpretar predicciones. Los modelos no garantizan ventas ni beneficios.

## 8. Cómo comprobar que funciona

```powershell
python test_sushi.py
```

Este script comprueba lectura de datos, integridad económica, ausencia de valores nulos en las características, carga de modelos, predicciones futuras y fórmula del beneficio.
