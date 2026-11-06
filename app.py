"""Dashboard Streamlit: Sushi | demanda, facturación y rentabilidad.

Ejecutar desde la carpeta del proyecto: python -m streamlit run app.py
No se entrenan modelos en esta aplicación: se cargan modelos guardados.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import xgboost as xgb

from xgboost import XGBRegressor

from backend import (MODELOS, cargar_datos, resumen_diario, medias_historicas,
                     cargar_modelo, predecir_futuro, calcular_rentabilidad,
                     punto_equilibrio)

st.set_page_config(page_title="Mi restaurante | Sushi ML", page_icon="🍣", layout="wide")

@st.cache_data
# Leo los CSV una vez; Streamlit guarda el resultado mientras no cambien.
def datos():
    pedidos, costes = cargar_datos()
    diario = resumen_diario(pedidos, costes)
    referencias = medias_historicas(pedidos, costes)
    return pedidos, costes, diario, referencias

@st.cache_resource
# Cargo el modelo desde disco una sola vez por nombre.
def modelo(tipo, nombre):
    return cargar_modelo(tipo, nombre)


def euro(valor):
    return f"{valor:,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")


def grafico_comparacion(df, titulo, y_real="real", y_pred="predicho"):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["date"], df[y_real], label="Real", alpha=0.75)
    ax.plot(df["date"], df[y_pred], label="Predicción", alpha=0.85)
    ax.set(title=titulo, xlabel="Fecha", ylabel="Cantidad")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.autofmt_xdate()
    st.pyplot(fig)
    plt.close(fig)

try:
    pedidos, costes, diario, referencias = datos()
except Exception as error:
    st.error("No puedo leer los datos. Verifica que los dos CSV estén en la carpeta data/.")
    st.exception(error)
    st.stop()

metadata_path = MODELOS / "metadata.json"
if not metadata_path.exists():
    st.error("Faltan los modelos. Ejecuta primero desde esta carpeta: `python train_models.py`.")
    st.stop()

metadata = json.loads(metadata_path.read_text(encoding="utf8"))
comparacion_d = pd.read_csv(MODELOS / "comparacion_demanda.csv")
comparacion_f = pd.read_csv(MODELOS / "comparacion_facturacion.csv")

st.title("🍣 Mi restaurante | Predicciones con Machine Learning")
st.caption("Caso de estudio: restaurante de sushi (datos históricos 2015–2024). No corresponde al restaurante africano que quiero abrir.")
st.warning("El histórico termina el 31/12/2024. Las previsiones arrancan el 01/01/2025·")

st.sidebar.header("Configuración de la previsión")
modelo_d = st.sidebar.selectbox("Modelo de demanda", comparacion_d["Modelo"].tolist(), index=0)
modelo_f = st.sidebar.selectbox("Modelo de facturación", comparacion_f["Modelo"].tolist(), index=0)
dias = st.sidebar.slider("Días consecutivos previstos", min_value=1, max_value=30, value=14)

st.sidebar.divider()
st.sidebar.header("Escenario económico (editable)")
ticket = st.sidebar.number_input("Ticket medio (€ por pedido)", min_value=0.0,
                                  value=round(referencias["ticket_medio"], 2), step=1.0)
cogs_pct = st.sidebar.slider("Costes variables / ingredientes (%)", 0.0, 100.0,
                             float(round(referencias["porcentaje_coste_variable"], 1)), 0.5)
dias_apertura = st.sidebar.number_input("Días abiertos al mes para repartir costes", 1, 31, 30)
personal_mensual = st.sidebar.number_input("Coste mensual de personal (€)", min_value=0.0,
                                value=round(referencias["personal_diario"] * 30, 2), step=100.0)
fijos_mensuales = st.sidebar.number_input("Otros costes fijos al mes (€)", min_value=0.0,
                               value=round(referencias["costes_fijos_diarios"] * 30, 2), step=50.0)
modo_facturacion = st.sidebar.radio("Facturación que se utiliza para el beneficio",
                                    ["Pedidos × ticket medio", "Predicción directa ML"], index=0)

try:
    forecast = predecir_futuro(diario, modelo("demanda", modelo_d), modelo("facturacion", modelo_f), dias)
    escenario = calcular_rentabilidad(forecast, ticket, cogs_pct, personal_mensual,
                                      fijos_mensuales, dias_apertura,
                                      usar_facturacion_ml=(modo_facturacion == "Predicción directa ML"))
except Exception as error:
    st.error("No puedo calcular la previsión. Revisa los modelos y los parámetros.")
    st.exception(error)
    st.stop()

# Pestañas en el orden del proyecto: análisis, demanda, ingresos, rentabilidad.
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Exploración", "2. Demanda", "3. Facturación", "4. Rentabilidad", "5. Modelos"
])

with tab1:
    st.subheader("Conocer los datos antes de predecir")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pedidos históricos", f"{len(pedidos):,}".replace(",", "."))
    c2.metric("Facturación histórica", euro(pedidos["revenue"].sum()))
    c3.metric("Ticket medio histórico", euro(pedidos["revenue"].mean()))
    c4.metric("Días registrados", len(diario))
    anuales = diario.assign(año=diario["date"].dt.year).groupby("año", as_index=False).agg(
        pedidos=("pedidos_diarios", "sum"),
        facturacion=("facturacion_diaria", "sum"),
        beneficio=("operating_profit", "sum"))
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Evolución anual de pedidos**")
        st.bar_chart(anuales.set_index("año")["pedidos"])
    with col2:
        st.markdown("**Evolución anual de facturación**")
        st.bar_chart(anuales.set_index("año")["facturacion"])
    st.markdown("**Pedidos diarios y media móvil (30 días)**")
    serie = diario.set_index("date")[["pedidos_diarios"]].rename(columns={"pedidos_diarios": "Pedidos"})
    serie["Media móvil 30 días"] = serie["Pedidos"].rolling(30, min_periods=1).mean()
    st.line_chart(serie)
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Pedidos según día de la semana (media por jornada)**")
        d = diario.assign(dia=diario["date"].dt.dayofweek).groupby("dia")["pedidos_diarios"].mean()
        d.index = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
        st.bar_chart(d)
    with col4:
        st.markdown("**Pedidos según canal**")
        st.bar_chart(pedidos["channel"].value_counts())
    st.markdown("**Vista de los datos originales (primeras filas)**")
    st.dataframe(pedidos.head(20), use_container_width=True)
    st.info("No calculo demanda por horas porque el archivo sushi_orders.csv no incluye una columna de hora de pedido.")

with tab2:
    st.subheader("Predicción de demanda: pedidos diarios")
    st.write("**Variable objetivo (y):** `pedidos_diarios` = pedidos únicos por fecha. "
             "**Variables predictoras (X):** día, mes, fin de semana y pedidos pasados (ayer, hace siete días y media de siete días anteriores).")
    d1, d2, d3 = st.columns(3)
    d1.metric("Pedidos previstos en el período", f"{escenario['pedidos_previstos'].sum():,.0f}")
    d2.metric("Media diaria prevista", f"{escenario['pedidos_previstos'].mean():.1f}")
    d3.metric("Modelo seleccionado", modelo_d)
    st.markdown("**Pronóstico diario — desde el último día del histórico**")
    demanda_graf = escenario.set_index("date")[["pedidos_previstos"]]
    st.line_chart(demanda_graf)
    st.dataframe(escenario[["date", "pedidos_previstos"]].assign(
        pedidos_previstos=lambda x: x["pedidos_previstos"].round(1)), use_container_width=True, hide_index=True)
    st.caption("Previsión recursiva: los días 2–30 reutilizan predicciones anteriores. El error puede acumularse.")

with tab3:
    st.subheader("Predicción de facturación")
    st.latex(r"\text{Facturación por ticket} = \text{Pedidos previstos}\times\text{Ticket medio}")
    st.write("Comparo dos maneras: la fórmula de mi proyecto original (pedidos × ticket medio editable) "
             "y una predicción directa de facturación diaria con los nueve regresores.")
    f1, f2, f3 = st.columns(3)
    f1.metric("Facturación por ticket", euro(escenario["facturacion_por_ticket"].sum()))
    f2.metric("Facturación con modelo ML", euro(escenario["facturacion_ml"].sum()))
    f3.metric("Método usado en rentabilidad", modo_facturacion)
    st.markdown("**Comparación de métodos**")
    st.line_chart(escenario.set_index("date")[["facturacion_por_ticket", "facturacion_ml"]])
    st.markdown("**Facturación prevista (método seleccionado)**")
    st.bar_chart(escenario.set_index("date")["facturacion_prevista"])
    st.caption("El ticket del escenario procede por defecto de 2024; para otro restaurante hay que introducir un ticket propio. La predicción ML directa permanece anclada en los precios y operaciones históricos del sushi.")

with tab4:
    st.subheader("Rentabilidad: facturación menos costes")
    st.latex(r"\text{Beneficio operativo} = \text{Facturación} - \text{Ingredientes} - \text{Personal} - \text{Costes fijos}")
    total_ventas = escenario["facturacion_prevista"].sum()
    total_beneficio = escenario["beneficio_operativo"].sum()
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Ingresos previstos", euro(total_ventas))
    r2.metric("Costes variables", euro(escenario["costes_variables"].sum()))
    r3.metric("Personal + otros costes", euro((escenario["costes_personal"] + escenario["costes_fijos"]).sum()))
    r4.metric("Beneficio operativo", euro(total_beneficio))
    st.markdown("**Beneficio o pérdida por día**")
    st.bar_chart(escenario.set_index("date")["beneficio_operativo"])
    st.markdown("**Costes e ingresos del escenario**")
    st.dataframe(escenario[["date", "pedidos_previstos", "facturacion_prevista", "costes_variables",
                           "costes_personal", "costes_fijos", "beneficio_operativo", "margen_operativo_pct"]].round(2),
                 use_container_width=True, hide_index=True)
    equilibrio = punto_equilibrio(ticket, cogs_pct, personal_mensual, fijos_mensuales, dias_apertura)
    if equilibrio is None:
        st.error("El margen por pedido es cero o negativo: no se alcanza el punto de equilibrio.")
    else:
        st.metric("Pedidos diarios necesarios para cubrir costes (escenario)", equilibrio)
    st.write("**Contexto real:** los datos históricos incluyen `cogs`, `labour_cost`, "
             "`fixed_ops_cost` y `operating_profit`. No estoy entrenando un modelo adicional de beneficio: "
             "se calcula con las predicciones y los supuestos editables de costes.")
    st.markdown("**Beneficio operativo histórico real, por año**")
    st.bar_chart(costes.assign(año=costes["date"].dt.year).groupby("año")["operating_profit"].sum())
    csv = escenario.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Descargar previsiones y rentabilidad (CSV)", csv,
                       file_name="predicciones_sushi.csv", mime="text/csv")

with tab5:
    st.subheader("Comparación de los nueve modelos")
    st.write("Los resultados corresponden al último 20 % del histórico. "
             "El modelo con menor MAE se coloca primero. R² más alto suele indicar mejor ajuste.")
    e1, e2 = st.columns(2)
    with e1:
        st.markdown("**Demanda — error en número de pedidos**")
        st.dataframe(comparacion_d.round(3), hide_index=True, use_container_width=True)
        st.bar_chart(comparacion_d.set_index("Modelo")["MAE"])
    with e2:
        st.markdown("**Facturación — error en importe monetario**")
        st.dataframe(comparacion_f.round(3), hide_index=True, use_container_width=True)
        st.bar_chart(comparacion_f.set_index("Modelo")["MAE"])
    tipo = st.radio("Comparación de valores reales y predichos", ["demanda", "facturacion"], horizontal=True)
    sel = modelo_d if tipo == "demanda" else modelo_f
    pred_test = pd.read_csv(MODELOS / f"predicciones_test_{tipo}.csv", parse_dates=["date"])
    pred_test["predicho"] = pred_test[sel]
    # Las primeras 60 jornadas permiten ver mejor la discrepancia entre series.
    grafico_comparacion(pred_test.head(60), f"Prueba temporal: {sel} | {tipo}")
    if tipo == "facturacion":
        st.markdown("**Evaluación de la fórmula pedidos × ticket medio**")
        st.dataframe(pd.read_csv(MODELOS / "evaluacion_formula_ticket.csv").round(2),
                     use_container_width=True, hide_index=True)
    st.caption("Las métricas se calculan a un día vista con información real de los días anteriores, no para un horizonte recursivo de 30 días. No interpretar un R² alto como garantía de beneficio futuro.")

st.caption("Archivos principales: backend.py (funciones), train_models.py (entrenamiento), app.py (Streamlit), data/ (CSV) y modelos/ (archivos .joblib).")
