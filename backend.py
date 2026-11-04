"""Funciones de datos y predicción. No depende de Streamlit.

Proyecto educativo: todos los datos provienen de sushi_orders.csv y
sushi_daily_pnl.csv. El histórico finaliza el 31/12/2024.
"""
from pathlib import Path
import math
import pandas as pd
import numpy as np
import joblib

CARPETA = Path(__file__).resolve().parent
DATOS = CARPETA / "data"
MODELOS = CARPETA / "modelos"

# Variables que utilizo para predecir pedidos diarios, como en mi notebook original.
COLUMNAS_DEMANDA = [
    "dia_semana", "mes", "fin_semana",
    "pedidos_ayer", "pedidos_hace_7_dias", "media_7_dias",
]
# Para predecir facturación directamente uso los importes de días previos.
COLUMNAS_FACTURACION = [
    "dia_semana", "mes", "fin_semana",
    "ventas_ayer", "ventas_hace_7_dias", "media_ventas_7_dias",
]


def cargar_datos():
    """Leo los dos CSV necesarios y compruebo sus columnas y fechas."""
    pedidos = pd.read_csv(DATOS / "sushi_orders.csv", parse_dates=["date"])
    costes = pd.read_csv(DATOS / "sushi_daily_pnl.csv", parse_dates=["date"])
    col_pedidos = {"order_id", "date", "revenue", "cogs", "channel"}
    col_costes = {"date", "gross_revenue", "cogs", "labour_cost", "fixed_ops_cost", "operating_profit"}
    if not col_pedidos.issubset(pedidos.columns):
        raise ValueError(f"Faltan columnas en sushi_orders.csv: {col_pedidos - set(pedidos.columns)}")
    if not col_costes.issubset(costes.columns):
        raise ValueError(f"Faltan columnas en sushi_daily_pnl.csv: {col_costes - set(costes.columns)}")
    if pedidos["order_id"].duplicated().any():
        raise ValueError("order_id debería ser único en sushi_orders.csv")
    if costes["date"].duplicated().any():
        raise ValueError("Las fechas de sushi_daily_pnl.csv deberían ser únicas")
    return pedidos, costes


def resumen_diario(pedidos, costes):
    """Una fila por fecha: pedidos, facturación y costes diarios."""
    ventas = (pedidos.groupby("date", as_index=False)
              .agg(pedidos_diarios=("order_id", "nunique"),
                   facturacion_diaria=("revenue", "sum")))
    resumen = ventas.merge(
        costes[["date", "cogs", "labour_cost", "fixed_ops_cost", "operating_profit"]],
        on="date", how="left", validate="one_to_one")
    resumen = resumen.sort_values("date").reset_index(drop=True)
    if resumen[["cogs", "labour_cost", "fixed_ops_cost", "operating_profit"]].isna().any().any():
        raise ValueError("Algunas fechas de pedidos no tienen datos de costes.")
    fechas_completas = pd.date_range(resumen["date"].min(), resumen["date"].max(), freq="D")
    if len(fechas_completas) != len(resumen):
        raise ValueError("Faltan fechas: revisa los días cerrados antes de crear variables lag.")
    # En este archivo concreto hay datos para todos los días de 2015 a 2024.
    return resumen


def preparar_variables(diario):
    """Creo variables con datos históricos PREVIOS, nunca del mismo día futuro."""
    df = diario.sort_values("date").copy()
    df["dia_semana"] = df["date"].dt.dayofweek
    df["mes"] = df["date"].dt.month
    df["fin_semana"] = (df["dia_semana"] >= 5).astype(int)

    df["pedidos_ayer"] = df["pedidos_diarios"].shift(1)
    df["pedidos_hace_7_dias"] = df["pedidos_diarios"].shift(7)
    df["media_7_dias"] = df["pedidos_diarios"].shift(1).rolling(7).mean()

    df["ventas_ayer"] = df["facturacion_diaria"].shift(1)
    df["ventas_hace_7_dias"] = df["facturacion_diaria"].shift(7)
    df["media_ventas_7_dias"] = df["facturacion_diaria"].shift(1).rolling(7).mean()
    return df.dropna(subset=COLUMNAS_DEMANDA + COLUMNAS_FACTURACION).reset_index(drop=True)


def separar_temporal(df, porcentaje_entrenamiento=0.8):
    """80 % de fechas antiguas para entrenar; 20 % recientes para evaluar."""
    limite = int(len(df) * porcentaje_entrenamiento)
    if limite < 10 or limite >= len(df):
        raise ValueError("No hay días suficientes para realizar la separación.")
    return df.iloc[:limite].copy(), df.iloc[limite:].copy()


def medias_historicas(pedidos, costes, dias=365):
    """Cifras observadas del período más reciente del dataset, no del restaurante del usuario."""
    ultima_fecha = costes["date"].max()
    desde = ultima_fecha - pd.Timedelta(days=dias - 1)
    p = pedidos[pedidos["date"] >= desde]
    c = costes[costes["date"] >= desde]
    if p.empty or c.empty:
        raise ValueError("No hay información histórica suficiente para el escenario.")
    return {
        "ticket_medio": float(p["revenue"].sum() / p["order_id"].nunique()),
        "porcentaje_coste_variable": float(c["cogs"].sum() / c["gross_revenue"].sum() * 100),
        "personal_diario": float(c["labour_cost"].mean()),
        "costes_fijos_diarios": float(c["fixed_ops_cost"].mean()),
        "fecha_ultima": ultima_fecha,
    }


def ruta_modelo(tipo, nombre):
    """Ruta estable de modelos; los nombres se definen en train_models.py."""
    from train_models import CLAVES_MODELOS
    return MODELOS / f"{tipo}_{CLAVES_MODELOS[nombre]}.joblib"


def cargar_modelo(tipo, nombre):
    """Cargo un modelo ya entrenado para que Streamlit no entrene otra vez."""
    return joblib.load(ruta_modelo(tipo, nombre))


def predecir_futuro(diario, modelo_demanda, modelo_facturacion, dias=14):
    """Preveo los siguientes 1-30 días en orden cronológico.

    IMPORTANTE: a partir del segundo día uso MIS PROPIAS predicciones previas
    en los lags (no las ventas reales de días que todavía no han sucedido).
    Los decimales en pedidos sirven para preservar información en la recursión;
    al visualizar se redondean a pedidos enteros.
    """
    if not 1 <= dias <= 30:
        raise ValueError("Se admiten horizontes de 1 a 30 días.")
    historial = diario.sort_values("date").copy()
    fechas = list(pd.to_datetime(historial["date"]))
    pedidos = list(historial["pedidos_diarios"].astype(float))
    ventas = list(historial["facturacion_diaria"].astype(float))
    if len(fechas) < 8:
        raise ValueError("Necesito al menos ocho fechas consecutivas.")
    esperado = pd.date_range(fechas[0], fechas[-1], freq="D")
    if len(esperado) != len(fechas):
        raise ValueError("El historial contiene días sin registros.")
    salida = []
    for _ in range(dias):
        fecha = fechas[-1] + pd.Timedelta(days=1)
        base = {
            "dia_semana": fecha.dayofweek,
            "mes": fecha.month,
            "fin_semana": int(fecha.dayofweek >= 5),
        }
        x_demanda = pd.DataFrame([{**base,
            "pedidos_ayer": pedidos[-1],
            "pedidos_hace_7_dias": pedidos[-7],
            "media_7_dias": float(np.mean(pedidos[-7:]))}])[COLUMNAS_DEMANDA]
        x_ventas = pd.DataFrame([{**base,
            "ventas_ayer": ventas[-1],
            "ventas_hace_7_dias": ventas[-7],
            "media_ventas_7_dias": float(np.mean(ventas[-7:]))}])[COLUMNAS_FACTURACION]
        pred_pedidos = max(0.0, float(modelo_demanda.predict(x_demanda)[0]))
        pred_ventas = max(0.0, float(modelo_facturacion.predict(x_ventas)[0]))
        fechas.append(fecha)
        pedidos.append(pred_pedidos)
        ventas.append(pred_ventas)
        salida.append({"date": fecha, "pedidos_previstos": pred_pedidos,
                       "facturacion_ml": pred_ventas})
    return pd.DataFrame(salida)


def calcular_rentabilidad(predicciones, ticket_medio, porcentaje_coste_variable,
                          personal_mensual, costes_fijos_mensuales,
                          dias_apertura_mes=30, usar_facturacion_ml=False):
    """Beneficio operativo = ingresos - coste variable - personal - fijos.

    Es un escenario contable calculado con SUPUESTOS editables; no se presenta
    como si existiese un modelo de ML que predijera costes desconocidos.
    """
    if not 0 <= porcentaje_coste_variable <= 100:
        raise ValueError("Los costes variables deben estar entre 0 y 100 %.")
    if dias_apertura_mes <= 0:
        raise ValueError("Los días de apertura deben ser mayores que cero.")
    if any(x < 0 for x in [ticket_medio, personal_mensual, costes_fijos_mensuales]):
        raise ValueError("Los costes y el ticket deben ser no negativos.")
    d = predicciones.copy()
    d["facturacion_por_ticket"] = d["pedidos_previstos"] * ticket_medio
    d["facturacion_prevista"] = (d["facturacion_ml"] if usar_facturacion_ml
                                 else d["facturacion_por_ticket"])
    d["costes_variables"] = d["facturacion_prevista"] * porcentaje_coste_variable / 100
    d["costes_personal"] = personal_mensual / dias_apertura_mes
    d["costes_fijos"] = costes_fijos_mensuales / dias_apertura_mes
    d["beneficio_operativo"] = (d["facturacion_prevista"] - d["costes_variables"]
                                - d["costes_personal"] - d["costes_fijos"])
    d["margen_operativo_pct"] = np.where(d["facturacion_prevista"] > 0,
        d["beneficio_operativo"] / d["facturacion_prevista"] * 100, np.nan)
    return d


def punto_equilibrio(ticket_medio, porcentaje_coste_variable,
                     personal_mensual, costes_fijos_mensuales, dias_apertura_mes):
    """Pedidos diarios necesarios para que beneficio operativo >= 0."""
    contribucion = ticket_medio * (1 - porcentaje_coste_variable / 100)
    fijo_diario = (personal_mensual + costes_fijos_mensuales) / dias_apertura_mes
    if contribucion <= 0:
        return None
    return math.ceil(fijo_diario / contribucion)
