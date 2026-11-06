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

# Variables que utilizo para predecir pedidos diarios.
COLUMNAS_DEMANDA = [
    "dia_semana", "mes", "fin_semana",
    "pedidos_ayer", "pedidos_hace_7_dias", "media_7_dias",
]
# Para predecir facturación directamente uso los importes de días previos.
COLUMNAS_FACTURACION = [
    "dia_semana", "mes", "fin_semana",
    "ventas_ayer", "ventas_hace_7_dias", "media_ventas_7_dias",
]

# Los nuevos enfoques NO cambian los nombres ni el contenido de las columnas
# originales. Los modelos .joblib actuales siguen funcionando con "actual".
ENFOQUES = ("actual", "anual", "combinado")
_COLUMNAS_CALENDARIO = ["dia_semana", "mes", "fin_semana"]
COLUMNAS_DEMANDA_POR_ENFOQUE = {
    "actual": COLUMNAS_DEMANDA,
    "anual": _COLUMNAS_CALENDARIO + ["pedidos_ano_anterior"],
    "combinado": COLUMNAS_DEMANDA + ["pedidos_ano_anterior"],
}
COLUMNAS_FACTURACION_POR_ENFOQUE = {
    "actual": COLUMNAS_FACTURACION,
    "anual": _COLUMNAS_CALENDARIO + ["ventas_ano_anterior"],
    "combinado": COLUMNAS_FACTURACION + ["ventas_ano_anterior"],
}


def columnas_para(tipo, enfoque="actual"):
    """Devuelve el mismo orden de variables utilizado al entrenar."""
    if enfoque not in ENFOQUES:
        raise ValueError(f"Enfoque desconocido: {enfoque}. Opciones: {ENFOQUES}")
    if tipo == "demanda":
        return COLUMNAS_DEMANDA_POR_ENFOQUE[enfoque]
    if tipo == "facturacion":
        return COLUMNAS_FACTURACION_POR_ENFOQUE[enfoque]
    raise ValueError("El tipo tiene que ser 'demanda' o 'facturacion'.")


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


def preparar_variables(diario, enfoque="actual"):
    """Crea variables previas sin filtrar información del futuro.

    actual: últimos 1/7 días (idéntico al proyecto original).
    anual: mismo día natural del año anterior, con ajuste de 29/02 -> 28/02.
    combinado: las dos fuentes de información.

    Para evaluar los tres enfoques justamente, conviene usar las mismas fechas
    (filas del enfoque "combinado") en entrenamiento y prueba.
    """
    df = diario.sort_values("date").copy()
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].duplicated().any():
        raise ValueError("Hay fechas duplicadas en el histórico.")
    df["dia_semana"] = df["date"].dt.dayofweek
    df["mes"] = df["date"].dt.month
    df["fin_semana"] = (df["dia_semana"] >= 5).astype(int)

    df["pedidos_ayer"] = df["pedidos_diarios"].shift(1)
    df["pedidos_hace_7_dias"] = df["pedidos_diarios"].shift(7)
    df["media_7_dias"] = df["pedidos_diarios"].shift(1).rolling(7).mean()

    df["ventas_ayer"] = df["facturacion_diaria"].shift(1)
    df["ventas_hace_7_dias"] = df["facturacion_diaria"].shift(7)
    df["media_ventas_7_dias"] = df["facturacion_diaria"].shift(1).rolling(7).mean()

    if enfoque in ("anual", "combinado"):
        # Es una búsqueda por FECHA, no shift(365); respeta los años bisiestos.
        origen = df["date"] - pd.DateOffset(years=1)
        referencia = df.set_index("date")
        df["pedidos_ano_anterior"] = origen.map(referencia["pedidos_diarios"])
        df["ventas_ano_anterior"] = origen.map(referencia["facturacion_diaria"])

    obligatorias = columnas_para("demanda", enfoque) + columnas_para("facturacion", enfoque)
    return df.dropna(subset=obligatorias).reset_index(drop=True)


def separar_temporal(df, porcentaje_entrenamiento=0.8):
    """80 % de fechas antiguas para entrenar; 20 % recientes para evaluar."""
    limite = int(len(df) * porcentaje_entrenamiento)
    if limite < 10 or limite >= len(df):
        raise ValueError("No hay días suficientes para realizar la separación.")
    return df.iloc[:limite].copy(), df.iloc[limite:].copy()


def medias_historicas(pedidos, costes, dias=365):
    """Cifras observadas del período más reciente del dataset."""
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


def predecir_futuro(diario, modelo_demanda, modelo_facturacion, dias=29, enfoque="actual"):
    """Predice entre 1 y 30 días sin utilizar datos reales futuros.

    Por defecto usa exactamente el comportamiento anterior ("actual").
    Con "anual" usa únicamente fechas anteriores en un año; no es recursivo.
    Con "combinado" consulta ambos: el año anterior y los lags recursivos.
    IMPORTANTE: los modelos que se pasen deben estar ENTRENADOS con las
    columnas del enfoque solicitado; no se pueden reciclar .joblib antiguos
    cambiando simplemente `enfoque`.
    """
    if not 1 <= dias <= 30:
        raise ValueError("Se admiten horizontes de 1 a 30 días.")
    columnas_d = columnas_para("demanda", enfoque)
    columnas_f = columnas_para("facturacion", enfoque)
    historial = diario.sort_values("date").copy()
    fechas = list(pd.to_datetime(historial["date"]))
    pedidos = list(historial["pedidos_diarios"].astype(float))
    ventas = list(historial["facturacion_diaria"].astype(float))
    if len(fechas) < 8:
        raise ValueError("Necesito al menos ocho fechas consecutivas.")
    if len(pd.date_range(fechas[0], fechas[-1], freq="D")) != len(fechas):
        raise ValueError("El historial contiene días sin registros.")
    if len(set(fechas)) != len(fechas):
        raise ValueError("El histórico contiene fechas duplicadas.")

    referencia_d = dict(zip(fechas, pedidos))
    referencia_f = dict(zip(fechas, ventas))
    salida = []
    for _ in range(dias):
        fecha = fechas[-1] + pd.Timedelta(days=1)
        base = {
            "dia_semana": fecha.dayofweek,
            "mes": fecha.month,
            "fin_semana": int(fecha.dayofweek >= 5),
        }
        xd, xf = base.copy(), base.copy()
        if enfoque in ("actual", "combinado"):
            xd.update({
                "pedidos_ayer": pedidos[-1],
                "pedidos_hace_7_dias": pedidos[-7],
                "media_7_dias": float(np.mean(pedidos[-7:])),
            })
            xf.update({
                "ventas_ayer": ventas[-1],
                "ventas_hace_7_dias": ventas[-7],
                "media_ventas_7_dias": float(np.mean(ventas[-7:])),
            })
        if enfoque in ("anual", "combinado"):
            fecha_anterior = fecha - pd.DateOffset(years=1)
            if fecha_anterior not in referencia_d:
                raise ValueError(
                    f"No tengo información del {fecha_anterior:%d/%m/%Y} "
                    f"para predecir {fecha:%d/%m/%Y}."
                )
            xd["pedidos_ano_anterior"] = referencia_d[fecha_anterior]
            xf["ventas_ano_anterior"] = referencia_f[fecha_anterior]

        x_demanda = pd.DataFrame([xd])[columnas_d]
        x_ventas = pd.DataFrame([xf])[columnas_f]
        pred_pedidos = max(0.0, float(modelo_demanda.predict(x_demanda)[0]))
        pred_ventas = max(0.0, float(modelo_facturacion.predict(x_ventas)[0]))
        fechas.append(fecha)
        pedidos.append(pred_pedidos)
        ventas.append(pred_ventas)
        salida.append({"date": fecha, "pedidos_previstos": pred_pedidos,
                       "facturacion_ml": pred_ventas})
    return pd.DataFrame(salida)



# -----------------------------------------------------------------------------
# Proyección de años completos: se entrena con atributos de calendario, sin lags
# de ventas de un 2025/2026 que no tenemos. Es un escenario, no una certeza.
# -----------------------------------------------------------------------------
COLUMNAS_CALENDARIO_LARGO = [
    "dia_semana", "mes", "fin_semana", "dia_mes", "seno_anual", "coseno_anual",
]


def variables_calendario_largo(fechas):
    """Variables conocidas para cualquier fecha futura; no consultan ventas futuras."""
    fechas = pd.DatetimeIndex(pd.to_datetime(fechas))
    dias_anio = np.where(fechas.is_leap_year, 366, 365)
    angulo = 2 * np.pi * (fechas.dayofyear.to_numpy() - 1) / dias_anio
    return pd.DataFrame({
        "dia_semana": fechas.dayofweek,
        "mes": fechas.month,
        "fin_semana": (fechas.dayofweek >= 5).astype(int),
        "dia_mes": fechas.day,
        "seno_anual": np.sin(angulo),
        "coseno_anual": np.cos(angulo),
    })[COLUMNAS_CALENDARIO_LARGO]


def entrenar_modelos_calendario_largo(diario):
    """Compara tres regresores sobre el ÚLTIMO año y reentrena el mejor.

    No reutiliza los modelos con variables lag de 1/7 días porque el horizonte
    2027 está separado del último dato real (2024) por años sin observaciones.
    Devuelve ({'demanda': estimador, 'facturacion': estimador}, métricas).
    """
    from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.base import clone

    hist = diario[["date", "pedidos_diarios", "facturacion_diaria"]].copy()
    hist["date"] = pd.to_datetime(hist["date"])
    hist = hist.sort_values("date").reset_index(drop=True)
    if hist["date"].duplicated().any():
        raise ValueError("El histórico tiene fechas duplicadas.")
    if (hist["date"].max() - hist["date"].min()).days < 730:
        raise ValueError("Se necesitan al menos dos años de datos para validar una proyección anual.")
    ultimo_anio = int(hist["date"].dt.year.max())
    train_mask = hist["date"].dt.year < ultimo_anio
    test_mask = hist["date"].dt.year == ultimo_anio
    if train_mask.sum() < 365 or test_mask.sum() < 180:
        raise ValueError("Se necesita un año de entrenamiento y otro de evaluación suficientemente completo.")

    X = variables_calendario_largo(hist["date"])
    candidatos = {
        "Random Forest": RandomForestRegressor(
            n_estimators=90, max_depth=12, min_samples_leaf=8,
            random_state=42, n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=110, max_leaf_nodes=15, min_samples_leaf=25,
            learning_rate=0.05, random_state=42,
        ),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=20.0)),
    }
    resultados = []
    modelos_finales = {}
    for tipo, objetivo in (("demanda", "pedidos_diarios"),
                           ("facturacion", "facturacion_diaria")):
        y = hist[objetivo].astype(float).reset_index(drop=True)
        mae_mejor, nombre_mejor = float("inf"), None
        for nombre, estimador in candidatos.items():
            prueba = clone(estimador)
            prueba.fit(X.loc[train_mask], y.loc[train_mask])
            pred = np.maximum(0.0, prueba.predict(X.loc[test_mask]))
            mae = float(mean_absolute_error(y.loc[test_mask], pred))
            resultados.append({
                "Objetivo": tipo, "Modelo": nombre,
                "MAE": mae,
                "RMSE": float(np.sqrt(mean_squared_error(y.loc[test_mask], pred))),
                "R2": float(r2_score(y.loc[test_mask], pred)),
                "Anio_validacion": ultimo_anio,
            })
            if mae < mae_mejor:
                mae_mejor, nombre_mejor = mae, nombre
        modelo_final = clone(candidatos[nombre_mejor])
        modelo_final.fit(X, y)
        modelos_finales[tipo] = modelo_final

    return modelos_finales, pd.DataFrame(resultados).sort_values(["Objetivo", "MAE"])


def predecir_anio_completo(diario, anio=2027, modelos=None):
    """Proyecta 01/01 a 31/12 del año elegido usando solo calendario conocido.

    Sin costes, salarios, promociones ni inflación futuros: se configuran aparte.
    Si se facilitan 'modelos', deben proceder de entrenar_modelos_calendario_largo.
    """
    if not 2025 <= int(anio) <= 2035:
        raise ValueError("El año de previsión debe estar entre 2025 y 2035.")
    ultimo_anio_real = int(pd.to_datetime(diario["date"]).dt.year.max())
    if int(anio) <= ultimo_anio_real:
        raise ValueError("Selecciona un año posterior al último del histórico.")
    if modelos is None:
        modelos, _ = entrenar_modelos_calendario_largo(diario)
    fechas = pd.date_range(f"{anio}-01-01", f"{anio}-12-31", freq="D")
    X = variables_calendario_largo(fechas)
    return pd.DataFrame({
        "date": fechas,
        "pedidos_previstos": np.maximum(0.0, modelos["demanda"].predict(X)),
        "facturacion_ml": np.maximum(0.0, modelos["facturacion"].predict(X)),
    })

def calcular_rentabilidad(predicciones, ticket_medio, porcentaje_coste_variable,
                          personal_mensual, costes_fijos_mensuales,
                          dias_apertura_mes=30, usar_facturacion_ml=False,
                          repartir_coste_mensual=False):
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
    if repartir_coste_mensual:
        # En el año completo deben imputarse exactamente 12 mensualidades,
        # no 365 / dias_apertura_mes mensualidades.
        d["date"] = pd.to_datetime(d["date"])
        dias_mes = d["date"].dt.days_in_month
        d["costes_personal"] = personal_mensual / dias_mes
        d["costes_fijos"] = costes_fijos_mensuales / dias_mes
    else:
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
