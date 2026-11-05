"""Entreno los mismos 9 regresores que tenía en mi proyecto original.

Ejecutar una sola vez: python train_models.py
Se usan DOS objetivos: pedidos diarios y facturación diaria.
El beneficio operativo se calcula con costes y no requiere otro modelo.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from backend import (CARPETA, MODELOS, COLUMNAS_DEMANDA, COLUMNAS_FACTURACION,
                     cargar_datos, resumen_diario, preparar_variables, separar_temporal,
                     medias_historicas)

CLAVES_MODELOS = {
    "Linear Regression": "linear_regression",
    "Ridge": "ridge",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
    "Gradient Boosting": "gradient_boosting",
    "HistGradientBoosting": "hist_gradient_boosting",
    "Decision Tree": "decision_tree",
    "KNN": "knn",
    "SVR": "svr",
}


def crear_modelos():
    """Devuelvo modelos sin entrenar."""
    return {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42, n_jobs=2, objective="reg:squarederror"),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "KNN": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5)),
        "SVR": make_pipeline(StandardScaler(), SVR()),
    }


def entrenar_objetivo(train, test, columnas, objetivo, prefijo):
    """Entreno, predigo y comparo MAE, RMSE y R² sin mezclar fechas."""
    X_train, y_train = train[columnas], train[objetivo]
    X_test, y_test = test[columnas], test[objetivo]
    resultados = []
    predicciones = pd.DataFrame({"date": test["date"].values, "real": y_test.values})
    for nombre, modelo in crear_modelos().items():
        print(f"  {prefijo}: {nombre}", flush=True)
        modelo.fit(X_train, y_train)
        pred = np.maximum(modelo.predict(X_test), 0)
        resultados.append({
            "Modelo": nombre,
            "MAE": float(mean_absolute_error(y_test, pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
            "R2": float(r2_score(y_test, pred)),
        })
        predicciones[nombre] = pred
        joblib.dump(modelo, MODELOS / f"{prefijo}_{CLAVES_MODELOS[nombre]}.joblib")
    resultados = pd.DataFrame(resultados).sort_values("MAE").reset_index(drop=True)
    resultados.to_csv(MODELOS / f"comparacion_{prefijo}.csv", index=False)
    predicciones.to_csv(MODELOS / f"predicciones_test_{prefijo}.csv", index=False)
    return resultados


def main():
    MODELOS.mkdir(parents=True, exist_ok=True)
    pedidos, costes = cargar_datos()
    diario = resumen_diario(pedidos, costes)
    df_ml = preparar_variables(diario)
    train, test = separar_temporal(df_ml, 0.8)
    print("Histórico:", diario["date"].min().date(), "a", diario["date"].max().date())
    print("Registros entrenamiento / prueba:", len(train), '/', len(test))
    print("Corte temporal:", test["date"].iloc[0].date())
    print("\nEntrenando MODELOS DE DEMANDA")
    rd = entrenar_objetivo(train, test, COLUMNAS_DEMANDA, "pedidos_diarios", "demanda")
    print("\nEntrenando MODELOS DE FACTURACIÓN")
    rf = entrenar_objetivo(train, test, COLUMNAS_FACTURACION, "facturacion_diaria", "facturacion")
    print("\nDemanda:\n", rd.round(2).to_string(index=False))
    print("\nFacturación:\n", rf.round(2).to_string(index=False))

    #En este código estoy haciendo una predicción de la facturación diaria del restaurante a partir de los pedidos previstos por Machine Learning.
    #Lo importante es que no estoy entrenando otro modelo de Machine Learning para predecir la facturación. 
    #Estoy utilizando las predicciones de demanda del mejor modelo y multiplicándolas por el ticket medio.
    #Lo hago así porque quiero calcular cuánto dinero podría facturar el restaurante a partir del número de pedidos que predice el modelo de Machine Learning.

    ticket_train = train["facturacion_diaria"].sum() / train["pedidos_diarios"].sum()
    pred_demanda_best = pd.read_csv(MODELOS / "predicciones_test_demanda.csv")
    best = rd.iloc[0]["Modelo"]
    facturacion_ticket = pred_demanda_best[best] * ticket_train
    error_ticket = pd.DataFrame([{
        "Método": "Pedidos previstos × ticket medio del entrenamiento",
        "Ticket medio entrenamiento": ticket_train,
        "MAE": mean_absolute_error(test["facturacion_diaria"], facturacion_ticket),
        "RMSE": np.sqrt(mean_squared_error(test["facturacion_diaria"], facturacion_ticket)),
        "R2": r2_score(test["facturacion_diaria"], facturacion_ticket),
    }])
    error_ticket.to_csv(MODELOS / "evaluacion_formula_ticket.csv", index=False)
    metadata = {
        "primer_dia": str(diario["date"].min().date()),
        "ultimo_dia": str(diario["date"].max().date()),
        "inicio_test": str(test["date"].min().date()),
        "filas_historico": int(len(diario)),
        "filas_train": int(len(train)),
        "filas_test": int(len(test)),
        "mejor_demanda": str(rd.iloc[0]["Modelo"]),
        "mejor_facturacion": str(rf.iloc[0]["Modelo"]),
        "ticket_entrenamiento": float(ticket_train),
    }
    (MODELOS / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf8")
    diario.to_csv(MODELOS / "historico_diario.csv", index=False)
    print("\nEntrenamiento finalizado: archivos en modelos/")
    print("Advertencia: prueba = predicción a 1 día con lags reales anteriores;")
    print("las predicciones a varios días en Streamlit son recursivas y más inciertas.")


if __name__ == "__main__":
    main()
