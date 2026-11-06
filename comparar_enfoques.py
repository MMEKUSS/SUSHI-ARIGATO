"""Comparación reproducible: reciente, anual y combinado.

Ejecutar desde la raíz del proyecto: python comparar_enfoques.py
Para comparar los nueve modelos originales: python comparar_enfoques.py --todos

NO sobrescribe los modelos originales; guarda todo en modelos/enfoques/.
"""

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from backend import (
    ENFOQUES, MODELOS, cargar_datos, cargar_modelo, columnas_para,
    predecir_futuro, preparar_variables, resumen_diario, separar_temporal,
    entrenar_modelos_calendario_largo,
)

DESTINO = MODELOS / "enfoques"
OBJETIVOS = {"demanda": "pedidos_diarios", "facturacion": "facturacion_diaria"}
ETIQUETAS = {"actual": "Reciente (ayer / 7 días)",
             "anual": "Mismo día del año anterior",
             "combinado": "Reciente + año anterior"}


def metricas(y, pred):
    return {
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "R2": float(r2_score(y, pred)) if len(y) > 1 else float("nan"),
    }


def slug(nombre):
    """Identificador estable para evitar caracteres raros y colisiones."""
    compacto = "".join(c.lower() if c.isalnum() else "_" for c in nombre).strip("_")[:32]
    sufijo = hashlib.sha1(nombre.encode("utf-8")).hexdigest()[:8]
    return f"{compacto}_{sufijo}"


def entrenar_comparar(diario, seleccion, horizonte):
    # La intersección de fechas es exactamente la misma para A, B y C.
    # El primer año se excluye de TODOS los enfoques para compararlos limpiamente.
    df = preparar_variables(diario, "combinado")
    train, test = separar_temporal(df, 0.8)
    if train.empty or test.empty:
        raise ValueError("No hay datos suficientes para comparar los tres enfoques.")
    if (test["date"].min() - train["date"].max()).days != 1:
        raise ValueError("El período de prueba no sigue inmediatamente al de entrenamiento.")

    DESTINO.mkdir(parents=True, exist_ok=True)
    informe, errores = [], []
    modelos = {tipo: {} for tipo in OBJETIVOS}
    manifiesto = {
        "version": 1,
        "fecha_entrenamiento_hasta": str(train["date"].max().date()),
        "prueba_desde": str(test["date"].min().date()),
        "prueba_hasta": str(test["date"].max().date()),
        "enfoques": {},
    }

    for enfoque in ENFOQUES:
        manifiesto["enfoques"][enfoque] = {tipo: [] for tipo in OBJETIVOS}
        print(f"\n== Enfoque: {ETIQUETAS[enfoque]} ==", flush=True)
        for tipo, objetivo in OBJETIVOS.items():
            columnas = columnas_para(tipo, enfoque)
            x_train, y_train = train[columnas], train[objetivo]
            x_test, y_test = test[columnas], test[objetivo]
            for nombre in seleccion[tipo]:
                print(f"  Entrenando {tipo}: {nombre} ...", flush=True)
                try:
                    # Clonamos la familia y parámetros del modelo guardado,
                    # PERO lo entrenamos de cero con las columnas del enfoque.
                    plantilla = cargar_modelo(tipo, nombre)
                    estimador = clone(plantilla)
                    estimador.fit(x_train, y_train)
                    pred = np.maximum(0.0, np.asarray(estimador.predict(x_test), dtype=float))
                    ruta = DESTINO / f"{tipo}_{enfoque}_{slug(nombre)}.joblib"
                    joblib.dump(estimador, ruta)
                    manifiesto["enfoques"][enfoque][tipo].append({
                        "modelo": nombre, "archivo": ruta.name,
                    })
                    modelos[(enfoque, tipo, nombre)] = estimador
                    informe.append({
                        "Objetivo": tipo, "Enfoque": enfoque, "Modelo": nombre,
                        **metricas(y_test, pred), "Dias_entrenamiento": len(train),
                        "Dias_prueba": len(test),
                    })
                except Exception as exc:
                    errores.append((enfoque, tipo, nombre, str(exc)))
                    warnings.warn(f"No se pudo entrenar {tipo}/{enfoque}/{nombre}: {exc}")

    if not informe:
        raise RuntimeError("Ningún modelo pudo entrenarse. Comprueba tus modelos y train_models.py.")

    resultados = pd.DataFrame(informe).sort_values(["Objetivo", "MAE"])
    resultados.to_csv(DESTINO / "comparacion_enfoques.csv", index=False, encoding="utf-8-sig")
    (DESTINO / "manifest.json").write_text(
        json.dumps(manifiesto, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== Métricas de un día vista (mismas fechas de prueba) ===")
    print(resultados[["Objetivo", "Enfoque", "Modelo", "MAE", "RMSE", "R2"]]
          .round(3).to_string(index=False), flush=True)

    # Validación práctica de previsiones de 1..horizonte días:
    # Avanzamos por bloques, sin consultar los días reales DENTRO de cada bloque.
    # Al comenzar el bloque siguiente sí conocemos el bloque anterior.
    filas_30 = []
    for enfoque in ENFOQUES:
        sub = resultados[resultados["Enfoque"] == enfoque]
        elegido = {}
        for tipo in OBJETIVOS:
            opciones = sub[sub["Objetivo"] == tipo]
            if not opciones.empty:
                nombre = str(opciones.iloc[0]["Modelo"])
                elegido[tipo] = (nombre, modelos[(enfoque, tipo, nombre)])
        if len(elegido) < 2:
            print(f"No se puede evaluar el horizonte del enfoque {enfoque}: faltan modelos.")
            continue

        pred_bloques = []
        for indice in range(0, len(test), horizonte):
            bloque = test.iloc[indice:indice + horizonte].copy()
            origen = pd.Timestamp(bloque["date"].iloc[0])
            historia_disponible = diario[diario["date"] < origen]
            pred = predecir_futuro(
                historia_disponible, elegido["demanda"][1],
                elegido["facturacion"][1], dias=len(bloque), enfoque=enfoque,
            )
            # Comparación fecha a fecha; garantizamos orden y ninguna fecha ausente.
            if not pred["date"].reset_index(drop=True).equals(
                bloque["date"].reset_index(drop=True)
            ):
                raise ValueError("Las fechas del pronóstico y la prueba no coinciden.")
            pred["origen_pronostico"] = origen
            pred["horizonte_dia"] = np.arange(1, len(pred) + 1)
            pred["pedidos_reales"] = bloque["pedidos_diarios"].to_numpy()
            pred["facturacion_real"] = bloque["facturacion_diaria"].to_numpy()
            pred_bloques.append(pred)

        datos_pred = pd.concat(pred_bloques, ignore_index=True)
        datos_pred.to_csv(
            DESTINO / f"predicciones_horizonte_{enfoque}.csv",
            index=False, encoding="utf-8-sig",
        )
        for tipo, col_real, col_pred in (
            ("demanda", "pedidos_reales", "pedidos_previstos"),
            ("facturacion", "facturacion_real", "facturacion_ml"),
        ):
            filas_30.append({
                "Objetivo": tipo, "Enfoque": enfoque,
                "Modelo": elegido[tipo][0], "Horizonte_max_dias": horizonte,
                "Dias_evaluados": len(datos_pred),
                **metricas(datos_pred[col_real], datos_pred[col_pred]),
            })

    if filas_30:
        comparacion_bloques = pd.DataFrame(filas_30).sort_values(["Objetivo", "MAE"])
        comparacion_bloques.to_csv(
            DESTINO / "comparacion_horizonte.csv", index=False, encoding="utf-8-sig"
        )
        print(f"\n=== Backtest en bloques de hasta {horizonte} días (recursivo cuando aplica) ===")
        print(comparacion_bloques[["Objetivo", "Enfoque", "Modelo", "MAE", "RMSE", "R2"]]
              .round(3).to_string(index=False), flush=True)

    if errores:
        ruta_errores = DESTINO / "errores_entrenamiento.txt"
        ruta_errores.write_text(
            "\n".join(f"{e}/{t}/{n}: {mensaje}" for e, t, n, mensaje in errores), encoding="utf-8"
        )
        print(f"\nAlgunos modelos fallaron. Revisa {ruta_errores}")
    print(f"\nResultados guardados en: {DESTINO.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Compara reciente, anual y combinado.")
    parser.add_argument("--calendario", action="store_true",
                        help="Valida por separado el modelo de año completo y guarda métricas.")
    parser.add_argument("--todos", action="store_true",
                        help="Entrena los nueve modelos originales de cada objetivo y enfoque.")
    parser.add_argument("--horizonte", type=int, default=30,
                        help="Horizonte por bloque (1..30 días). Por defecto: 30.")
    args = parser.parse_args()
    if args.calendario:
        pedidos, costes = cargar_datos()
        diario = resumen_diario(pedidos, costes)
        _, resultados = entrenar_modelos_calendario_largo(diario)
        DESTINO.mkdir(parents=True, exist_ok=True)
        ruta = DESTINO / "comparacion_calendario_largo_plazo.csv"
        resultados.to_csv(ruta, index=False, encoding="utf-8-sig")
        print(resultados.round(3).to_string(index=False))
        print(f"\nGuardado en: {ruta.resolve()}")
        return
    if not 1 <= args.horizonte <= 30:
        parser.error("--horizonte tiene que estar entre 1 y 30")
    comparaciones = {
        tipo: pd.read_csv(MODELOS / f"comparacion_{tipo}.csv")
        for tipo in OBJETIVOS
    }
    seleccion = {}
    for tipo, tabla in comparaciones.items():
        if not {"Modelo", "MAE"}.issubset(tabla.columns):
            raise ValueError(f"Faltan columnas Modelo y MAE en comparacion_{tipo}.csv")
        nombres = tabla.sort_values("MAE")["Modelo"].dropna().astype(str).tolist()
        seleccion[tipo] = list(dict.fromkeys(nombres if args.todos else nombres[:1]))
        if not seleccion[tipo]:
            raise ValueError(f"No hay modelos en comparacion_{tipo}.csv")
    pedidos, costes = cargar_datos()
    diario = resumen_diario(pedidos, costes)
    print("Estás reutilizando las familias e hiperparámetros de tus modelos guardados;")
    print("cada enfoque se reentrena desde cero para evitar incompatibilidades.")
    entrenar_comparar(diario, seleccion, args.horizonte)


if __name__ == "__main__":
    main()
