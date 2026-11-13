"""Sushi Arigato — interfaz Restaurant Management para Streamlit.

La previsión anual se entrena con modelos estacionales, mientras que la de
corto plazo conserva los modelos originales guardados.

Ejecutar: python -m streamlit run funciones.py
"""

from pathlib import Path
import json
import base64
import joblib

import altair as alt
import pandas as pd
import streamlit as st

# El gráfico diario puede superar 5.000 filas al seleccionar los diez años.
alt.data_transformers.disable_max_rows()

from backend import (
    MODELOS,
    calcular_rentabilidad,
    cargar_datos,
    cargar_modelo,
    medias_historicas,
    predecir_futuro,
    entrenar_modelos_calendario_largo,
    predecir_anio_completo,
    punto_equilibrio,
    resumen_diario,
)

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN Y ESTILO VISUAL — TEMA JAPONÉS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sushi Arigato | Restaurant Management",
    page_icon="🍣",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paleta japonesa inspirada en tinta sumi-e, pergamino y sol rojo.
# No cambiamos las predicciones, los datos ni los modelos del backend.
ROJO = "#B3261E"
ROJO_OSCURO = "#871E1B"
DORADO = "#C39A4C"
CREMA = "#F4E8C8"
MARFIL = "#FFF9EC"
TINTA = "#251D18"
MARRON = "#382D27"
GRIS = "#796A5B"
BORDE = "#D8C6A1"

# Ilustración vectorial ligera. Va integrada en este mismo .py, sin imágenes
# externas ni carpetas adicionales: sol japonés, montañas y cañas de bambú.
# El resultado es decorativo y no reproduce literalmente la fotografía original.
ARTE_JAPONES = r"""<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="320" viewBox="0 0 1400 320">
<defs>
  <linearGradient id="sun" x1="0" x2="0.75" y1="0" y2="1">
    <stop offset="0" stop-color="#C45742"/><stop offset="1" stop-color="#AD2A20"/>
  </linearGradient>
  <linearGradient id="haze" x1="0" x2="0" y1="0" y2="1">
    <stop offset="0" stop-color="#998E7D" stop-opacity="0.02"/>
    <stop offset="1" stop-color="#6E6858" stop-opacity="0.2"/>
  </linearGradient>
</defs>
<circle cx="1132" cy="132" r="87" fill="url(#sun)" opacity=".76"/>
<circle cx="1132" cy="132" r="93" fill="none" stroke="#A5362A" stroke-width="1" opacity=".14"/>
<path d="M700 290 L796 234 L835 251 L910 210 L960 242 L1040 191 L1110 245 L1218 201 L1273 230 L1400 172 L1400 320 L700 320Z" fill="url(#haze)"/>
<path d="M594 309 Q760 270 842 294 Q934 261 1030 295 Q1130 261 1225 281 Q1311 248 1400 277 L1400 320 L594 320Z" fill="#726B60" opacity=".14"/>
<path d="M680 315 Q960 285 1098 304 T1400 295" fill="none" stroke="#8A7860" stroke-width="3" opacity=".16"/>
<!-- Tinta de un pequeño templo de cinco aleros, al pie de las montañas. -->
<g transform="translate(897,218)" fill="#403A32" opacity=".25">
  <rect x="31" y="30" width="21" height="58"/><rect x="22" y="53" width="39" height="35"/>
  <path d="M5 28 Q41 34 79 28 L67 40 L18 40Z"/>
  <path d="M12 49 Q41 54 72 49 L63 58 L20 58Z"/>
  <path d="M18 68 Q41 74 66 68 L58 77 L24 77Z"/>
  <rect x="39" y="11" width="3" height="21"/>
</g>
<!-- Bambú en tinta, a la derecha de la cabecera. -->
<g fill="none" stroke="#302D25" stroke-linecap="round" opacity=".62">
  <path d="M1300 -20 C1280 89 1301 187 1277 350" stroke-width="7"/>
  <path d="M1367 -25 C1338 90 1368 201 1348 340" stroke-width="5"/>
  <path d="M1276 145 Q1216 94 1161 93" stroke-width="4"/>
  <path d="M1294 70 Q1340 38 1390 39" stroke-width="4"/>
  <path d="M1358 191 Q1307 177 1263 199" stroke-width="4"/>
  <path d="M1305 226 Q1246 211 1207 236" stroke-width="3"/>
</g>
<g fill="#252820" opacity=".56">
  <path d="M1278 132 Q1215 78 1179 78 Q1206 109 1278 132Z"/>
  <path d="M1262 120 Q1221 64 1193 57 Q1212 92 1262 120Z"/>
  <path d="M1250 106 Q1196 98 1156 113 Q1192 120 1250 106Z"/>
  <path d="M1300 74 Q1321 14 1368 -4 Q1356 34 1300 74Z"/>
  <path d="M1318 54 Q1381 12 1419 20 Q1377 51 1318 54Z"/>
  <path d="M1315 56 Q1336 4 1332 -29 Q1307 1 1315 56Z"/>
  <path d="M1347 176 Q1297 148 1261 151 Q1290 177 1347 176Z"/>
  <path d="M1339 182 Q1294 200 1275 226 Q1314 211 1339 182Z"/>
  <path d="M1304 225 Q1262 201 1225 202 Q1249 226 1304 225Z"/>
  <path d="M1280 231 Q1220 225 1191 256 Q1234 248 1280 231Z"/>
  <path d="M1352 109 Q1394 75 1419 84 Q1398 109 1352 109Z"/>
  <path d="M1354 127 Q1393 121 1422 148 Q1390 145 1354 127Z"/>
  <path d="M1353 260 Q1391 225 1430 231 Q1399 259 1353 260Z"/>
</g>
<g stroke="#625847" stroke-opacity=".17" fill="none" stroke-width="1.5">
  <path d="M710 297 Q817 280 906 288"/><path d="M936 310 Q1013 295 1101 301"/>
  <path d="M1157 308 Q1227 290 1321 299"/>
</g>
</svg>"""

ARTE_JAPONES_URL = (
    "data:image/svg+xml;base64,"
    + base64.b64encode(ARTE_JAPONES.encode("utf-8")).decode("ascii")
)

CSS_JAPONES = r"""
<style>
/* Un tema uniforme: fondo pergamino + tinta + dorado + rojo japonés. */
:root {
    --japan-red: #B3261E;
    --japan-dark-red: #871E1B;
    --japan-gold: #C39A4C;
    --japan-ink: #251D18;
    --japan-brown: #382D27;
    --japan-ivory: #FFF9EC;
    --japan-border: #D8C6A1;
}
html, body, [data-testid="stAppViewContainer"] {
    font-family: "Segoe UI", "Noto Sans", sans-serif;
}
.stApp {
    color: #251D18;
    background-color: #F4E8C8;
    background-image:
        radial-gradient(circle at 1px 1px, rgba(111, 79, 39, .055) 0.7px, transparent 1px),
        radial-gradient(ellipse at 17% 10%, #FAF1DB 0%, #F4E8C8 63%, #E8D5AB 100%);
    background-size: 7px 7px, auto;
}
[data-testid="stHeader"] { background: rgba(246, 236, 213, .93); }
[data-testid="stAppViewContainer"] > .main { background: transparent; }
.block-container { max-width: 1450px; padding-top: 1.7rem; padding-bottom: 3.5rem; }
h1, h2, h3, h4 { color: #251D18; letter-spacing: -.025em; }
h1 { font-weight: 800; }
h3 { font-weight: 750; }
hr { border-color: #D8C6A1; }
/* Evita texto verde heredado del estilo anterior. */
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] { color: #43372F; }
[data-testid="stAppViewContainer"] a { color: #972620; }
/* Navegación inspirada en tinta sumi-e. */
[data-testid="stSidebar"] {
    background: linear-gradient(170deg, #382D27 0%, #2D2723 100%);
    border-right: 2px solid #B38C50;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #F8EEDC;
}
[data-testid="stSidebar"] hr { border-color: rgba(244, 231, 207, .23); }
[data-testid="stSidebar"] [role="radiogroup"] label {
    border-radius: 10px;
    padding: 9px 10px;
    transition: background 150ms ease;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: rgba(255, 241, 211, .11);
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background: rgba(179, 38, 30, .27);
    box-shadow: inset 3px 0 #D1A357;
}
[data-testid="stSidebar"] [role="radiogroup"] input { accent-color: #C44A39; }
/* Entradas claras sobre el menú oscuro, para mayor legibilidad. */
[data-testid="stSidebar"] input:not([type="radio"]),
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-testid="stNumberInput"] > div,
[data-testid="stSidebar"] [data-testid="stNumberInput"] button {
    background: #FFF9EC;
    color: #251D18;
    border-color: #D8C6A1;
}
[data-testid="stSidebar"] input:not([type="radio"]) { color: #251D18; }
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color: #F8EEDC; }
[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
    background: #B3261E;
    border-color: #F4E8C8;
}
/* Identidad y cabecera. */
.brand { display:flex; align-items:center; gap:13px; padding: 13px 0 20px; }
.brand-icon {
    display:flex; align-items:center; justify-content:center;
    width:51px; height:51px;
    background:#6E2822; border:1px solid #D9A75F;
    border-radius:14px; font-size:28px;
}
.brand-name { font-weight:850; letter-spacing:.07em; color:#FFF8E8; font-size:17px; }
.brand-subtitle { font-size:10px; letter-spacing:.16em; color:#D9B67B; }
.eyebrow { letter-spacing:.16em; text-transform:uppercase; font-size:11px;
    color:#9F2B23; font-weight:850; margin-bottom:9px; }
.hero {
    position:relative; overflow:hidden; isolation:isolate;
    background-color:#F2E2BF;
    background-image: linear-gradient(90deg, rgba(249,240,220,.98) 0%,
        rgba(249,240,220,.96) 44%, rgba(249,240,220,.38) 76%,
        rgba(249,240,220,.10) 100%), url("__ARTE_JAPONES__");
    background-position:center, right center;
    background-size:cover, cover;
    border:1px solid #DDC69E;
    border-radius:20px;
    padding:34px 36px; margin-bottom:25px; min-height:207px;
    box-shadow:0 10px 28px rgba(77,50,27,.10);
}
.hero h1 { color:#201A16; font-size:clamp(28px, 3vw, 42px);
    max-width:880px; margin:0 0 10px; line-height:1.16; font-weight:850; }
.hero p { color:#493A2F; max-width:810px; font-size:15px;
    margin:0; line-height:1.55; font-weight:500; }
.hero-badge { display:inline-block; margin-top:17px;
    border:1px solid #BFA579; background:rgba(255,250,238,.83);
    color:#713C2C; padding:7px 13px; border-radius:25px;
    font-size:11px; font-weight:700; }
.section-heading { margin-top:16px; margin-bottom:13px;
    font-size:21px; font-weight:800; color:#2D251E; letter-spacing:-.025em; }
.helper { color:#705F4E; font-size:13px; line-height:1.65; margin:5px 0 18px; }
.notice { color:#51382C; padding:13px 16px; background:#F3E0BC;
    border-left:4px solid #B3261E; border-radius:9px; margin:10px 0 20px;
    font-size:13px; line-height:1.5; }
.footer-note { font-size:12px; color:#7C6A55; margin-top:22px; }
.mini-label { letter-spacing:.08em; text-transform:uppercase;
    font-size:11px; color:#A22E26; font-weight:800; }
/* KPI / métricas. */
[data-testid="stMetric"] {
    background:#FFF9EC; border:1px solid #DFCA9F;
    border-radius:15px; padding:20px; min-height:125px;
    box-shadow:0 5px 16px rgba(84,55,27,.07);
}
[data-testid="stMetricLabel"] { color:#78644F; font-size:14px; }
[data-testid="stMetricValue"] { color:#AE2E24; font-size:clamp(22px, 2.45vw, 31px); font-weight:750; }
[data-testid="stMetricDelta"] { font-size:12px; }
/* Tablas, gráficos, controles. */
[data-testid="stVerticalBlockBorderWrapper"] > div { border-radius:14px; }
[data-testid="stDataFrame"] {
    background:#FFF9EC; border:1px solid #DFCDAE; border-radius:12px;
}
[data-testid="stVegaLiteChart"] {
    background:#FFF9EC; padding:12px; border:1px solid #E4D3B4;
    border-radius:12px;
}
.stButton > button[kind="primary"], .stDownloadButton > button {
    background:#B3261E; color:#FFF9EC; border:1px solid #B3261E;
    border-radius:10px; padding:9px 15px; font-weight:700;
}
.stButton > button[kind="primary"]:hover, .stDownloadButton > button:hover {
    background:#871E1B; border-color:#871E1B; color:#FFFFFF;
}
.stButton > button[kind="secondary"] { border-color:#D2BA94; color:#473429; }
.stButton > button[kind="secondary"]:hover { border-color:#B3261E; color:#B3261E; }
[data-testid="stAppViewContainer"] input,
[data-testid="stAppViewContainer"] [data-baseweb="select"] > div {
    border-radius:9px;
}
.stAlert { border-radius:12px; }
@media (max-width:800px) {
    .hero { padding:24px 21px; min-height:180px;
        background-image:linear-gradient(90deg, rgba(249,240,220,.98) 0%,
            rgba(249,240,220,.91) 100%), url("__ARTE_JAPONES__"); }
    .hero h1 { font-size:29px; }
    .block-container { padding-top:1.1rem; }
}
</style>
"""
st.markdown(
    CSS_JAPONES.replace("__ARTE_JAPONES__", ARTE_JAPONES_URL),
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 2. FUNCIONES DEL PROYECTO (MISMA LÓGICA QUE EL ORIGINAL)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Leyendo el histórico...")
def datos():
    pedidos, costes = cargar_datos()
    diario = resumen_diario(pedidos, costes)
    referencias = medias_historicas(pedidos, costes)
    return pedidos, costes, diario, referencias


@st.cache_resource(show_spinner="Cargando modelo...")
def modelo(tipo, nombre, enfoque="original", version=0):
    """Los modelos originales nunca se sustituyen por los del experimento."""
    if enfoque == "original":
        return cargar_modelo(tipo, nombre)
    directorio = MODELOS / "enfoques"
    manifiesto = json.loads((directorio / "manifest.json").read_text(encoding="utf-8"))
    for entrada in manifiesto["enfoques"][enfoque][tipo]:
        if entrada["modelo"] == nombre:
            return joblib.load(directorio / entrada["archivo"])
    raise ValueError(f"No hay modelo para {tipo}, {enfoque} y {nombre}.")


@st.cache_resource(show_spinner="Entrenando y validando modelos estacionales...")
def modelos_largo_plazo():
    # Una sola vez por sesión de caché, no cada vez que cambian los costes.
    _, _, diario, _ = datos()
    return entrenar_modelos_calendario_largo(diario)


def euro(valor):
    return f"{valor:,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")


def entero(valor):
    return f"{valor:,.0f}".replace(",", ".")


def porcentaje(valor):
    return f"{valor:.1f} %".replace(".", ",")


def hero(titulo, descripcion, etiqueta="CASO DE ESTUDIO · RESTAURANTE DE SUSHI"):
    st.markdown(
        f"""<div class="hero"><div class="eyebrow">{etiqueta}</div>
        <h1>{titulo}</h1><p>{descripcion}</p>
        <span class="hero-badge">Datos 2015–2024 · {f'Proyección {anio_prediccion}' if es_anual else 'Pronóstico de 1 a 30 días desde 2025'}</span></div>""",
        unsafe_allow_html=True,
    )


def seccion(titulo, descripcion=None):
    st.markdown(f'<div class="section-heading">{titulo}</div>', unsafe_allow_html=True)
    if descripcion:
        st.markdown(f'<div class="helper">{descripcion}</div>', unsafe_allow_html=True)


def aviso(mensaje):
    st.markdown(f'<div class="notice">{mensaje}</div>', unsafe_allow_html=True)


def estilo_grafico(grafico):
    """Estilo japonés para todos los gráficos Altair, manteniendo sus datos."""
    return (
        grafico.properties(background=MARFIL)
        .configure_view(stroke=None)
        .configure_axis(
            labelColor=TINTA,
            titleColor=TINTA,
            gridColor="#E8DBC2",
            tickColor="#D8C6A1",
            domainColor="#D8C6A1",
            labelFontSize=12,
            titleFontSize=13,
        )
        .configure_legend(labelColor=TINTA, titleColor=TINTA, labelFontSize=12)
    )


def linea(data, x, y, *, nombres=None, titulo_y=None, alto=310, puntos=False):
    """Gráfico Altair interactivo: muestra valores al pasar el cursor."""
    nombres = nombres or {col: col for col in y}
    columnas = data[[x, *y]].copy()
    columnas = columnas.rename(columns=nombres)
    series = [nombres.get(col, col) for col in y]
    largo = columnas.melt(id_vars=[x], value_vars=series,
                          var_name="Serie", value_name="Valor")
    paleta = [ROJO, DORADO, "#705548", "#D9AC70"][:len(series)]
    base = alt.Chart(largo).encode(
        x=alt.X(f"{x}:T", title=None, axis=alt.Axis(labelAngle=0, format="%d %b %Y")),
        y=alt.Y("Valor:Q", title=titulo_y, scale=alt.Scale(zero=False)),
        color=alt.Color("Serie:N", scale=alt.Scale(domain=series, range=paleta),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=[alt.Tooltip(f"{x}:T", title="Fecha", format="%d/%m/%Y"),
                 alt.Tooltip("Serie:N", title="Serie"),
                 alt.Tooltip("Valor:Q", title="Valor", format=",.2f")],
    )
    grafico = base.mark_line(point=puntos, strokeWidth=2.8).properties(height=alto)
    st.altair_chart(estilo_grafico(grafico.interactive()), use_container_width=True)


def barras(data, x, y, *, color=ROJO, alto=265, eje_titulo=None, orden=None):
    grafico = alt.Chart(data).mark_bar(
        color=color, cornerRadiusTopLeft=5, cornerRadiusTopRight=5,
    ).encode(
        x=alt.X(f"{x}:O", title=None, sort=orden,
                axis=alt.Axis(labelAngle=0, labelLimit=100)),
        y=alt.Y(f"{y}:Q", title=eje_titulo),
        tooltip=[alt.Tooltip(f"{x}:O", title=x),
                 alt.Tooltip(f"{y}:Q", title=y, format=",.2f")],
    ).properties(height=alto)
    st.altair_chart(estilo_grafico(grafico), use_container_width=True)


def tabla_csv(data, archivo, etiqueta="Descargar datos (CSV)"):
    st.download_button(
        etiqueta,
        data.to_csv(index=False).encode("utf-8-sig"),
        file_name=archivo,
        mime="text/csv",
    )


# -----------------------------------------------------------------------------
# 3. LECTURA DE ARCHIVOS Y CONTROLES
# -----------------------------------------------------------------------------
try:
    pedidos, costes, diario, referencias = datos()
except Exception as error:
    st.error("No se han podido cargar los datos. Mantén los CSV originales en `data/`.")
    st.code("data/sushi_orders.csv\ndata/sushi_daily_pnl.csv")
    st.exception(error)
    st.stop()

necesarios = ["metadata.json", "comparacion_demanda.csv", "comparacion_facturacion.csv"]
faltantes = [nombre for nombre in necesarios if not (MODELOS / nombre).exists()]
if faltantes:
    st.error("Faltan archivos de modelos: " + ", ".join(faltantes))
    st.info("Conserva tu carpeta `modelos/` y, si no existe, ejecuta `python train_models.py`.")
    st.stop()

try:
    comparacion_d = pd.read_csv(MODELOS / "comparacion_demanda.csv")
    comparacion_f = pd.read_csv(MODELOS / "comparacion_facturacion.csv")
except Exception as error:
    st.error("No se han podido leer los resultados de comparación de modelos.")
    st.exception(error)
    st.stop()

st.sidebar.markdown(
    """<div class="brand"><div class="brand-icon">🍣</div><div>
    <div class="brand-name">SUSHI ARIGATO</div>
    <div class="brand-subtitle">RESTAURANT MANAGEMENT</div>
    </div></div>""",
    unsafe_allow_html=True,
)

PAGINAS = [
    "🏠  Dashboard",
    "📊  Histórico",
    "📈  Demanda",
    "💰  Facturación",
    "💵  Rentabilidad",
    "🧠  Modelos ML",
]
st.sidebar.markdown("**MENÚ PRINCIPAL**")
pagina = st.sidebar.radio("Selecciona la sección", PAGINAS, label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.markdown("**CONFIGURACIÓN DE PREVISIONES**")
modo_plazo = st.sidebar.radio(
    "Horizonte de predicción",
    ["Año completo", "Corto plazo (1–30 días)"],
    horizontal=False,
)
es_anual = modo_plazo == "Año completo"
if es_anual:
    anio_prediccion = st.sidebar.selectbox(
        "Año que quieres predecir", list(range(2025, 2036)), index=2
    )
    dias = len(pd.date_range(f"{anio_prediccion}-01-01", f"{anio_prediccion}-12-31"))
    st.sidebar.caption("Modelos estacionales: usan calendario y datos observados de 2015–2024.")
else:
    dias = st.sidebar.slider("Días que quieres predecir", 1, 30, 14)
    anio_prediccion = None

# Antes de ejecutar comparar_enfoques.py se mantiene el proyecto TAL CUAL.
# Al ejecutar el script aparecen tres enfoques experimentales adicionales.
carpeta_enfoques = MODELOS / "enfoques"
ruta_manifiesto = carpeta_enfoques / "manifest.json"
manifest_experimento = None
if ruta_manifiesto.exists():
    try:
        manifest_experimento = json.loads(ruta_manifiesto.read_text(encoding="utf-8"))
    except (ValueError, KeyError, OSError):
        st.sidebar.warning("El manifiesto de la comparación no se puede leer.")

opciones_enfoque = {"Original (sin cambios)": "original"}
if manifest_experimento:
    for clave, etiqueta in [
        ("actual", "A · Reciente (reentrenado)"),
        ("anual", "B · Mismo día del año anterior"),
        ("combinado", "C · Reciente + anual"),
    ]:
        conjuntos = manifest_experimento.get("enfoques", {}).get(clave, {})
        if conjuntos.get("demanda") and conjuntos.get("facturacion"):
            opciones_enfoque[etiqueta] = clave

if es_anual:
    etiqueta_enfoque = "D · Estacionalidad (validación en 2024)"
    enfoque_elegido = "calendario_largo"
    modelo_d = "Selección automática por MAE (2024)"
    modelo_f = "Selección automática por MAE (2024)"
else:
    etiqueta_enfoque = st.sidebar.selectbox(
        "Cómo quieres predecir", list(opciones_enfoque), index=0
    )
    enfoque_elegido = opciones_enfoque[etiqueta_enfoque]
    if enfoque_elegido == "original":
        opciones_d = comparacion_d["Modelo"].tolist()
        opciones_f = comparacion_f["Modelo"].tolist()
    else:
        conjuntos = manifest_experimento["enfoques"][enfoque_elegido]
        opciones_d = [item["modelo"] for item in conjuntos["demanda"]]
        opciones_f = [item["modelo"] for item in conjuntos["facturacion"]]
    with st.sidebar.expander("Modelos de Machine Learning", expanded=False):
        modelo_d = st.selectbox("Modelo de demanda", opciones_d)
        modelo_f = st.selectbox("Modelo de facturación", opciones_f)
    if not manifest_experimento:
        st.sidebar.caption("Para activar la comparación anual: python comparar_enfoques.py")

with st.sidebar.expander("Editar escenario económico", expanded=False):
    ticket = st.number_input("Ticket medio (€ / pedido)", min_value=0.0,
                             value=round(float(referencias["ticket_medio"]), 2), step=1.0)
    cogs_pct = st.slider("Costes variables (%)", 0.0, 100.0,
                         float(round(referencias["porcentaje_coste_variable"], 1)), 0.5)
    dias_apertura = st.number_input("Días abiertos al mes", min_value=1,
                                    max_value=31, value=30)
    personal_mensual = st.number_input("Personal al mes (€)", min_value=0.0,
                          value=round(float(referencias["personal_diario"]) * 30, 2), step=100.0)
    fijos_mensuales = st.number_input("Otros costes fijos al mes (€)", min_value=0.0,
                          value=round(float(referencias["costes_fijos_diarios"]) * 30, 2), step=50.0)
    modo_facturacion = st.radio(
        "Método de facturación para el beneficio",
        ["Pedidos × ticket medio", "Predicción directa ML"],
    )

st.sidebar.caption("Caso educativo · No son previsiones de un restaurante nuevo.")

try:
    # La versión invalida la caché si vuelves a generar el manifiesto.
    version_modelos = ruta_manifiesto.stat().st_mtime_ns if ruta_manifiesto.exists() else 0
    if es_anual:
        modelos_anuales, metricas_anuales = modelos_largo_plazo()
        forecast = predecir_anio_completo(diario, anio_prediccion, modelos_anuales)
        modelo_d = str(metricas_anuales[metricas_anuales["Objetivo"] == "demanda"].iloc[0]["Modelo"])
        modelo_f = str(metricas_anuales[metricas_anuales["Objetivo"] == "facturacion"].iloc[0]["Modelo"])
    else:
        metricas_anuales = None
        forecast = predecir_futuro(
            diario,
            modelo("demanda", modelo_d, enfoque_elegido, version_modelos),
            modelo("facturacion", modelo_f, enfoque_elegido, version_modelos),
            dias,
            enfoque="actual" if enfoque_elegido == "original" else enfoque_elegido,
        )
    escenario = calcular_rentabilidad(
        forecast, ticket, cogs_pct, personal_mensual, fijos_mensuales,
        dias_apertura, usar_facturacion_ml=(modo_facturacion == "Predicción directa ML"),
        repartir_coste_mensual=es_anual,
    )
except Exception as error:
    st.error("No se ha podido calcular la previsión. Revisa los modelos y parámetros.")
    st.exception(error)
    st.stop()

fecha_inicio = pd.Timestamp(escenario["date"].min()).strftime("%d/%m/%Y")
fecha_fin = pd.Timestamp(escenario["date"].max()).strftime("%d/%m/%Y")
total_ventas = float(escenario["facturacion_prevista"].sum())
total_pedidos = float(escenario["pedidos_previstos"].sum())
total_beneficio = float(escenario["beneficio_operativo"].sum())
margen = total_beneficio / total_ventas * 100 if total_ventas > 0 else 0.0


# -----------------------------------------------------------------------------
# 4. PÁGINAS INTERACTIVAS
# -----------------------------------------------------------------------------
if pagina == PAGINAS[0]:
    hero(
        "Una visión completa de tu restaurante",
        "Analiza el histórico, consulta pronósticos y simula la rentabilidad desde un solo lugar.",
    )
    aviso(
        f"<b>Horizonte seleccionado:</b> {fecha_inicio} – {fecha_fin} ({dias} días). "
        "Es una proyección del modelo, no una facturación real ni una garantía de resultados."
    )

    seccion("Resumen del escenario", "Métricas del período que acabas de seleccionar.")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Pedidos previstos", entero(total_pedidos))
    k2.metric("Ingresos previstos", euro(total_ventas))
    k3.metric("Beneficio operativo", euro(total_beneficio))
    k4.metric("Margen operativo", porcentaje(margen))

    if es_anual:
        seccion(f"Resumen mensual de {anio_prediccion}")
        mensual = (escenario.assign(Mes=escenario["date"].dt.strftime("%Y-%m"))
                   .groupby("Mes", as_index=False).agg(
                       Pedidos=("pedidos_previstos", "sum"),
                       Facturacion=("facturacion_prevista", "sum"),
                       Costes_personal=("costes_personal", "sum"),
                       Beneficio=("beneficio_operativo", "sum")))
        st.dataframe(mensual.round(2), hide_index=True, use_container_width=True)
        tabla_csv(mensual, f"prevision_mensual_{anio_prediccion}.csv")

    a, b = st.columns([1.6, 1])
    with a:
        if es_anual:
            seccion("Pedidos previstos por mes", f"Proyección estacional de {anio_prediccion}.")
            mensual = (escenario.assign(date=escenario["date"].dt.to_period("M").dt.to_timestamp())
                      .groupby("date", as_index=False)["pedidos_previstos"].sum())
            linea(mensual, "date", ["pedidos_previstos"],
                  nombres={"pedidos_previstos": "Pedidos previstos"},
                  titulo_y="Pedidos / mes", puntos=True)
        else:
            seccion("Evolución de pedidos", "Pronóstico diario y últimos 21 días reales.")
            previo = diario.tail(21)[["date", "pedidos_diarios"]].rename(
                columns={"pedidos_diarios": "Histórico"})
            futuro = escenario[["date", "pedidos_previstos"]].rename(
                columns={"pedidos_previstos": "Predicción"})
            combinado = pd.concat([previo, futuro], ignore_index=True)
            linea(combinado, "date", ["Histórico", "Predicción"],
                  titulo_y="Pedidos / día", puntos=True)
    with b:
        seccion("Ingresos y costes", "Distribución del importe acumulado previsto.")
    
        composicion = pd.DataFrame({
            "Concepto": ["Costes variables", "Personal", "Costes fijos", "Beneficio"],
            "Importe": [
                escenario["costes_variables"].sum(),
                escenario["costes_personal"].sum(),
                escenario["costes_fijos"].sum(),
                total_beneficio,
            ],
        })
    
        if total_beneficio >= 0 and total_ventas > 0:
            # Colores similares a los de la captura actual.
            colores = {
                "Costes variables": "#CDB377",  # dorado
                "Personal": "#735949",          # marrón
                "Costes fijos": "#E6D8B8",      # crema
                "Beneficio": "#A9342A",         # rojo
            }
    
            composicion["Porcentaje"] = (
                100 * composicion["Importe"] / composicion["Importe"].sum()
            )
    
            grafico = (
                alt.Chart(composicion)
                .mark_arc(
                    innerRadius=88,
                    outerRadius=128,
                    stroke="#FBF8EF",
                    strokeWidth=3,
                )
                .encode(
                    theta=alt.Theta("Importe:Q"),
                    color=alt.Color(
                        "Concepto:N",
                        scale=alt.Scale(
                            domain=list(colores.keys()),
                            range=list(colores.values()),
                        ),
                        legend=None,  # Leyenda HTML independiente: no se recorta.
                    ),
                    tooltip=[
                        alt.Tooltip("Concepto:N", title="Concepto"),
                        alt.Tooltip("Importe:Q", title="Importe (€)", format=",.2f"),
                        alt.Tooltip("Porcentaje:Q", title="Porcentaje (%)", format=".1f"),
                    ],
                )
                .properties(height=300, background="#FBF8EF")
                .configure_view(stroke=None)
            )
    
            st.altair_chart(grafico, use_container_width=True)
    
            # Leyenda en una cuadrícula 2 x 2 para columnas estrechas.
            items = []
            for _, fila in composicion.iterrows():
                concepto = fila["Concepto"]
                porcentaje = fila["Porcentaje"]
                color = colores[concepto]
                items.append(
                    f'<div style="display:flex; align-items:center; gap:8px; min-width:0;">'
                    f'<span style="display:inline-block; flex-shrink:0; '
                    f'width:11px; height:11px; border-radius:50%; background:{color};"></span>'
                    f'<span style="font-size:13px; color:#554A42; line-height:1.4;">'
                    f'{concepto} <strong>{porcentaje:.1f}%</strong></span>'
                    f'</div>'
                )
    
            st.markdown(
                '<div style="display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); '
                'gap:14px 10px; padding:8px 8px 12px;">'
                + ''.join(items) +
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info(
                "El beneficio es negativo o los ingresos son cero; "
                "consulta el desglose en Rentabilidad."
            )
            st.dataframe(composicion, hide_index=True, use_container_width=True)

    seccion("Indicadores de referencia")
    k5, k6, k7 = st.columns(3)
    k5.metric("Pedidos históricos", entero(pedidos["order_id"].nunique()))
    k6.metric("Ventas históricas", euro(pedidos["revenue"].sum()))
    k7.metric("Ticket medio histórico", euro(pedidos["revenue"].mean()))
    st.caption("Ticket medio histórico = facturación total / número de pedidos; el período histórico llega hasta 2024.")

elif pagina == PAGINAS[1]:
    hero(
        "Explora lo que ocurrió realmente",
        "Filtra los años, compara la evolución de ventas y descubre los patrones de consumo.",
        etiqueta="EXPLORACIÓN · ANÁLISIS HISTÓRICO",
    )
    años = sorted(int(x) for x in diario["date"].dt.year.unique())
    minimo, maximo = min(años), max(años)
    rango = st.slider("Intervalo histórico", min_value=minimo, max_value=maximo,
                      value=(max(minimo, maximo - 4), maximo))
    d = diario[diario["date"].dt.year.between(*rango)].copy()
    p = pedidos[pedidos["date"].dt.year.between(*rango)].copy()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Pedidos", entero(p["order_id"].nunique()))
    k2.metric("Facturación", euro(p["revenue"].sum()))
    k3.metric("Ticket medio", euro(p["revenue"].mean()))
    k4.metric("Días registrados", entero(len(d)))

    anuales = d.assign(Año=d["date"].dt.year).groupby("Año", as_index=False).agg(
        Pedidos=("pedidos_diarios", "sum"),
        Facturacion=("facturacion_diaria", "sum"),
        Beneficio=("operating_profit", "sum"),
    )
    a, b = st.columns(2)
    with a:
        seccion("Pedidos por año")
        barras(anuales, "Año", "Pedidos")
    with b:
        seccion("Facturación por año")
        barras(anuales, "Año", "Facturacion", color=DORADO)

    seccion("Actividad diaria", "Pasa el cursor para consultar valores y utiliza el zoom del gráfico.")
    detalle = d[["date", "pedidos_diarios"]].rename(columns={"pedidos_diarios": "Pedidos"})
    detalle["Media móvil (30 días)"] = detalle["Pedidos"].rolling(
        30, min_periods=1).mean()
    linea(detalle, "date", ["Pedidos", "Media móvil (30 días)"], titulo_y="Pedidos")

    a, b = st.columns(2)
    with a:
        seccion("Demanda por día de la semana")
        medias = d.assign(dia=d["date"].dt.dayofweek).groupby("dia", as_index=False).agg(
            Pedidos=("pedidos_diarios", "mean"))
        medias["Día"] = medias["dia"].map(dict(enumerate(
            ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"])))
        barras(medias, "Día", "Pedidos", orden=["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"])
    with b:
        seccion("Canales de compra")
        canal_counts = p.groupby("channel", as_index=False)["order_id"].nunique().rename(
            columns={"channel": "Canal", "order_id": "Pedidos"})
        barras(canal_counts, "Canal", "Pedidos", color=DORADO)

    seccion("Inspeccionar operaciones")
    opciones = ["Todos los canales"] + sorted(p["channel"].dropna().astype(str).unique().tolist())
    canal = st.selectbox("Filtrar tabla por canal", opciones)
    vista = p if canal == "Todos los canales" else p[p["channel"].astype(str) == canal]
    st.dataframe(vista.tail(150), hide_index=True, use_container_width=True)
    tabla_csv(vista, "historico_sushi_filtrado.csv", "Exportar histórico filtrado")
    st.caption("Los CSV no incluyen la hora del pedido, por lo que no se calcula la demanda por franjas horarias.")

elif pagina == PAGINAS[2]:
    hero(
        "Predicción de la demanda",
        "Consulta los pedidos previstos día a día y analiza cómo evoluciona la demanda futura.",
        etiqueta="INTELIGENCIA PREDICTIVA · DEMANDA",
    )
    if es_anual:
        aviso(f"Proyección del año {anio_prediccion} con modelos ML entrenados sobre calendario. "
              "Se validaron con el último año real (2024); no asumen conocer ventas de 2025 o 2026. "
              "La incertidumbre crece cuanto más lejano es el año de predicción.")
    elif enfoque_elegido == "anual":
        aviso("Se consulta cada fecha del mismo día del año anterior. "
              "Este enfoque no utiliza predicciones recursivas de días anteriores.")
    else:
        aviso("La previsión parte del día siguiente al último dato real. A partir del segundo día, "
              "el modelo utiliza también sus propias predicciones anteriores.")
    k1, k2, k3 = st.columns(3)
    k1.metric("Pedidos previstos", entero(total_pedidos))
    k2.metric("Media de pedidos al día", f"{escenario['pedidos_previstos'].mean():.1f}")
    k3.metric("Días previstos", dias)
    st.caption(f"Enfoque: {etiqueta_enfoque} · Modelo: {modelo_d} · Periodo: {fecha_inicio}–{fecha_fin}")

    seccion("Evolución de la demanda prevista")
    datos_grafico_demanda = (escenario.assign(
        date=escenario["date"].dt.to_period("M").dt.to_timestamp()
    ).groupby("date", as_index=False)["pedidos_previstos"].sum() if es_anual else escenario)
    linea(datos_grafico_demanda, "date", ["pedidos_previstos"],
          nombres={"pedidos_previstos": "Pedidos previstos"},
          titulo_y="Pedidos / día", puntos=True)
    seccion("Detalle por fecha", "Selecciona un día y descarga los resultados si lo necesitas.")
    dias_disponibles = escenario["date"].dt.strftime("%d/%m/%Y").tolist()
    fecha_elegida = st.selectbox("Inspeccionar día previsto", options=dias_disponibles)
    fila = escenario.loc[escenario["date"].dt.strftime("%d/%m/%Y") == fecha_elegida].iloc[0]
    st.metric(f"Pedidos previstos para {fecha_elegida}", entero(fila["pedidos_previstos"]))
    st.dataframe(escenario[["date", "pedidos_previstos"]].round(1),
                 use_container_width=True, hide_index=True)
    tabla_csv(escenario[["date", "pedidos_previstos"]], "prevision_demanda.csv")
    if es_anual:
        st.caption("Objetivo (y): pedidos diarios. Variables (X) del horizonte anual: "
                   "día de semana, mes, fin de semana, día de mes y estacionalidad. No hay lags futuros.")
    else:
        st.caption("Objetivo (y): pedidos diarios. Variables (X): calendario y, según el enfoque, "
                   "datos recientes, mismo día del año anterior o ambos.")

elif pagina == PAGINAS[3]:
    hero(
        "Cuánto podrías facturar",
        "Compara dos enfoques para estimar ingresos sin modificar tus modelos entrenados.",
        etiqueta="INTELIGENCIA PREDICTIVA · INGRESOS",
    )
    k1, k2, k3 = st.columns(3)
    total_ticket = float(escenario["facturacion_por_ticket"].sum())
    total_ml = float(escenario["facturacion_ml"].sum())
    k1.metric("Pedidos × ticket medio", euro(total_ticket))
    k2.metric("Predicción directa ML", euro(total_ml))
    k3.metric("Diferencia entre métodos", euro(total_ml - total_ticket))

    aviso(f"<b>Método para la rentabilidad:</b> {modo_facturacion}. "
          "Puedes cambiarlo en «Editar escenario económico», en el menú lateral.")
    seccion("Comparación mensual" if es_anual else "Comparación diaria")
    datos_grafico_ventas = (escenario.assign(
        date=escenario["date"].dt.to_period("M").dt.to_timestamp()
    ).groupby("date", as_index=False)[["facturacion_por_ticket", "facturacion_ml"]].sum()
        if es_anual else escenario)
    linea(datos_grafico_ventas, "date", ["facturacion_por_ticket", "facturacion_ml"],
          nombres={"facturacion_por_ticket": "Pedidos × ticket", "facturacion_ml": "ML directo"},
          titulo_y="Ingresos (€)", puntos=True)

    seccion("Previsión con el método elegido")
    tabla = escenario[["date", "pedidos_previstos", "facturacion_por_ticket",
                      "facturacion_ml", "facturacion_prevista"]].round(2)
    st.dataframe(tabla, hide_index=True, use_container_width=True)
    tabla_csv(tabla, "prevision_facturacion.csv")
    st.latex(r"\text{Facturación por ticket} = \text{Pedidos previstos} \times \text{Ticket medio}")
    st.caption("El método ML directo utiliza patrones de importes del restaurante de sushi; "
               "no adapta automáticamente sus precios a otro negocio.")

elif pagina == PAGINAS[4]:
    hero(
        "Simula la rentabilidad",
        "Ajusta los costes, compara escenarios y comprueba cuántos pedidos necesitas para cubrirlos.",
        etiqueta="GESTIÓN FINANCIERA · RENTABILIDAD",
    )
    aviso("Este resultado <b>no procede de un tercer modelo de ML</b>: "
          "el beneficio se calcula a partir de los ingresos previstos y los costes que configures.")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Ingresos previstos", euro(total_ventas))
    k2.metric("Costes variables", euro(escenario["costes_variables"].sum()))
    k3.metric("Personal + fijos", euro((escenario["costes_personal"] + escenario["costes_fijos"]).sum()))
    k4.metric("Beneficio operativo", euro(total_beneficio))

    a, b = st.columns([1.35, 1])
    with a:
        seccion("Beneficio mensual estimado" if es_anual else "Beneficio diario estimado")
        datos_grafico_beneficio = (escenario.assign(
            date=escenario["date"].dt.to_period("M").dt.to_timestamp()
        ).groupby("date", as_index=False)["beneficio_operativo"].sum()
            if es_anual else escenario)
        linea(datos_grafico_beneficio, "date", ["beneficio_operativo"],
              nombres={"beneficio_operativo": "Beneficio (€)"},
              titulo_y="Beneficio operativo (€)", puntos=True)
    with b:
        seccion("Punto de equilibrio")
        equilibrio = punto_equilibrio(ticket, cogs_pct, personal_mensual,
                                     fijos_mensuales, dias_apertura)
        if equilibrio is None:
            st.error("El margen de contribución por pedido es cero o negativo.")
        else:
            st.metric("Pedidos necesarios al día", entero(equilibrio))
            st.write("Es el número aproximado de pedidos diarios para cubrir "
                     "costes, calculado con el **ticket medio editable**.")
        st.metric("Margen operativo previsto", porcentaje(margen))

    seccion("Comparador de escenarios", "Prueba qué ocurriría si cambias el porcentaje de costes variables.")
    c1, c2 = st.columns([1.25, 1])
    with c1:
        cogs_alterno = st.slider("Costes variables en escenario alternativo (%)", 0.0, 100.0,
                                 value=float(cogs_pct), step=0.5)
        alternativa = calcular_rentabilidad(
            forecast, ticket, cogs_alterno, personal_mensual, fijos_mensuales,
            dias_apertura, usar_facturacion_ml=(modo_facturacion == "Predicción directa ML"),
            repartir_coste_mensual=es_anual,
        )
    with c2:
        beneficio_alterno = float(alternativa["beneficio_operativo"].sum())
        st.metric("Beneficio con nuevos costes", euro(beneficio_alterno),
                  delta=euro(beneficio_alterno - total_beneficio))
    compara = pd.DataFrame({
        "Escenario": ["Actual", "Alternativo"],
        "Beneficio": [total_beneficio, beneficio_alterno],
    })
    barras(compara, "Escenario", "Beneficio", alto=190)

    seccion("Desglose del resultado")
    columnas = ["date", "pedidos_previstos", "facturacion_prevista", "costes_variables",
                "costes_personal", "costes_fijos", "beneficio_operativo", "margen_operativo_pct"]
    st.dataframe(escenario[columnas].round(2), use_container_width=True, hide_index=True)
    tabla_csv(escenario, "predicciones_y_rentabilidad.csv", "Descargar previsión y rentabilidad")
    st.latex(r"\text{Beneficio operativo} = \text{Facturación} - \text{Costes variables} - \text{Personal} - \text{Costes fijos}")

    with st.expander("Ver beneficio real del histórico"):
        historia = costes.assign(Año=costes["date"].dt.year).groupby("Año", as_index=False).agg(
            Beneficio=("operating_profit", "sum"))
        barras(historia, "Año", "Beneficio", color=DORADO)

elif pagina == PAGINAS[5]:
    hero(
        "Explora los modelos de Machine Learning",
        "Compara el rendimiento del conjunto de prueba y examina valores reales frente a predichos.",
        etiqueta="LABORATORIO DE MODELOS · EVALUACIÓN",
    )
    aviso("La evaluación corresponde al último 20 % del histórico y a predicciones de un día vista. "
          "No mide el error acumulado de pronósticos recursivos a 30 días.")
    tipo = st.segmented_control(
        "¿Qué quieres evaluar?", options=["Demanda", "Facturación"], default="Demanda"
    )
    comp = comparacion_d if tipo == "Demanda" else comparacion_f
    clave = "demanda" if tipo == "Demanda" else "facturacion"
    metrica = st.radio("Ordenar por", ["MAE", "RMSE", "R2"], horizontal=True)
    if metrica not in comp.columns:
        st.warning(f"La columna {metrica} no aparece en el archivo de comparación.")
    else:
        seccion(f"Resultados de {tipo.lower()}", "MAE y RMSE más bajos son mejores; R² más alto suele ser mejor.")
        col1, col2 = st.columns([1.1, 1])
        with col1:
            st.dataframe(comp.round(3), hide_index=True, use_container_width=True)
        with col2:
            ordenado = comp.sort_values(metrica, ascending=(metrica != "R2"))
            diagrama = alt.Chart(ordenado).mark_bar(color=ROJO, cornerRadiusEnd=4).encode(
                y=alt.Y("Modelo:N", sort=ordenado["Modelo"].tolist(), title=None),
                x=alt.X(f"{metrica}:Q", title=metrica),
                tooltip=["Modelo:N", alt.Tooltip(f"{metrica}:Q", format=",.3f")],
            ).properties(height=330)
            st.altair_chart(estilo_grafico(diagrama), use_container_width=True)

    st.divider()
    seccion("Reales frente a predichos", "Muestra de los primeros 60 días del conjunto de prueba.")
    pred_path = MODELOS / f"predicciones_test_{clave}.csv"
    if pred_path.exists():
        pred_test = pd.read_csv(pred_path, parse_dates=["date"])
        modelos_disponibles = [m for m in comp["Modelo"].tolist() if m in pred_test.columns]
        if modelos_disponibles and "real" in pred_test.columns:
            elegido = st.selectbox("Modelo que quieres inspeccionar", modelos_disponibles)
            pred = pred_test.head(60)[["date", "real", elegido]].rename(
                columns={"real": "Valor real", elegido: "Predicho"})
            linea(pred, "date", ["Valor real", "Predicho"], titulo_y="Pedidos" if clave == "demanda" else "Euros")
        else:
            st.warning("El archivo de prueba no contiene las columnas necesarias para la comparación.")
    else:
        st.warning("No se ha encontrado el CSV de predicciones del conjunto de prueba.")

    if clave == "facturacion":
        formula_path = MODELOS / "evaluacion_formula_ticket.csv"
        if formula_path.exists():
            seccion("Método adicional: pedidos × ticket")
            st.dataframe(pd.read_csv(formula_path).round(2), use_container_width=True, hide_index=True)
    # Resultados del experimento anual, independientes de la evaluación original.
    comparacion_enfoques_path = carpeta_enfoques / "comparacion_enfoques.csv"
    comparacion_horizonte_path = carpeta_enfoques / "comparacion_horizonte.csv"
    if comparacion_enfoques_path.exists():
        st.divider()
        seccion("Comparación de enfoques temporales",
                "A (reciente), B (mismo día del año anterior) y C (combinado). "
                "Los tres se evalúan sobre las mismas fechas.")
        enfoque_tipo = st.radio(
            "Comparar variable", ["demanda", "facturacion"], horizontal=True,
            key="enfoque_tipo"
        )
        resultados = pd.read_csv(comparacion_enfoques_path)
        vista_resultados = resultados[resultados["Objetivo"] == enfoque_tipo]
        st.markdown("**Evaluación a un día vista**")
        st.dataframe(vista_resultados[["Enfoque", "Modelo", "MAE", "RMSE", "R2"]].round(3),
                     use_container_width=True, hide_index=True)
        if comparacion_horizonte_path.exists():
            st.markdown("**Evaluación de predicciones de varios días (backtest por bloques)**")
            bloques = pd.read_csv(comparacion_horizonte_path)
            sub_bloques = bloques[bloques["Objetivo"] == enfoque_tipo]
            st.dataframe(sub_bloques[["Enfoque", "Modelo", "Horizonte_max_dias",
                                     "MAE", "RMSE", "R2"]].round(3),
                         use_container_width=True, hide_index=True)
            st.caption("En cada bloque, los modelos solo utilizan datos reales anteriores "
                       "al inicio del bloque. Un menor MAE/RMSE indica menos error.")
    st.divider()
    seccion("Modelos de proyección anual (2027 por defecto)",
            "Validados con el último año disponible (2024), entrenados en 2015–2023; "
            "para predecir se ajusta el ganador con 2015–2024.")
    modelos_cal, resultados_cal = modelos_largo_plazo()
    st.dataframe(resultados_cal.round(3), hide_index=True, use_container_width=True)
    tabla_csv(resultados_cal, "comparacion_calendario_largo_plazo.csv", "Descargar evaluación anual")
    st.caption("Los enfoques anteriores siguen disponibles para corto plazo. "
               "El modelo anual se entrena con calendario, sin ventas reales de 2025/2026.")

st.markdown(
    '<div class="footer-note">Sushi Arigato · Aplicación educativa en Streamlit · '
    'Datos históricos 2015–2024 · Pronóstico anual estacional o de corto plazo.</div>',
    unsafe_allow_html=True,
)
