"""
Dashboard: TDA en la Salud Materno-Infantil
Impacto del Ácido Fólico en Mujeres Chilenas
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components

from data_loader import (
    load_data, load_mapper_clusters, get_mapper_html,
    load_stats, get_images,
)

# ── Configuración global ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="TDA Salud Materno-Infantil",
    page_icon="🤰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-title {
        font-size: 2rem; font-weight: 700; color: #5B4A8A;
        border-bottom: 3px solid #C8A2C8; padding-bottom: 0.4rem;
        margin-bottom: 0.8rem;
    }
    .subtitle { font-size: 1rem; color: #666; margin-top: -0.5rem; }
    .metric-card {
        background: #F8F4FF; border-radius: 12px; padding: 1rem 1.5rem;
        border-left: 5px solid #9B7EBD; margin-bottom: 0.5rem;
    }
    .cluster-card {
        background: white; border-radius: 10px; padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 1rem;
    }
    .riesgo-tag {
        background: #FDECEA; color: #C0392B; border-radius: 6px;
        padding: 2px 8px; font-size: 0.82rem; font-weight: 600;
    }
    .protector-tag {
        background: #EAF6EA; color: #1E8449; border-radius: 6px;
        padding: 2px 8px; font-size: 0.82rem; font-weight: 600;
    }
    .faq-question { font-weight: 600; color: #5B4A8A; }
    .kpi-value { font-size: 2.2rem; font-weight: 800; color: #5B4A8A; }
    .kpi-label { font-size: 0.9rem; color: #888; margin-top: 0.1rem; }
    .info-box {
        background: #EDE7F6; border-radius: 10px; padding: 1rem 1.4rem;
        border-left: 4px solid #7E57C2; margin: 0.8rem 0; font-size: 0.92rem;
    }
    .interp-box {
        background: #F0FFF4; border-radius: 10px; padding: 0.85rem 1.3rem;
        border-left: 4px solid #43A047; margin: 0.4rem 0 0.9rem 0; font-size: 0.91rem;
        color: #1B5E20;
    }
    .interp-box b { color: #1B5E20; }
    .warn-box {
        background: #FFF8E1; border-radius: 10px; padding: 1rem 1.4rem;
        border-left: 4px solid #FFC107; margin: 0.8rem 0; font-size: 0.92rem;
    }
    div[data-testid="stExpander"] { border: 1px solid #E8DEF8; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ── Paleta clínica ─────────────────────────────────────────────────────────────
P = {
    "primary":   "#7E57C2",
    "secondary": "#E57373",
    "accent":    "#FFB74D",
    "green":     "#66BB6A",
    "light":     "#B39DDB",
    "gray":      "#90A4AE",
    "purple":    "#5B4A8A",
    "bg":        "#F8F4FF",
}

# ── Glosario clínico ───────────────────────────────────────────────────────────
GLOSARIO = {
    "Ácido Fólico (AF)": {
        "icono": "•",
        "definicion": (
            "Vitamina B9 esencial para la síntesis de ADN, la división celular "
            "y el desarrollo del tubo neural fetal."
        ),
        "relevancia": (
            "Su deficiencia periconcepcional se asocia con defectos del tubo neural "
            "(espina bífida, anencefalia), parto prematuro y bajo peso al nacer."
        ),
        "parametros": (
            "OMS recomienda 400 µg/día antes y durante el primer trimestre. "
            "En Chile la harina de trigo está fortificada con 220 µg/100 g (DS 977/96)."
        ),
    },
    "TDA — Análisis Topológico de Datos": {
        "icono": "•",
        "definicion": (
            "Rama de las matemáticas aplicadas que estudia la 'forma' de los datos "
            "usando herramientas topológicas como la homología persistente."
        ),
        "relevancia": (
            "Detecta subpoblaciones ocultas y relaciones no lineales que los métodos "
            "estadísticos clásicos no capturan."
        ),
        "parametros": (
            "En este proyecto se usó el algoritmo Mapper (grafo topológico) "
            "y homología persistente H0/H1 con el complejo de Vietoris-Rips."
        ),
    },
    "Algoritmo Mapper": {
        "icono": "•",
        "definicion": (
            "Método TDA que resume la topología de datos multidimensionales en un grafo, "
            "donde cada nodo es un clúster de observaciones similares."
        ),
        "relevancia": (
            "Identifica subpoblaciones con patrones similares de ingesta de AF "
            "y resultados neonatales sin asumir distribuciones paramétricas."
        ),
        "parametros": (
            "Lente: PCA 2D · Cubierta: n_cubes=15, overlap=50% · "
            "Clustering local: AgglomerativeClustering(k=4, ward) · "
            "Colorización: AF_exposure."
        ),
    },
    "Homología Persistente": {
        "icono": "•",
        "definicion": (
            "Técnica TDA que mide características topológicas (componentes conexas H0, "
            "ciclos H1) a diferentes escalas de resolución."
        ),
        "relevancia": (
            "Permite comparar formalmente la estructura del espacio de ingesta "
            "entre grupos de suplementación mediante distancias de Wasserstein."
        ),
        "parametros": (
            "Complejo Vietoris-Rips sobre 8 variables normalizadas con RobustScaler. "
            "H0 = componentes, H1 = ciclos."
        ),
    },
    "Parto Prematuro": {
        "icono": "•",
        "definicion": "Nacimiento antes de las 37 semanas completas de gestación.",
        "relevancia": (
            "Principal causa de mortalidad neonatal. "
            "La suplementación adecuada de AF reduce su incidencia."
        ),
        "parametros": (
            "Umbral: EG < 37 sem. Rangos: <28 sem (extremo), 28–31 (muy prematuro), "
            "32–36 (tardío), ≥37 (término)."
        ),
    },
    "Bajo Peso al Nacer": {
        "icono": "•",
        "definicion": "Peso al nacer inferior a 2 500 g, independiente de la EG.",
        "relevancia": (
            "Marcador de restricción del crecimiento intrauterino. "
            "Asociado a ingesta deficiente de micronutrientes maternos."
        ),
        "parametros": "Umbral clínico: < 2 500 g. Rango normal: 2 500–4 000 g.",
    },
    "Resultado Adverso Perinatal": {
        "icono": "•",
        "definicion": (
            "Variable compuesta: parto prematuro (EG < 37 sem) "
            "y/o bajo peso al nacer (< 2 500 g)."
        ),
        "relevancia": (
            "Indicador global de salud neonatal usado para comparar subpoblaciones "
            "en el análisis TDA."
        ),
        "parametros": (
            "resultado_adverso = 1 si prematuro = 1 OR bajo_peso_rn = 1."
        ),
    },
    "AF_exposure (Score Total)": {
        "icono": "•",
        "definicion": (
            "Variable compuesta: uf_af (suplemento directo, µg/día) "
            "+ score_pan (N° panes × 160 µg)."
        ),
        "relevancia": (
            "Estima la exposición total a AF integrando la fuente farmacológica "
            "y la dietética (pan de trigo fortificado)."
        ),
        "parametros": (
            "score_pan = N° panes × 160 µg (DS 977/96). "
            "Dosis adecuada: AF_exposure ≥ 1 000 µg/día."
        ),
    },
    "Distancia de Wasserstein": {
        "icono": "•",
        "definicion": (
            "Métrica que cuantifica la distancia entre dos diagramas de persistencia, "
            "midiendo el esfuerzo mínimo para transformar uno en el otro."
        ),
        "relevancia": (
            "Compara formalmente la topología del espacio de ingesta entre subgrupos "
            "de suplementación."
        ),
        "parametros": (
            "Mayor distancia = topologías más distintas. "
            "Se calcula para H0 (componentes) y H1 (ciclos)."
        ),
    },
    "Edad Gestacional (EG)": {
        "icono": "•",
        "definicion": (
            "Duración del embarazo en semanas completas desde el primer día "
            "de la última menstruación."
        ),
        "relevancia": "Determinante principal de la madurez neonatal.",
        "parametros": "Umbral prematuridad: < 37 sem. Rango dataset: 32–42 sem.",
    },
}

# ── Preguntas frecuentes ────────────────────────────────────────────────────────
FAQS = [
    {
        "pregunta": "¿Por qué TDA y no regresión logística simple?",
        "respuesta": (
            "La regresión logística asume relaciones lineales y no captura la estructura "
            "interna del espacio de datos. TDA (Mapper) identifica subpoblaciones con "
            "patrones de riesgo heterogéneos sin asumir ninguna distribución. La homología "
            "persistente cuantificó diferencias topológicas entre grupos de suplementación "
            "que la estadística clásica no detectó."
        ),
        "categoria": "Metodología",
    },
    {
        "pregunta": "¿Qué diferencia hay entre los clusters Mapper y los clusters K-Means topológicos?",
        "respuesta": (
            "Los **clusters Mapper** son componentes conexas del grafo topológico: agrupan mujeres "
            "que comparten observaciones en el espacio PCA-AF y reflejan la geometría global.\n\n"
            "Los **clusters K-Means topológicos** se determinan a partir de H0: se cuentan "
            "las componentes con persistencia > percentil 90 y ese K se usa en K-Means sobre "
            "el espacio escalado. Son más sensibles a la densidad local."
        ),
        "categoria": "Metodología",
    },
    {
        "pregunta": "¿La suplementación con AF reduce el riesgo de parto prematuro en este dataset?",
        "respuesta": (
            "Los tests de Kruskal-Wallis y chi-cuadrado muestran diferencias significativas "
            "en EG y resultados adversos entre clusters. Los clusters con mayor AF_exposure "
            "presentan menores tasas de prematuridad. Sin embargo, el diseño es observacional "
            "— no se puede establecer causalidad directa. La regresión OLS ajustada (Tab 4) "
            "muestra una asociación positiva entre AF_exposure y semanas de gestación."
        ),
        "categoria": "Hallazgos clínicos",
    },
    {
        "pregunta": "¿Por qué aparece el pan como variable relevante en el análisis?",
        "respuesta": (
            "El DS 977/96 de Chile obliga a fortificar la harina de trigo con 220 µg de AF "
            "por 100 g. Un pan (~100 g) aporta ~160–220 µg de AF. Para mujeres sin acceso "
            "a suplementos farmacológicos, el pan es la principal fuente de AF. "
            "El score_pan (N° panes × 160 µg) se usó como proxy de exposición dietética."
        ),
        "categoria": "Contexto clínico-nutricional",
    },
    {
        "pregunta": "¿Qué significa una mayor norma L2 del paisaje de persistencia H1?",
        "respuesta": (
            "La norma L2 del paisaje de persistencia H1 mide la complejidad topológica: "
            "cuántos ciclos no triviales existen en el espacio de ingesta de ese subgrupo. "
            "Una norma mayor indica mayor heterogeneidad — las mujeres siguen trayectorias "
            "más diversas de ingesta de AF. El grupo 'Dosis adecuada (≥1000 µg/d)' mostró "
            "la mayor norma L2, sugiriendo mayor variabilidad en las fuentes de AF."
        ),
        "categoria": "Interpretación TDA",
    },
    {
        "pregunta": "¿Qué recomendaciones clínicas emergen del análisis?",
        "respuesta": (
            "1. **Identificación temprana**: priorizar mujeres del cluster de bajo AF y alta tasa "
            "de resultados adversos para intervención nutricional periconcepcional.\n\n"
            "2. **Rol del pan**: 3+ panes/día puede compensar parcialmente la deficiencia "
            "en poblaciones con baja adherencia a suplementos.\n\n"
            "3. **Timing de suplementación**: iniciar antes del embarazo mostró el mayor "
            "impacto en los clusters protectores.\n\n"
            "4. **Personalización**: la heterogeneidad topológica sugiere que intervenciones "
            "uniformes son subóptimas — se requiere segmentación por perfil de ingesta."
        ),
        "categoria": "Recomendaciones clínicas",
    },
    {
        "pregunta": "¿El modelo Random Forest combina bien variables clínicas con clusters TDA?",
        "respuesta": (
            "Sí. La variante 'Combinado' (ingesta clínica + membresía al cluster TDA) obtuvo "
            "el mayor AUC-ROC en validación cruzada 5-fold, superando a cada conjunto de "
            "variables por separado. La importancia Gini muestra que los clusters topológicos "
            "aportan señal predictiva independiente y complementaria a la dosis de AF, "
            "validando el valor agregado del enfoque TDA."
        ),
        "categoria": "Resultados ML",
    },
    {
        "pregunta": "¿Cómo interpretar la distancia de Wasserstein H1 entre subgrupos?",
        "respuesta": (
            "Mide el 'costo topológico' de transformar el diagrama H1 de un subgrupo en el de "
            "otro. Si la distancia entre 'Sin suplemento' y 'Dosis adecuada' es mayor que entre "
            "'Sin suplemento' y 'Dosis baja', la suplementación adecuada genera un espacio de "
            "ingesta topológicamente más distinto del grupo no suplementado — cambia no solo "
            "el nivel sino la estructura completa del patrón de ingesta."
        ),
        "categoria": "Interpretación TDA",
    },
]


# ── Carga de datos ─────────────────────────────────────────────────────────────
@st.cache_data
def get_data() -> pd.DataFrame:
    df       = load_data()
    clusters = load_mapper_clusters()
    df = df.merge(clusters, left_index=True, right_on="orig_index", how="left")
    return df


@st.cache_data
def get_stats() -> dict:
    return load_stats()


df    = get_data()
STATS = get_stats()


# ── Helpers de columnas ────────────────────────────────────────────────────────
def _col(*candidates: str, src: pd.DataFrame | None = None) -> "str | None":
    frame = src if src is not None else df
    for c in candidates:
        if c in frame.columns:
            return c
    return None


AF_COL    = _col("uf_af", "uf-af")
PN_COL    = _col("pnacer_num", "pnacer")
EG_COL    = _col("eg_num", "eg_raw")
PAN_COL   = _col("n_panes_num", "N° PANES")
PREM_COL  = _col("prematuro")
BP_COL    = _col("bajo_peso_rn", "bajo_peso")
ADV_COL   = _col("resultado_adverso")
SUPL_COL  = _col("cat_supl")
CLMAP_COL = _col("cluster_mapper")
EDAD_COL  = _col("edad")
NEDUC_COL = _col("neduc")
AF_EXP    = _col("AF_exposure")
REGION_COL = _col("region", "region_chile")

TIMING_COLS = [c for c in [
    "Antes del embarazo", "Durante todo el embarazo",
    "1-3 meses", "4-6 meses", "7-9 meses",
] if c in df.columns]


# ── Sidebar: filtros globales ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Filtros")

    if REGION_COL:
        regiones = ["Todas"] + sorted(df[REGION_COL].dropna().unique().tolist())
        region_sel = st.selectbox("Región de Chile", regiones)
    else:
        region_sel = "Todas"

    if EDAD_COL:
        edad_min, edad_max = int(df[EDAD_COL].min()), int(df[EDAD_COL].max())
        edad_rango = st.slider("Edad materna (años)", edad_min, edad_max, (edad_min, edad_max))
    else:
        edad_rango = (0, 99)

    if EG_COL:
        eg_vals  = df[EG_COL].dropna()
        eg_rango = st.slider(
            "Semanas de gestación",
            int(eg_vals.min()), int(eg_vals.max()),
            (int(eg_vals.min()), int(eg_vals.max())),
        )
    else:
        eg_rango = (0, 45)

    if SUPL_COL:
        cats = (
            df[SUPL_COL].cat.categories.tolist()
            if hasattr(df[SUPL_COL], "cat")
            else sorted(df[SUPL_COL].dropna().unique().tolist())
        )
        cat_sel = st.selectbox("Nivel de suplementación", ["Todas"] + cats)
    else:
        cat_sel = "Todas"

    st.markdown("---")
    st.markdown(
        "**Proyecto:** TDA en la Salud Materno-Infantil  \n"
        "**Dataset:** 1 170 mujeres embarazadas chilenas  \n"
        "**Clusters Mapper:** 4 componentes válidas (n ≥ 20)"
    )

# ── Aplicar filtros ────────────────────────────────────────────────────────────
df_f = df.copy()
if region_sel != "Todas" and REGION_COL:
    df_f = df_f[df_f[REGION_COL] == region_sel]
if EDAD_COL:
    df_f = df_f[df_f[EDAD_COL].between(*edad_rango)]
if EG_COL:
    df_f = df_f[df_f[EG_COL].between(*eg_rango)]
if cat_sel != "Todas" and SUPL_COL:
    df_f = df_f[df_f[SUPL_COL] == cat_sel]


# ── Navegación ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Inicio",
    "Mapa Topológico",
    "Perfiles de Clusters",
    "Estadística Clínica",
    "Glosario Clínico",
    "Preguntas Frecuentes",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — INICIO
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(
        '<div class="main-title"> TDA en la Salud Materno-Infantil</div>'
        '<div class="subtitle">Impacto del Ácido Fólico en Mujeres Embarazadas Chilenas '
        '· Análisis Topológico de Datos</div>',
        unsafe_allow_html=True,
    )

    # Aviso si hay filtros activos
    n_filtrado = len(df_f)
    if n_filtrado < len(df):
        st.info(f"Mostrando **{n_filtrado}** de **{len(df)}** registros según los filtros activos.")

    st.markdown("")

    # ── KPIs ──────────────────────────────────────────────────────────────────
    n_total  = n_filtrado
    pct_prem = df_f[PREM_COL].mean() * 100 if PREM_COL and df_f[PREM_COL].notna().any() else 0.0
    pct_adv  = df_f[ADV_COL].mean()  * 100 if ADV_COL  and df_f[ADV_COL].notna().any()  else 0.0
    pct_supl = (
        (df_f[SUPL_COL] == "Dosis adecuada (≥1000)").mean() * 100
        if SUPL_COL else 0.0
    )
    af_med   = df_f[AF_EXP].median() if AF_EXP else 0.0

    k1, k2, k3, k4 = st.columns(4)
    for col_w, value, label, threshold, hi_bad in [
        (k1, n_total,  "Mujeres en análisis", None, None),
        (k2, pct_prem, "Partos prematuros",   10,   True),
        (k3, pct_adv,  "Resultados adversos", 15,   True),
        (k4, pct_supl, "Suplementación adecuada", 50, False),
    ]:
        if threshold is None:
            color = P["purple"]
        elif hi_bad:
            color = P["secondary"] if value > threshold else P["green"]
        else:
            color = P["green"] if value >= threshold else P["secondary"]

        suffix = "%" if isinstance(value, float) else ""
        fmt    = f"{value:.1f}{suffix}" if isinstance(value, float) else f"{value:,}"
        with col_w:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="kpi-value" style="color:{color}">{fmt}</div>'
                f'<div class="kpi-label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("")

    # ── Fila principal: descripción + histogramas ──────────────────────────────
    desc_col, charts_col = st.columns([1, 1.8], gap="large")

    with desc_col:
        st.markdown("""
        <div class="info-box">
        <b>¿Qué es este dashboard?</b><br>
        Visualiza los resultados del análisis topológico aplicado a 1 170 mujeres
        embarazadas chilenas para estudiar cómo el ácido fólico impacta los
        resultados perinatales.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("##### Cómo navegar")
        st.markdown("""
        | Pestaña | Contenido |
        |---------|-----------|
        | Mapa Topológico | Grafo Mapper interactivo |
        | Perfiles | Subpoblaciones y riesgo |
        | Estadística | Regresión OLS y boxplots |
        | Glosario | Términos clave |
        | Preguntas Frecuentes | Metodología y hallazgos |
        """)

        if SUPL_COL and df_f[SUPL_COL].notna().any():
            st.markdown("##### Distribución por suplementación")
            supl_counts = df_f[SUPL_COL].value_counts()
            fig_pie = px.pie(
                values=supl_counts.values,
                names=supl_counts.index,
                color_discrete_sequence=[P["secondary"], P["accent"], P["green"]],
                hole=0.45,
            )
            fig_pie.update_traces(textposition="outside", textinfo="percent+label",
                                  textfont_size=11)
            fig_pie.update_layout(
                showlegend=False, height=250,
                margin=dict(t=10, b=10, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_pie, use_container_width=True, key="pie_suplementacion_inicio")
            st.markdown("""
            <div class="interp-box">
            <b>¿Qué nos dice esta gráfica?</b> Muestra cuántas mujeres tomaron
            suplemento de ácido fólico y en qué dosis. Si la mayoría aparece en
            "Sin suplemento" o "Dosis baja", significa que gran parte del grupo
            no cumplió la recomendación mínima de 1 000 µg/día durante el embarazo.
            </div>
            """, unsafe_allow_html=True)

    with charts_col:
        st.markdown("##### Distribución de indicadores clave")
        hist_fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Peso al Nacer (g)", "Semanas de Gestación"),
        )
        if PN_COL:
            hist_fig.add_trace(
                go.Histogram(x=df_f[PN_COL].dropna(), nbinsx=40,
                             marker_color=P["primary"], opacity=0.75, name="Peso RN"),
                row=1, col=1,
            )
            hist_fig.add_vline(x=2500, line_dash="dash", line_color=P["secondary"],
                               annotation_text="2 500 g", row=1, col=1)
        if EG_COL:
            hist_fig.add_trace(
                go.Histogram(x=df_f[EG_COL].dropna(), nbinsx=25,
                             marker_color=P["light"], opacity=0.75, name="EG"),
                row=1, col=2,
            )
            hist_fig.add_vline(x=37, line_dash="dash", line_color=P["secondary"],
                               annotation_text="37 sem", row=1, col=2)
        hist_fig.update_layout(
            showlegend=False, height=290,
            margin=dict(t=40, b=20, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=P["bg"],
        )
        st.plotly_chart(hist_fig, use_container_width=True, key="histogramas_inicio")
        st.markdown("""
        <div class="interp-box">
        <b>¿Qué nos dicen estos histogramas?</b><br>
        • <b>Peso al nacer:</b> la línea roja punteada marca los 2 500 g, que es el límite
        clínico del bajo peso al nacer. Las barras a la izquierda de esa línea representan
        bebés con bajo peso. Entre más barras a la izquierda, mayor riesgo en el grupo.<br>
        • <b>Edad gestacional:</b> la línea roja marca las 37 semanas (umbral de parto prematuro).
        Si hay muchas barras antes de ese punto, indica alta prevalencia de prematuridad en
        el grupo estudiado.
        </div>
        """, unsafe_allow_html=True)

        # Scatter AF_exposure vs EG
        if AF_EXP and EG_COL and SUPL_COL and df_f[[AF_EXP, EG_COL]].dropna().shape[0] > 5:
            st.markdown("##### Exposición a AF vs Edad Gestacional")
            scatter_df = df_f[[AF_EXP, EG_COL, SUPL_COL]].dropna()
            fig_sc = px.scatter(
                scatter_df, x=AF_EXP, y=EG_COL, color=SUPL_COL,
                color_discrete_sequence=[P["secondary"], P["accent"], P["green"]],
                opacity=0.4,
                labels={AF_EXP: "Exposición total AF (µg/día)", EG_COL: "Semanas de gestación"},
            )
            # Línea de tendencia global (numpy, sin statsmodels)
            _x = scatter_df[AF_EXP].values
            _y = scatter_df[EG_COL].values
            _m, _b = np.polyfit(_x, _y, 1)
            _xr = np.array([_x.min(), _x.max()])
            fig_sc.add_trace(go.Scatter(
                x=_xr, y=_m * _xr + _b,
                mode="lines", line=dict(color=P["purple"], width=2, dash="dot"),
                name="Tendencia global", showlegend=True,
            ))
            fig_sc.add_hline(y=37, line_dash="dash", line_color=P["secondary"],
                             annotation_text="Umbral prematuro")
            fig_sc.update_layout(
                height=270,
                margin=dict(t=10, b=20, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=P["bg"],
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1, font_size=10),
            )
            st.plotly_chart(fig_sc, use_container_width=True, key="scatter_af_eg_inicio")
            st.markdown("""
            <div class="interp-box">
            <b>¿Qué nos dice esta gráfica?</b> Cada punto es una mujer. El eje horizontal
            es su consumo total de ácido fólico (suplemento + ácido fólico del pan) y el eje
            vertical son las semanas que duró su embarazo. La línea punteada morada es la
            tendencia general: si sube hacia la derecha, significa que a mayor consumo de AF
            tendieron a tener embarazos más largos (menos prematuros). La línea roja horizontal
            marca las 37 semanas — los puntos por debajo son partos prematuros. El color de
            cada punto indica el nivel de suplementación de esa mujer.
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        '<div style="text-align:center;color:#aaa;font-size:0.82rem;">'
        "Dataset: Ingesta_AF_clean.csv · 1 170 mujeres embarazadas chilenas · "
        "KeplerMapper &amp; Ripser · Reto Topología 2025"
        "</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MAPA TOPOLÓGICO
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(
        '<div class="main-title"> Mapa Topológico — Grafo Mapper</div>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    <div class="info-box">
    El grafo Mapper resume la topología del espacio multidimensional de ingesta de ácido fólico.
    Cada <b>nodo</b> agrupa mujeres con perfiles similares; las <b>aristas</b> conectan nodos que
    comparten observaciones. El <b>color</b> refleja la exposición promedio a AF por nodo.
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Nodos", "564")
    m2.metric("Aristas", "491")
    m3.metric("Cubos (cubierta)", "225 (15 × 15)")
    m4.metric("Solapamiento", "50 %")

    st.markdown("---")

    html_content = get_mapper_html(canvas_height=620)
    if html_content:
        components.html(html_content, height=720, scrolling=True)
    else:
        st.warning(
            "Archivo HTML del Mapper no encontrado. "
            "Ejecuta primero `Mapper/Mapper.ipynb` para generarlo."
        )
        st.info("Archivo esperado: `Mapper/mapper_af_embarazo_semifinal.html`")

    st.markdown("---")

    leg_col, tech_col = st.columns(2)
    with leg_col:
        st.markdown("""
        ##### Cómo leer el mapa
        - **Color azul/frío** → baja exposición a AF
        - **Color cálido/rojo** → alta exposición a AF
        - **Nodos grandes** → más mujeres en el grupo
        - **Nodos conectados** → patrones de ingesta similares
        - **Clusters separados** → subpoblaciones estructuralmente distintas
        """)
    with tech_col:
        st.markdown("""
        ##### Parámetros del Mapper
        - **Lente:** Proyección PCA 2D (10 variables de ingesta)
        - **Cubierta:** n_cubes=15, perc_overlap=50 %
        - **Clustering local:** AgglomerativeClustering(k=4, ward)
        - **Colorización:** AF_exposure (suplemento + pan)
        - **Filtro:** componentes con n ≥ 20 observaciones
        """)

    # Usar el perfil pre-computado de StatsMapper (4 clusters del análisis TDA)
    _perf = STATS.get("perfil_mapper")
    if _perf is not None and not _perf.empty:
        st.markdown("##### Distribución de observaciones por cluster (análisis TDA)")
        _cl_ids  = _perf["cluster_mapper"].astype(int).tolist()
        _cl_n    = _perf["n"].astype(int).tolist()
        _af_vals = _perf["uf_af_media"].tolist()
        _bar_col = [
            P["green"]     if v >= 1000 else
            P["accent"]    if v >= 500  else
            P["secondary"]
            for v in _af_vals
        ]
        fig_cl = go.Figure(go.Bar(
            x=[f"Cl.{c}" for c in _cl_ids],
            y=_cl_n,
            marker_color=_bar_col,
            text=_cl_n,
            textposition="outside",
            customdata=list(zip(_af_vals, [round(v*100,1) for v in _perf["pct_adverso"]])),
            hovertemplate=(
                "Cluster %{x}<br>n = %{y}<br>"
                "AF medio = %{customdata[0]:.0f} µg/d<br>"
                "% Adverso = %{customdata[1]:.1f}%<extra></extra>"
            ),
        ))
        fig_cl.update_layout(
            height=300,
            xaxis_title="Cluster (análisis TDA — StatsMapper.ipynb)",
            yaxis_title="N° mujeres",
            margin=dict(t=20, b=20, l=30, r=30),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=P["bg"],
        )
        # Leyenda de color manual
        st.plotly_chart(fig_cl, use_container_width=True, key="bar_dist_mapper_clusters")
        st.caption(
            "🟢 AF ≥ 1 000 µg/d  ·  🟠 AF 500–999 µg/d  ·  🔴 AF < 500 µg/d  "
            "— basado en 4 clusters del análisis TDA de StatsMapper.ipynb"
        )
        st.markdown("""
        <div class="interp-box">
        <b>¿Qué nos dice esta gráfica?</b> El algoritmo TDA identificó
        <b>4 grupos naturales</b> de mujeres según sus patrones de ingesta de ácido fólico.
        Cada barra muestra cuántas mujeres hay en ese grupo. El color indica el consumo
        promedio de AF: <b>verde</b> = dosis adecuada (≥ 1 000 µg/d), <b>naranja</b> = dosis
        baja, <b>rojo</b> = consumo muy bajo o sin suplemento. Al pasar el cursor sobre una
        barra también se puede ver el porcentaje de resultados adversos de ese grupo.
        </div>
        """, unsafe_allow_html=True)

    # ── Análisis TDA — imágenes de StatsMapper.ipynb ──────────────────────────
    # ── Imágenes TDA organizadas por grupo temático ──────────────────────────────
    tda_imgs_all = {img["title"]: img for img in get_images("tda")}
    if tda_imgs_all:
        st.markdown("---")
        st.markdown("#### Análisis Topológico Detallado")

        TDA_GROUPS = [
            {
                "label": "Espacio PCA — ¿Dónde se ubica cada mujer?",
                "keys": ["PCA 2D — Resultados neonatales"],
                "caption": (
                    "Esta figura resume en un plano 2D los perfiles de ingesta de ácido fólico "
                    "de las 84 mujeres del subconjunto de análisis. Cada punto es una mujer; "
                    "puntos cercanos tienen patrones de consumo parecidos. Los colores muestran "
                    "si hubo resultado adverso (parto prematuro o bajo peso). Si los puntos de "
                    "colores distintos forman grupos separados, significa que el patrón de "
                    "consumo de AF está relacionado con el resultado del embarazo."
                ),
                "full_width": True,
            },
            {
                "label": "Homología Persistente — ¿Qué 'formas' tienen los datos?",
                "keys": [
                    "Homología Persistente — Diagrama y Barcodes",
                    "Homología por Patrón de Suplementación",
                    "Paisajes de Persistencia H1",
                ],
                "caption": (
                    "La homología persistente es una herramienta matemática que detecta "
                    "'huecos' o 'bucles' en los datos — sin importar la escala. Cada barra "
                    "en el diagrama de barcodes representa una estructura que aparece y "
                    "desaparece en los datos: cuanto más larga la barra, más relevante y "
                    "estable es esa estructura. Comparar estas barras entre mujeres con "
                    "distinto patrón de suplementación permite identificar diferencias "
                    "profundas que los promedios no capturan."
                ),
                "full_width": False,
            },
            {
                "label": "Prematuro vs No-Prematuro — ¿Son topológicamente distintos?",
                "keys": [
                    "Comparación Topológica: Prematuro vs No",
                    "Distribución de Lifetimes H1",
                ],
                "caption": (
                    "Aquí se comparan directamente las 'formas' de los datos entre mujeres "
                    "con parto prematuro y sin parto prematuro. Si las distribuciones son "
                    "distintas, significa que los patrones de ingesta de AF se organizan de "
                    "manera diferente en ambos grupos — lo cual refuerza la hipótesis de que "
                    "el ácido fólico tiene un rol protector. La distancia entre formas se mide "
                    "con la 'distancia de Wasserstein': mayor distancia = grupos más distintos."
                ),
                "full_width": False,
            },
        ]

        for group in TDA_GROUPS:
            imgs_in_group = [tda_imgs_all[k] for k in group["keys"] if k in tda_imgs_all]
            if not imgs_in_group:
                continue
            with st.expander(group["label"], expanded=True):
                st.caption(group["caption"])
                if group["full_width"] or len(imgs_in_group) == 1:
                    for img in imgs_in_group:
                        st.image(img["path"], caption=img["title"], use_container_width=True)
                else:
                    cols = st.columns(len(imgs_in_group), gap="medium")
                    for col_w, img in zip(cols, imgs_in_group):
                        with col_w:
                            st.image(img["path"], caption=img["title"], use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PERFILES DE CLUSTERS
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(
        '<div class="main-title">Perfiles de Subpoblaciones </div>',
        unsafe_allow_html=True,
    )

    # ── Heatmap desde perfil pre-computado por StatsMapper.ipynb ─────────────
    perfil_mapper = STATS.get("perfil_mapper")
    merge_df      = STATS.get("merge")

    if perfil_mapper is not None and not perfil_mapper.empty:
        st.markdown("#### Comparación global de perfiles por subgrupo (clusters TDA)")

        col_rename = {
            "uf_af_media":   "AF medio (µg/d)",
            "n_panes_media": "Panes/día",
            "pct_sin_supl":  "% Sin supl.",
            "eg_media":      "EG media (sem)",
            "pct_prematuro": "% Prematuro",
            "pct_bajo_peso": "% Bajo peso",
            "pct_adverso":   "% Adverso",
        }
        heat_src = perfil_mapper.copy()
        heat_src = heat_src.rename(columns=col_rename)
        heat_src["cluster_mapper"] = heat_src["cluster_mapper"].astype(int)
        heat_src = heat_src.set_index("cluster_mapper")

        # Formatear porcentajes para mostrar en celdas
        display_src = heat_src.copy()
        for c in ["% Sin supl.", "% Prematuro", "% Bajo peso", "% Adverso"]:
            if c in display_src.columns:
                display_src[c] = (display_src[c] * 100).round(1)

        num_cols_h = list(col_rename.values())
        num_cols_h = [c for c in num_cols_h if c in display_src.columns]
        heat_norm  = display_src[num_cols_h].copy().astype(float)
        for c in num_cols_h:
            rng = heat_norm[c].max() - heat_norm[c].min()
            heat_norm[c] = (heat_norm[c] - heat_norm[c].min()) / rng if rng > 0 else 0.5

        x_labels = [
            f"Cl.{int(idx)} (n={int(heat_src.loc[idx,'n'])})"
            for idx in heat_src.index
        ]
        fig_heat = go.Figure(go.Heatmap(
            z=heat_norm[num_cols_h].values.T,
            x=x_labels,
            y=num_cols_h,
            colorscale="RdYlGn_r",
            text=display_src[num_cols_h].round(2).values.T,
            texttemplate="%{text}",
            textfont={"size": 11},
            hovertemplate="Cluster: %{x}<br>Métrica: %{y}<br>Valor: %{text}<extra></extra>",
            colorbar=dict(title="Normalizado", tickvals=[0, 0.5, 1],
                          ticktext=["Bajo", "Medio", "Alto"]),
        ))
        fig_heat.update_layout(
            height=max(320, len(num_cols_h) * 46),
            margin=dict(t=20, b=20, l=160, r=60),
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(side="top"),
        )
        st.plotly_chart(fig_heat, use_container_width=True,
                        key="heatmap_clusters_comparacion")
        st.markdown("""
        <div class="interp-box">
        <b>¿Cómo leer esta tabla de colores (heatmap)?</b> Cada columna es un grupo de
        mujeres (cluster) y cada fila es un indicador clínico. El color <b>verde</b> indica
        que ese grupo tiene un valor favorable en ese indicador (por ejemplo, mayor consumo
        de AF o menor porcentaje de prematuros). El color <b>rojo</b> indica el peor valor
        dentro de los 4 grupos. Los números dentro de cada celda son los valores reales.
        Esto permite ver de un vistazo qué grupo tiene mayor riesgo y en qué indicadores.
        </div>
        """, unsafe_allow_html=True)

        # ── Perfiles individuales desde merge_df ─────────────────────────────
        st.markdown("---")
        st.markdown("#### Perfil detallado por cluster")

        cluster_ids_pre = sorted(heat_src.index.tolist())
        for idx, cid in enumerate(cluster_ids_pre):
            row_p = heat_src.loc[cid]
            n_sub        = int(row_p["n"])
            pct_adv_cl   = float(display_src.loc[cid, "% Adverso"])   if "% Adverso"   in display_src.columns else 0.0
            pct_prem_cl  = float(display_src.loc[cid, "% Prematuro"]) if "% Prematuro" in display_src.columns else 0.0
            af_mean_cl   = float(row_p["AF medio (µg/d)"])             if "AF medio (µg/d)" in row_p.index     else 0.0

            if pct_adv_cl > 20 or pct_prem_cl > 15:
                tag = '<span class="riesgo-tag">⚠️ RIESGO ALTO</span>'
            elif pct_adv_cl < 8 and af_mean_cl > 800:
                tag = '<span class="protector-tag">✅ PERFIL PROTECTOR</span>'
            else:
                tag = (
                    '<span style="background:#EEF2FF;color:#5B4A8A;border-radius:6px;'
                    'padding:2px 8px;font-size:0.82rem;font-weight:600;">'
                    '🔵 RIESGO MODERADO</span>'
                )

            with st.expander(
                f"Cluster {cid}  ·  n={n_sub}  ·  "
                f"AF={af_mean_cl:.0f} µg/d  ·  %Adverso={pct_adv_cl:.1f}%",
                expanded=(idx == 0),
            ):
                st.markdown(f"**Clasificación:** {tag}", unsafe_allow_html=True)
                st.markdown("")

                c_left, c_right = st.columns(2, gap="medium")

                with c_left:
                    bar_labels = ["% Adverso", "% Prematuro", "% Bajo peso",
                                  "AF medio (µg/d)", "EG media (sem)"]
                    bar_labels = [l for l in bar_labels if l in display_src.columns]
                    bar_vals   = [round(float(display_src.loc[cid, l]), 1) for l in bar_labels]
                    bar_colors_l = [
                        P["secondary"] if any(k in l for k in ("Adverso","Prematuro","Bajo"))
                        else P["green"] if "AF" in l
                        else P["primary"]
                        for l in bar_labels
                    ]
                    fig_bar = go.Figure(go.Bar(
                        x=bar_labels, y=bar_vals,
                        marker_color=bar_colors_l,
                        text=[f"{v:.1f}" for v in bar_vals],
                        textposition="outside",
                    ))
                    fig_bar.update_layout(
                        height=280, showlegend=False,
                        margin=dict(t=20, b=10, l=10, r=10),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=P["bg"],
                    )
                    st.plotly_chart(fig_bar, use_container_width=True,
                                    key=f"bar_cluster_{cid}")
                    st.markdown("""
                    <div class="interp-box" style="font-size:0.87rem;">
                    Las barras <b style="color:#C0392B">rojas</b> indican riesgo
                    (prematuridad, bajo peso, resultado adverso — valores altos son malos).
                    La barra <b style="color:#1E8449">verde</b> (AF medio) indica consumo
                    promedio de ácido fólico — valores altos son protectores.
                    Las barras <b style="color:#5B4A8A">moradas</b> muestran la edad
                    gestacional media en semanas (mayor = mejor).
                    </div>
                    """, unsafe_allow_html=True)

                with c_right:
                    # Timing desde merge_df si está disponible
                    if merge_df is not None and "cluster_mapper" in merge_df.columns:
                        sub_m = merge_df[merge_df["cluster_mapper"] == cid]
                        tcols = [c for c in TIMING_COLS if c in sub_m.columns]
                        if tcols and len(sub_m) > 0:
                            t_vals = [round(sub_m[t].mean() * 100, 1) for t in tcols]
                            t_colors = [P["primary"], P["light"], P["accent"],
                                        P["gray"], P["secondary"]][:len(tcols)]
                            fig_t = go.Figure(go.Bar(
                                x=tcols, y=t_vals,
                                marker_color=t_colors,
                                text=[f"{v:.0f}%" for v in t_vals],
                                textposition="outside",
                            ))
                            fig_t.update_layout(
                                title_text="Timing de suplementación (%)",
                                height=280, showlegend=False,
                                margin=dict(t=40, b=10, l=10, r=10),
                                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=P["bg"],
                                yaxis=dict(title="%", range=[0, 105]),
                            )
                            st.plotly_chart(fig_t, use_container_width=True,
                                            key=f"timing_cluster_{cid}")
                            st.markdown("""
                            <div class="interp-box" style="font-size:0.87rem;">
                            Muestra en qué momento del embarazo las mujeres de este grupo
                            comenzaron a tomar suplemento de AF. Lo ideal es que la mayoría
                            empiece <b>antes del embarazo</b> o en los primeros meses. Si el
                            porcentaje mayor corresponde a etapas tardías (7-9 meses o durante
                            todo el embarazo sin inicio precoz), el periodo crítico del
                            desarrollo fetal ya habrá ocurrido sin la protección adecuada.
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.info("Sin datos de timing para este cluster.")
                    else:
                        st.info("Datos de timing no disponibles.")

    else:
        # Fallback: calcular desde datos filtrados en tiempo real
        if not CLMAP_COL or df_f[CLMAP_COL].isna().all():
            st.error("No se encontraron clusters. Ejecuta StatsMapper.ipynb y recarga.")
        else:
            df_cl = df_f.dropna(subset=[CLMAP_COL]).copy()
            df_cl[CLMAP_COL] = df_cl[CLMAP_COL].astype(int)
            for cid in sorted(df_cl[CLMAP_COL].unique()):
                sub = df_cl[df_cl[CLMAP_COL] == cid]
                st.markdown(f"**Cluster {cid}** — n={len(sub)}")

    # ── Imágenes de clusters desde StatsMapper.ipynb ──────────────────────────
    # ── Imágenes de clusters organizadas: panorámicas + individuales ─────────────
    cluster_imgs_all = {img["title"]: img for img in get_images("clusters")}
    if cluster_imgs_all:
        st.markdown("---")
        st.markdown("#### Visualizaciones de subpoblaciones por el análisis TDA")

        # Panorámica — imágenes de visión general (ancho completo)
        OVERVIEW_KEYS = [
            "Clústeres Topológicos K-Means",
            "Indicadores Neonatales por Clúster TDA",
            "Perfil Clínico por Cluster Mapper",
            "Perfil Clínico por Subpoblación",
            "Timing de Suplementación por Subpoblación",
        ]
        overview_imgs = [cluster_imgs_all[k] for k in OVERVIEW_KEYS if k in cluster_imgs_all]
        OVERVIEW_INTERP = {
            "Clústeres Topológicos K-Means": (
                "Vista general de los 4 grupos identificados por el algoritmo TDA en el espacio "
                "de ingesta de AF. Cada color representa un cluster distinto. Los grupos bien "
                "separados indican que existen subpoblaciones con patrones de consumo claramente "
                "diferentes entre sí."
            ),
            "Indicadores Neonatales por Clúster TDA": (
                "Comparación de los resultados del bebé (peso al nacer, semanas de gestación, "
                "porcentaje de prematuros) entre los 4 grupos. Permite ver si los grupos con "
                "mayor consumo de AF tienen mejores resultados neonatales."
            ),
            "Perfil Clínico por Cluster Mapper": (
                "Tabla de colores que resume todos los indicadores clínicos de cada grupo "
                "al mismo tiempo. Verde = indicador favorable, rojo = indicador de riesgo. "
                "Es la forma más rápida de identificar qué grupo está en mayor riesgo."
            ),
            "Perfil Clínico por Subpoblación": (
                "Detalle del perfil de salud de cada subpoblación: cuántas mujeres hay, "
                "cuánto AF consumieron y cuáles fueron sus resultados perinatales. Útil para "
                "comparar grupos de forma directa."
            ),
            "Timing de Suplementación por Subpoblación": (
                "Muestra cuándo empezó a suplementarse cada subpoblación. El inicio previo al "
                "embarazo o en el primer trimestre es el más protector. Un inicio tardío indica "
                "que el período más crítico del desarrollo fetal transcurrió sin protección."
            ),
        }
        if overview_imgs:
            with st.expander("Visión general de subpoblaciones", expanded=True):
                st.markdown("""
                <div class="interp-box">
                Las siguientes figuras muestran una visión completa de las 4 subpoblaciones
                identificadas por el análisis TDA. <b>No es necesario entender la matemática</b>
                para leer estas gráficas: simplemente compare los colores y los valores
                numéricos entre grupos para identificar cuál tiene mayor riesgo clínico.
                </div>
                """, unsafe_allow_html=True)
                for img in overview_imgs:
                    st.image(img["path"], use_container_width=True)
                    interp = OVERVIEW_INTERP.get(img["title"], "")
                    if interp:
                        st.caption(f"🔍 {interp}")

        # Comparativa por cluster — 5 indicadores en cuadrícula 2–3 columnas
        DETAIL_KEYS = [
            "% Resultado Adverso por Cluster",
            "AF Suplemento Medio por Cluster",
            "Edad Gestacional Media por Cluster",
            "Timing de Suplementación por Cluster",
            "Peso al Nacer Medio por Cluster",
        ]
        DETAIL_CAPTIONS = [
            "🔴 % Resultado adverso: proporción de mujeres con parto prematuro o bebé con bajo peso en cada grupo. El grupo con la barra más alta tiene mayor riesgo clínico.",
            "🟢 AF medio (µg/día): consumo promedio de ácido fólico en suplemento. Comparar con el umbral de 1 000 µg/d recomendado. A mayor barra, mejor protección.",
            "🔵 Edad gestacional media (semanas): duración promedio del embarazo en cada grupo. Lo deseable es ≥ 37 semanas. Barras más cortas indican mayor prematuridad.",
            "Timing de suplementación: momento en que las mujeres del grupo comenzaron a tomar AF. Inicio previo al embarazo es el más protector — evalúa si el grupo llegó al periodo fetal crítico con protección.",
            "Peso al nacer medio (g): peso promedio del bebé al nacer por grupo. El umbral clínico de bajo peso es 2 500 g. Grupos con barras por debajo de ese valor tienen mayor prevalencia de bajo peso.",
        ]
        detail_imgs = [(cluster_imgs_all[k], cap) for k, cap in zip(DETAIL_KEYS, DETAIL_CAPTIONS)
                       if k in cluster_imgs_all]
        if detail_imgs:
            with st.expander("Comparativa por subgrupo — Indicadores individuales", expanded=True):
                n_cols = 2
                for row_start in range(0, len(detail_imgs), n_cols):
                    row_items = detail_imgs[row_start:row_start + n_cols]
                    cols = st.columns(len(row_items), gap="medium")
                    for col_w, (img, cap) in zip(cols, row_items):
                        with col_w:
                            st.image(img["path"], use_container_width=True)
                            st.caption(cap)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ESTADÍSTICA CLÍNICA
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown(
        '<div class="main-title"> Estadística Clínica</div>',
        unsafe_allow_html=True,
    )

    # ── AUC-ROC desde StatsMapper.ipynb ───────────────────────────────────────
    auc_df      = STATS.get("auc")
    features_df = STATS.get("features")
    wasser_df   = STATS.get("wasserstein")
    normas_df   = STATS.get("normas")

    if auc_df is not None:
        st.markdown("#### ¿Puede predecirse el riesgo perinatal con el consumo de AF?")
        st.markdown("""
        <div class="info-box">
        Se entrenaron modelos de inteligencia artificial para predecir si una mujer tendrá
        un resultado adverso (parto prematuro o bajo peso), usando distintos conjuntos de
        información: solo datos clínicos de ingesta, solo el cluster TDA al que pertenece,
        o ambos combinados. Los resultados muestran qué tan bien funcionó cada modelo.
        </div>
        """, unsafe_allow_html=True)

        auc_l, auc_r = st.columns([1.4, 1.6], gap="large")

        with auc_l:
            # Heatmap AUC pivot
            try:
                auc_pivot = auc_df.pivot(index="Clasificador", columns="Features",
                                         values="AUC-ROC")
                feat_order = [c for c in
                              ["Ingesta (clínica)", "TDA (clúster)", "Combinado"]
                              if c in auc_pivot.columns]
                auc_pivot  = auc_pivot[feat_order] if feat_order else auc_pivot

                fig_auc_h = go.Figure(go.Heatmap(
                    z=auc_pivot.values,
                    x=auc_pivot.columns.tolist(),
                    y=auc_pivot.index.tolist(),
                    colorscale="YlGn",
                    zmin=0.45, zmax=0.75,
                    text=auc_pivot.round(3).values,
                    texttemplate="%{text}",
                    textfont={"size": 12},
                    colorbar=dict(title="AUC-ROC"),
                ))
                fig_auc_h.update_layout(
                    title="AUC-ROC por modelo y features",
                    height=280, margin=dict(t=40, b=20, l=20, r=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    xaxis_title="Conjunto de features",
                )
                st.plotly_chart(fig_auc_h, use_container_width=True,
                                key="heatmap_auc_clasificacion")
            except Exception:
                st.dataframe(auc_df, use_container_width=True)

        with auc_r:
            # Barras horizontales ordenadas
            auc_sorted = auc_df.sort_values("AUC-ROC", ascending=True)
            feat_color_map = {
                "Ingesta (clínica)": P["gray"],
                "TDA (clúster)":     P["primary"],
                "Combinado":         P["green"],
            }
            bar_c = [feat_color_map.get(f, P["primary"]) for f in auc_sorted["Features"]]
            fig_auc_b = go.Figure(go.Bar(
                x=auc_sorted["AUC-ROC"],
                y=[f"{r['Clasificador']}\n({r['Features']})"
                   for _, r in auc_sorted.iterrows()],
                orientation="h",
                marker_color=bar_c,
                error_x=dict(type="data", array=auc_sorted["Std"].tolist(),
                             color=P["gray"], thickness=1.5),
                text=[f"{v:.3f}" for v in auc_sorted["AUC-ROC"]],
                textposition="outside",
            ))
            fig_auc_b.add_vline(x=0.5, line_dash="dash", line_color=P["secondary"],
                                annotation_text="Azar")
            fig_auc_b.update_layout(
                title="Ranking de modelos",
                height=320,
                margin=dict(t=40, b=20, l=20, r=40),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=P["bg"],
                xaxis=dict(title="AUC-ROC", range=[0.4, 0.8]),
            )
            st.plotly_chart(fig_auc_b, use_container_width=True,
                            key="grafico_regresion_ols")

        st.markdown("""
        <div class="interp-box">
        <b>¿Cómo leer el AUC-ROC?</b> Es una medida de qué tan bien el modelo distingue
        entre mujeres con y sin resultado adverso. Va de 0 a 1:
        <b>0.5 = azar puro</b> (la línea roja punteada), <b>0.7+ = buena predicción</b>,
        <b>1.0 = predicción perfecta</b>. Si el modelo con información del cluster TDA
        supera al modelo solo con datos clínicos, significa que los patrones topológicos
        aportan información que los datos de ingesta simples no capturan.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Importancia de features desde StatsMapper.ipynb ───────────────────────
    if features_df is not None:
        st.markdown("#### Importancia de Features (Características) — Random Forest ")
        feat_plot = features_df.sort_values("importancia_gini", ascending=True)
        feat_colors = [
            P["primary"] if any(k in n for k in ("Clúster", "cluster", "cl_"))
            else P["gray"]
            for n in feat_plot["Feature"]
        ]
        fig_feat = go.Figure(go.Bar(
            x=feat_plot["importancia_gini"],
            y=feat_plot["Feature"],
            orientation="h",
            marker_color=feat_colors,
            text=[f"{v:.3f}" for v in feat_plot["importancia_gini"]],
            textposition="outside",
        ))
        fig_feat.update_layout(
            title="Importancia Gini (morado = feature topológica TDA)",
            height=max(300, len(feat_plot) * 30),
            margin=dict(t=50, b=20, l=20, r=60),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=P["bg"],
            xaxis_title="Importancia Gini",
        )
        st.plotly_chart(fig_feat, use_container_width=True,
                        key="boxplot_peso_suplementacion")
        st.markdown("""
        <div class="interp-box">
        <b>¿Qué nos dice esta gráfica?</b> El modelo de Random Forest analiza todas las
        variables disponibles y asigna un puntaje a cada una según cuánto contribuye a
        predecir el resultado adverso. Las barras más largas son las variables más
        importantes. Las barras <b style="color:#5B4A8A">moradas</b> son variables
        derivadas del análisis topológico TDA (el cluster al que pertenece la mujer).
        Si aparecen entre las más importantes, confirma que la información topológica
        tiene valor clínico predictivo real — más allá del simple consumo de AF.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Wasserstein + Normas L2 ───────────────────────────────────────────────
    w_col, n_col = st.columns(2, gap="large")

    with w_col:
        st.markdown("#### ¿Qué tan distintos son los grupos de suplementación?")
        if wasser_df is not None:
            st.markdown("""
            <div class="info-box">
            <b>Distancias de Wasserstein H1:</b> mide qué tan diferentes son los patrones
            de ingesta entre dos grupos de mujeres desde una perspectiva matemática profunda
            (no solo comparando promedios). Un valor alto indica que los dos grupos tienen
            estructuras de consumo muy distintas.
            </div>
            """, unsafe_allow_html=True)
            wasser_display = wasser_df.copy()
            wasser_display["wasserstein_H1"] = wasser_display["wasserstein_H1"].round(4)
            wasser_display.columns = ["Subgrupo A", "Subgrupo B", "Wasserstein H1"]
            st.dataframe(wasser_display, use_container_width=True)
            st.markdown("""
            <div class="interp-box" style="font-size:0.87rem;">
            <b>¿Cómo interpretar los números?</b> Cuanto mayor sea el valor de Wasserstein H1,
            más diferentes son los patrones de ingesta de AF entre esos dos grupos — lo que
            sugiere que no solo consumen cantidades distintas, sino que organizan su consumo
            de manera estructuralmente diferente a lo largo del embarazo.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Ejecuta StatsMapper.ipynb para calcular distancias Wasserstein.")

    with n_col:
        st.markdown("#### ¿Qué tan heterogéneo es el consumo dentro de cada grupo?")
        if normas_df is not None:
            st.markdown("""
            <div class="info-box">
            <b>Normas L2 de paisajes de persistencia:</b> indica qué tan variable o complejo
            es el patrón de ingesta de AF dentro de cada subgrupo. Un valor alto significa
            que dentro del grupo hay mucha variabilidad — las mujeres no siguen un patrón
            uniforme de consumo.
            </div>
            """, unsafe_allow_html=True)
            normas_display = normas_df.copy()
            normas_display["norma_L2_H1"] = normas_display["norma_L2_H1"].round(4)
            normas_display["pct_resultado_adverso"] = (
                normas_display["pct_resultado_adverso"] * 100
            ).round(1)
            normas_display.columns = ["Subgrupo", "Norma L2 H1", "% Adverso"]
            st.dataframe(normas_display, use_container_width=True)
            st.markdown("""
            <div class="interp-box" style="font-size:0.87rem;">
            Compare la columna <b>Norma L2 H1</b> con la columna <b>% Adverso</b>:
            si los grupos con norma alta también tienen mayor % de resultados adversos,
            sugiere que la <em>heterogeneidad</em> en el patrón de consumo (no solo la
            cantidad) puede estar asociada al riesgo perinatal.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Ejecuta StatsMapper.ipynb para calcular normas L2.")

    st.markdown("---")

    # ── Boxplot interactivo: peso al nacer por suplementación ─────────────────
    st.markdown("#### ¿Las mujeres que toman más AF tienen bebés con mayor peso al nacer?")

    box_l, box_r = st.columns([1.7, 1.3], gap="large")

    with box_l:
        if PN_COL and SUPL_COL and df_f[[PN_COL, SUPL_COL]].dropna().shape[0] > 0:
            ORDER = ["Sin suplemento", "Dosis baja (<1000 µg/d)", "Dosis adecuada (≥1000)"]
            ORDER = [c for c in ORDER if c in df_f[SUPL_COL].unique()]
            box_df = df_f[[PN_COL, SUPL_COL]].dropna()
            fig_box = px.box(
                box_df, x=SUPL_COL, y=PN_COL,
                category_orders={SUPL_COL: ORDER},
                color=SUPL_COL,
                color_discrete_sequence=[P["secondary"], P["accent"], P["green"]],
                labels={SUPL_COL: "Suplementación", PN_COL: "Peso al nacer (g)"},
                points="outliers",
            )
            fig_box.add_hline(y=2500, line_dash="dash", line_color=P["secondary"],
                              annotation_text="Umbral bajo peso (2 500 g)")
            fig_box.update_layout(
                height=380, showlegend=False,
                margin=dict(t=20, b=20, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=P["bg"],
            )
            st.plotly_chart(fig_box, use_container_width=True,
                            key="boxplot_peso_suplementacion_tab4")
        else:
            st.info("No hay suficientes datos con los filtros actuales.")

    with box_r:
        st.markdown("""
        <div class="interp-box">
        <b>¿Cómo leer este diagrama de cajas?</b><br>
        • <b>Línea del medio</b> = peso más típico (mediana)<br>
        • <b>La caja</b> = rango donde cae el 50 % central de las mujeres<br>
        • <b>Las líneas largas</b> (bigotes) = valores normales<br>
        • <b>Puntos sueltos</b> = casos inusuales (muy alto o bajo peso)<br>
        • <b>Línea roja punteada</b> = umbral de bajo peso (2 500 g)
        </div>
        <div class="interp-box" style="margin-top:0.5rem;">
        Si la caja del grupo "Dosis adecuada (≥1000)" está más alta que la del grupo
        "Sin suplemento", confirma que un mayor consumo de AF está asociado con mayor
        peso al nacer a nivel de toda la muestra.
        </div>
        """, unsafe_allow_html=True)
        if SUPL_COL and PN_COL and df_f[[SUPL_COL, PN_COL]].dropna().shape[0] > 0:
            st.markdown("##### Estadísticas por grupo")
            stats_tab = (
                df_f.groupby(SUPL_COL, observed=True)[PN_COL]
                .agg(n="count", Mediana="median", Media="mean", DE="std",
                     P25=lambda x: x.quantile(0.25), P75=lambda x: x.quantile(0.75))
                .round(1).reset_index()
            )
            stats_tab.columns = ["Suplementación", "n", "Mediana", "Media", "DE", "P25", "P75"]
            st.dataframe(stats_tab, use_container_width=True, height=170)

    st.markdown("---")

    # ── Violin: AF por resultado adverso ─────────────────────────────────────
    st.markdown("#### ¿Las mujeres con resultados adversos consumieron menos ácido fólico?")
    if AF_EXP and ADV_COL and df_f[[AF_EXP, ADV_COL]].dropna().shape[0] > 5:
        viol_df = df_f[[AF_EXP, ADV_COL]].dropna().copy()
        viol_df["Resultado"] = viol_df[ADV_COL].map(
            {0: "Sin resultado adverso", 1: "Resultado adverso"}
        )
        fig_viol = px.violin(
            viol_df, x="Resultado", y=AF_EXP, color="Resultado",
            color_discrete_map={
                "Sin resultado adverso": P["green"],
                "Resultado adverso":    P["secondary"],
            },
            box=True, points="outliers",
            labels={AF_EXP: "Exposición total AF (µg/día)", "Resultado": ""},
        )
        fig_viol.add_hline(y=1000, line_dash="dash", line_color=P["gray"],
                           annotation_text="Umbral 1 000 µg/d")
        fig_viol.update_layout(
            height=370, showlegend=False,
            margin=dict(t=20, b=20, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=P["bg"],
        )
        st.plotly_chart(fig_viol, use_container_width=True, key="violin_af_resultado")
        st.markdown("""
        <div class="interp-box">
        <b>¿Cómo leer esta gráfica de violín?</b> El ancho de cada "violín" muestra cuántas
        mujeres tienen ese nivel de consumo de AF: donde el violín es más ancho, hay más
        mujeres con esa cantidad. La caja pequeña del interior es el boxplot (mediana y
        rango intercuartílico). La línea punteada marca los 1 000 µg/d recomendados.
        <br><br>
        <b>¿Qué buscar?</b> Si el violín del grupo "Resultado adverso" es más ancho en la
        parte baja (menos de 1 000 µg/d), confirma que las mujeres que tuvieron parto
        prematuro o bebé de bajo peso tendieron a consumir menos ácido fólico.
        </div>
        """, unsafe_allow_html=True)

    # ── Imágenes de estadística desde StatsMapper.ipynb ───────────────────────
    stats_imgs = get_images("stats")
    if stats_imgs:
        st.markdown("---")
        st.markdown("#### Visualizaciones de clasificación")
        STATS_INTERP = {
            "Clasificación ML — AUC-ROC": (
                "Resumen visual de qué tan bien predice cada modelo el resultado adverso. "
                "Barras más largas = mejor capacidad predictiva. La columna 'Combinado' "
                "usa tanto datos clínicos como información topológica (cluster TDA)."
            ),
            "Importancia de Features (Random Forest)": (
                "Lista ordenada de las variables más relevantes para predecir el riesgo. "
                "Las barras moradas son variables topológicas — si están arriba, confirma "
                "que el análisis TDA aporta valor predictivo real más allá de los datos clínicos."
            ),
        }
        cols = st.columns(min(len(stats_imgs), 2), gap="medium")
        for i, img in enumerate(stats_imgs):
            with cols[i % len(cols)]:
                st.image(img["path"], use_container_width=True)
                interp = STATS_INTERP.get(img["title"], "")
                if interp:
                    st.caption(f"🔍 {interp}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — GLOSARIO CLÍNICO
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown(
        '<div class="main-title"> Glosario Clínico y Metodológico</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Referencia rápida de términos clínicos, estadísticos y topológicos "
        "usados en este análisis."
    )

    search = st.text_input(
        "🔍 Buscar término…",
        placeholder="Ej: ácido fólico, mapper, prematuro, Wasserstein",
    )

    filtered_glosario = {
        k: v for k, v in GLOSARIO.items()
        if not search
        or search.lower() in k.lower()
        or search.lower() in v["definicion"].lower()
        or search.lower() in v["relevancia"].lower()
    }

    if not filtered_glosario:
        st.warning(f"No se encontraron términos que coincidan con «{search}».")
    else:
        items = list(filtered_glosario.items())
        mid   = (len(items) + 1) // 2
        col_a, col_b = st.columns(2, gap="large")

        for col_w, chunk in [(col_a, items[:mid]), (col_b, items[mid:])]:
            with col_w:
                for term, data in chunk:
                    with st.expander(f"{data['icono']} {term}", expanded=False):
                        st.markdown(f"**• Definición:** {data['definicion']}")
                        st.markdown(f"**• Relevancia clínica:** {data['relevancia']}")
                        st.markdown(f"**•Parámetros / Dosis:** {data['parametros']}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — PREGUNTAS FRECUENTES
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown(
        '<div class="main-title"> Preguntas Frecuentes</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Respuestas sobre metodología TDA, hallazgos clínicos e "
        "interpretación de resultados."
    )
    st.markdown("")

    all_cats = sorted({f["categoria"] for f in FAQS})
    cat_filter = st.multiselect(
        "Filtrar por categoría",
        options=all_cats,
        default=all_cats,
        key="faq_cat_filter",
    )

    CAT_COLORS = {
        "Metodología":                  P["primary"],
        "Interpretación TDA":           P["purple"],
        "Hallazgos clínicos":           P["green"],
        "Contexto clínico-nutricional": P["accent"],
        "Recomendaciones clínicas":     P["secondary"],
        "Resultados ML":                P["gray"],
    }

    faqs_vis = [f for f in FAQS if f["categoria"] in cat_filter]

    if not faqs_vis:
        st.warning("No hay preguntas en las categorías seleccionadas.")
    else:
        for i, faq in enumerate(faqs_vis):
            color = CAT_COLORS.get(faq["categoria"], P["primary"])
            with st.expander(faq["pregunta"], expanded=(i == 0)):
                st.markdown(
                    f'<span style="background:{color}22;color:{color};'
                    f'border-radius:6px;padding:2px 10px;'
                    f'font-size:0.78rem;font-weight:600;">'
                    f'{faq["categoria"]}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown("")
                st.markdown(faq["respuesta"])
