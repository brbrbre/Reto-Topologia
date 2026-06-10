"""data_loader.py — Carga, preprocesamiento y resultados de StatsMapper para el dashboard."""
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent

DATA_PATH     = BASE / "Data" / "Ingesta_AF_clean.csv"
CLUSTERS_PATH = BASE / "Mapper" / "mapper_clusters.csv"
OUTPUTS_DIR   = BASE / "Mapper" / "outputs"          # generado por StatsMapper.ipynb

MAPPER_HTML_CANDIDATES = [
    BASE / "Mapper" / "mapper_af_embarazo_semifinal.html",
    BASE / "Mapper" / "mapper_af_embarazo (2) (2).html",
    BASE / "Mapper" / "mapper_af_embarazo.html",
]

REGION_MAP = {
    1: "Región Metropolitana",
    2: "Valparaíso",
    3: "Biobío",
    4: "O'Higgins",
    9: "La Araucanía",
}

SUPL_CATS = ["Sin suplemento", "Dosis baja (<1000 µg/d)", "Dosis adecuada (≥1000)"]

# ── Metadatos de imágenes generadas por StatsMapper.ipynb ─────────────────────
IMAGES_META = [
    # (filename,                        title,                                          tab)
    ("01_pca2d_resultados.png",         "PCA 2D — Resultados neonatales",               "tda"),
    ("02_homologia_persistente.png",    "Homología Persistente — Diagrama y Barcodes",  "tda"),
    ("03_homologia_por_supl.png",       "Homología por Patrón de Suplementación",       "tda"),
    ("04_paisajes_persistencia.png",    "Paisajes de Persistencia H1",                  "tda"),
    ("05_clusters_topologicos.png",     "Clústeres Topológicos K-Means",                "clusters"),
    ("06_indicadores_por_cluster.png",  "Indicadores Neonatales por Clúster TDA",       "clusters"),
    ("07_clasificacion_ml.png",         "Clasificación ML — AUC-ROC",                   "stats"),
    ("08_importancia_features.png",     "Importancia de Features (Random Forest)",      "stats"),
    ("09a_comparacion_topologica_prem.png", "Comparación Topológica: Prematuro vs No",  "tda"),
    ("09b_lifetimes_h1.png",            "Distribución de Lifetimes H1",                 "tda"),
    ("10_perfil_clinico_mapper.png",    "Perfil Clínico por Cluster Mapper",            "clusters"),
    ("11_perfil_por_subpoblacion.png",  "Perfil Clínico por Subpoblación",              "clusters"),
    ("12_timing_supl.png",              "Timing de Suplementación por Subpoblación",    "clusters"),
    ("13a_adverso_cluster.png",         "% Resultado Adverso por Cluster",              "clusters"),
    ("13b_af_medio_cluster.png",        "AF Suplemento Medio por Cluster",              "clusters"),
    ("13c_eg_media_cluster.png",        "Edad Gestacional Media por Cluster",           "clusters"),
    ("13d_timing_cluster.png",          "Timing de Suplementación por Cluster",         "clusters"),
    ("13e_peso_cluster.png",            "Peso al Nacer Medio por Cluster",              "clusters"),
]


# ── Funciones de carga del dataset principal ───────────────────────────────────
def load_data() -> pd.DataFrame:
    """Carga el CSV principal y deriva todas las variables de análisis."""
    df = pd.read_csv(DATA_PATH, low_memory=False)

    for col in ["pnacer_num", "eg_num", "uf_af", "n_panes_num", "neduc", "prematuro"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["paridad"] = pd.to_numeric(df["paridad"], errors="coerce")

    for col in ["Antes del embarazo", "Durante todo el embarazo",
                "1-3 meses", "4-6 meses", "7-9 meses"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["bajo_peso_rn"] = (df["pnacer_num"] < 2500).astype(float)
    df.loc[df["pnacer_num"].isna(), "bajo_peso_rn"] = np.nan

    df["resultado_adverso"] = (
        (df["prematuro"] == 1) | (df["bajo_peso_rn"] == 1)
    ).astype(float)
    df.loc[df["prematuro"].isna() & df["bajo_peso_rn"].isna(), "resultado_adverso"] = np.nan

    df["score_pan"]   = df["n_panes_num"].fillna(0) * 160
    df["AF_exposure"] = df["uf_af"].fillna(0) + df["score_pan"]

    conds = [
        df["NO consumio"] == 1,
        (df["NO consumio"] == 0) & (df["uf_af"] < 1000),
        (df["NO consumio"] == 0) & (df["uf_af"] >= 1000),
    ]
    df["cat_supl"] = np.select(conds, SUPL_CATS, default=None)
    df["cat_supl"] = pd.Categorical(df["cat_supl"], categories=SUPL_CATS, ordered=True)

    df["region"] = df["region_chile"].map(REGION_MAP).fillna(
        df["region_chile"].astype(str)
    )
    return df


def load_mapper_clusters() -> pd.DataFrame:
    """Carga mapper_clusters.csv; conserva sólo componentes con n ≥ 20."""
    clusters = pd.read_csv(CLUSTERS_PATH)
    counts = clusters["cluster_mapper"].value_counts()
    valid  = counts[counts >= 20].index
    clusters.loc[~clusters["cluster_mapper"].isin(valid), "cluster_mapper"] = np.nan
    return clusters


# ── Carga de resultados pre-computados por StatsMapper.ipynb ──────────────────
def _read(filename: str) -> "pd.DataFrame | None":
    path = OUTPUTS_DIR / filename
    if path.exists():
        return pd.read_csv(path)
    return None


def load_stats() -> dict:
    """Devuelve un dict con todos los DataFrames exportados por StatsMapper.ipynb."""
    stats: dict = {}

    # Perfil por cluster Mapper (heatmap + barras)
    df = _read("stats_05_perfil_clusters_mapper.csv")
    if df is not None:
        df["cluster_mapper"] = df["cluster_mapper"].astype(int)
        stats["perfil_mapper"] = df

    # Datos fusionados TDA + Mapper (para desgloses individuales)
    df = _read("stats_06_datos_merge_mapper.csv")
    if df is not None:
        df = df.rename(columns={"Unnamed: 0": "idx"})
        df["cluster_mapper"] = pd.to_numeric(df["cluster_mapper"], errors="coerce")
        stats["merge"] = df

    # AUC-ROC de clasificación ML
    df = _read("stats_02_auc_clasificacion.csv")
    if df is not None:
        stats["auc"] = df

    # Importancia de features Random Forest
    df = _read("stats_03_importancia_features.csv")
    if df is not None:
        df = df.rename(columns={"Unnamed: 0": "Feature"})
        stats["features"] = df.sort_values("importancia_gini", ascending=False)

    # Distancias Wasserstein H1
    df = _read("stats_07_wasserstein_subgrupos.csv")
    if df is not None:
        stats["wasserstein"] = df

    # Normas L2 paisajes de persistencia
    df = _read("stats_08_normas_paisajes.csv")
    if df is not None:
        stats["normas"] = df

    return stats


def get_images(tab: "str | None" = None) -> list[dict]:
    """Lista las imágenes disponibles en outputs/.
    Si tab se especifica ('tda', 'clusters', 'stats'), filtra por categoría.
    """
    result = []
    for fname, title, img_tab in IMAGES_META:
        if tab and img_tab != tab:
            continue
        path = OUTPUTS_DIR / fname
        if path.exists():
            result.append({"path": str(path), "title": title, "tab": img_tab})
    return result


# ── HTML del Mapper (parcheado para iframe) ────────────────────────────────────
def get_mapper_html(canvas_height: int = 600) -> "str | None":
    """Devuelve el HTML del Mapper parcheado para renderizar dentro de un iframe."""
    for path in MAPPER_HTML_CANDIDATES:
        if not path.exists():
            continue
        html = path.read_text(encoding="utf-8", errors="replace")

        # Añadir height explícito a #canvas
        html = html.replace(
            "#canvas {\n  width: 100%;\n}",
            f"#canvas {{\n  width: 100%;\n  height: {canvas_height}px;\n}}",
        )
        if f"height: {canvas_height}px" not in html:
            html = html.replace(
                "#canvas {",
                f"#canvas {{\n  height: {canvas_height}px;",
                1,
            )

        # Inyectar resize trigger para que D3 recalcule dimensiones
        html = html.replace(
            "</body>",
            "<script>\nwindow.addEventListener('load',function(){"
            "setTimeout(function(){window.dispatchEvent(new Event('resize'));},200);});\n"
            "</script>\n</body>",
            1,
        )
        return html
    return None


# ── Regresión OLS ──────────────────────────────────────────────────────────────
def _compute_ols() -> dict:
    try:
        import statsmodels.formula.api as smf

        df = load_data()
        df_ols = df[["eg_num", "AF_exposure", "edad", "neduc", "paridad"]].dropna()
        model = smf.ols(
            "eg_num ~ AF_exposure + edad + neduc + paridad", data=df_ols
        ).fit(cov_type="HC3")

        ci = model.conf_int()
        return {
            var: {
                "coef":    round(float(model.params[var]),  5),
                "std_err": round(float(model.bse[var]),     5),
                "pval":    round(float(model.pvalues[var]), 5),
                "ci_low":  round(float(ci.loc[var, 0]),     5),
                "ci_high": round(float(ci.loc[var, 1]),     5),
            }
            for var in model.params.index
        }
    except Exception:
        return {}


OLS_RESULTS: dict = _compute_ols()
