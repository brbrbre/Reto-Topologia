# Reto Topología — Análisis Topológico de Datos aplicado a Salud Materno-Infantil

Proyecto de investigación que aplica **Análisis Topológico de Datos (TDA)** para identificar subpoblaciones de mujeres embarazadas según sus patrones de ingesta de **Ácido Fólico (AF)** y su asociación con resultados perinatales adversos (prematuridad y bajo peso al nacer) en Chile.

---

## Contexto

- **Dataset**: 1.170 mujeres embarazadas chilenas, 73 variables clínicas y nutricionales
- **Exposición principal**: ingesta de AF (suplementos + pan enriquecido, en µg/día)
- **Outcomes primarios**: semanas de gestación (`eg_num`), peso al nacer (`pnacer_num`), prematuridad (<37 sem) y bajo peso al nacer (<2.500 g)
- **Dato faltante clave**: dosis de suplemento (`uf_af`) con ~55% de missing (MCAR)

---

## Estructura del repositorio

```
Reto-Topologia/
├── app.py                              # Dashboard interactivo (Streamlit)
├── data_loader.py                      # Carga y preprocesamiento de datos
├── requirements.txt
│
├── Data/
│   ├── Ingesta_AF_embarazo.xlsx        # Dataset original
│   ├── Ingesta_AF_clean.csv            # Dataset limpio (entrada principal)
│   └── Ingesta_AF_df.csv               # Procesamiento intermedio
│
├── Preprocesamiento/
│   ├── Etapa2_CRISPDM_AF_Embarazo.ipynb    # Limpieza y EDA (metodología CRISP-DM)
│   └── Etapa2_Analisis_Descriptivo.ipynb   # Estadística descriptiva
│
├── Mapper/
│   ├── Mapper.ipynb                         # Algoritmo Mapper principal
│   ├── Mapper_Busqueda_Configuraciones.ipynb # Búsqueda de hiperparámetros
│   ├── StatsMapper.ipynb                    # Validación estadística y ML
│   ├── mapper_clusters.csv                  # Asignación de clusters (n=1.170)
│   ├── mapper_graph.pkl                     # Grafo Mapper serializado
│   ├── mapper_resultados_completos.csv      # Resultados completos
│   ├── mapper_af_embarazo_semifinal.html    # Visualización interactiva (D3)
│   └── outputs/                             # Figuras y tablas estadísticas generadas
│
├── Estadística Clásica/
│   └── EstadisticaClasicaAF.ipynb          # Validación con métodos clásicos
│
└── Homología Persistente/
    └── HomologiaPersistente.ipynb           # Análisis de homología persistente
```

---

## Pipeline de análisis

```
Raw data (Excel)
      │
      ▼
Preprocesamiento (CRISP-DM)
  - Detección y corrección de errores (8 tipos)
  - Feature engineering: prematuro, bajo_peso_rn, resultado_adverso, AF_exposure
  - Imputación de medianas, encoding categórico
      │
      ▼
TDA — Algoritmo Mapper
  - Espacio 10D normalizado (StandardScaler + imputación mediana)
  - Lente: PCA 2D
  - Cover: 15 cubos, 50% overlap (225 hipercubos)
  - Clustering local: AgglomerativeClustering (Ward, k=4)
  - Salida: 564 nodos, 491 aristas → 4 componentes conexas (n≥20)
      │
      ▼
Homología Persistente
  - Complejo de Vietoris-Rips en 8 variables de ingesta
  - H0: 184 componentes conexas | H1: 209 ciclos
  - Distancias de Wasserstein entre grupos de suplementación
  - Normas L2 de paisajes de persistencia
      │
      ▼
Validación estadística
  - Kruskal-Wallis (variables continuas) + χ² (variables binarias)
  - Clasificación ML (Logistic, Random Forest, Gradient Boosting, 5-fold CV)
  - AUC-ROC combinado (datos clínicos + cluster TDA): 0.5815
      │
      ▼
Dashboard interactivo (Streamlit)
```

---

## Dashboard (app.py)

La aplicación Streamlit ofrece 6 secciones:

| Sección | Contenido |
|---|---|
| **Inicio** | KPIs generales, histogramas, dispersión AF vs. semanas gestación |
| **Mapa Topológico** | Grafo Mapper interactivo (D3), distribución de clusters |
| **Perfiles de Clusters** | Heatmap de características, perfiles detallados, timing de suplementación |
| **Estadística Clínica** | AUC-ROC, importancia de features, distancias de Wasserstein, paisajes de persistencia |
| **Glosario Clínico** | 12 términos clínicos y metodológicos con definiciones buscables |
| **Preguntas Frecuentes** | 8 FAQs sobre metodología, hallazgos e interpretación TDA |

Filtros interactivos: región, edad materna, semanas de gestación, nivel de suplementación.

### Ejecución local

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Ejecutar el dashboard
streamlit run app.py
```

---

## Hallazgos clave

- **4 subpoblaciones** identificadas por Mapper con patrones distintos de ingesta de AF
- Clusters con exposición ≥1.000 µg/día presentan menor tasa de prematuridad (~6–7%)
- Clusters con exposición <500 µg/día muestran mayor tasa de resultados adversos (~14%)
- Las distancias de Wasserstein confirman que el grupo "Dosis adecuada" tiene una topología de ingesta estructuralmente diferente al grupo "Sin suplemento"
- El modelo combinado (datos clínicos + membresía topológica) alcanza AUC=0.5815 para predecir resultados adversos

---

## Requisitos técnicos

- Python 3.9+
- Ver [requirements.txt](requirements.txt) para dependencias completas

---

## Colaboradores

Brisma Alvarez Valdez
Valeria Arciga Valencia
Paulina Castellanos Chávez 
Ximena Montes Bautista
Emiliano Ruiz López
