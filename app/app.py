"""Tablero Streamlit — Hackathon IA: Calidad de Vino.

Ejecutar desde la raíz del repo:  streamlit run app/app.py
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import BASE_FEATURES  # noqa: E402  (necesario para cargar los pipelines)
from model import build_final_model, build_pca_pipeline  # noqa: E402

st.set_page_config(page_title="Wine Quality · Hackathon IA", page_icon="🍷", layout="wide")
COLORS = {"0 · estándar": "#9aa5b1", "1 · superior": "#8e1b3a"}


@st.cache_data
def load_data():
    train = pd.read_csv(ROOT / "data" / "train_public.csv")
    test = pd.read_csv(ROOT / "data" / "test_public_features.csv")
    res = pd.read_csv(ROOT / "entregas" / "resultados_cv.csv")
    return train, test, res


@st.cache_resource
def load_models(_X, _y):
    try:
        return (joblib.load(ROOT / "models" / "modelo_final.joblib"),
                joblib.load(ROOT / "models" / "pca_pipeline.joblib"))
    except Exception:
        # Si la versión de scikit-learn no coincide con la del .joblib, se reentrena
        # (k-NN "entrena" en milisegundos: solo memoriza los datos escalados).
        return build_final_model().fit(_X, _y), build_pca_pipeline().fit(_X)


train, test, res = load_data()
X, y = train[BASE_FEATURES], train["target"]
model, pca_pipe = load_models(X, y)
best = res.sort_values("f1_macro", ascending=False).iloc[0]

# ---------------------------------------------------------------- encabezado
st.title("🍷 Clasificación de calidad de vino")
st.caption("Hackathon — Métodos y Modelos Matemáticos para IA · Métrica: F1-Score Macro · Validación: StratifiedKFold (5)")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Muestras de entrenamiento", f"{len(train):,}")
c2.metric("Clase 1 (calidad superior)", f"{y.mean():.1%}")
c3.metric(f"Mejor F1-Macro (CV, ± {best.f1_std:.3f})", f"{best.f1_macro:.4f}")
c4.metric("Modelos evaluados", len(res))

tab_eda, tab_cmp, tab_sim = st.tabs(["📊 EDA y PCA", "🏁 Comparativa de modelos", "🔮 Simulador de predicción"])

# ---------------------------------------------------------------- (a) EDA / PCA
with tab_eda:
    left, right = st.columns(2)
    with left:
        st.subheader("Correlación de cada variable con el target")
        corr = train[BASE_FEATURES + ["target"]].corr()["target"].drop("target").sort_values()
        fig = go.Figure(go.Bar(x=corr.values, y=corr.index, orientation="h",
                               marker_color=np.where(corr.values > 0, "#8e1b3a", "#9aa5b1")))
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Correlación de Pearson")
        st.plotly_chart(fig, width="stretch")

        st.subheader("Distribución por clase")
        var = st.selectbox("Variable", BASE_FEATURES, index=BASE_FEATURES.index("total_sulfur_dioxide"))
        fig = px.histogram(train, x=var, color=train["target"].map({0: "0 · estándar", 1: "1 · superior"}),
                           barmode="overlay", histnorm="probability density", opacity=0.6, nbins=60,
                           labels={"color": "clase"}, color_discrete_map=COLORS)
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")

    with right:
        st.subheader("Proyección PCA (PC1 vs PC2)")
        Z = pca_pipe.transform(X)
        evr = pca_pipe["pca"].explained_variance_ratio_
        dfz = pd.DataFrame({"PC1": Z[:, 0], "PC2": Z[:, 1],
                            "clase": y.map({0: "0 · estándar", 1: "1 · superior"})})
        fig = px.scatter(dfz.sort_values("clase"), x="PC1", y="PC2", color="clase", opacity=0.55,
                         color_discrete_map=COLORS,
                         labels={"PC1": f"PC1 ({evr[0]:.1%})", "PC2": f"PC2 ({evr[1]:.1%})"})
        fig.update_traces(marker_size=4)
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")

        st.subheader("Varianza explicada acumulada")
        cum = np.cumsum(evr)
        fig = go.Figure()
        fig.add_bar(x=list(range(1, len(evr) + 1)), y=evr, name="individual")
        fig.add_scatter(x=list(range(1, len(evr) + 1)), y=cum, mode="lines+markers", name="acumulada")
        fig.add_hline(y=0.95, line_dash="dash", annotation_text="95 %")
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="componente")
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Se necesitan {int(np.argmax(cum >= .95)) + 1} componentes para retener el 95 % "
                   f"de la varianza (de {len(evr)} variables tras el feature engineering).")

# ---------------------------------------------------------------- (b) comparativa
with tab_cmp:
    st.subheader("Progresión de modelos — F1-Macro en validación cruzada estratificada (5 folds)")
    order = ["Baseline k-means", "Baseline k-NN", "Feature Engineering", "PCA", "SVM",
             "MLP (Red Neuronal)", "k-NN ponderado", "Ensamble k-NN"]
    best_fam = (res.sort_values("f1_macro", ascending=False).groupby("familia").head(1)
                .set_index("familia").reindex([o for o in order if o in set(res.familia)]).reset_index())
    fig = px.bar(best_fam, x="f1_macro", y="familia", orientation="h", error_x="f1_std",
                 text=best_fam["f1_macro"].map("{:.4f}".format), hover_data=["modelo"],
                 color_discrete_sequence=["#8e1b3a"])
    fig.update_traces(textposition="inside", insidetextanchor="start", textfont_color="white")
    fig.update_layout(height=380, xaxis_range=[0.7, 0.9], margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="F1-Macro (mejor modelo de cada familia)", yaxis_title="")
    st.plotly_chart(fig, width="stretch")

    fam = st.multiselect("Filtrar familias", sorted(res.familia.unique()), default=sorted(res.familia.unique()))
    show = res[res.familia.isin(fam)].sort_values("f1_macro", ascending=False).rename(columns={
        "modelo": "Modelo", "familia": "Familia", "f1_macro": "F1-Macro", "f1_std": "Desv. est.",
        "fit_time_s": "Entrenamiento (s/fold)", "pred_time_s": "Predicción (s/fold)"})
    st.dataframe(show.style.format({"F1-Macro": "{:.4f}", "Desv. est.": "{:.4f}",
                                    "Entrenamiento (s/fold)": "{:.3f}", "Predicción (s/fold)": "{:.3f}"})
                 .background_gradient(subset=["F1-Macro"], cmap="Greens"),
                 width="stretch", hide_index=True, height=420)
    st.info("**Modelo final:** ensamble *soft voting* de 3 k-NN ponderados por 1/d "
            "(Manhattan k=40 · Coseno k=15 · Manhattan k=25 con features fisicoquímicas). "
            "Todo el preprocesamiento está dentro de Pipelines (sin data leakage).")

# ---------------------------------------------------------------- (c) simulador
with tab_sim:
    st.subheader("Simulador interactivo")
    st.write("Ajusta las propiedades fisicoquímicas o carga una muestra del conjunto de prueba.")

    bounds = {f: (float(X[f].quantile(0.001)), float(X[f].quantile(0.999))) for f in BASE_FEATURES}

    def set_sample(row):
        for f in BASE_FEATURES:
            lo, hi = bounds[f]
            st.session_state[f"sl_{f}"] = float(np.clip(row[f], lo, hi))

    if "sl_alcohol" not in st.session_state:
        set_sample(X.median())

    b1, b2, _ = st.columns([1, 1, 3])
    if b1.button("🎲 Muestra aleatoria del test"):
        set_sample(test.sample(1).iloc[0])
    if b2.button("↺ Valores medianos"):
        set_sample(X.median())

    cols = st.columns(3)
    values = {}
    for i, f in enumerate(BASE_FEATURES):
        lo, hi = bounds[f]
        step = float(np.round((hi - lo) / 200, 5)) or 0.0001
        values[f] = cols[i % 3].slider(f, lo, hi, step=step, key=f"sl_{f}",
                                       format="%.4f" if f == "density" else "%.3f")

    x_new = pd.DataFrame([values])[BASE_FEATURES]
    proba = float(model.predict_proba(x_new)[0, 1])
    pred = int(proba >= 0.5)

    r1, r2 = st.columns([1, 2])
    with r1:
        if pred:
            st.success(f"### Calidad SUPERIOR (1)\nProbabilidad: **{proba:.1%}**")
        else:
            st.warning(f"### Calidad ESTÁNDAR (0)\nProbabilidad de clase 1: **{proba:.1%}**")
        st.caption("Umbral de decisión 0.5 (la curva F1 vs umbral es plana alrededor de 0.5).")
    with r2:
        fig = go.Figure(go.Indicator(mode="gauge+number", value=proba * 100, number={"suffix": " %"},
                                     gauge={"axis": {"range": [0, 100]}, "threshold": {"value": 50, "line": {"color": "black"}},
                                            "bar": {"color": "#2a7ab9" if pred else "#c0504d"}},
                                     title={"text": "P(calidad superior)"}))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig, width="stretch")

    with st.expander("Ver los vecinos más cercanos en train (k-NN Manhattan)"):
        knn_pipe = model.named_estimators_["knn_manhattan_k40"]
        d, idx = knn_pipe["knn"].kneighbors(knn_pipe["sc"].transform(x_new), n_neighbors=8)
        neigh = train.iloc[idx[0]][["id"] + BASE_FEATURES + ["target"]].copy()
        neigh.insert(1, "distancia_L1", d[0].round(3))
        st.dataframe(neigh, hide_index=True, width="stretch")
        st.caption("Cada vecino vota con peso 1/d: si existe una muestra idéntica (d = 0), su etiqueta decide.")
