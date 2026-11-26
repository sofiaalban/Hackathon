"""Transformadores de feature engineering para el reto Wine Quality.

Todos son compatibles con sklearn.Pipeline, así que se ajustan SOLO con los
datos de entrenamiento de cada fold (sin data leakage).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans

BASE_FEATURES = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "pH", "sulphates", "alcohol",
]

# Variables con fuerte asimetría positiva -> log1p las acerca a una gaussiana
SKEWED = ["fixed_acidity", "volatile_acidity", "residual_sugar", "chlorides",
          "free_sulfur_dioxide", "total_sulfur_dioxide", "sulphates"]


class WineFeatures(BaseEstimator, TransformerMixin):
    """Crea variables con sentido fisicoquímico (transformación sin estado).

    - acid_ratio      = fixed_acidity / volatile_acidity  (acidez "buena" vs avinagrada)
    - free_so2_ratio  = free_SO2 / total_SO2              (fracción activa del SO2)
    - alcohol_density = alcohol / density                 (cuerpo vs grado alcohólico)
    - sugar_alcohol   = residual_sugar / alcohol          (dulzor relativo)
    - bound_so2       = total_SO2 - free_SO2              (SO2 ligado)
    - total_acidity   = fixed + volatile + citric
    Además aplica log1p a las variables muy sesgadas.
    """

    def __init__(self, log_transform=True, add_ratios=True):
        self.log_transform = log_transform
        self.add_ratios = add_ratios

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=BASE_FEATURES).copy()
        out = X.copy()
        if self.add_ratios:
            out["acid_ratio"] = X["fixed_acidity"] / (X["volatile_acidity"] + 1e-3)
            out["free_so2_ratio"] = X["free_sulfur_dioxide"] / (X["total_sulfur_dioxide"] + 1e-3)
            out["alcohol_density"] = X["alcohol"] / X["density"]
            out["sugar_alcohol"] = X["residual_sugar"] / X["alcohol"]
            out["bound_so2"] = X["total_sulfur_dioxide"] - X["free_sulfur_dioxide"]
            out["total_acidity"] = X["fixed_acidity"] + X["volatile_acidity"] + X["citric_acid"]
        if self.log_transform:
            for c in SKEWED + (["acid_ratio", "sugar_alcohol", "bound_so2"] if self.add_ratios else []):
                out[c] = np.log1p(out[c].clip(lower=0))
        return out.values

    def get_feature_names_out(self, input_features=None):
        names = list(BASE_FEATURES)
        if self.add_ratios:
            names += ["acid_ratio", "free_so2_ratio", "alcohol_density",
                      "sugar_alcohol", "bound_so2", "total_acidity"]
        return np.array(names)


class KMeansDistances(BaseEstimator, TransformerMixin):
    """Añade la distancia euclidiana a cada uno de los k centroides de k-means.

    Se ajusta dentro del Pipeline (después del escalado), por lo que los
    centroides se calculan solo con el fold de entrenamiento.
    """

    def __init__(self, n_clusters=6, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X, y=None):
        self.km_ = KMeans(n_clusters=self.n_clusters, n_init=10,
                          random_state=self.random_state).fit(X)
        return self

    def transform(self, X):
        return np.hstack([X, self.km_.transform(X)])
