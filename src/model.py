"""Definición del modelo final (idéntica a la del notebook) para reconstruirlo si hace falta."""
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import WineFeatures


def build_final_model():
    """Ensamble soft-voting de 3 k-NN ponderados por 1/d."""
    return VotingClassifier([
        ("knn_manhattan_k40", Pipeline([("sc", StandardScaler()),
                                        ("knn", KNeighborsClassifier(40, p=1, weights="distance"))])),
        ("knn_coseno_k15", Pipeline([("sc", StandardScaler()),
                                     ("knn", KNeighborsClassifier(15, metric="cosine", algorithm="brute",
                                                                  weights="distance"))])),
        ("knn_FE_manhattan_k25", Pipeline([("fe", WineFeatures()), ("sc", StandardScaler()),
                                           ("knn", KNeighborsClassifier(25, p=1, weights="distance"))])),
    ], voting="soft")


def build_pca_pipeline():
    return Pipeline([("fe", WineFeatures()), ("sc", StandardScaler()), ("pca", PCA())])
