# 🍷 Hackathon IA — Clasificación de calidad de vino

Clasificación binaria (calidad **superior = 1** vs **estándar = 0**) sobre Wine Quality (5,197 muestras de train, 1,300 de test, 11 variables fisicoquímicas).
**Métrica:** F1-Score Macro · **Validación:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
Solo se usan los métodos permitidos: distancias, k-means, k-NN, PCA, SVM y MLP.

## Resultado

| | |
|---|---|
| **Modelo final** | Ensamble *soft voting* de 3 k-NN ponderados por distancia (1/d) |
| **F1-Macro CV** | **0.860 ± 0.008** |
| Baseline (k-NN k=3 Manhattan) | 0.807 ± 0.016 |
| Mejora sobre el baseline | **+0.053** |

## Estructura del repositorio

```
├── app/app.py                         # Tablero Streamlit (EDA/PCA, comparativa, simulador)
├── data/                              # train_public.csv, test_public_features.csv
├── entregas/
│   ├── submission.csv                 # id, prediction  (1,300 filas + encabezado)
│   └── resultados_cv.csv              # tabla completa de métricas en CV
├── models/                            # modelo_final.joblib, pca_pipeline.joblib
├── notebooks/01_eda_y_modelado.ipynb  # notebook ejecutado con toda la progresión
├── src/
│   ├── features.py                    # transformadores: WineFeatures, KMeansDistances
│   ├── model.py                       # definición del modelo final
│   └── build_notebook.py              # genera el notebook
├── PITCH.md                           # guion del pitch de 5 min + preguntas probables
└── requirements.txt
```

## Cómo ejecutar

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda_y_modelado.ipynb   # ~5 min
streamlit run app/app.py
```
Si la versión de scikit-learn no coincide con la de los `.joblib`, la app reentrena el modelo automáticamente (k-NN tarda milisegundos).

### Verificar que todo esté bien
```bash
python verificar.py
```
Revisa la estructura del repo, el formato de `submission.csv` (columnas, 1,300 filas, sin nulos, ids alineados), que el modelo reentrenado reproduzca exactamente el submission y el F1-Macro en CV. Debe terminar en `✅ TODO CORRECTO`.

### Google Colab
Abre `notebooks/00_colab_verificar_y_app.ipynb` en Colab (`Archivo → Subir notebook`, o desde GitHub). Sube `hack_entrega.zip` cuando lo pida, o pon la URL del repo en `REPO_URL`. El notebook verifica todo y levanta la app con un enlace público `trycloudflare.com`.
Cuando el repo esté en GitHub, este badge lo abre directo (reemplaza `USUARIO/REPO`):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USUARIO/REPO/blob/main/notebooks/00_colab_verificar_y_app.ipynb)

## Enfoque matemático (resumen ejecutivo)

1. **EDA.** Sin nulos. Desbalance 84/16, por eso F1-Macro y no accuracy. Escalas muy dispares (density ≈ 0.99 vs total SO₂ ≤ 440) y asimetrías fuertes (chlorides, skew ≈ 5.6). Hay **807 filas duplicadas** en train, y el **26 % del test** tiene una gemela exacta en train.
2. **Escalado.** `StandardScaler` dentro del Pipeline. Sin él, la distancia $d(x,z)=\sqrt{\sum_j (x_j-z_j)^2}$ queda dominada por las variables de mayor rango, y el gradiente de la MLP se vuelve mal condicionado.
3. **Baseline (obligatorio).** k-NN con k ∈ {3, 5, 7} × {Euclidiana, Manhattan}, más k-means con asignación de clase por mayoría del clúster.
4. **Feature engineering** (`src/features.py`): `acid_ratio` = fixed/volatile, `free_so2_ratio` = free/total SO₂, `alcohol_density`, `sugar_alcohol`, `bound_so2`, `total_acidity`, `log1p` para las variables sesgadas, y la **distancia a los centroides de k-means** (feature geométrica). Todo se ajusta dentro del Pipeline, así que no hay *data leakage*.
5. **PCA.** 9 de 17 componentes retienen el 95 % de la varianza. Mejora k-NN sobre features redundantes, pero empeora el SVM: PCA no es supervisado y descarta direcciones discriminantes. Se usa para visualizar, no en el modelo final.
6. **SVM** (GridSearch sobre kernel lineal, polinomial y RBF, C, γ y `class_weight`): el mejor es RBF, C = 3 (0.817).
7. **MLP** (GridSearch sobre arquitectura, activación y α): la mejor es (128, 128) con `tanh`, α = 1e-4 (0.831). *Early stopping* empeoró el resultado porque quita datos de entrenamiento.
8. **k-NN ponderado por 1/d.** Explota la estructura local (duplicados y casi-duplicados). Un $k$ grande da estabilidad y el peso 1/d conserva la información local.
9. **Ensamble soft voting** de k-NN Manhattan (k = 40), Coseno (k = 15) y Manhattan + FE (k = 25): promedia $\hat p(y=1\mid x)$ de tres geometrías distintas, lo que reduce la varianza. El análisis de umbral *out-of-fold* muestra una curva plana alrededor de 0.5, así que se deja el umbral por defecto.

## Resultados en validación cruzada (StratifiedKFold 5)

| Familia | Modelo | F1-Macro | ± std | Fit (s/fold) |
|---|---|---|---|---|
| Baseline k-means | k-means (k=60), mayoría, umbral 0.3 | 0.7954 | 0.0120 | 0.29 |
| Baseline k-NN | k-NN k=3 Euclidiana | 0.8034 | 0.0146 | 0.02 |
| Baseline k-NN | k-NN k=3 Manhattan | 0.8066 | 0.0165 | 0.01 |
| Baseline k-NN | k-NN k=7 Euclidiana / Manhattan | 0.8008 / 0.7985 | 0.005 / 0.015 | 0.01 |
| Feature Eng. | k-NN k=7 Manhattan + FE + dist. k-means | 0.7939 | 0.0098 | 0.09 |
| PCA | PCA 95 % + k-NN k=7 Manhattan | 0.8033 | 0.0111 | 0.02 |
| PCA | PCA 95 % + SVM RBF | 0.8078 | 0.0101 | 0.22 |
| SVM | SVM polinomial (mejor del grid) | 0.7903 | 0.0163 | 0.95 |
| SVM | SVM lineal (mejor del grid) | 0.8068 | 0.0104 | 0.21 |
| SVM | SVM RBF C=3 (mejor del grid) | 0.8170 | 0.0140 | 0.14 |
| MLP | MLP (128,128) tanh α=1e-4 + FE | 0.8310 | 0.0205 | 11.8 |
| k-NN ponderado | k-NN k=15 Coseno, 1/d | 0.8517 | 0.0172 | 0.01 |
| k-NN ponderado | k-NN k=25 Manhattan + FE, 1/d | 0.8527 | 0.0115 | 0.01 |
| k-NN ponderado | k-NN k=40 Manhattan, 1/d | 0.8552 | 0.0092 | 0.01 |
| **Ensamble** | **Soft voting de 3 k-NN** | **0.8599** | **0.0077** | 0.03 |

La tabla completa (29 configuraciones) está en `entregas/resultados_cv.csv`.

## Roles sugeridos (equipo de 4)
- **Data/EDA:** sección 1 del notebook y visualizaciones.
- **Baseline + FE + PCA:** secciones 2–4 y `src/features.py`.
- **SVM + MLP + ensamble:** secciones 5–7 y `submission.csv`.
- **App + README + pitch:** `app/app.py`, este README y `PITCH.md`.

Hagan *commits* desde la cuenta de cada integrante para que el historial sea colaborativo (la rúbrica lo evalúa).
