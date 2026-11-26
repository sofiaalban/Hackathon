"""Verificación rápida de los entregables.  Uso:  python verificar.py

Revisa estructura del repo, formato de submission.csv, que el modelo se pueda
reconstruir y reproduzca exactamente el submission, y el F1-Macro en CV.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

OK, FAIL = "  [OK]  ", "  [FALLA]"
errores = 0


def check(cond, msg):
    global errores
    print((OK if cond else FAIL), msg)
    errores += 0 if cond else 1
    return cond


print("\n1) Estructura del repositorio")
for f in ["notebooks/01_eda_y_modelado.ipynb", "app/app.py", "entregas/submission.csv", "README.md",
          "data/train_public.csv", "data/test_public_features.csv", "src/features.py", "src/model.py",
          "entregas/resultados_cv.csv", "requirements.txt"]:
    check((ROOT / f).exists(), f)

print("\n2) Librerías")
try:
    import pandas as pd, numpy as np, sklearn  # noqa: E401
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    check(True, f"pandas {pd.__version__} · numpy {np.__version__} · scikit-learn {sklearn.__version__}")
except ImportError as e:
    check(False, f"falta una librería: {e}  ->  pip install -r requirements.txt")
    sys.exit(1)
for lib in ["streamlit", "plotly", "matplotlib", "seaborn"]:
    try:
        __import__(lib)
        check(True, lib)
    except ImportError:
        check(False, f"{lib} no instalado  ->  pip install -r requirements.txt")

from features import BASE_FEATURES  # noqa: E402
from model import build_final_model  # noqa: E402

train = pd.read_csv(ROOT / "data/train_public.csv")
test = pd.read_csv(ROOT / "data/test_public_features.csv")
sub = pd.read_csv(ROOT / "entregas/submission.csv")

print("\n3) Formato de entregas/submission.csv")
check(list(sub.columns) == ["id", "prediction"], f"columnas exactas id, prediction  -> {list(sub.columns)}")
check(len(sub) == 1300, f"1,300 observaciones (+ encabezado = 1,301 líneas)  -> {len(sub)}")
check(sub.isna().sum().sum() == 0, "sin valores nulos")
check(set(sub["prediction"].unique()) <= {0, 1}, "predicciones binarias 0/1")
check((sub["id"].values == test["id"].values).all(), "ids alineados con test_public_features.csv")
print(f"         % de clase 1 predicha: {sub.prediction.mean():.1%}  (train: {train.target.mean():.1%})")

print("\n4) Reproducibilidad del modelo final")
t = time.time()
model = build_final_model().fit(train[BASE_FEATURES], train["target"])
pred = model.predict(test[BASE_FEATURES])
check((pred == sub["prediction"].values).all(),
      f"el modelo reentrenado reproduce el submission ({(pred == sub.prediction.values).mean():.2%} igual, {time.time()-t:.2f}s)")

print("\n5) F1-Macro en validación cruzada estratificada (5 folds)")
cv = StratifiedKFold(5, shuffle=True, random_state=42)
s = cross_val_score(build_final_model(), train[BASE_FEATURES], train["target"], cv=cv, scoring="f1_macro")
check(s.mean() > 0.85, f"F1-Macro = {s.mean():.4f} ± {s.std():.4f}   (por fold: {np.round(s, 4).tolist()})")

print("\n" + ("✅ TODO CORRECTO" if errores == 0 else f"❌ {errores} verificación(es) fallaron"))
sys.exit(1 if errores else 0)
