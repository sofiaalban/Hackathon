# Pitch técnico — 5 minutos

## Guion (≈1 min por bloque)

**1. Problema y métrica (0:00–0:45)**
"Clasificamos vinos de calidad superior (16 %) frente a estándar (84 %). Con este desbalance, un modelo que siempre diga 'estándar' saca 84 % de accuracy y es inútil. Por eso optimizamos F1-Macro, el promedio sin ponderar del F1 de cada clase."

**2. EDA y decisiones de preprocesamiento (0:45–1:45)** → mostrar la pestaña *EDA y PCA*
- Escalas de 0.99 a 440: sin escalar, la distancia euclidiana es básicamente "distancia en SO₂". Por eso usamos StandardScaler dentro del Pipeline.
- Hallazgo clave: 807 filas duplicadas en train y 26 % del test con gemela exacta en train. La geometría local va a ser muy informativa.
- PCA: PC1 separa dos poblaciones químicas (perfil SO₂/azúcar frente a perfil acidez/sulfatos/cloruros, que es la firma blanco frente a tinto), y la clase 1 se concentra en una.

**3. Progresión de modelos (1:45–3:30)** → mostrar la pestaña *Comparativa*
- Baseline k-NN uniforme: 0.807. k-means por mayoría: 0.795, por debajo porque no usa la etiqueta al formar los clústeres.
- Feature engineering: 6 ratios fisicoquímicos + log1p + distancias a centroides k-means.
- PCA 95 % (9 componentes): ayuda a k-NN con features redundantes, pero le quita al SVM porque es no supervisado. No lo usamos en el modelo final.
- SVM RBF: 0.817. MLP (128,128, tanh, L2): 0.831, la mejor de las "avanzadas", aunque es la más lenta (~12 s/fold) y la más inestable (±0.02).
- El salto: k-NN **ponderado por 1/d**, 0.855. Ensamble de 3 k-NN (Manhattan, Coseno, Manhattan+FE): **0.860 ± 0.008**, el mejor y el más estable.

**4. Por qué gana el k-NN ponderado (3:30–4:30)**
- Con peso $w_i = 1/d_i$, una muestra a distancia ≈ 0 domina el voto. Recupera duplicados y casi-duplicados sin memorizar ruido.
- Así podemos usar un k grande (40), que da menos varianza, sin perder resolución local. Es un mejor equilibrio sesgo–varianza que k = 3.
- Las tres distancias ven cosas distintas: L1 (robusta a outliers), coseno (proporciones de la composición, no su magnitud) y L1 sobre features químicas. Promediar probabilidades reduce la varianza.
- Umbral: la curva F1 vs umbral es plana en 0.5, así que no lo sobreajustamos.

**5. Cierre (4:30–5:00)** → mostrar el *Simulador* con una muestra aleatoria y los vecinos más cercanos
"Todo vive en Pipelines (sin leakage), validado con StratifiedKFold, y supera el baseline por +0.053 de F1-Macro."

---

## Preguntas probables del docente

**¿Por qué no ganó la MLP si es un aproximador universal?**
Ser aproximador universal habla de capacidad, no de generalización con 5k muestras. La señal dominante aquí es local (vecindarios casi idénticos). La MLP aprende una frontera suave y global y regulariza justo esas irregularidades locales. Además tiene más hiperparámetros y más varianza entre folds (±0.02 frente a ±0.008).

**¿No es trampa aprovechar los duplicados?**
No. El test viene de la misma distribución (partición aleatoria 80/20), así que los duplicados también existen en producción. k-NN con peso 1/d es un método permitido y la CV estratificada (que también tiene duplicados entre folds) estima de forma honesta su desempeño en el test oculto.

**¿Cómo evitaron data leakage?**
El escalador, el PCA, el k-means de las features geométricas y los transformadores de FE están dentro de `Pipeline`. En cada fold se ajustan solo con los datos de entrenamiento. El umbral se evaluó con predicciones out-of-fold.

**¿Por qué Manhattan sobre Euclidiana?**
$L_1$ suma diferencias absolutas, así que un outlier en una sola variable pesa linealmente y no cuadráticamente. Con variables sesgadas (chlorides, sulphates) eso la hace más robusta.

**¿Qué hace γ en el SVM RBF?**
$K(x,z)=e^{-\gamma\|x-z\|^2}$: γ es el inverso del radio de influencia de cada vector de soporte. Con γ alto (≥3) la frontera rodea cada punto y sobreajusta (F1 cae a ≈0.69). C controla cuánto penalizamos violar el margen.

**¿Por qué no `class_weight='balanced'`?**
Lo probamos en SVM. Sube el recall de la clase 1 pero baja su precisión, y en F1-Macro el efecto neto fue nulo o negativo. En k-NN el desbalance se manejó vía probabilidades y la revisión de umbral.

**¿Qué ganaron con PCA?**
Interpretabilidad (PC1 = eje químico blanco/tinto) y menos dimensiones (17 → 9 con 95 %). Para k-NN, decorrelacionar las features redundantes mejoró el F1. Para SVM lo empeoró, así que no entra al modelo final.
