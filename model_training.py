import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report

# ------------------------------
# Crear carpetas si no existen
# ------------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ------------------------------
# Cargar datos
# ------------------------------
df = pd.read_csv("data/historico_inventario.csv", parse_dates=["fecha"])
catalogo = pd.read_csv("data/catalogo_productos.csv")

# ------------------------------
# Feature engineering
# ------------------------------
def season_factor(val):
    return {"alta": 1.2, "media": 1.0, "baja": 0.85}.get(str(val).lower(), 1.0)

def trend_factor(val):
    return {"subiendo": 1.15, "estable": 1.0, "bajando": 0.9}.get(str(val).lower(), 1.0)

df = df.sort_values(["producto", "fecha"])
df["ventas_promedio_dia"] = df.groupby("producto")["unidades_vendidas"].rolling(7, min_periods=1).mean().reset_index(0, drop=True)

df["factor_estacionalidad"] = df["estacionalidad"].apply(season_factor)
df["factor_tendencia"] = df["tendencia"].apply(trend_factor)
df["dlt_ajustada"] = df["ventas_promedio_dia"] * df["tiempo_entrega_dias"] * df["factor_estacionalidad"] * df["factor_tendencia"]

std_ventas = df.groupby("producto")["unidades_vendidas"].transform(lambda s: s.rolling(14, min_periods=1).std().fillna(0))
k = 1.28  # nivel de servicio ~90%
df["stock_seguridad"] = (k * std_ventas * np.sqrt(df["tiempo_entrega_dias"].clip(lower=0.0001))).clip(lower=0)
df["punto_reorden"] = (df["dlt_ajustada"] + df["stock_seguridad"]).round()

df["hacer_pedido"] = (df["inventario"] <= df["punto_reorden"]).astype(int)

df = df.merge(catalogo[["producto", "es_basico", "stock_minimo"]], on="producto", how="left")

def cantidad_objetivo(row):
    base = max(row["dlt_ajustada"] + row["stock_seguridad"] - row["inventario"], 0)
    if row.get("es_basico", 0) == 1:
        return float(max(base, row.get("stock_minimo", 0)))
    return float(base)

df["cantidad_sugerida_obj"] = df.apply(cantidad_objetivo, axis=1)

# ------------------------------
# Preparar datos para modelado
# ------------------------------
features = ["inventario", "ventas_promedio_dia", "tiempo_entrega_dias", "estacionalidad", "tendencia", "precio", "proyeccion_eventos"]
X = df[features].copy()
y_cls = df["hacer_pedido"]
y_reg = df["cantidad_sugerida_obj"]

cat_cols = ["estacionalidad", "tendencia"]
num_cols = ["inventario", "ventas_promedio_dia", "tiempo_entrega_dias", "precio", "proyeccion_eventos"]

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", StandardScaler(), num_cols)
])

# ------------------------------
# Modelos
# ------------------------------
clf = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    min_samples_leaf=20,
    class_weight={0: 1.0, 1: 2.0},  # penaliza falsos negativos
    random_state=42
)
pipe_cls = Pipeline([("prep", preprocess), ("model", clf)])

reg = RandomForestRegressor(
    n_estimators=200,
    max_depth=6,
    random_state=42,
    n_jobs=-1
)
pipe_reg = Pipeline([("prep", preprocess), ("model", reg)])

# ------------------------------
# Entrenamiento
# ------------------------------
X_train, X_test, y_cls_train, y_cls_test = train_test_split(X, y_cls, test_size=0.2, random_state=42, stratify=y_cls)
_, _, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)

pipe_cls.fit(X_train, y_cls_train)
pipe_reg.fit(X_train, y_reg_train)

# ------------------------------
# Evaluación
# ------------------------------
y_pred = pipe_cls.predict(X_test)
print("Evaluación del modelo de clasificación:")
print(classification_report(y_cls_test, y_pred, digits=3))

# ------------------------------
# Guardar modelos
# ------------------------------
joblib.dump(pipe_cls, "models/modelo_clasificacion.pkl")
joblib.dump(pipe_reg, "models/modelo_regresion.pkl")
joblib.dump(preprocess, "models/pipeline_features.pkl")
print("✅ Modelos guardados en /models")
