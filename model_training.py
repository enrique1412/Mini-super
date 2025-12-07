import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report
import joblib
import os

# ------------------------------
# Cargar datos
# ------------------------------
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

hist_path = "data/historico_inventario.csv"
catalogo_path = "data/catalogo_productos.csv"

hist = pd.read_csv(hist_path, parse_dates=["fecha"])
catalogo = pd.read_csv(catalogo_path)

# ------------------------------
# Feature engineering
# ------------------------------
def season_factor(val):
    return {"alta": 1.2, "media": 1.0, "baja": 0.85}.get(str(val).lower(), 1.0)

def trend_factor(val):
    return {"subiendo": 1.15, "estable": 1.0, "bajando": 0.9}.get(str(val).lower(), 1.0)

hist = hist.sort_values(["producto", "fecha"])
hist["ventas_promedio_dia"] = hist.groupby("producto")["unidades_vendidas"].rolling(7, min_periods=1).mean().reset_index(0, drop=True)

hist["factor_estacionalidad"] = hist["estacionalidad"].apply(season_factor)
hist["factor_tendencia"] = hist["tendencia"].apply(trend_factor)
hist["dlt_ajustada"] = hist["ventas_promedio_dia"] * hist["tiempo_entrega_dias"] * hist["factor_estacionalidad"] * hist["factor_tendencia"]

std_ventas = hist.groupby("producto")["unidades_vendidas"].transform(lambda s: s.rolling(14, min_periods=1).std().fillna(0))
k_nivel_servicio = 1.28
hist["stock_seguridad"] = (k_nivel_servicio * std_ventas * np.sqrt(hist["tiempo_entrega_dias"].clip(lower=0.0001))).clip(lower=0)
hist["punto_reorden"] = (hist["dlt_ajustada"] + hist["stock_seguridad"]).round()

hist["hacer_pedido"] = (hist["inventario"] <= hist["punto_reorden"]).astype(int)

hist = hist.merge(catalogo[["producto", "es_basico", "stock_minimo"]], on="producto", how="left")

def cantidad_objetivo(row):
    base = max(row["dlt_ajustada"] + row["stock_seguridad"] - row["inventario"], 0)
    if row.get("es_basico", 0) == 1:
        return float(max(base, row.get("stock_minimo", 0)))
    return float(base)

hist["cantidad_sugerida_obj"] = hist.apply(cantidad_objetivo, axis=1)

# ------------------------------
# Features y etiquetas
# ------------------------------
feature_cols = ["inventario","ventas_promedio_dia","tiempo_entrega_dias","estacionalidad","tendencia","precio","proyeccion_eventos"]
X = hist[feature_cols].copy()
y_cls = hist["hacer_pedido"].copy()
y_reg = hist["cantidad_sugerida_obj"].copy()

cat_cols = ["estacionalidad","tendencia"]
num_cols = ["inventario","ventas_promedio_dia","tiempo_entrega_dias","precio","proyeccion_eventos"]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols),
    ]
)

clf = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    min_samples_leaf=20,
    class_weight={0:1.0, 1:2.0},
    random_state=42
)
pipe_cls = Pipeline(steps=[("prep", preprocess), ("model", clf)])

reg = RandomForestRegressor(
    n_estimators=200,
    max_depth=6,
    random_state=42,
    n_jobs=-1
)
pipe_reg = Pipeline(steps=[("prep", preprocess), ("model", reg)])

X_train, X_test, y_cls_train, y_cls_test = train_test_split(X, y_cls, test_size=0.2, random_state=42, stratify=y_cls)
_, _, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)

pipe_cls.fit(X_train, y_cls_train)
pipe_reg.fit(X_train, y_reg_train)

y_pred = pipe_cls.predict(X_test)
print("Reporte de clasificación:")
print(classification_report(y_cls_test, y_pred, digits=3))

joblib.dump(pipe_cls, "models/modelo_clasificacion.pkl")
joblib.dump(pipe_reg, "models/modelo_regresion.pkl")
print("Modelos guardados en /models")
