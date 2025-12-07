import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from utils_reports import export_excel, export_pdf
from utils_dashboard import calcular_metricas_basicas, preparar_series_promedio

# ------------------------------
# Configuración general
# ------------------------------
st.set_page_config(page_title="SATD Pro", page_icon="🛒", layout="wide")

# ------------------------------
# Usuario por defecto (sin autenticación)
# ------------------------------
# Cambia aquí el perfil por defecto si quieres simular "Compras"
rol = "Admin"   # "Admin" o "Compras"
usuario = "admin"
nombre = "Administrador"
st.sidebar.success(f"Bienvenido, {nombre}")

# ------------------------------
# Cargar modelos
# ------------------------------
try:
    pipe_cls = joblib.load("models/modelo_clasificacion.pkl")
    pipe_reg = joblib.load("models/modelo_regresion.pkl")
except Exception as e:
    st.error(f"No se pudieron cargar los modelos. Verifica la carpeta /models. Detalle: {e}")
    st.stop()

# ------------------------------
# Menú lateral
# ------------------------------
st.sidebar.title("Menú")
menu = st.sidebar.radio(
    "Selecciona",
    ["🏠 Bienvenida", "🛒 Predicciones", "📊 Dashboard", "📑 Reportes", "📦 Catálogo", "📂 Subir archivo", "⚙️ Configuración", "📥 Descargar ejemplo"]
)

# ------------------------------
# Utilidades internas
# ------------------------------
def season_factor(estacionalidad: str) -> float:
    return {"alta": 1.2, "media": 1.0, "baja": 0.85}.get(str(estacionalidad).lower(), 1.0)

def trend_factor(tendencia: str) -> float:
    return {"subiendo": 1.15, "estable": 1.0, "bajando": 0.9}.get(str(tendencia).lower(), 1.0)

def decision_y_cantidad(inventario, ventas_prom, tiempo_entrega, estacionalidad, tendencia, precio, proyeccion_eventos, k_ss=1.28):
    entrada = pd.DataFrame([{
        "inventario": inventario,
        "ventas_promedio_dia": ventas_prom,
        "tiempo_entrega_dias": tiempo_entrega,
        "estacionalidad": estacionalidad,
        "tendencia": tendencia,
        "precio": precio,
        "proyeccion_eventos": proyeccion_eventos
    }])

    pred_bin = int(pipe_cls.predict(entrada)[0])
    cantidad_pred = max(0.0, float(pipe_reg.predict(entrada)[0]))

    # Punto de reorden ROP (explicativo)
    f_est = season_factor(estacionalidad)
    f_trend = trend_factor(tendencia)
    dlt_aj = ventas_prom * tiempo_entrega * f_est * f_trend + proyeccion_eventos
    ss = max(0.0, k_ss * (ventas_prom * 0.2) * np.sqrt(max(tiempo_entrega, 0.0001)))  # aproximación simple
    rop = round(dlt_aj + ss)

    hacer_pedido_regla = int(inventario <= rop)
    hacer_pedido_final = 1 if (pred_bin == 1 or hacer_pedido_regla == 1) else 0

    cantidad_por_rop = max(0.0, (dlt_aj + ss) - inventario)
    cantidad_final = float(np.ceil(max(cantidad_pred, cantidad_por_rop, 0.0)))

    return {
        "pred_bin": pred_bin,
        "hacer_pedido_regla": hacer_pedido_regla,
        "hacer_pedido_final": hacer_pedido_final,
        "cantidad_pred": cantidad_pred,
        "cantidad_final": cantidad_final,
        "dlt_aj": dlt_aj,
        "ss": ss,
        "rop": rop
    }

# ------------------------------
# Pantalla de bienvenida
# ------------------------------
if menu == "🏠 Bienvenida":
    st.title("SATD Pro – Sistema de Apoyo a la Toma de Decisiones")
    st.markdown("""
    Optimiza la planeación de compras y reabastecimiento para tu mini supermercado:
    - **Predicciones**: decisión de pedido (Sí/No) y cantidad sugerida.
    - **Dashboard**: métricas clave y tendencias.
    - **Reportes**: exportación a PDF y Excel.
    - **Catálogo**: gestión de productos.
    - **Subir archivo**: análisis en lote con IA.
    - **Configuración**: parámetros del modelo.
    """)

# ------------------------------
# Predicciones individuales
# ------------------------------
elif menu == "🛒 Predicciones":
    st.header("Predicción de reabastecimiento (individual)")
    col1, col2, col3 = st.columns(3)

    with col1:
        producto = st.text_input("Producto", "Leche 1L")
        inventario = st.number_input("Inventario actual (unidades)", min_value=0, value=35)
        ventas_prom = st.number_input("Ventas promedio por día", min_value=0.0, value=12.0)
    with col2:
        tiempo_entrega = st.number_input("Tiempo de entrega (días)", min_value=0.0, value=2.0)
        estacionalidad = st.selectbox("Estacionalidad", ["alta", "media", "baja"], index=0)
        tendencia = st.selectbox("Tendencia", ["subiendo", "estable", "bajando"], index=0)
    with col3:
        precio = st.number_input("Precio (MXN)", min_value=0.0, value=25.0)
        proyeccion_eventos = st.number_input("Proyección de eventos (extra demanda)", min_value=0.0, value=0.0)

    k_ss = st.slider("Nivel de servicio (k para stock de seguridad)", 0.5, 3.0, 1.28, 0.01)

    resultados = decision_y_cantidad(
        inventario, ventas_prom, tiempo_entrega, estacionalidad, tendencia, precio, proyeccion_eventos, k_ss=k_ss
    )

    texto = "HACER PEDIDO" if resultados["hacer_pedido_final"] == 1 else "NO HACER PEDIDO"
    color = "#34A853" if resultados["hacer_pedido_final"] == 1 else "#5F6368"
    st.markdown(f"<h2 style='color:{color}'>{texto}</h2>", unsafe_allow_html=True)

    st.subheader("Cantidad sugerida")
    st.write(f"{int(resultados['cantidad_final'])} unidades")

    st.subheader("Justificación")
    st.write(f"- Inventario: {inventario}")
    st.write(f"- DLT ajustada: {resultados['dlt_aj']:.2f}")
    st.write(f"- Stock de seguridad (aprox.): {resultados['ss']:.2f}")
    st.write(f"- Punto de reorden (ROP): {resultados['rop']}")
    st.write(f"- Regla ROP sugiere pedido: {'Sí' if resultados['hacer_pedido_regla'] else 'No'}")
    st.write(f"- Clasificador sugiere pedido: {'Sí' if resultados['pred_bin']==1 else 'No'}")

    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=inventario,
        title={"text": "Inventario actual"},
        delta={"reference": resultados['rop'], "valueformat": ".0f",
               "increasing": {"color": "red"}, "decreasing": {"color": "green"}}
    ))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Descargar reporte")
    resumen = {
        "Producto": producto,
        "Decisión": texto,
        "Cantidad sugerida (unidades)": int(resultados['cantidad_final']),
        "Inventario actual": inventario,
        "DLT ajustada": f"{resultados['dlt_aj']:.2f}",
        "Stock de seguridad (aprox.)": f"{resultados['ss']:.2f}",
        "Punto de reorden (ROP)": resultados['rop'],
        "Clasificador": "Hacer" if resultados['pred_bin']==1 else "No Hacer",
        "Regla ROP": "Hacer" if resultados['hacer_pedido_regla']==1 else "No Hacer"
    }
    df_reporte = pd.DataFrame([{
        "producto": producto,
        "decision": texto,
        "cantidad_sugerida": int(resultados['cantidad_final']),
        "inventario": inventario,
        "dlt_ajustada": resultados['dlt_aj'],
        "stock_seguridad_aprox": resultados['ss'],
        "rop": resultados['rop'],
        "clasificador": "Hacer" if resultados['pred_bin']==1 else "No Hacer",
        "regla_rop": "Hacer" if resultados['hacer_pedido_regla']==1 else "No Hacer"
    }])

    colA, colB = st.columns(2)
    with colA:
        if st.button("Exportar Excel"):
            path = export_excel(df_reporte, "reporte_satd_individual.xlsx")
            with open(path, "rb") as f:
                st.download_button("Descargar Excel", f, file_name="reporte_satd_individual.xlsx")
    with colB:
        if st.button("Exportar PDF"):
            path = export_pdf(resumen, "reporte_satd_individual.pdf")
            with open(path, "rb") as f:
                st.download_button("Descargar PDF", f, file_name="reporte_satd_individual.pdf", mime="application/pdf")

# ------------------------------
# Dashboard de métricas
# ------------------------------
elif menu == "📊 Dashboard":
    st.header("Dashboard de métricas")
    try:
        hist = pd.read_csv("data/historico_inventario.csv", parse_dates=["fecha"])
        st.dataframe(hist.tail(50), use_container_width=True)

        # Métricas básicas (utils_dashboard)
        met = calcular_metricas_basicas(hist)
        st.metric("Nivel de servicio", f"{met['nivel_servicio']:.1f}%")
        st.metric("Rotación promedio (unid/día)", f"{met['rotacion_prom']:.2f}")
        st.metric("Productos críticos (ROP excedido)", f"{met['productos_criticos']}")

        # Inventario histórico
        fig_inv = px.line(hist, x="fecha", y="inventario", color="producto", title="Inventario histórico")
        st.plotly_chart(fig_inv, use_container_width=True)

        # Ventas promedio móviles
        hist_prom = preparar_series_promedio(hist)
        fig_vent = px.line(hist_prom, x="fecha", y="ventas_prom_7d", color="producto", title="Ventas promedio móvil (7 días)")
        st.plotly_chart(fig_vent, use_container_width=True)
    except Exception as e:
        st.warning(f"No se pudo cargar histórico: {e}")

# ------------------------------
# Reportes históricos
# ------------------------------
elif menu == "📑 Reportes":
    st.header("Reportes y análisis histórico")
    try:
        hist = pd.read_csv("data/historico_inventario.csv", parse_dates=["fecha"])
        st.dataframe(hist, use_container_width=True)

        prod_sel = st.selectbox("Producto", sorted(hist["producto"].unique()))
        dfp = hist[hist["producto"] == prod_sel].copy()
        dfp["ventas_prom_7d"] = dfp["unidades_vendidas"].rolling(7, min_periods=1).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dfp["fecha"], y=dfp["inventario"], name="Inventario"))
        fig.add_trace(go.Scatter(x=dfp["fecha"], y=dfp["ventas_prom_7d"], name="Ventas promedio 7d"))
        st.plotly_chart(fig, use_container_width=True)

        if st.button("Exportar Excel (histórico)"):
            path = export_excel(dfp, "reporte_hist_producto.xlsx")
            with open(path, "rb") as f:
                st.download_button("Descargar Excel", f, file_name="reporte_hist_producto.xlsx")
    except Exception as e:
        st.warning(f"No se pudo cargar histórico: {e}")

# ------------------------------
# Catálogo de productos
# ------------------------------
elif menu == "📦 Catálogo":
    st.header("Catálogo de productos")
    try:
        catalogo = pd.read_csv("data/catalogo_productos.csv")
        st.dataframe(catalogo, use_container_width=True)

        if rol == "Admin":
            st.subheader("Editar catálogo")
            with st.form("form_nuevo_prod", clear_on_submit=True):
                nuevo_prod = st.text_input("Nombre del producto")
                categoria = st.text_input("Categoría", value="General")
                es_basico = st.selectbox("Producto básico", [0, 1], index=0)
                stock_minimo = st.number_input("Stock mínimo", min_value=0, value=10)
                proveedor = st.text_input("Proveedor", value="Proveedor X")
                lead_time = st.number_input("Tiempo de entrega promedio (días)", min_value=0, value=2)
                submitted = st.form_submit_button("Agregar")

            if submitted and nuevo_prod.strip():
                catalogo.loc[len(catalogo)] = [nuevo_prod, categoria, es_basico, stock_minimo, proveedor, lead_time]
                catalogo.to_csv("data/catalogo_productos.csv", index=False)
                st.success(f"Producto agregado: {nuevo_prod}")
    except Exception as e:
        st.warning(f"No se pudo cargar catálogo: {e}")

# ------------------------------
# Subir archivo para análisis en lote con IA
# ------------------------------
elif menu == "📂 Subir archivo":
    st.header("Subir archivo de inventario/ventas (CSV/Excel)")
    uploaded_file = st.file_uploader("Carga tu archivo", type=["csv","xlsx"])

    if uploaded_file is not None:
        # Leer archivo
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("Archivo cargado correctamente ✅")
        st.dataframe(df.head(), use_container_width=True)

        # Cargar catálogo
        catalogo = pd.read_csv("data/catalogo_productos.csv")

        # Enriquecer datos con catálogo
        df = df.merge(catalogo[["producto","lead_time","precio"]], on="producto", how="left")
        df["tiempo_entrega_dias"] = df["lead_time"].fillna(2)
        df["precio"] = df["precio"].fillna(25)

        # Calcular estacionalidad y tendencia automáticamente
        mes = pd.Timestamp.today().month
        if mes in [11,12,1]:
            df["estacionalidad"] = "alta"
        elif mes in [6,7,8]:
            df["estacionalidad"] = "media"
        else:
            df["estacionalidad"] = "baja"

        df["tendencia"] = np.where(df["ventas_promedio_dia"].diff().fillna(0) > 0, "subiendo", "estable")
        df["proyeccion_eventos"] = 0

        # Aplicar modelo a cada producto
        resultados = []
        for _, row in df.iterrows():
            res = decision_y_cantidad(
                row["inventario"],
                row["ventas_promedio_dia"],
                row["tiempo_entrega_dias"],
                row["estacionalidad"],
                row["tendencia"],
                row["precio"],
                row["proyeccion_eventos"]
            )
            resultados.append({
                "producto": row["producto"],
                "decision": "Hacer Pedido" if res["hacer_pedido_final"]==1 else "No Hacer Pedido",
                "cantidad_sugerida": int(res["cantidad_final"]),
                "ventas_promedio_dia": row["ventas_promedio_dia"]
            })

        df_result = pd.DataFrame(resultados)
        st.subheader("Resultados del análisis")
        st.dataframe(df_result, use_container_width=True)

        # Exportar
        if st.button("Exportar Excel"):
            path = export_excel(df_result, "reporte_batch.xlsx")
            with open(path, "rb") as f:
                st.download_button("Descargar Excel", f, file_name="reporte_batch.xlsx")

        # ------------------------------
        # Visualización global
        # ------------------------------
        st.subheader("Visualización global de resultados")

        # Top productos por ventas promedio
        top_ventas = df_result.sort_values("ventas_promedio_dia", ascending=False).head(10)
        fig1 = px.bar(top_ventas, x="producto", y="ventas_promedio_dia",
                      title="Top 10 productos por ventas promedio/día",
                      labels={"ventas_promedio_dia":"Ventas promedio/día"})
        st.plotly_chart(fig1, use_container_width=True)

        # Top productos por cantidad sugerida
        top_cant = df_result.sort_values("cantidad_sugerida", ascending=False).head(10)
        fig2 = px.bar(top_cant, x="producto", y="cantidad_sugerida",
                      title="Top 10 productos por cantidad sugerida",
                      labels={"cantidad_sugerida":"Cantidad sugerida"})
        st.plotly_chart(fig2, use_container_width=True)

        # Distribución de decisiones
        fig3 = px.pie(df_result, names="decision", title="Distribución de decisiones (Hacer vs No Hacer Pedido)")
        st.plotly_chart(fig3, use_container_width=True)

# ------------------------------
# Descargar Excel de ejemplo
# ------------------------------
elif menu == "📥 Descargar ejemplo":
    st.header("Descargar Excel de ejemplo")

    # Selector para cantidad de productos por categoría
    cantidad = st.slider("Número de productos por categoría", 50, 200, 100, 10)

    categorias = {
        "Abarrotes": [f"Abarrote_{i}" for i in range(1, cantidad+1)],
        "Bebidas": [f"Bebida_{i}" for i in range(1, cantidad+1)],
        "Limpieza/Higiene": [f"Limpieza_{i}" for i in range(1, cantidad+1)]
    }

    productos = []
    for cat, items in categorias.items():
        for prod in items:
            productos.append({
                "producto": prod,
                "categoria": cat,
                "inventario": np.random.randint(10, 100),
                "ventas_promedio_dia": np.random.randint(3, 30)
            })

    df_ejemplo = pd.DataFrame(productos)
    st.dataframe(df_ejemplo.head(20), use_container_width=True)

    if st.button("Generar Excel de ejemplo"):
        path = export_excel(df_ejemplo, "inventario_ventas_ejemplo.xlsx")
        with open(path, "rb") as f:
            st.download_button(
                "Descargar Excel",
                f,
                file_name="inventario_ventas_ejemplo.xlsx"
            )


# ------------------------------
# Configuración del sistema
# ------------------------------
elif menu == "⚙️ Configuración":
    st.header("Configuración del sistema y modelo")
    if rol == "Admin":
        st.markdown("- Ajusta parámetros y recuerda reentrenar el modelo ejecutando `model_training.py`.")
        max_depth = st.slider("Profundidad máxima sugerida del árbol (informativo)", 2, 10, 4)
        nivel_servicio_k = st.slider("Nivel de servicio (k para SS aprox.)", 0.5, 3.0, 1.28, 0.01)
        st.info("Estos controles son informativos para documentación. Los parámetros reales se ajustan en model_training.py.")
        st.markdown("Sube nuevos datos a la carpeta `data/` y vuelve a entrenar para actualizar los modelos.")
    else:
        st.warning("Solo el administrador puede modificar configuración.")

        
