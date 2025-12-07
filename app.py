import streamlit as st
import streamlit_authenticator as stauth
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
# Login con roles
# ------------------------------
names = ['Admin', 'Compras']
usernames = ['admin', 'compras']
# En producción, usa Settings → Secrets: passwords = ["admin123","compras123"]
passwords = st.secrets.get("passwords", ['admin123', 'compras123'])

credentials = {
    "usernames": {
        "admin": {
            "name": "Administrador",
            "password": "12345"
        },
        "compras": {
            "name": "Compras",
            "password": "67890"
        }
    }
}

authenticator = stauth.Authenticate(
    credentials,
    "satd_cookie",
    "satd_signature",
    cookie_expiry_days=1
)

name, authentication_status, username = authenticator.login("Iniciar sesión", "main")


if not authentication_status:
    st.warning("Acceso restringido. Inicia sesión.")
    st.stop()

authenticator.logout('Cerrar sesión', 'sidebar')
st.sidebar.success(f'Bienvenido, {name}')
rol = "Admin" if username == "admin" else "Compras"

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
    ["🏠 Bienvenida", "🛒 Predicciones", "📊 Dashboard", "📑 Reportes", "📦 Catálogo", "📂 Subir archivo", "⚙️ Configuración"]
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
    st.header("Subir archivo de análisis (CSV/Excel)")
    st.write("Columnas mínimas: inventario, ventas_promedio_dia, tiempo_entrega_dias, estacionalidad, tendencia, precio, proyeccion_eventos, producto (opcional).")

    uploaded_file = st.file_uploader("Carga tu archivo CSV o Excel", type=["csv", "xlsx"])

    if uploaded_file is not None:
        # Leer archivo
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.success("Archivo cargado correctamente ✅")
        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")
            st.stop()

        st.dataframe(df.head(20), use_container_width=True)

        # Validación básica de columnas
        required_cols = ["inventario","ventas_promedio_dia","tiempo_entrega_dias","estacionalidad","tendencia","precio","proyeccion_eventos"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"Faltan columnas requeridas: {missing}")
        else:
            # Aplicar IA en lote
            try:
                pred_bin = pipe_cls.predict(df[required_cols])
                cantidad_pred = pipe_reg.predict(df[required_cols])
            except Exception as e:
                st.error(f"Error al aplicar modelos: {e}")
                st.stop()

            df_result = df.copy()
            df_result["decision"] = ["Hacer Pedido" if int(x)==1 else "No Hacer Pedido" for x in pred_bin]
            df_result["cantidad_sugerida"] = np.ceil(np.maximum(cantidad_pred, 0)).astype(int)

            st.subheader("Resultados del análisis con IA")
            cols_show = ["producto"] if "producto" in df_result.columns else []
            cols_show += ["inventario","ventas_promedio_dia","tiempo_entrega_dias","estacionalidad","tendencia","decision","cantidad_sugerida"]
            st.dataframe(df_result[cols_show], use_container_width=True)

            # Exportar resultados
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Exportar Excel (lote)"):
                    path = export_excel(df_result, "reporte_batch.xlsx")
                    with open(path, "rb") as f:
                        st.download_button("Descargar Excel", f, file_name="reporte_batch.xlsx")
            with c2:
                resumen_lote = {
                    "Total filas": len(df_result),
                    "Pedidos recomendados": int((df_result["decision"] == "Hacer Pedido").sum()),
                    "Productos con cantidad > 0": int((df_result["cantidad_sugerida"] > 0).sum())
                }
                if st.button("Exportar PDF (resumen lote)"):
                    path = export_pdf(resumen_lote, "reporte_batch_resumen.pdf")
                    with open(path, "rb") as f:
                        st.download_button("Descargar PDF", f, file_name="reporte_batch_resumen.pdf", mime="application/pdf")

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
