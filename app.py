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
    ["🏠 Bienvenida", "🛒 Predicciones", "📊 Dashboard", "📑 Reportes",
     "📦 Catálogo", "📂 Subir archivo", "⚙️ Configuración", "📥 Descargar ejemplo"]
)

# ------------------------------
# Funciones internas
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

    f_est = season_factor(estacionalidad)
    f_trend = trend_factor(tendencia)
    dlt_aj = ventas_prom * tiempo_entrega * f_est * f_trend + proyeccion_eventos
    ss = max(0.0, k_ss * (ventas_prom * 0.2) * np.sqrt(max(tiempo_entrega, 0.0001)))
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

def asignar_estacionalidad_por_mes(mes_num: int) -> str:
    if mes_num in [11, 12, 1]:
        return "alta"
    elif mes_num in [6, 7, 8]:
        return "media"
    else:
        return "baja"

# ------------------------------
# Bienvenida
# ------------------------------
if menu == "🏠 Bienvenida":
    st.title("SATD Pro – Sistema de Apoyo a la Toma de Decisiones")
    st.markdown("""
    Optimiza la planeación de compras y reabastecimiento para tu mini supermercado:
    - Predicciones: decisión de pedido (Sí/No) y cantidad sugerida.
    - Dashboard: métricas clave y tendencias (se actualiza con tu Excel).
    - Reportes: exportación a PDF y Excel.
    - Catálogo: gestión de productos (autoactualizable).
    - Subir archivo: análisis en lote con IA.
    - Configuración: parámetros del sistema.
    """)

# ------------------------------
# Predicciones individuales (usa Excel subido si existe)
# ------------------------------
elif menu == "🛒 Predicciones":
    st.header("Predicción de reabastecimiento (individual)")

    # si ya hay datos en session_state, úsalos
    df_ind = None
    if "df_dashboard" in st.session_state:
        df_ind = st.session_state["df_dashboard"].copy()
        st.info("Usando el Excel subido previamente para seleccionar producto ✅")

    st.subheader("Sube tu archivo con productos (opcional si ya subiste)")
    archivo_ind = st.file_uploader("Archivo Excel o CSV", type=["csv", "xlsx"], key="uploader_ind")

    if archivo_ind is not None:
        if archivo_ind.name.endswith(".csv"):
            df_ind = pd.read_csv(archivo_ind)
        else:
            df_ind = pd.read_excel(archivo_ind)
        st.session_state["df_dashboard"] = df_ind.copy()  # sincroniza para el Dashboard
        st.success("Archivo cargado correctamente ✅")

    if df_ind is not None:
        # Normalizar columnas
        colmap = {"Producto":"producto", "Categoria":"categoria", "Inventario":"inventario",
                  "Ventas_promedio_dia":"ventas_promedio_dia", "Mes":"mes", "Lead_time":"lead_time", "Precio":"precio"}
        df_ind = df_ind.rename(columns={k:v for k,v in colmap.items() if k in df_ind.columns})

        st.dataframe(df_ind.head(), use_container_width=True)

        if "producto" not in df_ind.columns:
            st.error("La columna 'producto' es obligatoria.")
        else:
            producto_sel = st.selectbox("Selecciona un producto", sorted(df_ind["producto"].unique()))
            datos = df_ind[df_ind["producto"] == producto_sel].iloc[0]

            col1, col2, col3 = st.columns(3)
            with col1:
                inventario = st.number_input("Inventario actual (unidades)", min_value=0, value=int(datos.get("inventario", 0)))
                ventas_prom = st.number_input("Ventas promedio por día", min_value=0.0, value=float(datos.get("ventas_promedio_dia", 0)))
            with col2:
                tiempo_entrega = st.number_input("Tiempo de entrega (días)", min_value=0.0, value=float(datos.get("lead_time", 2)))
                estacionalidad = st.selectbox("Estacionalidad", ["alta", "media", "baja"], index=0)
                tendencia = st.selectbox("Tendencia", ["subiendo", "estable", "bajando"], index=0)
            with col3:
                precio = st.number_input("Precio (MXN)", min_value=0.0, value=float(datos.get("precio", 25)))
                proyeccion_eventos = st.number_input("Proyección de eventos (extra demanda)", min_value=0.0, value=float(datos.get("proyeccion_eventos", 0)))

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
    else:
        st.info("Sube un archivo en esta sección o en '📂 Subir archivo' para habilitar la predicción individual.")

# ------------------------------
# Dashboard (conecta a session_state si existe)
# ------------------------------
elif menu == "📊 Dashboard":
    st.header("Dashboard de métricas")
    try:
        if "df_dashboard" in st.session_state:
            hist = st.session_state["df_dashboard"].copy()
            st.info("Usando datos del Excel subido ✅")
        else:
            hist = pd.read_csv("data/historico_inventario.csv", parse_dates=["fecha"])
            st.info("Usando histórico local 📂")

        st.dataframe(hist.head(50), use_container_width=True)

        # Métricas básicas
        try:
            met = calcular_metricas_basicas(hist.rename(columns={"ventas_promedio_dia":"unidades_vendidas"}))
            st.metric("Nivel de servicio", f"{met['nivel_servicio']:.1f}%")
            st.metric("Rotación promedio (unid/día)", f"{met['rotacion_prom']:.2f}")
            st.metric("Productos críticos (ROP excedido)", f"{met['productos_criticos']}")
        except Exception:
            st.metric("Total productos", len(hist))
            st.metric("Promedio ventas/día", f"{hist['ventas_promedio_dia'].mean():.2f}")
            st.metric("Pedidos sugeridos", int((hist['inventario'] < hist['ventas_promedio_dia']).sum()))

        # Gráficas (si hay 'mes')
        if "mes" in hist.columns:
            fig_inv = px.line(hist, x="mes", y="inventario", color="producto", title="Inventario por mes")
            st.plotly_chart(fig_inv, use_container_width=True)

            fig_vent = px.bar(hist, x="mes", y="ventas_promedio_dia", color="producto", title="Ventas promedio por mes")
            st.plotly_chart(fig_vent, use_container_width=True)

            ventas_mes = hist.groupby(["mes","producto"])["ventas_promedio_dia"].sum().reset_index()
            top_mes = ventas_mes.sort_values(["mes","ventas_promedio_dia"], ascending=[True,False]).groupby("mes").head(1)
            st.subheader("Producto más vendido por mes")
            st.dataframe(top_mes, use_container_width=True)
            fig_top_mes = px.bar(top_mes, x="mes", y="ventas_promedio_dia", color="producto", title="Top producto por mes")
            st.plotly_chart(fig_top_mes, use_container_width=True)

    except Exception as e:
        st.warning(f"No se pudo cargar datos: {e}")

# ------------------------------
# Reportes históricos (archivo local)
# ------------------------------
elif menu == "📑 Reportes":
    st.header("Reportes y análisis histórico")
    try:
        # Si hay datos subidos en sesión, usa esos; si no, usa el histórico
        if "df_dashboard" in st.session_state:
            hist = st.session_state["df_dashboard"].copy()
            st.info("Usando datos del Excel subido ✅")
        else:
            hist = pd.read_csv("data/historico_inventario.csv", parse_dates=["fecha"])
            st.info("Usando histórico local 📂")

        st.dataframe(hist, use_container_width=True)

        if "producto" in hist.columns:
            prod_sel = st.selectbox("Producto", sorted(hist["producto"].unique()))
            dfp = hist[hist["producto"] == prod_sel].copy()
            # Si no hay fecha, usamos mes para graficar
            if "fecha" in dfp.columns:
                dfp["ventas_prom_7d"] = dfp["unidades_vendidas"].rolling(7, min_periods=1).mean()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=dfp["fecha"], y=dfp["inventario"], name="Inventario"))
                fig.add_trace(go.Scatter(x=dfp["fecha"], y=dfp["ventas_prom_7d"], name="Ventas promedio 7d"))
            else:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=dfp.get("mes", ["Mes actual"]), y=dfp["inventario"], name="Inventario"))
                fig.add_trace(go.Bar(x=dfp.get("mes", ["Mes actual"]), y=dfp["ventas_promedio_dia"], name="Ventas prom/día"))
            st.plotly_chart(fig, use_container_width=True)

            if st.button("Exportar Excel (histórico)"):
                path = export_excel(dfp, "reporte_hist_producto.xlsx")
                with open(path, "rb") as f:
                    st.download_button("Descargar Excel", f, file_name="reporte_hist_producto.xlsx")
        else:
            st.warning("El conjunto de datos no incluye columna 'producto'.")
    except Exception as e:
        st.warning(f"No se pudo cargar histórico: {e}")

# ------------------------------
# Catálogo de productos (autoactualizable)
# ------------------------------
elif menu == "📦 Catálogo":
    st.header("Catálogo de productos")
    try:
        try:
            catalogo = pd.read_csv("data/catalogo_productos.csv")
        except Exception:
            catalogo = pd.DataFrame(columns=["producto","categoria","es_basico","stock_minimo","proveedor","lead_time","precio"])

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
                precio = st.number_input("Precio (MXN)", min_value=0.0, value=25.0)
                submitted = st.form_submit_button("Agregar")

            if submitted and nuevo_prod.strip():
                nuevo = {
                    "producto": nuevo_prod, "categoria": categoria, "es_basico": es_basico,
                    "stock_minimo": stock_minimo, "proveedor": proveedor, "lead_time": lead_time, "precio": precio
                }
                catalogo = pd.concat([catalogo, pd.DataFrame([nuevo])], ignore_index=True)
                catalogo.to_csv("data/catalogo_productos.csv", index=False)
                st.success(f"Producto agregado: {nuevo_prod}")

        # Integrar productos nuevos desde Excel subido
        if "df_dashboard" in st.session_state:
            df_src = st.session_state["df_dashboard"].copy()
            base_nuevos = df_src[["producto","categoria"]].drop_duplicates()
            base_nuevos["lead_time"] = df_src.get("lead_time", 2)
            base_nuevos["precio"] = df_src.get("precio", 25)
            base_nuevos["es_basico"] = 0
            base_nuevos["stock_minimo"] = 10
            base_nuevos["proveedor"] = "Proveedor X"

            nuevos = base_nuevos[~base_nuevos["producto"].isin(catalogo["producto"])]
            if not nuevos.empty:
                st.info(f"Productos nuevos detectados: {len(nuevos)}. Se agregarán al catálogo.")
                catalogo = pd.concat([catalogo, nuevos], ignore_index=True)
                catalogo.to_csv("data/catalogo_productos.csv", index=False)
                st.success("Catálogo actualizado con datos del Excel ✅")
            else:
                st.caption("No hay productos nuevos para agregar.")
    except Exception as e:
        st.warning(f"No se pudo cargar catálogo: {e}")

# ------------------------------
# Subir archivo para análisis en lote (guarda en session_state y genera reporte)
# ------------------------------
elif menu == "📂 Subir archivo":
    st.header("Subir archivo de inventario/ventas (CSV/Excel)")
    uploaded_file = st.file_uploader("Carga tu archivo", type=["csv","xlsx"], key="uploader_batch")

    if uploaded_file is not None:
        # Leer archivo
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Normalizar columnas
        colmap = {
            "Producto": "producto", "Categoria": "categoria", "Inventario": "inventario",
            "Ventas_promedio_dia": "ventas_promedio_dia", "Mes": "mes",
            "Lead_time": "lead_time", "Precio": "precio"
        }
        df = df.rename(columns={k: v for k, v in colmap.items() if k in df.columns})

        st.success("Archivo cargado correctamente ✅")
        st.dataframe(df.head(), use_container_width=True)

        # Editor interactivo
        st.subheader("Editar datos del archivo (interactivo)")
        df = st.data_editor(df, num_rows="dynamic", use_container_width=True)

        # Guardar en session_state para Dashboard y otras secciones
        st.session_state["df_dashboard"] = df.copy()
        st.info("Datos guardados en memoria. El Dashboard y Predicciones usarán este archivo.")

        # Enriquecer con catálogo
        try:
            catalogo = pd.read_csv("data/catalogo_productos.csv")
        except Exception:
            catalogo = pd.DataFrame(columns=["producto","categoria","lead_time","precio"])

        cols_existentes = [c for c in ["producto","lead_time","precio","categoria"] if c in catalogo.columns]
        df = df.merge(catalogo[cols_existentes], on="producto", how="left", suffixes=("", "_cat"))

        # Resolver valores con prioridad: archivo > catálogo > default
        df["lead_time"] = df["lead_time"].fillna(df.get("lead_time_cat")).fillna(2)
        df["precio"] = df["precio"].fillna(df.get("precio_cat")).fillna(25)
        df["categoria"] = df["categoria"].fillna(df.get("categoria_cat")).fillna("General")

        # Estacionalidad por mes si existe
        if "mes" in df.columns:
            meses_map = {
                "Enero":1,"Febrero":2,"Marzo":3,"Abril":4,"Mayo":5,"Junio":6,
                "Julio":7,"Agosto":8,"Septiembre":9,"Octubre":10,"Noviembre":11,"Diciembre":12
            }
            df["mes_num"] = df["mes"].map(meses_map).fillna(pd.Timestamp.today().month)
            df["estacionalidad"] = df["mes_num"].apply(asignar_estacionalidad_por_mes)
        else:
            mes_actual = pd.Timestamp.today().month
            df["estacionalidad"] = asignar_estacionalidad_por_mes(mes_actual)
            df["mes"] = "Mes actual"

        # Tendencia simple
        df["tendencia"] = np.where(df["ventas_promedio_dia"].diff().fillna(0) > 0, "subiendo", "estable")
        df["proyeccion_eventos"] = df.get("proyeccion_eventos", 0)

        # Aplicar modelo a cada producto
        resultados = []
        for _, row in df.iterrows():
            res = decision_y_cantidad(
                float(row.get("inventario", 0)),
                float(row.get("ventas_promedio_dia", 0)),
                float(row.get("lead_time", 2)),
                row.get("estacionalidad", "media"),
                row.get("tendencia", "estable"),
                float(row.get("precio", 25)),
                float(row.get("proyeccion_eventos", 0))
            )
            resultados.append({
                "producto": row.get("producto"),
                "categoria": row.get("categoria"),
                "mes": row.get("mes"),
                "decision": "Hacer Pedido" if res["hacer_pedido_final"]==1 else "No Hacer Pedido",
                "cantidad_sugerida": int(res["cantidad_final"]),
                "ventas_promedio_dia": float(row.get("ventas_promedio_dia", 0)),
                "inventario": float(row.get("inventario", 0)),
                "precio": float(row.get("precio", 0)),
                "rop": res["rop"]
            })

        df_result = pd.DataFrame(resultados)
        st.subheader("Resultados del análisis")
        st.dataframe(df_result, use_container_width=True)

        # Reporte automático
        st.subheader("Reporte de análisis")
        colR1, colR2 = st.columns(2)
        with colR1:
            if st.button("Exportar reporte Excel"):
                path = export_excel(df_result, "reporte_batch.xlsx")
                with open(path, "rb") as f:
                    st.download_button("Descargar reporte Excel", f, file_name="reporte_batch.xlsx")
        with colR2:
            if st.button("Exportar reporte PDF"):
                resumen = df_result.to_dict(orient="records")
                path = export_pdf(resumen, "reporte_batch.pdf")
                with open(path, "rb") as f:
                    st.download_button("Descargar reporte PDF", f, file_name="reporte_batch.pdf", mime="application/pdf")

        # Visualización global
        st.subheader("Visualización global de resultados")
        top_ventas = df_result.sort_values("ventas_promedio_dia", ascending=False).head(10)
        fig1 = px.bar(top_ventas, x="producto", y="ventas_promedio_dia",
                      title="Top 10 productos por ventas promedio/día", color="categoria")
        st.plotly_chart(fig1, use_container_width=True)

        top_cant = df_result.sort_values("cantidad_sugerida", ascending=False).head(10)
        fig2 = px.bar(top_cant, x="producto", y="cantidad_sugerida",
                      title="Top 10 productos por cantidad sugerida", color="categoria")
        st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.pie(df_result, names="decision", title="Distribución de decisiones (Hacer vs No Hacer Pedido)")
        st.plotly_chart(fig3, use_container_width=True)

        # Dashboard con datos del archivo subido (inline)
        st.subheader("Dashboard con datos del archivo subido")
        try:
            met = calcular_metricas_basicas(df.rename(columns={"ventas_promedio_dia":"unidades_vendidas"}))
            st.metric("Nivel de servicio", f"{met['nivel_servicio']:.1f}%")
            st.metric("Rotación promedio (unid/día)", f"{met['rotacion_prom']:.2f}")
            st.metric("Productos críticos (ROP excedido)", f"{met['productos_criticos']}")
        except Exception:
            st.metric("Total productos", len(df_result))
            st.metric("Promedio ventas/día", f"{df_result['ventas_promedio_dia'].mean():.2f}")
            st.metric("Pedidos sugeridos", int((df_result['decision'] == "Hacer Pedido").sum()))

        if "mes" in df_result.columns:
            fig_inv = px.line(df_result, x="mes", y="inventario", color="producto", title="Inventario por mes")
            st.plotly_chart(fig_inv, use_container_width=True)

            fig_vent_mes = px.bar(df_result, x="mes", y="ventas_promedio_dia", color="producto",
                                  title="Ventas promedio por mes")
            st.plotly_chart(fig_vent_mes, use_container_width=True)

            st.subheader("Producto más vendido por mes")
            ventas_mes = df_result.groupby(["mes","producto"])["ventas_promedio_dia"].sum().reset_index()
            top_mes = ventas_mes.sort_values(["mes","ventas_promedio_dia"], ascending=[True,False]).groupby("mes").head(1)
            st.dataframe(top_mes, use_container_width=True)
            fig_top_mes = px.bar(top_mes, x="mes", y="ventas_promedio_dia", color="producto",
                                 title="Top producto por mes")
            st.plotly_chart(fig_top_mes, use_container_width=True)

        # Actualizar catálogo automáticamente
        st.subheader("Actualizar catálogo automáticamente")
        try:
            catalogo = pd.read_csv("data/catalogo_productos.csv")
        except Exception:
            catalogo = pd.DataFrame(columns=["producto","categoria","es_basico","stock_minimo","proveedor","lead_time","precio"])

        base_nuevos = df_result[["producto","categoria"]].drop_duplicates().copy()
        base_nuevos["lead_time"] = df.get("lead_time", 2)
        base_nuevos["precio"] = df_result["precio"]
        base_nuevos["es_basico"] = 0
        base_nuevos["stock_minimo"] = 10
        base_nuevos["proveedor"] = "Proveedor X"

        nuevos = base_nuevos[~base_nuevos["producto"].isin(catalogo["producto"])]
        if not nuevos.empty:
            st.info(f"Se encontraron {len(nuevos)} productos nuevos; se agregarán al catálogo.")
            catalogo = pd.concat([catalogo, nuevos], ignore_index=True)
            catalogo.to_csv("data/catalogo_productos.csv", index=False)
            st.success("Catálogo actualizado ✅")
        else:
            st.caption("No hay productos nuevos para agregar al catálogo.")

# ------------------------------
# Descargar Excel de ejemplo con productos reales
# ------------------------------
elif menu == "📥 Descargar ejemplo":
    st.header("Descargar Excel de ejemplo con productos reales")

    cantidad = st.slider("Número de productos por categoría", 50, 100, 50, 10)
    categorias = {
        "Abarrotes": ["Arroz 1kg","Frijol negro 1kg","Harina de trigo","Pasta espagueti","Aceite vegetal 1L",
                      "Azúcar 1kg","Sal 1kg","Atún en lata","Galletas María","Café molido 250g",
                      "Sopa instantánea","Pan de caja","Leche en polvo","Manteca vegetal","Harina de maíz",
                      "Sardina en lata","Mayonesa","Catsup","Mermelada fresa","Avena 1kg",
                      "Chocolate en polvo","Vinagre","Salsa picante","Maíz para palomitas","Cereal hojuelas"],
        "Bebidas": ["Agua 1L","Refresco Coca-Cola 2L","Jugo naranja 1L","Leche entera 1L","Yogurt bebible",
                    "Cerveza Corona 355ml","Vino tinto","Té helado","Red Bull","Café instantáneo",
                    "Agua mineral","Refresco Pepsi 2L","Jugo manzana 1L","Cerveza Modelo","Whisky 750ml",
                    "Sidra","Agua saborizada","Electrolit","Leche deslactosada 1L","Kombucha"],
        "Limpieza/Higiene": ["Detergente en polvo","Jabón de barra","Shampoo 750ml","Pasta dental","Papel higiénico 12pz",
                             "Cloro 1L","Limpiador multiusos","Toallas sanitarias","Desodorante","Gel antibacterial",
                             "Suavizante de telas","Jabón líquido","Crema corporal","Rastrillos","Toallas húmedas",
                             "Fibras limpiadoras","Guantes","Lavatrastes","Limpiador piso","Aromatizante"]
    }
    meses = ["Enero","Febrero","Marzo","Abril","Mayo","Junio","Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"]

    productos = []
    for cat, items in categorias.items():
        for prod in items[:cantidad]:
            productos.append({
                "producto": prod,
                "categoria": cat,
                "inventario": np.random.randint(10, 120),
                "ventas_promedio_dia": np.random.randint(3, 40),
                "mes": np.random.choice(meses),
                "lead_time": np.random.randint(1, 5),
                "precio": np.random.randint(10, 250)
            })

    df_ejemplo = pd.DataFrame(productos)
    st.dataframe(df_ejemplo.head(20), use_container_width=True)

    if st.button("Generar Excel de ejemplo"):
        path = export_excel(df_ejemplo, "inventario_ventas_reales.xlsx")
        with open(path, "rb") as f:
            st.download_button("Descargar Excel", f, file_name="inventario_ventas_reales.xlsx")

# ------------------------------
# Configuración
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
