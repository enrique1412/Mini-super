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
rol = "Admin"
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
    dlt_aj = ventas_prom * tiempo_entrega * f_est * f_trend + proyeccion_eventos  # DLT ajustada
    ss = max(0.0, k_ss * (ventas_prom * 0.2) * np.sqrt(max(tiempo_entrega, 0.0001)))  # Stock de seguridad
    rop = round(dlt_aj + ss)  # Punto de reorden

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

MESES_TXT = ["Enero","Febrero","Marzo","Abril","Mayo","Junio","Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"]
MESES_MAP = {m:i+1 for i,m in enumerate(MESES_TXT)}

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
    st.subheader("Glosario de abreviaturas")
    st.markdown("""
    - DLT (Demanda durante el Lead Time)
    - ROP (Reorder Point / Punto de Reorden)
    - SS (Stock de Seguridad – inventario adicional)
    """)

# ------------------------------
# Predicciones (individual)
# ------------------------------
elif menu == "🛒 Predicciones":
    st.header("Predicción de reabastecimiento (individual)")

    df_ind = st.session_state.get("df_dashboard")
    st.subheader("Sube tu archivo con productos (opcional)")
    archivo_ind = st.file_uploader("Archivo Excel o CSV", type=["csv", "xlsx"], key="uploader_ind")

    if archivo_ind is not None:
        if archivo_ind.name.endswith(".csv"):
            df_ind = pd.read_csv(archivo_ind)
        else:
            df_ind = pd.read_excel(archivo_ind)
        st.session_state["df_dashboard"] = df_ind.copy()
        st.success("Archivo cargado correctamente ✅")

    if df_ind is not None:
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

            k_ss = st.slider("Nivel de servicio (k para Stock de Seguridad – SS)", 0.5, 3.0, 1.28, 0.01)

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
            st.write(f"- DLT ajustada (Demanda durante el Lead Time): {resultados['dlt_aj']:.2f}")
            st.write(f"- Stock de seguridad (SS – inventario adicional): {resultados['ss']:.2f}")
            st.write(f"- Punto de reorden (ROP – Reorder Point): {resultados['rop']}")
            st.write(f"- Regla ROP sugiere pedido: {'Sí' if resultados['hacer_pedido_regla'] else 'No'}")
            st.write(f"- Clasificador sugiere pedido: {'Sí' if resultados['pred_bin']==1 else 'No'}")
    else:
        st.info("Sube un archivo en esta sección o en '📂 Subir archivo' para habilitar la predicción individual.")

# ------------------------------
# Dashboard
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
            st.metric("Nivel de servicio (%)", f"{met['nivel_servicio']:.1f}%")
            st.metric("Rotación promedio (unid/día)", f"{met['rotacion_prom']:.2f}")
            st.metric("Productos críticos (ROP – Punto de Reorden excedido)", f"{met['productos_criticos']}")
        except Exception:
            st.metric("Total productos", len(hist))
            st.metric("Promedio ventas/día", f"{hist['ventas_promedio_dia'].mean():.2f}")
            st.metric("Pedidos sugeridos", int((hist['inventario'] < hist['ventas_promedio_dia']).sum()))

        # Selector de mes y top producto del mes
        if "mes" in hist.columns:
            st.subheader("Análisis por mes")
            mes_sel = st.selectbox("Selecciona mes", sorted(hist["mes"].unique(), key=lambda m: MESES_MAP.get(m, 13)))
            df_mes = hist[hist["mes"] == mes_sel].copy()

            fig_inv = px.line(df_mes, x="producto", y="inventario", color="categoria", title=f"Inventario - {mes_sel}")
            st.plotly_chart(fig_inv, use_container_width=True)

            fig_vent = px.bar(df_mes, x="producto", y="ventas_promedio_dia", color="categoria", title=f"Ventas promedio/día - {mes_sel}")
            st.plotly_chart(fig_vent, use_container_width=True)

            ventas_mes = df_mes.groupby(["producto"])["ventas_promedio_dia"].sum().reset_index()
            top_prod = ventas_mes.sort_values("ventas_promedio_dia", ascending=False).head(1)
            if not top_prod.empty:
                st.metric("Producto más vendido en el mes", top_prod.iloc[0]["producto"])
                st.metric("Ventas promedio/día del top", f"{top_prod.iloc[0]['ventas_promedio_dia']:.2f}")

            # Exportar SOLO el mes seleccionado
            st.subheader(f"Exportar productos del mes {mes_sel}")
            if st.button(f"Generar Excel ({mes_sel})"):
                df_export = df_mes.sort_values("ventas_promedio_dia", ascending=False).head(100)
                path = export_excel(df_export, f"productos_{mes_sel}.xlsx")
                with open(path, "rb") as f:
                    st.download_button(f"Descargar Excel {mes_sel}", f, file_name=f"productos_{mes_sel}.xlsx")

    except Exception as e:
        st.warning(f"No se pudo cargar datos: {e}")


# ------------------------------
# Reportes
# ------------------------------
elif menu == "📑 Reportes":
    st.header("Reportes y análisis histórico")
    try:
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
            fig = go.Figure()
            if "fecha" in dfp.columns and "unidades_vendidas" in dfp.columns:
                dfp["ventas_prom_7d"] = dfp["unidades_vendidas"].rolling(7, min_periods=1).mean()
                fig.add_trace(go.Scatter(x=dfp["fecha"], y=dfp["inventario"], name="Inventario"))
                fig.add_trace(go.Scatter(x=dfp["fecha"], y=dfp["ventas_prom_7d"], name="Ventas promedio 7d (DLT – Demanda durante Lead Time)"))
            else:
                fig.add_trace(go.Bar(x=dfp.get("mes", ["Mes actual"]), y=dfp["inventario"], name="Inventario"))
                fig.add_trace(go.Bar(x=dfp.get("mes", ["Mes actual"]), y=dfp["ventas_promedio_dia"], name="Ventas prom/día"))
            st.plotly_chart(fig, use_container_width=True)

            if st.button("Exportar Excel (producto)"):
                path = export_excel(dfp, "reporte_hist_producto.xlsx")
                with open(path, "rb") as f:
                    st.download_button("Descargar Excel", f, file_name="reporte_hist_producto.xlsx")
        else:
            st.warning("El conjunto de datos no incluye columna 'producto'.")
    except Exception as e:
        st.warning(f"No se pudo cargar histórico: {e}")

# ------------------------------
# Catálogo
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

        if "df_dashboard" in st.session_state:
            df_src = st.session_state["df_dashboard"].copy()
            base_nuevos = df_src[["producto","categoria"]].drop_duplicates()
            if "lead_time" not in df_src.columns: df_src["lead_time"] = 2
            if "precio" not in df_src.columns: df_src["precio"] = 25
            base_nuevos = base_nuevos.merge(df_src[["producto","lead_time","precio"]], on="producto", how="left")
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
# Subir archivo
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
        colmap = {"Producto":"producto", "Categoria":"categoria", "Inventario":"inventario",
                  "Ventas_promedio_dia":"ventas_promedio_dia", "Mes":"mes", "Lead_time":"lead_time", "Precio":"precio"}
        df = df.rename(columns={k: v for k, v in colmap.items() if k in df.columns})

        st.success("Archivo cargado correctamente ✅")
        st.dataframe(df.head(), use_container_width=True)

        # Editor interactivo
        st.subheader("Editar datos del archivo (interactivo)")
        df = st.data_editor(df, num_rows="dynamic", use_container_width=True)

        # Guardar en session_state
        st.session_state["df_dashboard"] = df.copy()
        st.info("Datos guardados en memoria. El Dashboard y Predicciones usarán este archivo.")

        # Cargar catálogo seguro
        try:
            catalogo = pd.read_csv("data/catalogo_productos.csv")
        except Exception:
            catalogo = pd.DataFrame(columns=["producto","categoria","lead_time","precio"])

        for col in ["producto","categoria","lead_time","precio"]:
            if col not in catalogo.columns:
                catalogo[col] = np.nan

        # Merge con catálogo
        df = df.merge(catalogo[["producto","categoria","lead_time","precio"]].rename(
            columns={"categoria":"categoria_cat","lead_time":"lead_time_cat","precio":"precio_cat"}
        ), on="producto", how="left")

        # Completar valores
        if "lead_time" not in df.columns: df["lead_time"] = np.nan
        if "precio" not in df.columns: df["precio"] = np.nan
        if "categoria" not in df.columns: df["categoria"] = np.nan

        df["lead_time"] = df["lead_time"].combine_first(df["lead_time_cat"]).fillna(2)
        df["precio"] = df["precio"].combine_first(df["precio_cat"]).fillna(25)
        df["categoria"] = df["categoria"].combine_first(df["categoria_cat"]).fillna("General")

        # Estacionalidad y tendencia
        if "mes" in df.columns:
            df["mes_num"] = df["mes"].map(MESES_MAP).fillna(pd.Timestamp.today().month)
            df["estacionalidad"] = df["mes_num"].apply(asignar_estacionalidad_por_mes)
        else:
            mes_actual = pd.Timestamp.today().month
            df["estacionalidad"] = asignar_estacionalidad_por_mes(mes_actual)
            df["mes"] = "Mes actual"

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

        # Guardar datos subidos
        st.session_state["df_dashboard"] = df.copy()

        # Exportar 100 productos por mes
        st.subheader("Exportar 100 productos por cada mes (ventas e inventario)")
        if "mes" in df.columns and st.button("Generar Excel por mes (100 productos por mes)"):
            frames = []
            for mes_val in sorted(df["mes"].unique(), key=lambda m: MESES_MAP.get(m, 13)):
                df_m = df[df["mes"] == mes_val].copy()
                df_m = df_m.sort_values("ventas_promedio_dia", ascending=False).head(100)
                df_m["mes_export"] = mes_val
                frames.append(df_m[["mes_export","producto","categoria","inventario","ventas_promedio_dia","lead_time","precio"]])
            df_export = pd.concat(frames, ignore_index=True)
            path = export_excel(df_export, "productos_por_mes_100.xlsx")
            with open(path, "rb") as f:
                st.download_button("Descargar Excel por mes", f, file_name="productos_por_mes_100.xlsx")

        # Actualizar catálogo automáticamente
        st.subheader("Actualizar catálogo automáticamente")
        try:
            catalogo = pd.read_csv("data/catalogo_productos.csv")
        except Exception:
            catalogo = pd.DataFrame(columns=["producto","categoria","es_basico","stock_minimo","proveedor","lead_time","precio"])

        base_nuevos = df[["producto","categoria","lead_time","precio"]].drop_duplicates().copy()
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
# Descargar ejemplo
# ------------------------------
elif menu == "📥 Descargar ejemplo":
    st.header("Descargar Excel de ejemplo con 150 productos y ventas mensuales")

    categorias = {
        "Abarrotes": ["Arroz 1kg","Frijol negro 1kg","Harina de trigo","Pasta espagueti","Aceite vegetal 1L",
                      "Azúcar 1kg","Sal 1kg","Atún en lata","Galletas María","Café molido 250g",
                      "Sopa instantánea","Pan de caja","Leche en polvo","Manteca vegetal","Harina de maíz",
                      "Sardina en lata","Mayonesa","Catsup","Mermelada fresa","Avena 1kg"],
        "Bebidas": ["Agua 1L","Refresco sabor cola 2L","Jugo naranja 1L","Leche entera 1L","Yogurt bebible",
                    "Cerveza clara 355ml","Vino tinto","Té helado","Red Bull","Café instantáneo",
                    "Agua mineral","Refresco sabor limón 2L","Jugo manzana 1L","Cerveza oscura","Whisky escocés",
                    "Sidra","Agua saborizada","Electrolit","Leche deslactosada 1L","Kombucha"],
        "Limpieza/Higiene": ["Detergente en polvo","Jabón de barra","Shampoo anticaspa","Pasta dental","Papel higiénico 12 rollos",
                             "Cloro líquido","Limpiador multiusos","Toallas sanitarias","Desodorante","Gel antibacterial",
                             "Suavizante de telas","Jabón líquido","Crema corporal","Rastrillos","Toallas húmedas",
                             "Fibras limpiadoras","Guantes","Lavatrastes","Limpiador piso","Aromatizante"],
        "Frescos": ["Tomate","Papa","Cebolla","Zanahoria","Manzana","Plátano","Naranja","Pollo entero","Carne molida de res","Filete de pescado",
                    "Calabacita","Pepino","Lechuga romana","Espinaca","Uva verde","Mango","Sandía","Melón","Huevo blanco docena","Queso fresco"],
        "Otros básicos": ["Jamón de pavo","Crema ácida","Mantequilla","Yogurt natural","Pan dulce","Tortillas de maíz","Tortillas de harina","Chorizo","Salchicha","Queso Oaxaca",
                          "Queso panela","Queso manchego","Gelatina en polvo","Flan de vainilla","Chocolate de mesa","Salsa verde","Salsa roja","Mole en pasta","Caldo de pollo en cubos","Caldo de res en cubos"]
    }

    meses = ["Enero","Febrero","Marzo","Abril","Mayo","Junio","Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"]

    productos = []
    for cat, items in categorias.items():
        for prod in items:
            for mes in meses:
                productos.append({
                    "producto": prod,
                    "categoria": cat,
                    "mes": mes,
                    "inventario": np.random.randint(10, 120),
                    "ventas_promedio_dia": np.random.randint(3, 40),
                    "lead_time": np.random.randint(1, 5),
                    "precio": np.random.randint(10, 250)
                })

    df_ejemplo = pd.DataFrame(productos)
    st.dataframe(df_ejemplo.head(20), use_container_width=True)

    if st.button("Generar Excel de ejemplo"):
        path = export_excel(df_ejemplo, "inventario_ventas_150productos.xlsx")
        with open(path, "rb") as f:
            st.download_button("Descargar Excel", f, file_name="inventario_ventas_150productos.xlsx")

# ------------------------------
# Configuración
# ------------------------------
elif menu == "⚙️ Configuración":
    st.header("Configuración del sistema y modelo")
    if rol == "Admin":
        st.markdown("- Ajusta parámetros y recuerda reentrenar el modelo ejecutando `model_training.py`.")
        max_depth = st.slider("Profundidad máxima sugerida del árbol (informativo)", 2, 10, 4)
        nivel_servicio_k = st.slider("Nivel de servicio (k para SS – Stock de Seguridad)", 0.5, 3.0, 1.28, 0.01)
        st.info("Estos controles son informativos para documentación. Los parámetros reales se ajustan en model_training.py.")
        st.markdown("Sube nuevos datos a la carpeta `data/` y vuelve a entrenar para actualizar los modelos.")
    else:
        st.warning("Solo el administrador puede modificar configuración.")
