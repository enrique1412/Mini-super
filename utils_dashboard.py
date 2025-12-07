import pandas as pd

def calcular_metricas_basicas(hist: pd.DataFrame) -> dict:
    nivel_servicio = float((hist["inventario"] >= hist["unidades_vendidas"]).mean() * 100)
    rotacion_prom = float(hist.groupby("producto")["unidades_vendidas"].mean().mean())
    prom7 = hist.groupby("producto")["unidades_vendidas"].rolling(7, min_periods=1).mean().reset_index(0, drop=True)
    crit = int((hist["inventario"] < (prom7 * hist["tiempo_entrega_dias"])).sum())
    return {"nivel_servicio": nivel_servicio, "rotacion_prom": rotacion_prom, "productos_criticos": crit}

def preparar_series_promedio(hist: pd.DataFrame) -> pd.DataFrame:
    hist_prom = hist.sort_values(["producto","fecha"]).copy()
    hist_prom["ventas_prom_7d"] = hist_prom.groupby("producto")["unidades_vendidas"].rolling(7, min_periods=1).mean().reset_index(0, drop=True)
    return hist_prom
