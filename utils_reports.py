import pandas as pd
from fpdf import FPDF
from datetime import datetime

def export_excel(df: pd.DataFrame, path: str) -> str:
    df.to_excel(path, index=False)
    return path

def export_pdf(resumen: dict, path: str) -> str:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Reporte SATD - Decisión de Reabastecimiento", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(5)
    for k, v in resumen.items():
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, f"{k}:", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 8, str(v))
        pdf.ln(2)
    pdf.output(path)
    return path
