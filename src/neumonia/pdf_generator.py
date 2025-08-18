# -*- coding: utf-8 -*-
"""
Generación de reportes en PDF
Realiza capturas de pantalla de la interfaz.
"""

import pyautogui
from tkinter.messagebox import showinfo
from PIL import Image


def create_pdf(root, report_id=0):
    """
    Genera un archivo PDF a partir de una captura de pantalla de la ventana Tkinter.

    Parameters
    ----------
    root : Tk
        Ventana principal de Tkinter de la que se capturará la pantalla.
    report_id : int, optional
        Identificador numérico para el archivo generado (por defecto 0).

    Returns
    -------
    str
        Ruta del archivo PDF generado.
    """
    # Capturar pantalla (solo la parte visible de la ventana)
    x = root.winfo_rootx()
    y = root.winfo_rooty()
    w = root.winfo_width()
    h = root.winfo_height()

    screenshot = pyautogui.screenshot(region=(x, y, w, h))
    img_path = f"Reporte{report_id}.jpg"
    screenshot.save(img_path)

    img = Image.open(img_path).convert("RGB")
    pdf_path = f"Reporte{report_id}.pdf"
    img.save(pdf_path)

    showinfo(title="PDF", message="El PDF fue generado con éxito.")
    return pdf_path