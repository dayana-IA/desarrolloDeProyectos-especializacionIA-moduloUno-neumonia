import os
import json
from pathlib import Path
import pyautogui
from PIL import Image


class PDFGenerator:
    """
    Clase que genera reportes en PDF.
    """

    def __init__(self, config_path: str = "config.json"):
        with open(config_path, "r") as f:
            config = json.load(f)

        pdf_path_str = config.get("pdf_path")
        if pdf_path_str is None:
            raise ValueError("El archivo de configuración no contiene 'pdf_path'")

        # Convertir a Path y crear carpeta si no existe
        self.pdf_path = Path(pdf_path_str)
        self.pdf_path.mkdir(parents=True, exist_ok=True)

    def create_pdf(self, root, report_id=0) -> str:
        x = root.winfo_rootx()
        y = root.winfo_rooty()
        w = root.winfo_width()
        h = root.winfo_height()

        screenshot = pyautogui.screenshot(region=(x, y, w, h))
        img_path = self.pdf_path / f"Reporte{report_id}.jpg"
        screenshot.save(img_path)

        img = Image.open(img_path).convert("RGB")
        pdf_path = self.pdf_path / f"Reporte{report_id}.pdf"
        img.save(pdf_path)

        return str(pdf_path)
