import os
import json
from pathlib import Path
import pyautogui
from PIL import Image


class PDFGenerator:
    """
    Clase para generar reportes en PDF a partir de capturas de pantalla.

    Esta clase utiliza la librería ``pyautogui`` para capturar una sección de la
    ventana gráfica (por ejemplo, una GUI con Tkinter) y genera un archivo PDF
    con dicha captura.
    """

    def __init__(self, config_path: str = "config.json"):
        """
        Inicializa el generador de PDFs leyendo la configuración desde un archivo JSON.

        Parameters
        ----------
        config_path : str, optional
            Ruta al archivo de configuración JSON (por defecto "config.json").
            El archivo debe contener la clave ``pdf_path`` que indica la carpeta
            donde se guardarán los reportes generados.

        Raises
        ------
        ValueError
            Si el archivo de configuración no contiene la clave ``pdf_path``.
        FileNotFoundError
            Si el archivo JSON no existe en la ruta indicada.
        json.JSONDecodeError
            Si el archivo JSON no tiene un formato válido.
        """
        with open(config_path, "r") as f:
            config = json.load(f)

        pdf_path_str = config.get("pdf_path")
        if pdf_path_str is None:
            raise ValueError("El archivo de configuración no contiene 'pdf_path'")

        # Convertir a Path y crear carpeta si no existe
        self.pdf_path = Path(pdf_path_str)
        self.pdf_path.mkdir(parents=True, exist_ok=True)

    def create_pdf(self, root, report_id: int = 0) -> str:
        """
        Genera un archivo PDF a partir de una captura de pantalla de un widget.

        La función toma las coordenadas y dimensiones del widget `root`,
        realiza una captura de pantalla de esa región y la guarda como PDF.

        Parameters
        ----------
        root : tkinter.Tk o tkinter.Widget
            Objeto raíz o widget del cual se tomará la captura de pantalla.
        report_id : int, optional
            Identificador numérico para el nombre del reporte. Se usará en
            los archivos generados (por defecto 0).

        Returns
        -------
        str
            Ruta absoluta del archivo PDF generado.

        Notes
        -----
        - Se genera primero una imagen JPG temporal antes de convertirla en PDF.
        - El archivo resultante se guarda en la carpeta definida por ``pdf_path`` en el JSON.
        """
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
