"""
Pruebas para la función create_pdf.

Este módulo contiene pruebas unitarias para la función create_pdf
usando pytest. Se simulan dependencias externas como Tkinter,
pyautogui y PIL para evitar efectos secundarios de I/O y GUI.
"""

from unittest.mock import patch, MagicMock
from generator_pdf import create_pdf


@patch("generator_pdf.pyautogui.screenshot")
@patch("generator_pdf.Image.open")
@patch("generator_pdf.showinfo")
def test_create_pdf(mock_showinfo, mock_Image_open, mock_screenshot):
    """
    Prueba create_pdf con captura de pantalla y guardado en PDF simulados.

    Parámetros
    ----------
    mock_showinfo : MagicMock
        Simulación de tkinter.showinfo.
    mock_Image_open : MagicMock
        Simulación de PIL.Image.open.
    mock_screenshot : MagicMock
        Simulación de pyautogui.screenshot.
    """
    mock_root = MagicMock()
    mock_root.winfo_rootx.return_value = 0
    mock_root.winfo_rooty.return_value = 0
    mock_root.winfo_width.return_value = 100
    mock_root.winfo_height.return_value = 100

    mock_img_screenshot = MagicMock()
    mock_screenshot.return_value = mock_img_screenshot
    mock_img_screenshot.save.return_value = None

    mock_img_pil = MagicMock()
    mock_Image_open.return_value = mock_img_pil
    mock_img_pil.convert.return_value = mock_img_pil
    mock_img_pil.save.return_value = None

    pdf_path = create_pdf(mock_root, report_id=1)

    assert pdf_path == "Reporte1.pdf"