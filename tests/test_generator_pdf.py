"""
Pruebas para la función `create_pdf`.

Este módulo contiene pruebas unitarias para la función `create_pdf`
usando pytest. Se simulan dependencias externas como Tkinter,
pyautogui y PIL para evitar efectos secundarios de I/O y GUI.
"""

from unittest.mock import patch, MagicMock
from src.neumonia.pdf_generator import create_pdf


@patch("src.neumonia.pdf_generator.pyautogui.screenshot")
@patch("src.neumonia.pdf_generator.Image.open")
@patch("src.neumonia.pdf_generator.showinfo")
def test_create_pdf(mock_showinfo, mock_Image_open, mock_screenshot):
    """
    Test para la función `create_pdf`.

    Simula la captura de pantalla, apertura de imagen y guardado
    de PDF sin generar archivos reales.

    Parameters
    ----------
    mock_showinfo : MagicMock
        Simulación de tkinter.showinfo.
    mock_Image_open : MagicMock
        Simulación de PIL.Image.open.
    mock_screenshot : MagicMock
        Simulación de pyautogui.screenshot.

    Returns
    -------
    None
        Este test no retorna valor, falla si el PDF generado no cumple con el nombre esperado.
    """
    # Simular propiedades del root
    mock_root = MagicMock()
    mock_root.winfo_rootx.return_value = 0
    mock_root.winfo_rooty.return_value = 0
    mock_root.winfo_width.return_value = 100
    mock_root.winfo_height.return_value = 100

    # Simular screenshot
    mock_img_screenshot = MagicMock()
    mock_screenshot.return_value = mock_img_screenshot
    mock_img_screenshot.save.return_value = None

    # Simular PIL Image
    mock_img_pil = MagicMock()
    mock_Image_open.return_value = mock_img_pil
    mock_img_pil.convert.return_value = mock_img_pil
    mock_img_pil.save.return_value = None

    # Ejecutar la función
    pdf_path = create_pdf(mock_root, report_id=1)

    # Verificación
    assert pdf_path == "Reporte1.pdf", "El PDF generado debe llamarse 'Reporte1.pdf'"
