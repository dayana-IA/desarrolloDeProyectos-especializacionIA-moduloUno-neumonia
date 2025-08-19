# tests/test_pdf_generator.py
import pytest
from unittest.mock import patch, MagicMock
from src.neumonia.pdf_generator import PDFGenerator


@pytest.fixture
def mock_root():
    """
    Fixture que simula una ventana Tkinter para pruebas.
    """
    root = MagicMock()
    root.winfo_rootx.return_value = 0
    root.winfo_rooty.return_value = 0
    root.winfo_width.return_value = 800
    root.winfo_height.return_value = 600
    return root


def test_create_pdf(tmp_path, mock_root):
    """
    Testea la generación de PDF usando mocks para evitar la creación real de archivos.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Carpeta temporal proporcionada por pytest.
    mock_root : MagicMock
        Mock de la ventana Tkinter.
    """
    # Crear un config.json temporal apuntando a la carpeta tmp_path
    config_path = tmp_path / "config.json"
    config_content = {
        "pdf_path": str(tmp_path / "reportes")
    }
    config_path.write_text(str(config_content).replace("'", '"'))

    pdf_generator = PDFGenerator(config_path=str(config_path))

    # Parchear pyautogui.screenshot y PIL.Image.open
    with patch("pyautogui.screenshot") as mock_screenshot, \
         patch("PIL.Image.open") as mock_image_open:

        mock_img = MagicMock()
        mock_screenshot.return_value = mock_img
        mock_image_open.return_value = mock_img

        pdf_path = pdf_generator.create_pdf(mock_root, report_id="test")

        # Verificar que screenshot se llamó con la región correcta
        mock_screenshot.assert_called_once_with(region=(0, 0, 800, 600))
        # Verificar que Image.open se llamó
        mock_image_open.assert_called_once()
        # Verificar que se devolvió la ruta esperada (sin espacio)
        expected_path = tmp_path / "reportes" / "Reportetest.pdf"
        assert str(expected_path) == pdf_path
