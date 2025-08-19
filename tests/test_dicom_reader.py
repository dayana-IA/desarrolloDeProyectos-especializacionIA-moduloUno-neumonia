import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.neumonia.dicom_reader import DicomReader
from PIL import Image

def test_read_dicom_file():
    """
    Test para la función `DicomReader.read`.

    Verifica que la función lea correctamente un archivo DICOM simulado
    y devuelva dos formatos de imagen:
    1. Imagen en RGB como `np.ndarray`.
    2. Imagen en formato PIL para visualización.

    Se utiliza un mock de `pydicom.dcmread` para no depender de un archivo real.

    Returns
    -------
    None
        Este test no retorna valor, pero falla si la salida no cumple con los tipos y dimensiones esperadas.
    """
    mock_pixel_array = np.random.randint(0, 256, (64, 64), dtype=np.uint16)
    mock_dicom = MagicMock()
    mock_dicom.pixel_array = mock_pixel_array

    with patch("pydicom.dcmread", return_value=mock_dicom):
        img_rgb, img_pil = DicomReader.read("dummy_path.dcm")
    
    assert isinstance(img_rgb, np.ndarray), "La imagen RGB debe ser un np.ndarray"
    assert img_rgb.shape[2] == 3, "La imagen RGB debe tener 3 canales"
    assert isinstance(img_pil, Image.Image), "La imagen PIL debe ser un objeto Image.Image"
    assert img_rgb.shape[0:2] == mock_pixel_array.shape, "Las dimensiones deben coincidir con el pixel_array original"
