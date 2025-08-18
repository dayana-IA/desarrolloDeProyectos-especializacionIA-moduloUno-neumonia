"""
Pruebas para la clase DicomReader.

Este módulo contiene pruebas unitarias para el método
DicomReader.read_dicom_file usando pytest. Las dependencias externas
como dicom, cv2 y PIL son simuladas para evitar operaciones de I/O
y de GUI.
"""

import numpy as np
from unittest.mock import patch, MagicMock
from dicom_reader import DicomReader


@patch("dicom_reader.dicom")
@patch("dicom_reader.Image")
@patch("dicom_reader.cv2")
def test_read_dicom_file(mock_cv2, mock_Image, mock_dicom):
    """
    Prueba read_dicom_file con datos DICOM simulados.

    Parámetros
    ----------
    mock_cv2 : MagicMock
        Simulación del módulo cv2.
    mock_Image : MagicMock
        Simulación del módulo PIL.Image.
    mock_dicom : MagicMock
        Simulación del módulo dicom.
    """
    mock_dataset = MagicMock()
    mock_dataset.pixel_array = np.array([[0, 255], [128, 64]], dtype=np.uint8)
    mock_dicom.dcmread.return_value = mock_dataset

    mock_img_pil = MagicMock()
    mock_Image.fromarray.return_value = mock_img_pil

    mock_cv2.cvtColor.return_value = np.zeros((2, 2, 3), dtype=np.uint8)

    img_rgb, img2show = DicomReader.read_dicom_file("ruta_falsa.dcm")

    assert isinstance(img_rgb, np.ndarray)
    assert img2show == mock_img_pil