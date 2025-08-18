"""
Pruebas para la clase Preprocessor.

Este módulo contiene pruebas unitarias para el método
Preprocessor.preprocess usando pytest. Se simulan dependencias
externas de cv2 para garantizar aislamiento.
"""

import numpy as np
from unittest.mock import patch, MagicMock
from preprocessor import Preprocessor


@patch("preprocessor.cv2")
def test_preprocess(mock_cv2):
    """
    Prueba preprocess con operaciones de OpenCV simuladas.

    Parámetros
    ----------
    mock_cv2 : MagicMock
        Simulación del módulo cv2.
    """
    fake_img = np.ones((100, 100, 3), dtype=np.uint8)

    mock_cv2.resize.return_value = fake_img
    mock_cv2.cvtColor.return_value = np.ones((512, 512), dtype=np.uint8)

    mock_clahe = MagicMock()
    mock_clahe.apply.return_value = np.ones((512, 512), dtype=np.uint8)
    mock_cv2.createCLAHE.return_value = mock_clahe

    result = Preprocessor.preprocess(fake_img)

    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 512, 512, 1)
    assert np.all((0 <= result) & (result <= 1))