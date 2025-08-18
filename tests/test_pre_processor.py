"""
Pruebas para la clase `Preprocessor`.

Este módulo contiene pruebas unitarias para el método
`Preprocessor.preprocess` usando pytest. Se simulan dependencias
externas de OpenCV (cv2) para garantizar aislamiento y evitar
operaciones reales de procesamiento de imagen.
"""

import numpy as np
from unittest.mock import patch, MagicMock
from src.neumonia.pre_processor import Preprocessor


@patch("src.neumonia.pre_processor.cv2")
def test_preprocess(mock_cv2):
    """
    Test para `Preprocessor.preprocess`.

    Simula las operaciones de OpenCV (resize, color conversion y CLAHE)
    para verificar que la función retorna un array normalizado con
    la forma correcta y valores entre 0 y 1.

    Parameters
    ----------
    mock_cv2 : MagicMock
        Simulación del módulo OpenCV (cv2).

    Returns
    -------
    None
        Este test no retorna valor, falla si la salida no cumple las
        dimensiones o rango esperado.
    """
    # Imagen simulada
    fake_img = np.ones((100, 100, 3), dtype=np.uint8)

    # Simular funciones de cv2
    mock_cv2.resize.return_value = fake_img
    mock_cv2.cvtColor.return_value = np.ones((512, 512), dtype=np.uint8)

    mock_clahe = MagicMock()
    mock_clahe.apply.return_value = np.ones((512, 512), dtype=np.uint8)
    mock_cv2.createCLAHE.return_value = mock_clahe

    # Ejecutar preprocess
    result = Preprocessor.preprocess(fake_img)

    # Verificaciones
    assert isinstance(result, np.ndarray), "La salida debe ser un np.ndarray"
    assert result.shape == (1, 512, 512, 1), "La salida debe tener forma (1, 512, 512, 1)"
    assert np.all((0 <= result) & (result <= 1)), "Todos los valores deben estar normalizados entre 0 y 1"
