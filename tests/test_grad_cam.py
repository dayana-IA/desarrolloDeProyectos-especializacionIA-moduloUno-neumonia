"""
Pruebas unitarias para la clase GradCAMModel.

Se usan mocks para TensorFlow, OpenCV y pydicom para aislar la funcionalidad
y evitar cálculos costosos o dependencias externas.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.neumonia.grad_cam import GradCAMModel
from PIL import Image


@pytest.fixture
def dummy_image():
    """
    Imagen simulada para pruebas.

    Returns
    -------
    np.ndarray
        Imagen de tamaño 512x512x3 con valores aleatorios.
    """
    return np.random.randint(0, 256, (512, 512), dtype=np.uint8)


@pytest.fixture
def mock_model(monkeypatch):
    """
    Crea un mock de tf.keras.models.load_model.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Herramienta para parchear funciones/modulos.

    Returns
    -------
    MagicMock
        Modelo simulado con métodos predict y get_layer.
    """
    mock = MagicMock()
    mock.predict.return_value = np.array([[0.1, 0.7, 0.2]])
    mock.get_layer.return_value.output = np.random.rand(1, 16, 16, 32)
    
    # Patch de load_model para devolver el mock
    monkeypatch.setattr("tensorflow.keras.models.load_model", lambda path, compile=False: mock)
    return mock


def test_read_dicom(dummy_image, monkeypatch):
    """
    Prueba la función read_dicom de GradCAMModel.

    Parameters
    ----------
    dummy_image : np.ndarray
        Imagen de prueba.
    monkeypatch : pytest.MonkeyPatch
        Herramienta para parchear funciones/modulos.
    """
    mock_dcm = MagicMock()
    mock_dcm.pixel_array = dummy_image
    monkeypatch.setattr("pydicom.dcmread", lambda path: mock_dcm)
    
    img_rgb, img_pil = GradCAMModel.read_dicom("dummy_path.dcm")
    assert isinstance(img_rgb, np.ndarray)
    assert isinstance(img_pil, Image.Image)
    assert img_rgb.shape[2] == 3


def test_read_jpg(dummy_image, monkeypatch):
    """
    Prueba la función read_jpg de GradCAMModel.

    Parameters
    ----------
    dummy_image : np.ndarray
        Imagen de prueba.
    monkeypatch : pytest.MonkeyPatch
        Herramienta para parchear funciones/modulos.
    """
    monkeypatch.setattr("cv2.imread", lambda path: dummy_image)
    img_array, img_pil = GradCAMModel.read_jpg("dummy_path.jpg")
    assert isinstance(img_array, np.ndarray)
    assert isinstance(img_pil, Image.Image)
    assert img_array.shape == dummy_image.shape


def test_preprocess(dummy_image):
    """
    Prueba la función preprocess de GradCAMModel.

    Parameters
    ----------
    dummy_image : np.ndarray
        Imagen de prueba.
    """
    batch = GradCAMModel.preprocess(dummy_image)
    assert batch.shape == (1, 512, 512, 1)
    assert np.all((0 <= batch) & (batch <= 1))


def test_predict_and_grad_cam(dummy_image, mock_model):
    """
    Prueba la predicción y generación de Grad-CAM.

    Parameters
    ----------
    dummy_image : np.ndarray
        Imagen de prueba.
    mock_model : MagicMock
        Modelo simulado para evitar cálculo real de TensorFlow.
    """
    grad_cam_model = GradCAMModel("conv_MLP_84.h5")
    
    # Patch grad_cam para no hacer cálculo real
    grad_cam_model.grad_cam = lambda x, layer_name="conv10_thisone": np.zeros((512, 512, 3), dtype=np.uint8)
    
    label, prob, heatmap = grad_cam_model.predict(dummy_image)
    assert label in ["bacteriana", "normal", "viral"]
    assert 0 <= prob <= 100
    assert heatmap.shape == (512, 512, 3)
