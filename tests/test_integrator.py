import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.neumonia.integrator import Integrator

@pytest.fixture
def integrator():
    with patch("src.neumonia.load_model.ModelLoader.load_model") as mock_load_model, \
         patch("src.neumonia.grad_cam.GradCAMModel") as mock_gradcam_class, \
         patch("src.neumonia.pre_processor.PreProcessor.read_jpg") as mock_read_jpg, \
         patch("src.neumonia.pre_processor.PreProcessor.read_dicom") as mock_read_dicom, \
         patch("src.neumonia.pre_processor.PreProcessor.preprocess") as mock_preprocess, \
         patch("src.neumonia.csv_handler.CSVHandler.save") as mock_csv_save, \
         patch("src.neumonia.pdf_generator.PDFGenerator.create_pdf") as mock_create_pdf:

        # Mock del modelo
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.1, 0.8, 0.1]])
        mock_load_model.return_value = mock_model

        # Mock de Grad-CAM
        mock_gradcam_instance = MagicMock()
        mock_gradcam_instance.generate.return_value = np.zeros((224, 224, 3), dtype=np.uint8)
        mock_gradcam_class.return_value = mock_gradcam_instance

        # Mock de imágenes
        dummy_array = np.zeros((224, 224, 3))
        dummy_image = MagicMock()
        mock_read_jpg.return_value = (dummy_array, dummy_image)
        mock_read_dicom.return_value = (dummy_array, dummy_image)

        # Mock de preprocesamiento
        mock_preprocess.return_value = np.expand_dims(dummy_array, axis=0)

        # Mock CSV y PDF
        mock_csv_save.return_value = None
        mock_create_pdf.return_value = "dummy_report.pdf"

        integrator_instance = Integrator(config_path="tests/test_config.json")
        yield integrator_instance

def test_load_image_jpg(integrator):
    array, img = integrator.load_image("image.jpg")
    assert isinstance(array, np.ndarray)
    assert img is not None

def test_process_image(integrator):
    label, prob, heatmap = integrator.process_image("image.jpg", "patient_01")
    assert label in ["bacteriana", "normal", "viral"]
    assert 0 <= prob <= 100
    assert isinstance(heatmap, np.ndarray)

def test_save_result(integrator):
    integrator.save_result("patient_01", "bacteriana", 90.0)

def test_generate_pdf(integrator):
    class DummyRoot:
        pass
    pdf_path = integrator.generate_pdf(DummyRoot(), report_id=1)
    assert pdf_path == "dummy_report.pdf"
