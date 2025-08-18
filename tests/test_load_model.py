# tests/test_load_model.py
import os
import json
import pytest
import tensorflow as tf
from src.neumonia.load_model import ModelLoader

TEST_CONFIG = "test_config.json"
TEST_MODEL_PATH = "conv_MLP_84.h5"  # Modelo dummy


@pytest.fixture(scope="function")
def create_test_config():
    """
    Fixture para crear un archivo JSON temporal de configuración.
    """
    config = {"model_path": TEST_MODEL_PATH}
    with open(TEST_CONFIG, "w", encoding="utf-8") as f:
        json.dump(config, f)
    yield
    # Cleanup
    if os.path.exists(TEST_CONFIG):
        os.remove(TEST_CONFIG)
    # Resetear singleton para que no interfiera con otros tests
    ModelLoader._instance = None
    ModelLoader._model = None


def test_singleton(create_test_config):
    """
    Verifica que ModelLoader respete el patrón Singleton.

    Al crear múltiples instancias con la misma configuración,
    todas deben apuntar al mismo objeto.

    Parameters
    ----------
    create_test_config : fixture
        Fixture que crea un archivo de configuración temporal.
    """
    loader1 = ModelLoader(config_file=TEST_CONFIG)
    loader2 = ModelLoader(config_file=TEST_CONFIG)
    assert loader1 is loader2, "ModelLoader no está respetando el patrón Singleton"


def test_file_not_found():
    """
    Verifica que ModelLoader lance FileNotFoundError si el archivo de configuración no existe.
    """
    # Resetear singleton antes de la prueba
    ModelLoader._instance = None
    ModelLoader._model = None
    with pytest.raises(FileNotFoundError):
        ModelLoader(config_file="non_existent_config.json")


def test_model_file_not_found(tmp_path):
    """
    Verifica que load_model lance FileNotFoundError si el modelo especificado no existe.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directorio temporal proporcionado por pytest.
    """
    # Resetear singleton
    ModelLoader._instance = None
    ModelLoader._model = None

    # Crear config apuntando a modelo inexistente
    config_path = tmp_path / "config.json"
    config = {"model_path": "non_existent_model.h5"}
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f)

    loader = ModelLoader(config_file=str(config_path))
    with pytest.raises(FileNotFoundError):
        loader.load_model()
