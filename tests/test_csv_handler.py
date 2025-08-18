import pytest
from unittest.mock import patch, MagicMock
import os
import tempfile
import json
from src.neumonia.csv_handler import CSVHandler

@pytest.fixture
def config_file(tmp_path):
    """
    Crea un archivo de configuración temporal con la ruta del CSV.
    """
    csv_path = tmp_path / "historial.csv"
    config = {"csv_path": str(csv_path)}
    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
    return str(config_path), str(csv_path)

def test_csvhandler_save_result(config_file):
    config_path, csv_path = config_file

    with patch("src.neumonia.csv_handler.showinfo") as mock_show:
        handler = CSVHandler(config_path)
        handler.save_result("12345", "bacteriana", 87.65)

        # Verificar que el CSV se haya creado
        assert os.path.exists(csv_path)

        # Verificar contenido del CSV
        with open(csv_path, "r") as f:
            lines = f.readlines()
        assert len(lines) == 1
        assert lines[0].strip() == "12345-bacteriana-87.65%"

        # Verificar que showinfo fue llamado
        mock_show.assert_called_once()
