#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Módulo para manejar el guardado de resultados de predicción en CSV.

Este módulo lee la ruta de salida desde un archivo de configuración JSON,
asegura que la carpeta exista y guarda los resultados en formato CSV.
"""

import csv
import json
import os
from tkinter.messagebox import showinfo


class CSVHandler:
    """
    Clase para guardar resultados de predicción en un archivo CSV.

    Attributes
    ----------
    csv_path : str
        Ruta al archivo CSV donde se guardarán los resultados.
    """

    def __init__(self, config_path: str = "config.json"):
        """
        Inicializa la clase leyendo la ruta del CSV desde un archivo de configuración JSON.
        Crea la carpeta si no existe.

        Parameters
        ----------
        config_path : str, optional
            Ruta al archivo de configuración JSON (por defecto "config.json").
        """
        with open(config_path, "r") as f:
            config = json.load(f)
        self.csv_path = config.get("csv_path")
        if self.csv_path is None:
            raise ValueError("El archivo de configuración no contiene 'csv_path'")

        # Crear carpeta si no existe
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)

    def save_result(self, patient_id: str, label: str, probability: float):
        """
        Guarda un resultado de predicción en el archivo CSV.

        Parameters
        ----------
        patient_id : str
            Identificación del paciente.
        label : str
            Clase predicha.
        probability : float
            Probabilidad de la clase en porcentaje.
        """
        with open(self.csv_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter="-")
            writer.writerow([patient_id, label, f"{probability:.2f}%"])
        showinfo(title="Guardar", message="Los datos se guardaron con éxito.")
