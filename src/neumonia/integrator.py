#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Integrator: coordina todas las funcionalidades del proyecto para ser consumidas
por la UI. Separa responsabilidades y encapsula la lógica de negocio.
"""

import os
from pathlib import Path
from typing import Tuple
from PIL import Image
import numpy as np
import pydicom as dicom
import cv2

# Importar módulos funcionales
from src.neumonia.load_model import ModelLoader
from src.neumonia.pre_processor import PreProcessor
from src.neumonia.grad_cam import GradCAMModel
from src.neumonia.csv_handler import CSVHandler
from src.neumonia.pdf_generator import PDFGenerator


class Integrator:
    """
    Clase integradora que expone funcionalidades para la interfaz gráfica.
    """

    def __init__(self, config_path: str = "config.json"):
        """
        Inicializa el integrador con la configuración general.
        """
        # Instancias de módulos funcionales
        self.model_loader = ModelLoader()
        self.preprocessor = PreProcessor()
        self.model = self.model_loader.load_model()
        self.gradcam = GradCAMModel(self.model)
        self.csv_handler = CSVHandler(config_path=config_path)
        self.pdf_generator = PDFGenerator(config_path=config_path)

    def load_image(self, path: str):
        """
        Carga imagen DICOM y devuelve array y PIL.Image.
        Parameters
        ----------
        path : str
            Ruta al archivo de imagen.
        """
        return self.preprocessor.read_dicom(path)

    def process_image(self, image_path: str, patient_id: str) -> Tuple[str, float, np.ndarray]:
        """
        Procesa una imagen desde ruta de archivo.

        Parameters
        ----------
        image_path : str
            Ruta al archivo de imagen.
        patient_id : str
            Identificador del paciente.

        Returns
        -------
        label : str
        prob : float
        heatmap_array : np.ndarray
        """
        array, _ = self.load_image(image_path)
        # Preprocesar
        img_batch = self.preprocessor.preprocess(array)

        # Predecir
        preds = self.model.predict(img_batch, verbose=0)
        pred_class = int(np.argmax(preds[0]))
        prob = float(np.max(preds[0]) * 100)

        label_map = {0: "bacteriana", 1: "normal", 2: "viral"}
        label = label_map.get(pred_class, "desconocida")

        # Generar Grad-CAM
        heatmap_array = self.gradcam.grad_cam(array)

        return label, prob, heatmap_array

    def process_image_from_array(self, array: np.ndarray, patient_id: str) -> Tuple[str, float, np.ndarray]:
        """
        Procesa una imagen ya cargada como array: preprocesa, predice y genera Grad-CAM.

        Parameters
        ----------
        array : np.ndarray
            Imagen ya cargada en memoria.
        patient_id : str
            Identificador del paciente.

        Returns
        -------
        label : str
            Etiqueta predicha ('bacteriana', 'viral', 'normal').
        prob : float
            Probabilidad de la predicción (%).
        heatmap_array : np.ndarray
            Imagen con Grad-CAM superpuesto.
        """
        # Preprocesar
        img_batch = self.preprocessor.preprocess(array)

        # Predecir
        preds = self.model.predict(img_batch, verbose=0)
        pred_class = int(np.argmax(preds[0]))
        prob = float(np.max(preds[0]) * 100)

        label_map = {0: "bacteriana", 1: "normal", 2: "viral"}
        label = label_map.get(pred_class, "desconocida")

        # Generar Grad-CAM
        heatmap_array = self.gradcam.grad_cam(array)

        return label, prob, heatmap_array

    def save_result(self, patient_id: str, label: str, prob: float):
        """
        Guarda el resultado en CSV.
        """
        self.csv_handler.save_result(patient_id, label, prob)

    def generate_pdf(self, root, report_id: int) -> str:
        """
        Genera PDF de la ventana Tkinter.
        """
        return self.pdf_generator.create_pdf(root, report_id)
