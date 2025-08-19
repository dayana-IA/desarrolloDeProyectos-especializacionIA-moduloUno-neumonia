# -*- coding: utf-8 -*-
"""
Preprocesamiento de imágenes médicas.

Esta clase proporciona métodos para:
1. Leer imágenes desde archivos DICOM o JPG/PNG.
2. Preprocesar imágenes para entrada en modelos CNN:
   - Redimensionar a 512x512.
   - Convertir a escala de grises.
   - Aplicar CLAHE (ecualización adaptativa de histograma).
   - Normalizar valores a [0, 1].
   - Expandir dimensiones de batch y canal.
"""

import numpy as np
import cv2
from PIL import Image
import pydicom as dicom


class PreProcessor:
    """
    Clase para el preprocesamiento y lectura de imágenes médicas.

    Métodos
    -------
    read_dicom(path: str) -> tuple[np.ndarray, Image.Image]
        Lee un archivo DICOM y devuelve un array RGB y un objeto PIL.Image.
    
    read_jpg(path: str) -> tuple[np.ndarray, Image.Image]
        Lee un archivo JPG/PNG y devuelve un array y un objeto PIL.Image.

    preprocess(array: np.ndarray) -> np.ndarray
        Preprocesa una imagen para modelos CNN, aplicando resize, gris,
        CLAHE, normalización y expansión de dimensiones.
    """

    @staticmethod
    def read_dicom(path: str) -> tuple[np.ndarray, Image.Image]:
        """
        Lee un archivo DICOM y lo convierte a imagen RGB y PIL.Image.

        Parameters
        ----------
        path : str
            Ruta al archivo DICOM.

        Returns
        -------
        img_rgb : np.ndarray
            Imagen en formato RGB con shape (H, W, 3).
        img_pil : PIL.Image.Image
            Imagen en formato PIL.Image para mostrar en UI.
        """
        img = dicom.dcmread(path)
        img_array = img.pixel_array
        img_pil = Image.fromarray(img_array)
        img_norm = np.uint8((np.maximum(img_array, 0) / img_array.max()) * 255.0)
        img_rgb = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2RGB)
        return img_rgb, img_pil

    @staticmethod
    def read_jpg(path: str) -> tuple[np.ndarray, Image.Image]:
        """
        Lee un archivo JPG o PNG y lo convierte a array y PIL.Image.

        Parameters
        ----------
        path : str
            Ruta al archivo de imagen.

        Returns
        -------
        img_array : np.ndarray
            Imagen como array (H x W x 3).
        img_pil : PIL.Image.Image
            Imagen en formato PIL.Image.
        """
        img = cv2.imread(path)
        img_array = np.asarray(img)
        img_pil = Image.fromarray(img_array)
        return img_array, img_pil

    @staticmethod
    def preprocess(array: np.ndarray) -> np.ndarray:
        """
        Preprocesa una imagen para usarla en modelos CNN.

        Pasos
        -----
        1. Redimensionar a 512x512.
        2. Convertir a escala de grises si es RGB.
        3. Aplicar CLAHE (ecualización adaptativa de histograma).
        4. Normalizar valores a [0, 1].
        5. Expandir dimensiones de batch y canal para entrada CNN.

        Parameters
        ----------
        array : np.ndarray
            Imagen original en formato NumPy.

        Returns
        -------
        np.ndarray
            Imagen preprocesada lista para el modelo con shape (1, 512, 512, 1).
        """
        array_resized = cv2.resize(array, (512, 512))
        # Convertir a gris si es RGB
        if len(array_resized.shape) == 3 and array_resized.shape[2] == 3:
            gray = cv2.cvtColor(array_resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = array_resized
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        clahe_img = clahe.apply(gray)
        normalized = clahe_img / 255.0
        batch_array = np.expand_dims(normalized, axis=-1)  # canal
        batch_array = np.expand_dims(batch_array, axis=0)  # batch
        return batch_array
