# -*- coding: utf-8 -*-
"""
Preprocesamiento de imágenes médicas.
"""
import numpy as np
import cv2


class Preprocessor:
    """
    Clase para el preprocesamiento de imágenes médicas.

    Métodos
    -------
    preprocess(array: np.ndarray) -> np.ndarray:
        Preprocesa una imagen aplicando resize, escala de grises,
        CLAHE, normalización y expansión de dimensiones.
    """

    @staticmethod
    def preprocess(array: np.ndarray) -> np.ndarray:
        """
        Preprocesa una imagen para usarla en modelos CNN.

        Pasos:
        1. Redimensionar a 512x512
        2. Convertir a escala de grises
        3. Aplicar CLAHE (ecualización adaptativa del histograma)
        4. Normalizar valores [0, 1]
        5. Expandir dimensiones (batch y canal)

        Parameters
        ----------
        array : np.ndarray
            Imagen original en formato NumPy.

        Returns
        -------
        np.ndarray
            Imagen preprocesada lista para el modelo.
        """
        array = cv2.resize(array, (512, 512))
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        array = clahe.apply(array)
        array = array / 255.0
        array = np.expand_dims(array, axis=-1)
        array = np.expand_dims(array, axis=0)
        return array