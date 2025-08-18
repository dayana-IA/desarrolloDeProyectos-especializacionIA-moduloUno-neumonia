# -*- coding: utf-8 -*-
"""
Lectura de archivos DICOM.
"""
import cv2
import pydicom as dicom
import numpy as np
from PIL import Image


class DicomReader:
    """
    Clase para la lectura de imágenes DICOM.

    Métodos
    -------
    read(path: str) -> tuple[np.ndarray, Image]:
        Lee un archivo DICOM y devuelve una versión RGB (np.array)
        y otra versión en formato PIL (Image).
    """

    @staticmethod
    def read(path: str):
        """
        Lee un archivo DICOM y lo convierte en RGB y en formato PIL.

        Parameters
        ----------
        path : str
            Ruta del archivo DICOM.

        Returns
        -------
        tuple
            img_rgb : np.ndarray
                Imagen en formato RGB.
            img2show : PIL.Image
                Imagen en formato PIL para visualización.
        """
        img = dicom.dcmread(path)
        img_array = img.pixel_array
        img2show = Image.fromarray(img_array)

        img2 = img_array.astype(float)
        img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
        img2 = np.uint8(img2)
        img_rgb = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)

        return img_rgb, img2show