#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clase para la predicción de neumonía y generación de Grad-CAM.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import pydicom as dicom


class GradCAMModel:
    """
    Clase que encapsula la carga del modelo, preprocesamiento,
    predicción y generación de mapas de calor Grad-CAM.

    Attributes
    ----------
    model : tf.keras.Model
        Modelo de red neuronal convolucional cargado.
    """

    def __init__(self, model_path):
        """
        Inicializa la clase cargando el modelo desde un archivo .h5.

        Parameters
        ----------
        model_path : str
            Ruta del archivo de modelo.
        """
        self.model = tf.keras.models.load_model(model_path, compile=False)

    @staticmethod
    def read_dicom(path: str) -> tuple[np.ndarray, Image.Image]:
        """
        Lee un archivo DICOM y devuelve imagen RGB y PIL.Image.

        Parameters
        ----------
        path : str
            Ruta al archivo DICOM.

        Returns
        -------
        img_rgb : np.ndarray
            Imagen en formato RGB (512x512x3).
        img_pil : PIL.Image.Image
            Imagen en formato PIL.Image.
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
        Lee un archivo JPG/PNG y devuelve imagen como array y PIL.Image.

        Parameters
        ----------
        path : str
            Ruta al archivo de imagen.

        Returns
        -------
        img_array : np.ndarray
            Imagen como array (H x W x 3).
        img_pil : PIL.Image.Image
            Imagen como PIL.Image.
        """
        img = cv2.imread(path)
        img_array = np.asarray(img)
        img_pil = Image.fromarray(img_array)
        return img_array, img_pil

    @staticmethod
    def preprocess(array: np.ndarray) -> np.ndarray:
        """
        Preprocesa la imagen para la entrada del modelo.
        Convierte a escala de grises si es RGB, aplica CLAHE y normaliza.

        Parameters
        ----------
        array : np.ndarray
            Imagen original (H x W x C) o (H x W).

        Returns
        -------
        batch_array : np.ndarray
            Imagen preprocesada lista para el modelo con shape (1, 512, 512, 1).
        """
        array_resized = cv2.resize(array, (512, 512))
        # Si la imagen es RGB, convertir a gris
        if len(array_resized.shape) == 3 and array_resized.shape[2] == 3:
            gray = cv2.cvtColor(array_resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = array_resized  # ya es 2D
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        clahe_img = clahe.apply(gray)
        normalized = clahe_img / 255.0
        batch_array = np.expand_dims(normalized, axis=-1)
        batch_array = np.expand_dims(batch_array, axis=0)
        return batch_array

    def grad_cam(self, array: np.ndarray, layer_name: str = "conv10_thisone") -> np.ndarray:
        """
        Genera un mapa de calor Grad-CAM sobre la imagen.

        Parameters
        ----------
        array : np.ndarray
            Imagen original (H x W x C) o (H x W).
        layer_name : str, optional
            Nombre de la capa convolucional de interés.

        Returns
        -------
        superimposed_img : np.ndarray
            Imagen con mapa de calor aplicado (512x512x3).
        """
        img_input = self.preprocess(array)
        preds = self.model.predict(img_input)
        argmax = int(np.argmax(preds[0]))

        conv_layer = self.model.get_layer(layer_name)
        grad_model = tf.keras.models.Model(
            [self.model.inputs], [conv_layer.output, self.model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_input)
            loss = predictions[:, argmax]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0].numpy()
        pooled_grads = pooled_grads.numpy()

        for i in range(conv_outputs.shape[-1]):
            conv_outputs[:, :, i] *= pooled_grads[i]

        heatmap = np.mean(conv_outputs, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)

        heatmap = cv2.resize(heatmap, (img_input.shape[2], img_input.shape[1]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        img_resized = cv2.resize(array, (512, 512))
        superimposed_img = cv2.addWeighted(img_resized, 1.0, heatmap, 0.4, 0)
        return superimposed_img

    def predict(self, array: np.ndarray) -> tuple[str, float, np.ndarray]:
        """
        Predice la clase de neumonía y genera Grad-CAM.

        Parameters
        ----------
        array : np.ndarray
            Imagen original (H x W x C) o (H x W).

        Returns
        -------
        label : str
            Clase predicha ('bacteriana', 'normal', 'viral').
        probability : float
            Probabilidad de la clase en porcentaje.
        heatmap_img : np.ndarray
            Imagen con Grad-CAM superpuesto.
        """
        batch_img = self.preprocess(array)
        preds = self.model.predict(batch_img, verbose=0)
        pred_class = int(np.argmax(preds))
        probability = float(np.max(preds)) * 100
        label = {0: "bacteriana", 1: "normal", 2: "viral"}.get(pred_class, "desconocida")
        heatmap_img = self.grad_cam(array)
        return label, probability, heatmap_img
