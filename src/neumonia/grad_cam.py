#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clase para la predicción de neumonía y generación de Grad-CAM.
"""

import os
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import pydicom as dicom


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class GradCAMModel:
    """
    Clase que encapsula la carga del modelo, preprocesamiento,
    predicción y generación de mapas de calor Grad-CAM.
    """

    def __init__(self, model):
        """
        Inicializa la clase con un modelo Keras o una ruta a un modelo.
        """
        if isinstance(model, str):
            self.model = tf.keras.models.load_model(model, compile=False)
        else:
            self.model = model

    @staticmethod
    def read_dicom(path: str) -> tuple[np.ndarray, Image.Image]:
        """Lee imagen DICOM y devuelve array para modelo y objeto PIL para UI"""
        img = dicom.dcmread(path)
        img_array = img.pixel_array
        img_pil = Image.fromarray(img_array)
        img_norm = np.uint8((np.maximum(img_array, 0) / img_array.max()) * 255.0)
        img_rgb = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2RGB)
        return img_rgb, img_pil

    @staticmethod
    def read_jpg(path: str) -> tuple[np.ndarray, Image.Image]:
        """Lee imagen JPG/PNG y devuelve array para modelo y objeto PIL para UI"""
        img = cv2.imread(path)
        img_array = np.asarray(img)
        img_pil = Image.fromarray(img_array)
        return img_array, img_pil

    @staticmethod
    def preprocess(array: np.ndarray) -> np.ndarray:
        """Preprocesamiento para CNN: resize, gris, CLAHE, normalización, batch"""
        array_resized = cv2.resize(array, (512, 512))
        if len(array_resized.shape) == 3 and array_resized.shape[2] == 3:
            gray = cv2.cvtColor(array_resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = array_resized
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        clahe_img = clahe.apply(gray)
        normalized = clahe_img / 255.0
        batch_array = np.expand_dims(normalized, axis=-1)  # Canal
        batch_array = np.expand_dims(batch_array, axis=0)   # Batch
        return batch_array

    def grad_cam(self, array: np.ndarray, layer_name: str = "conv10_thisone") -> np.ndarray:
        """
        Genera un mapa de calor Grad-CAM sobre la imagen.
        """
        img_input = self.preprocess(array)
        conv_layer = self.model.get_layer(layer_name)
        grad_model = tf.keras.models.Model(
            inputs=self.model.inputs,
            outputs=[conv_layer.output, self.model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_input)
            # Asegurar tensor float32
            predictions = tf.convert_to_tensor(predictions, dtype=tf.float32)
            # Índice de la clase más probable
            argmax = tf.argmax(predictions[0])
            # Selección segura de la clase con tf.gather
            loss = tf.gather(predictions[0], argmax)

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
        """
        batch_img = self.preprocess(array)
        preds = self.model.predict(batch_img, verbose=0)
        pred_class = int(np.argmax(preds[0]))
        probability = float(np.max(preds[0])) * 100
        label = {0: "bacteriana", 1: "normal", 2: "viral"}.get(pred_class, "desconocida")
        heatmap_img = self.grad_cam(array)
        return label, probability, heatmap_img
