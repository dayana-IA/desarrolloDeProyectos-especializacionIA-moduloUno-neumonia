#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clase para la predicción de neumonía y generación de Grad-CAM.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import pydicom as dicom


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

    def grad_cam(self,img_input:np.ndarray, array: np.ndarray, layer_name: str = "conv10_thisone") -> np.ndarray:
        """
        Genera un mapa de calor Grad-CAM sobre la imagen.
        """
        
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

    
