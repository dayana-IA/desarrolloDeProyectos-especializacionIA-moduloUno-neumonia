#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Módulo para cargar un modelo de red neuronal convolucional previamente
entrenado usando el patrón Singleton. La ruta del modelo se obtiene
desde un archivo de configuración JSON.
"""

import os
import json
import tensorflow as tf


class ModelLoader:
    """
    Clase Singleton encargada de cargar y mantener una única instancia
    del modelo de red neuronal convolucional.
    """

    _instance = None
    _model = None

    def __new__(cls, config_file: str = "config.json"):
        """
        Controla la creación de instancias para garantizar
        el patrón Singleton.
        """
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance.config_file = config_file
            cls._instance.model_path = cls._instance._read_config()
        return cls._instance

    def _read_config(self) -> str:
        """
        Lee la ruta del modelo desde un archivo JSON.

        Returns:
            str: Ruta del archivo .h5 definida en config.json.

        Raises:
            FileNotFoundError: Si el archivo de configuración no existe.
            KeyError: Si falta la clave 'model_path'.
        """
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(
                f"No se encontró el archivo de configuración: {self.config_file}"
            )

        with open(self.config_file, "r", encoding="utf-8") as file:
            config = json.load(file)

        if "model_path" not in config:
            raise KeyError("El archivo JSON debe contener 'model_path'.")

        return config["model_path"]

    def load_model(self) -> tf.keras.Model:
        """
        Carga el modelo desde archivo si aún no está cargado.
        Si ya está cargado, devuelve la misma instancia.

        Returns:
            tf.keras.Model: Modelo cargado.
        """
        if self._model is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"No se encontró el archivo del modelo: {self.model_path}"
                )
            self._model = tf.keras.models.load_model(
                self.model_path, compile=False
            )
        return self._model
