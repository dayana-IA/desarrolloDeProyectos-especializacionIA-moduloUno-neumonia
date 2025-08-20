# src/neumonia/detector_neumonia.py
# -*- coding: utf-8 -*-
"""
UI para la detección de neumonía usando el integrador.
"""

from __future__ import annotations

# Stdlib
import os

# Terceros
import tkinter as tk
from tkinter import filedialog, font, ttk, messagebox
from PIL import Image, ImageTk

# Local
from .integrator import Integrator


# (Opcional) Silenciar logs de TF si este módulo llegara a importarlo de forma indirecta
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class App:
    """Aplicación Tkinter para cargar imágenes, predecir y exportar resultados."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Detección de Neumonía")

        font_bold = font.Font(weight="bold")
        self.root.geometry("815x560")
        self.root.resizable(0, 0)

        # Integrador
        self.integrator = Integrator()

        # Labels
        self.lab1 = ttk.Label(self.root, text="Imagen Radiográfica", font=font_bold)
        self.lab2 = ttk.Label(self.root, text="Imagen con Heatmap", font=font_bold)
        self.lab3 = ttk.Label(self.root, text="Resultado:", font=font_bold)
        self.lab4 = ttk.Label(self.root, text="Cédula Paciente:", font=font_bold)
        self.lab5 = ttk.Label(
            self.root,
            text="SOFTWARE PARA APOYO AL DIAGNÓSTICO",
            font=font_bold,
        )
        self.lab6 = ttk.Label(self.root, text="Probabilidad:", font=font_bold)

        # Estado
        self.patient_id_var = tk.StringVar()
        self.array = None
        self.report_id = 0
        self.label = ""
        self.proba = 0.0
        self.heatmap = None

        # Inputs
        self.text1 = ttk.Entry(self.root, textvariable=self.patient_id_var, width=10)
        self.text_img1 = tk.Text(self.root, width=31, height=15)
        self.text_img2 = tk.Text(self.root, width=31, height=15)
        self.text2 = tk.Text(self.root)
        self.text3 = tk.Text(self.root)

        # Botones
        self.button1 = ttk.Button(
            self.root, text="Predecir", state="disabled", command=self.run_model
        )
        self.button2 = ttk.Button(
            self.root, text="Cargar Imagen", command=self.load_img_file
        )
        self.button3 = ttk.Button(self.root, text="Borrar", command=self.clear_all)
        self.button4 = ttk.Button(self.root, text="PDF", command=self.create_pdf)
        self.button6 = ttk.Button(
            self.root, text="Guardar", command=self.save_results_csv
        )

        # Posicionamiento
        self.lab1.place(x=110, y=65)
        self.lab2.place(x=545, y=65)
        self.lab3.place(x=500, y=350)
        self.lab4.place(x=65, y=350)
        self.lab5.place(x=122, y=25)
        self.lab6.place(x=500, y=400)
        self.button1.place(x=220, y=460)
        self.button2.place(x=70, y=460)
        self.button3.place(x=670, y=460)
        self.button4.place(x=520, y=460)
        self.button6.place(x=370, y=460)
        self.text1.place(x=200, y=350)
        self.text2.place(x=610, y=350, width=90, height=30)
        self.text3.place(x=610, y=400, width=90, height=30)
        self.text_img1.place(x=65, y=90)
        self.text_img2.place(x=500, y=90)

        self.text1.focus_set()
        self.root.mainloop()

    # -----------------------
    # Eventos / acciones UI
    # -----------------------
    def load_img_file(self) -> None:
        """Abrir selector de archivos y cargar imagen (DICOM/JPG/PNG)."""
        filepath = filedialog.askopenfilename(
            initialdir="/",
            title="Seleccionar imagen",
            filetypes=(
                ("DICOM", "*.dcm"),
                ("JPEG", "*.jpeg"),
                ("JPG", "*.jpg"),
                ("PNG", "*.png"),
            ),
        )
        if not filepath:
            return

        self.array, img2show = self.integrator.load_image(filepath)

        img_resized = img2show.resize((250, 250), Image.Resampling.LANCZOS)
        self.img1 = ImageTk.PhotoImage(img_resized)
        self.text_img1.image_create(tk.END, image=self.img1)
        self.button1["state"] = "enabled"

    def run_model(self) -> None:
        """Ejecutar el modelo y mostrar resultados y heatmap."""
        patient_id = self.patient_id_var.get()

        if self.array is None:
            self.label, self.proba, self.heatmap = self.integrator.process_image(
                "", patient_id
            )
        else:
            self.label, self.proba, self.heatmap = (
                self.integrator.process_image_from_array(self.array, patient_id)
            )

        img2 = Image.fromarray(self.heatmap).resize(
            (250, 250), Image.Resampling.LANCZOS
        )
        self.img2 = ImageTk.PhotoImage(img2)
        self.text_img2.image_create(tk.END, image=self.img2)

        self.text2.insert(tk.END, self.label)
        self.text3.insert(tk.END, f"{self.proba:.2f}%")

    def save_results_csv(self) -> None:
        """Guardar resultados en CSV y notificar al usuario."""
        patient_id = self.patient_id_var.get()
        self.integrator.save_result(patient_id, self.label, self.proba)
        messagebox.showinfo(title="Guardar", message="Los datos se guardaron con éxito.")

    def create_pdf(self) -> None:
        """Generar PDF del reporte actual y notificar ubicación."""
        pdf_path = self.integrator.generate_pdf(self.root, self.report_id)
        self.report_id += 1
        messagebox.showinfo(title="PDF", message=f"PDF generado en {pdf_path}")

    def clear_all(self) -> None:
        """Limpiar todos los campos previa confirmación."""
        if messagebox.askokcancel(
            title="Confirmación",
            message="Se borrarán todos los datos.",
            icon=messagebox.WARNING,
        ):
            self.text1.delete(0, "end")
            self.text2.delete(1.0, "end")
            self.text3.delete(1.0, "end")
            self.text_img1.delete(1.0, "end")
            self.text_img2.delete(1.0, "end")
            messagebox.showinfo(title="Borrar", message="Datos borrados con éxito.")
