# src/neumonia/detector_neumonia.py
# -*- coding: utf-8 -*-
"""
UI para la detección de neumonía usando el integrador.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tkinter import *
from tkinter import ttk, font, filedialog
from tkinter.messagebox import askokcancel, showinfo, WARNING
from PIL import ImageTk, Image
import pyautogui

# Importar integrador
from .integrator import Integrator

class App:
    def __init__(self):
        self.root = Tk()
        self.root.title("Detección de Neumonía")
        fonti = font.Font(weight="bold")

        self.root.geometry("815x560")
        self.root.resizable(0, 0)

        # Crear instancia del integrador
        self.integrator = Integrator()

        # Labels
        self.lab1 = ttk.Label(self.root, text="Imagen Radiográfica", font=fonti)
        self.lab2 = ttk.Label(self.root, text="Imagen con Heatmap", font=fonti)
        self.lab3 = ttk.Label(self.root, text="Resultado:", font=fonti)
        self.lab4 = ttk.Label(self.root, text="Cédula Paciente:", font=fonti)
        self.lab5 = ttk.Label(self.root, text="SOFTWARE PARA APOYO AL DIAGNÓSTICO", font=fonti)
        self.lab6 = ttk.Label(self.root, text="Probabilidad:", font=fonti)

        # Variables
        self.ID = StringVar()
        self.array = None
        self.reportID = 0
        self.label = ""
        self.proba = 0.0
        self.heatmap = None

        # Inputs
        self.text1 = ttk.Entry(self.root, textvariable=self.ID, width=10)
        self.text_img1 = Text(self.root, width=31, height=15)
        self.text_img2 = Text(self.root, width=31, height=15)
        self.text2 = Text(self.root)
        self.text3 = Text(self.root)

        # Buttons
        self.button1 = ttk.Button(self.root, text="Predecir", state="disabled", command=self.run_model)
        self.button2 = ttk.Button(self.root, text="Cargar Imagen", command=self.load_img_file)
        self.button3 = ttk.Button(self.root, text="Borrar", command=self.delete)
        self.button4 = ttk.Button(self.root, text="PDF", command=self.create_pdf)
        self.button6 = ttk.Button(self.root, text="Guardar", command=self.save_results_csv)

        # Widget positions
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

    # Métodos
    def load_img_file(self):
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
        if filepath:
            self.array, img2show = self.integrator.load_image(filepath)
            self.img1 = img2show.resize((250, 250), Image.Resampling.LANCZOS)
            self.img1 = ImageTk.PhotoImage(self.img1)
            self.text_img1.image_create(END, image=self.img1)
            self.button1["state"] = "enabled"

    def run_model(self):
        patient_id = self.ID.get()
        self.label, self.proba, self.heatmap = self.integrator.process_image("", patient_id) if self.array is None else self.integrator.process_image_from_array(self.array, patient_id)
        
        self.img2 = Image.fromarray(self.heatmap)
        self.img2 = self.img2.resize((250, 250), Image.Resampling.LANCZOS)
        self.img2 = ImageTk.PhotoImage(self.img2)
        self.text_img2.image_create(END, image=self.img2)
        self.text2.insert(END, self.label)
        self.text3.insert(END, f"{self.proba:.2f}%")

    def save_results_csv(self):
        patient_id = self.ID.get()
        self.integrator.save_result(patient_id, self.label, self.proba)
        showinfo(title="Guardar", message="Los datos se guardaron con éxito.")

    def create_pdf(self):
        pdf_path = self.integrator.generate_pdf(self.root, self.reportID)
        self.reportID += 1
        showinfo(title="PDF", message=f"PDF generado en {pdf_path}")

    def delete(self):
        if askokcancel(title="Confirmación", message="Se borrarán todos los datos.", icon=WARNING):
            self.text1.delete(0, "end")
            self.text2.delete(1.0, "end")
            self.text3.delete(1.0, "end")
            self.text_img1.delete(1.0, "end")
            self.text_img2.delete(1.0, "end")
            showinfo(title="Borrar", message="Datos borrados con éxito.")
