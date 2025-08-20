#  Detección de Neumonía con Deep Learning y Grad-CAM

Esta aplicación usa Deep Learning* para analizar radiografías de tórax (DICOM y JPG/PNG).  
El sistema carga la imagen, la preprocesa (resize, normalización, CLAHE), ejecuta la inferencia con un modelo Keras (.h5), muestra el resultado con probabilidad y genera un **Grad-CAM** para interpretabilidad.  
Además, permite guardar resultados en CSV y exportar un reporte en PDF.  

El modelo actual clasifica en tres categorías:  
- Neumonía Bacteriana  
- Neumonía Viral 
- Sin Neumonía 

La arquitectura es modular: lectura, preprocesamiento, carga del modelo, integración y Grad-CAM, con interfaz gráfica en Tkinter.

---

## Requisitos previos e instalación

Se recomienda usar **Python 3.11** con un entorno virtual administrado por `uv` (reproducible y ligero). Alternativamente puede usarse Conda.

  ```
  # Crear y activar entorno
  uv venv
  # Windows (PowerShell)
  .\.venv\Scripts\Activate.ps1
  # macOS/Linux
  source .venv/bin/activate

  # Instalar dependencias
  uv pip install -r requirements.txt
 ```

## Modelo (.h5)

Descargar conv_MLP_84.h5 según la URL en models/download_model.txt y colocarlo en models/.
El archivo está ignorado en Git:

  models/*.h5

## Ejecución

Con el entorno activo

  python main.py

Se abrirá la interfaz. Ingrese la cédula del paciente, cargue una imagen (DICOM/JPG/PNG), presione Predecir para ver la clase y probabilidad, revise el heatmap, y use Guardar o PDF según necesidad.

## Estructura del proyecto

.
├── main.py
├── models/
│   ├── download_model.txt
│   └── conv_MLP_84.h5        # local, no versionado
├── outputs/
│   └── reportes/
│       ├── gradcam/          # capturas generadas por Grad-CAM
│       └── ui/               # capturas de la interfaz
├── src/
│   └── neumonia/
│       ├── detector_neumonia.py   # interfaz (Tkinter)
│       ├── integrator.py          # orquestación del flujo
│       ├── dicom_reader.py        # lectura DICOM/JPG
│       ├── pre_processor.py       # resize, CLAHE, normalización
│       ├── load_model.py          # carga del .h5
│       ├── grad_cam.py            # mapa de calor
│       ├── csv_handler.py         # guardado en CSV
│       └── pdf_generator.py       # generación de PDF
├── tests/                         # pruebas unitarias (pytest)
├── requirements.txt
└── README.md

## Pruebas y estilo

Ejecutar pruebas unitarias con:

  pytest -q

Verificar estilo y PEP8:

  uv pip install flake8
  flake8 src

## Docker (opcional)

Para portabilidad y reproducibilidad en cualquier equipo:

  FROM python:3.11-slim
  RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-tk tk tcl libgl1 libglib2.0-0 fonts-dejavu && rm -rf /var/lib/apt/lists/*
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt
  COPY . .
  CMD ["python", "main.py"]

Construir y ejecutar montando modelos/datos/reportes:

  docker build -t neumonia-app .
  docker run --rm \
    -v "$PWD/models:/app/models" \
    -v "$PWD/data:/app/data" \
    -v "$PWD/outputs:/app/outputs" \
  neumonia-app

Para mostrar la GUI desde Docker en Windows, usar un servidor X (p. ej. VcXsrv):

docker run -it --rm -e DISPLAY=host.docker.internal:0.0 \
docker run -it --rm -e DISPLAY=host.docker.internal:0.0 \
  -v "$PWD/models:/app/models" \
  -v "$PWD/data:/app/data" \
  -v "$PWD/outputs:/app/outputs" \
  neumonia-app

## Demo

Interfaz gráfica (Tkinter)  
![Interfaz](outputs/reportes/ui.png)

Ejemplo de Grad-CAM (regiones relevantes resaltadas)  
![Grad-CAM](outputs/reportes/gradcam.png)

## Contribución y créditos

Las contribuciones se hacen por ramas y Pull Requests.

Proyecto original

  Isabella Torres Revelo — https://github.com/isa-tr
  Nicolás Díaz Salazar — https://github.com/nicolasdiazsalazar

Adaptación y mejoras (Módulo 1 – 2025)

  Dayana Muñoz Muñoz — https://github.com/dayana-IA
  Jonatan Paz Guzmán — https://github.com/jonatan-paz-guzman-ia
  Daniel Carlosama Martínez — https://github.com/21danka

Aportes de esta versión

  Refactorización y organización modular
  Corrección de dependencias
  Integración del flujo (lectura → preprocesamiento → modelo → Grad-CAM → PDF)
  Estandarización PEP8 y docstrings
  Pruebas unitarias (pytest)
  Documentación (README) y Dockerización
  Entorno reproducible con uv

Docencia / Acompañamiento
  Jan Polanco Velasco (Senior Data Scientist)

## Licencia

MIT