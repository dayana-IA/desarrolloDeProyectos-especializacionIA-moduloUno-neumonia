# Imagen base ligera con Python 3.11
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependencias del sistema necesarias para Tkinter y OpenCV
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python3-tk tk tcl libgl1 libglib2.0-0 fonts-dejavu \
    libsm6 libxext6 libxrender-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Crear entorno virtual
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Establecer directorio de trabajo
WORKDIR /app

# Copiar el contenido del proyecto al contenedor
COPY . .

# Instalar dependencias
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Establecer variable de entorno para la GUI (Tkinter)
ENV DISPLAY=:0

# Comando por defecto para ejecutar la app
CMD ["python", "main.py"]