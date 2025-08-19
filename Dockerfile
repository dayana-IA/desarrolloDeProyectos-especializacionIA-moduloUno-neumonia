# Imagen base más completa (bullseye en vez de slim)
FROM python:3.10-bullseye

ENV DEBIAN_FRONTEND=noninteractive

# Actualizar e instalar dependencias necesarias
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-opencv python3-tk python3-dev \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Crear entorno virtual
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /home/src
COPY . .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

ENV DISPLAY=:0

CMD ["python", "main.py"]
