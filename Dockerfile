FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Mostrar los mensajes sin esperar y evitar la caché de pip.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Ubuntu 22.04 proporciona Python 3.10. Docker ya aísla este entorno del host.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       python3 python3-pip python3-dev python-is-python3 build-essential git \
       ca-certificates ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Herramientas compatibles con la instalación de fairseq utilizada.
RUN python -m pip install pip==24.0 setuptools==69.5.1 wheel==0.45.1 \
       Cython==0.29.37 ninja==1.11.1.3

WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install -r requirements.txt

# .dockerignore limita la copia al código y excluye modelos, datos y salidas.
COPY . .

ARG FAIRSEQ_COMMIT=3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99
RUN python -m pip install . \
    && python -m pip install --no-build-isolation \
       "fairseq @ git+https://github.com/facebookresearch/fairseq.git@${FAIRSEQ_COMMIT}"

CMD ["python", "multi_task_inference.py", "--help"]
