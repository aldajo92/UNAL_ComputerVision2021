# Use a more recent version of miniconda3
FROM continuumio/miniconda3:latest

# Set labels for documentation
LABEL maintainer="Computer Vision Project UNAL"
LABEL description="JupyterLab server for Computer Vision Project with OpenCV"

# Set working directory
WORKDIR /workspace

# Copy environment file first for better layer caching
COPY ./environment.yml ./environment.yml

# Install system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create conda environment with all dependencies including jupyter
RUN conda env create -f environment.yml && \
    conda clean -afy

# Initialize conda for bash shell
RUN conda init bash

# Install JupyterLab and nb_conda_kernels in base environment to manage kernels
RUN conda install -n base -c conda-forge jupyterlab=3 nb_conda_kernels -y && \
    conda clean -afy

# Create JupyterLab configuration directory
RUN mkdir -p /root/.jupyter/lab/user-settings/@jupyterlab/apputils-extension

# Copy JupyterLab configuration files
COPY ./.jupyter_config_files/jupyter_lab_config.py /root/.jupyter/jupyter_lab_config.py
COPY ./.jupyter_config_files/jupyter_lab_config.json /root/.jupyter/jupyter_lab_config.json
COPY ./.jupyter_config_files/themes.jupyterlab-settings /root/.jupyter/lab/user-settings/@jupyterlab/apputils-extension/themes.jupyterlab-settings

# Expose JupyterLab port
EXPOSE 8899

# Set the default working directory for JupyterLab
WORKDIR /workspace/ComputerVisionProject

# Start JupyterLab
# Using shell form to properly initialize conda
CMD ["bash", "-c", "source activate opencv && jupyter-lab --ip=0.0.0.0 --port=8899 --no-browser --allow-root --notebook-dir=/workspace/ComputerVisionProject"]
