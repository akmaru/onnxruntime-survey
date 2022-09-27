FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8
ENV PYTHONIOENCODING=utf-8

# Install ubuntu packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    git \
    gnupg \
    ninja-build \
    sudo \
    unzip \
    vim \
    wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY install/install_python_packages.sh /install/install_python_packages.sh
RUN bash /install/install_python_packages.sh

COPY install/install_onnx.sh /install/install_onnx.sh
RUN bash /install/install_onnx.sh


# Add a user that UID:GID will be updated by vscode
ARG USERNAME=developer
ARG GROUPNAME=developer
ARG UID=1000
ARG GID=1000
ARG PASSWORD=developer
RUN groupadd -g $GID $GROUPNAME && \
    useradd -m -s /bin/bash -u $UID -g $GID -G sudo $USERNAME && \
    echo $USERNAME:$PASSWORD | chpasswd && \
    echo "$USERNAME   ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER $USERNAME
ENV HOME /home/developer
ENV PATH $HOME/.local/bin:$PATH
ENV PYTHONPATH /workspaces/onnxruntime-survey/python:$PYTHONPATH


