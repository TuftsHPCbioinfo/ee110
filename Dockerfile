# Use the specified base image
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Author label
LABEL maintainer="Yucheng Zhang <Yucheng.Zhang@tufts.edu>"

# Help message
LABEL description="This container is for Tufts course EE110 by Dr. Mai Vu"


# Set environment variables
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends build-essential wget git && \
    wget https://github.com/conda-forge/miniforge/releases/download/25.3.1-0/Miniforge3-25.3.1-0-Linux-x86_64.sh  \
    && bash Miniforge3-25.3.1-0-Linux-x86_64.sh -b -p /opt/miniforge \
    && rm -f Miniforge3-25.3.1-0-Linux-x86_64.sh  
ENV PATH=/opt/miniforge/bin:$PATH

# Update conda and clean
RUN conda update --all \
    && conda clean --all --yes \
    && rm -rf /root/.cache/pip
# Update conda and clean
RUN conda update --all \
    && conda clean --all --yes \
    && rm -rf /root/.cache/pip


COPY project4_environment.yml .
RUN conda env create --prefix /opt/conda/env/ml-nlp -f project4_environment.yml
ENV PATH=/opt/conda/env/ml-nlp/bin:$PATH