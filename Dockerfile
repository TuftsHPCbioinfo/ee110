# Use the specified base image
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Author label
LABEL maintainer="Yucheng Zhang <Yucheng.Zhang@tufts.edu>"

# Help message
LABEL description="This container is for Tufts course EE110 by Dr. Mai Vu"

COPY requirements.txt .
RUN pip install --no-cache-dir -r transformer_requirement.txt
