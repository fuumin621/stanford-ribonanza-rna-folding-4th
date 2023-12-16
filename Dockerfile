FROM gcr.io/kaggle-gpu-images/python:latest
RUN pip uninstall torch torchvision torchaudio torchtext -y
RUN pip install torch torchvision