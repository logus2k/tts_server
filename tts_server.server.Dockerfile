FROM python:3.10.18-slim

USER root

RUN pip install --upgrade pip
RUN pip install transformers kokoro==0.9.4 nltk fastapi uvicorn python-socketio soundfile numpy torch

WORKDIR /tts_server
