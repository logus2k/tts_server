FROM python:3.10.18-slim

USER root

RUN pip install --upgrade pip
RUN pip install kokoro==0.9.4 fastapi aiohttp uvicorn python-socketio soundfile numpy torch

WORKDIR /tts_server
