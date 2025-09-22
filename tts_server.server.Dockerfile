FROM python:3.10.18-slim

USER root

RUN pip install --upgrade pip

RUN apt-get update && apt-get install -y cmake build-essential && rm -rf /var/lib/apt/lists/*

# Additional modules required to support Japanese language
RUN pip install pyopenjtalk
RUN pip install "fugashi[unidic]"
RUN python -m unidic download
RUN pip install jaconv
RUN pip install mojimoji

# Additional modules required to support Mandarin Chinese language
RUN pip install ordered_set
RUN pip install pypinyin
RUN pip install cn2an
RUN pip install jieba

RUN pip install kokoro==0.9.4 fastapi aiohttp uvicorn python-socketio soundfile numpy torch sacremoses

RUN pip install markdown emoji

RUN python -m spacy download en_core_web_sm

WORKDIR /tts_server
