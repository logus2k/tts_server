FROM tts_server-server:1.0

USER root

COPY kokoro_server.py /tts_server

EXPOSE 7700

WORKDIR /tts_server

CMD ["python", "kokoro_server.py"]
