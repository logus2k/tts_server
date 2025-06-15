FROM tts_server-server:1.0

USER root

COPY *.py /tts_server

EXPOSE 7700

WORKDIR /tts_server

CMD ["python", "tts_server.py"]
