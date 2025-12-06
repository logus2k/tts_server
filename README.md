# Multi-Language Real-Time Text-to-Speech (TTS) Server

A high-performance, asynchronous server designed for low-latency Text-to-Speech synthesis using [Kokoro TTS]()https://huggingface.co/hexgrad/Kokoro-82M. This project provides real-time audio streaming over [Socket.IO](https://socket.io/) WebSockets, supporting a wide range of voices and languages.

The server is built on a robust [FastAPI/Uvicorn](https://uvicorn.dev/) stack, utilizing asynchronous processing and dedicated thread pool execution to efficiently handle the synchronous nature of the TTS inference, ensuring high concurrency and optimal throughput.

---

## Features

* **Kokoro TTS Integration:** Utilizes the high-quality **Kokoro** library (`KPipeline`) for speech synthesis.
* **Multi-Language Support:** Pre-initializes pipelines for a comprehensive set of languages including American/British English, Japanese, Mandarin Chinese, Spanish, French, Hindi, Italian, and Brazilian Portuguese.
* **Real-Time Audio Streaming:** Streams synthesized audio in real-time chunks (WAV bytes or Base64 encoding) back to the client.
* **Advanced Sentence Splitting:** Employs [spaCy](https://spacy.io/) (for rule-based sentencizing) and [Sacremoses](https://github.com/hplt-project/sacremoses) (for detokenization) to ensure natural-sounding speech segments, respecting sentence and paragraph boundaries.
* **Asynchronous Architecture:** Built on [FastAPI](https://github.com/FastAPI/FastAPI) and **Uvicorn** for a highly concurrent and scalable web backend.
* **Dynamic Client Control:** Allows clients to dynamically set their preferred voice, playback speed, and audio format (binary or base64) via Socket.IO events.
* **Per-Client Buffering:** Implements per-client text buffering and idle-timeout flush logic to minimize latency and guarantee complete word pronunciation.

---

## Technology Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **TTS Engine** | **Kokoro** | Core library for multi-language, high-quality speech synthesis. |
| **Web Framework** | FastAPI | ASGI framework for the REST API (health, settings) and hosting Socket.IO. |
| **Asynchronous Server** | Uvicorn | High-performance ASGI server. |
| **Real-Time I/O** | python-socketio | Handles WebSocket communication for streaming audio data. |
| **NLP Utilities** | spaCy, Sacremoses | Used for robust sentence boundary detection and linguistic cleanup. |
| **Audio Processing** | NumPy, soundfile, io | Handling of raw audio data, conversion, and WAV encoding. |

---

## Installation and Setup

Follow these steps to set up the server environment and run the application.

### 1. Clone the repository

```bash
git clone [https://github.com/logus2k/tts_server.git](https://github.com/logus2k/tts_server.git)
cd tts_server
```

### 2\. Set up the Python Environment

It is highly recommended to use a virtual environment.

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

### 3\. Install Dependencies

You need `kokoro` (specific version recommended in the script), the core server packages, and NLP dependencies.

```bash
pip install kokoro==0.9.4 fastapi uvicorn python-socketio soundfile numpy torch aiohttp spacy sacremoses
# Required spaCy models (for English and Portuguese sentence splitting)
python -m spacy download en_core_web_sm
python -m spacy download pt_core_news_sm
```

### 4\. Configuration

The server loads operational and TTS model settings from a configuration file, typically located at `data/configuration/tts.server.settings.json`.

---

## Usage

### 1\. Run the Server

Execute the main server file. The server will initialize all enabled Kokoro pipelines concurrently before starting the network listener.

```bash
python tts_server.py
```

The server will log the active endpoints:

| Endpoint | Description |
| :--- | :--- |
| `http://0.0.0.0:7700` | FastAPI Root |
| `http://0.0.0.0:7700/health` | Server Health Check |
| `http://0.0.0.0:7700/languages` | Lists supported and active languages |
| `http://0.0.0.0:7700/socket.io/` | Socket.IO WebSocket Connection |

### 2\. Docker Deployment

Use the provided Docker files for consistent deployment. The `docker-compose.yml` orchestrates the environment.

```bash
docker-compose up --build
```

---

## Client-Server Protocol (Socket.IO Events)

Clients communicate with the server using the following Socket.IO events:

| Event Name | Direction | Payload | Description |
| :--- | :--- | :--- | :--- |
| `tts_text_chunk` | Client → Server | `{'client_id': str, 'text': str, ...}` | Streams a chunk of text to the server buffer. |
| `tts_text_final` | Client → Server | `{'client_id': str, ...}` | Signals the end of the text stream for the current request. |
| `audio_stream` | Server → Client | `bytes` (WAV) or `dict` (Base64) | Streams synthesized audio chunks in real-time. |
| `tts_sentence_complete` | Server → Client | `dict` | Signals a sentence is finished synthesizing. |
| `set_client_mode` | Client → Server | `{'client_id': str, 'mode': 'tts'|'avatar'}` | Updates the client's operational mode. |
| `register_audio_client` | Client → Server | `dict` | Maps a main client session ID to an audio connection ID and sets initial parameters. |

---

## License

This project is licensed under the **Apache License 2.0**.

---
