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

## One file from a text: `POST /synthesize`

`{"text", "voice", "speed"}` → one WAV (24 kHz mono PCM16) of the whole text, with `X-Duration-Seconds`. Added for
Cortex's Movie narration (2026-10-23); the socket.io stream above stays the way to speak live. An unknown voice is a
400. Voices are Kokoro's (`VOICE_LANGUAGE_MAP`) plus:

### European Portuguese: `pt_tuga`

Kokoro's own Portuguese is Brazilian (`pf_dora`, `pm_alex`, `pm_santa`). `pt_tuga` is the **European Portuguese**
voice of the `tts_eu_pt` package (Kokoro-82M fine-tuned by the author, TugaPhone G2P for the Lisbon lect, Apache-2.0;
`~/env/assets/tts_eu_pt`, PyPI `tts_eu_pt`), installed in the image (`tts_server.Dockerfile`) and run on the **CPU**,
so it costs no VRAM and is not part of the GPU lease below. Its weights (313 MB) are not in git: copy them from the
Hugging Face cache of `logus2k/kokoro_tts_eu_pt` (`tuga_kokoro.pth`, `tuga_voicepack.pt`) into `data/eu_pt/`
(git-ignored; `docker-compose.yml` names them in `TTS_EU_PT_MODEL` / `TTS_EU_PT_VOICEPACK`). Measured: a 20-word
sentence in 0.2 s after a 0.5 s first load. Only `/synthesize` offers it for now; the live stream's voice list is
Kokoro's.

## Releasing the GPU

Another application may ask for the card's memory back for a while (Cortex does while it ingests documents, when its
setting *Free the speech services' GPU during document ingestion* is on). Released, the shared Kokoro model (≈0.6 GB of the card; back in 0.3 s) waits in host memory. The
service keeps running: a sentence to speak that arrives meanwhile brings the model back first and is served, slower by that reload,
never refused. A release is a **lease**: it lapses after `minutes` (1-120, default 10) and the model comes back by
itself, so a caller that died cannot leave it away; releasing again renews it.

| Route | What |
| --- | --- |
| `POST /admin/release` `{"reason", "minutes", "by"}` | waits for the work in flight, moves the model to host memory, answers `{state, freed_mib, lease, gpu_mib}` |
| `POST /admin/load` | the model back now; the lease closed |
| `GET /admin/state` | `{state: loaded\|released, lease: {until, reason, by, seconds_left}, gpu_mib}` |

Every route needs the header `X-Admin-Secret` equal to `SPEECH_ADMIN_SECRET`, which `docker-compose.yml` reads
from `./.env` (git-ignored; generate one with `python3 -c 'import secrets; print(secrets.token_urlsafe(32))'`).
Without it the routes do not exist (404). Give the same value to the caller (Cortex: `CORTEX_TTS_ADMIN_SECRET`
in its `.env`, upper case). Port 7700 is published on the host: the secret is what protects the routes.

Check against the running service (about 2 minutes; it leaves the model loaded):

```bash
set -a; . ./.env; set +a
docker run --rm --network logus2k_network -e SPEECH_ADMIN_SECRET -v "$PWD/tests:/t:ro" --entrypoint python cortex:0.2 /t/release_check.py
```
