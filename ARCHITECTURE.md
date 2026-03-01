# Kokoro TTS Server — Architecture Guide

> Companion document for [`architecture.drawio`](architecture.drawio).
> Open the `.drawio` file in [draw.io](https://app.diagrams.net/) or the VS Code draw.io extension to view the diagram.

---

## Overview

The Kokoro TTS Server is a real-time, multi-language text-to-speech streaming server. Clients send text over WebSockets and receive synthesised audio chunks back in real time. The server is a single-process Python application built on an asynchronous stack (FastAPI + Uvicorn + python-socketio) that offloads the synchronous Kokoro model inference to a thread pool, keeping the event loop free for concurrent client I/O.

The diagram is organised top-to-bottom following the data flow: **Clients → Transport → API Surface → Text Processing → TTS Engine → Audio Output → back to Clients**.

---

## 1. Clients

Four distinct consumer types connect to the server:

| Client | Transport | Audio format | Notes |
|:--|:--|:--|:--|
| **Browser Client** | Socket.IO (WebSocket) | Base64-encoded WAV | Typical end-user in a web application. |
| **Avatar Server** | Socket.IO (WebSocket) | Raw binary WAV | Server-side avatar renderer consuming binary audio directly. |
| **Relay Client** | Socket.IO (WebSocket) | Either | An intermediary that proxies TTS requests on behalf of other sessions (uses `target_client_id`). |
| **REST Consumer** | HTTP GET | N/A (no audio) | Reads server status only (`/health`, `/languages`, `/voices`, `/settings`). |

Socket.IO clients maintain a persistent bidirectional connection for both sending text and receiving streamed audio. REST consumers only use the read-only HTTP endpoints.

---

## 2. Transport Layer

```
Client connections
        │
        ▼
   ┌─────────┐
   │ Uvicorn  │   ASGI server listening on 0.0.0.0:7700
   └────┬─────┘
        │
        ▼
 ┌──────────────┐
 │ ASGIApp      │   socketio.ASGIApp wrapper
 │ (router)     │   Routes /socket.io/ to Socket.IO,
 └──┬────────┬──┘   everything else to FastAPI
    │        │
    ▼        ▼
Socket.IO  FastAPI
```

**Uvicorn** is the high-performance ASGI server. It passes every incoming connection to a single `socketio.ASGIApp` instance that acts as a router:

- WebSocket upgrades on the `/socket.io/` path are handled by the **Socket.IO AsyncServer**.
- All other HTTP requests fall through to the **FastAPI** application.

Both the host, port, and Uvicorn-level settings (log level, access log, reload) are loaded from the configuration file at startup.

---

## 3. API Surface

### 3a. Socket.IO Event Handlers

These are the real-time bidirectional events that drive the TTS streaming workflow:

| Event | Direction | Purpose |
|:--|:--|:--|
| `connect` / `disconnect` | lifecycle | Initialise or tear down a client session with default voice, speed, mode, and format. On connect the server emits `tts_connected` with supported languages and settings. |
| `register_audio_client` | Client → Server | Maps a main application session ID to one or more audio Socket.IO connection IDs. Supports multiple audio sockets per logical client. |
| `set_client_mode` | Client → Server | Switches the client between `tts` mode (standard playback) and `avatar` mode (feeds an avatar renderer). |
| **`tts_text_chunk`** | Client → Server | **Main pipeline entry point.** Accepts `{ chunk, final, target_client_id }`. Buffers incoming text, detects sentence boundaries, and triggers TTS generation per complete sentence. When `final` is `true` the buffer is flushed. |
| `tts_configure` / `tts_configure_client` | Client → Server | Dynamically change voice, speed, or enabled state for a direct connection or a relay-proxied client respectively. Validates voice availability and clamps speed to the configured range. |
| `stop_generation` / `stop_current_generation` | Client → Server | Sends an immediate `tts_stop_immediate` signal to halt audio playback on the client side. |
| `tts_get_voices` / `tts_get_languages` / `tts_get_settings` | Client → Server | Query events that return voice catalogues, language status, or server configuration over the WebSocket channel. |
| `tts_client_disconnect` | Client → Server | Explicit cleanup for relay-proxied clients, removes session and mapping data. |

The highlighted `tts_text_chunk` handler is the critical path — it orchestrates the entire text-to-speech pipeline described in sections 5–7 below.

### 3b. REST Endpoints

Read-only HTTP endpoints exposed by FastAPI, useful for monitoring and integrations:

| Endpoint | Returns |
|:--|:--|
| `GET /health` | Pipeline availability, active client count, mode distribution, voice/language counts, version. |
| `GET /languages` | Per-language availability, enabled status, associated voices. |
| `GET /voices` | Per-voice language mapping, availability, and enabled status. |
| `GET /settings` | Current TTS configuration values (default voice/speed, limits, timeouts). |

### 3c. Lifespan Manager

FastAPI's `lifespan` context manager runs during server startup and shutdown:

- **Startup:** Calls `initialize_all_pipelines()` which concurrently loads a Kokoro `KPipeline` instance for every enabled language. Also starts the background `cleanup_stale_buffers()` task.
- **Shutdown:** Logs and releases resources.

---

## 4. Client Session Manager

Two in-memory dictionaries maintain per-client state:

| Structure | Key | Value |
|:--|:--|:--|
| `client_tts_sessions` | Socket.IO `sid` | `{ voice, speed, enabled, mode, format, connected_time, main_client_id, connection_type }` |
| `audio_client_mapping` | Logical `client_id` | `List[sid]` — one or more Socket.IO connection IDs that should receive audio for this client. |

Every Socket.IO event handler reads from and writes to these dictionaries to resolve which voice, speed, format, and target sockets to use. The mapping supports a one-to-many relationship: a single logical client can have multiple audio connections (e.g., both a browser tab and an avatar renderer).

---

## 5. Text Processing Pipeline

When `tts_text_chunk` receives text, it passes through four stages before reaching the TTS engine:

### 5a. Per-Client Text Buffer

```python
_tts_buffers[client_id]   # accumulated raw text
_tts_last_touch[client_id]  # timestamp of last chunk
```

Incoming chunks are appended using `_append_stream()`, which avoids inserting extra spaces (critical for languages like Japanese/Chinese) while collapsing redundant whitespace at chunk seams.

### 5b. spaCy Sentencizer

The accumulated buffer is passed to a rule-based spaCy sentencizer (`_split_sentences_spacy()`). The sentencizer supports:

- Standard punctuation: `. ! ? ...`
- CJK punctuation: `。？！`
- Newline-as-boundary mode (configurable via `treat_newline_as_boundary`).

Two spaCy models are loaded — `en_core_web_sm` and `pt_core_news_sm` — with all heavy components (`tagger`, `parser`, `ner`) excluded. For other languages a blank pipeline with a sentencizer pipe is used. Models are cached in `_SPACY_CACHE`.

The sentencizer splits the buffer into parts. All parts except the last are considered complete sentences. The last part is held as a tail in the buffer unless the `final` flag is set.

### 5c. Idle-Timeout Flush

If no new chunks arrive for `buffer_idle_timeout` seconds (default 0.25s) and the tail does not end mid-word (checked via a regex for alphanumeric/accented characters), the tail is promoted to a complete sentence and flushed. This prevents indefinite buffering when the upstream text source pauses.

### 5d. Sacremoses Detokenizer

Each complete sentence is passed through `_detok_sentence()` just before TTS generation. This uses the Sacremoses `MosesTokenizer` + `MosesDetokenizer` pair to normalise tokenisation artifacts (misplaced spaces around punctuation, etc.). Currently supported for English (`en`) and Portuguese (`pt`); other languages pass through unchanged. Tokenizers and detokenizers are cached in `_TOK_CACHE` and `_DETOK_CACHE`.

---

## 6. Kokoro TTS Engine

### 6a. Voice-Language Mapping

A static dictionary (`VOICE_LANGUAGE_MAP`) maps 53 voice identifiers to 9 language codes:

| Code | Language | Voices |
|:--|:--|:--|
| `a` | American English | 19 |
| `b` | British English | 8 |
| `j` | Japanese | 5 |
| `z` | Mandarin Chinese | 8 |
| `e` | Spanish | 3 |
| `f` | French | 1 |
| `h` | Hindi | 4 |
| `i` | Italian | 2 |
| `p` | Brazilian Portuguese | 3 |

`get_pipeline_for_voice()` resolves a voice to its language, looks up the pre-initialised pipeline, and falls back to American English (`a`) if the requested pipeline is unavailable.

### 6b. KPipeline Instances

During startup, `initialize_all_pipelines()` creates one `KPipeline` instance per enabled language. Each initialisation is wrapped in `asyncio.wait_for()` with a configurable timeout (`pipeline_timeout`, default 300s) and runs in the default thread-pool executor since the Kokoro constructor is synchronous.

At runtime, `KPipeline.__call__()` is similarly dispatched to the thread pool via `loop.run_in_executor()`, keeping the async event loop unblocked during model inference. The pipeline runs on GPU when available (NVIDIA CUDA via PyTorch).

### 6c. Audio Generators

Two async generator functions produce audio from a sentence:

- **`generate_tts_binary()`** — yields `(metadata_dict, wav_bytes)` tuples. Binary WAV data is sent directly over the WebSocket.
- **`generate_tts_base64()`** — yields `dict` payloads with a `audio_data` field containing a base64-encoded WAV string.

Both generators:
1. Look up the correct pipeline and validate the speed.
2. Call the pipeline synchronously in the thread pool.
3. Iterate over the returned chunks, converting each audio tensor to a NumPy float32 array.
4. Enforce configurable limits: `chunk_limit` (max chunks per sentence) and `chunk_timeout` (per-chunk conversion timeout).
5. Yield a `tts_sentence_complete` event at the end.

The choice between binary and base64 is determined by the client session's `format` field, set at connection time or via `register_audio_client`.

---

## 7. Audio Processing

Two utility functions handle the final conversion from raw NumPy arrays to client-ready audio:

| Function | Input | Output |
|:--|:--|:--|
| `audio_to_wav_bytes()` | `np.ndarray` (float32) | `bytes` — PCM_16 WAV at 24 kHz |
| `audio_to_base64()` | `np.ndarray` (float32) | `str` — base64-encoded PCM_16 WAV at 24 kHz |

Both normalise the amplitude if it exceeds 1.0, write to an in-memory `io.BytesIO` buffer using `soundfile`, and return the result. These run in the thread-pool executor to avoid blocking the event loop.

The output audio is emitted back to the client(s) via `sio.emit('tts_audio_chunk', ...)` targeted at the appropriate Socket.IO room(s) resolved from `audio_client_mapping`. This is the return path shown as a dashed line on the right side of the diagram.

---

## 8. BGE-M3 Boundary Detector (Optional Module)

The file `bge_api_boundary_detector.py` contains an alternative sentence-boundary detection system that uses semantic embeddings instead of rule-based punctuation. It is not wired into the main `tts_text_chunk` handler by default but is available as a drop-in replacement.

### 8a. APIBasedBGEBoundaryDetector

Computes a **completion confidence score** for a text buffer by comparing its embedding against pre-computed reference embeddings for "complete" and "incomplete" sentence patterns. The score is a weighted combination of max and average cosine similarities. It also checks for obvious incomplete patterns (text ending with prepositions, conjunctions, or technical phrases).

### 8b. APIStreamingSentenceBuffer

A per-client buffer (similar to the built-in `_tts_buffers`) that uses the semantic detector alongside traditional punctuation checks. Three completion strategies:

1. **Immediate** — traditional strong punctuation (`. ! ?`).
2. **Semantic** — BGE-M3 confidence above a conservative threshold (0.85 for immediate, 0.75 with a time delay).
3. **Forced** — timeout or max-length safety net.

### 8c. FlagEmbeddingAPIClient

An async HTTP client (`aiohttp`) with connection pooling that calls the external FlagEmbedding service:

| Endpoint | Purpose |
|:--|:--|
| `POST /embed` | Generate BGE-M3 embeddings for a batch of texts. |
| `POST /rerank` | Rerank results (available but not used by the boundary detector). |
| `GET /health` | Service health check. |

The service address is configured in `tts.server.settings.json` under the `FlagEmbedding` key (default: `http://flagembedding:8000`). The client includes a local embedding cache (FIFO, max 500 entries) to reduce API calls for repeated text.

---

## 9. Configuration

All operational settings are loaded at startup from `data/configuration/tts.server.settings.json` via `load_config()` and `initialize_tts_settings()`.

The configuration has two top-level sections:

### Server settings (root level)

| Key | Default | Purpose |
|:--|:--|:--|
| `host` | `0.0.0.0` | Uvicorn bind address. |
| `port` | `7700` | Uvicorn listen port. |
| `log_level` | `info` | Uvicorn log level. |
| `access_log` | `false` | Enable/disable HTTP access logging. |
| `reload` | `false` | Enable/disable auto-reload on file changes. |

### TTS settings (`TTS` key)

| Key | Default | Purpose |
|:--|:--|:--|
| `default_voice` | `af_heart` | Voice used when a client does not specify one. |
| `default_speed` | `1.0` | Playback speed used when a client does not specify one. |
| `enabled_languages` | all 9 codes | Which language pipelines to initialise. |
| `pipeline_timeout` | `300` | Seconds to wait for pipeline initialisation. |
| `max_sentence_length` | `180` | Characters; longer sentences are truncated. |
| `chunk_limit` | `40` | Maximum audio chunks per sentence. |
| `chunk_timeout` | `3.0` | Seconds; timeout for converting a single audio chunk. |
| `generation_timeout` | `25.0` | Seconds; timeout for the entire pipeline call. |
| `speed_min` / `speed_max` | `0.5` / `2.0` | Allowed playback speed range. |

### FlagEmbedding settings (`FlagEmbedding` key)

| Key | Default | Purpose |
|:--|:--|:--|
| `ServerAPIAddress` | `http://flagembedding:8000` | Base URL of the BGE-M3 embedding service. |
| `BatchSize` | `8` | Texts per API batch request. |
| `Dimension` | `1024` | Expected embedding vector dimension. |
| `Debug` | `false` | Enable detailed API timing logs. |

Configuration flows from this single file into three consumers shown in the diagram: the Uvicorn server, the TTS engine, and the BGE boundary detector.

---

## 10. Docker Infrastructure

The `docker-compose.yml` defines a single service:

```yaml
services:
  tts_server:
    image: tts_server:1.0
    ports: ["7700:7700"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: "all"
              capabilities: [gpu]
    volumes:
      - ~/env/assets/tts_server/data:/tts_server/data:ro
    networks:
      logus2k_network:

networks:
  logus2k_network:
    external: true
```

Key points:

- **GPU access:** All NVIDIA GPUs are passed through to the container for PyTorch/CUDA inference.
- **Read-only config mount:** The host `data/` directory (containing the settings file) is mounted read-only into the container.
- **External network:** The container joins `logus2k_network`, an externally managed Docker network that connects it to peer services (e.g., the FlagEmbedding service at `http://flagembedding:8000`).
- **Logging:** Container logs are capped at 10 MB with 3 rotated files.

---

## Data Flow Summary

```
1. Client sends text chunk via Socket.IO
                    │
2. tts_text_chunk handler receives it
                    │
3. Text appended to per-client buffer
                    │
4. spaCy splits buffer into sentences
                    │
5. Complete sentences detokenised (Sacremoses)
                    │
6. Each sentence → KPipeline (thread pool, GPU)
                    │
7. Audio chunks → NumPy → WAV bytes or base64
                    │
8. sio.emit('tts_audio_chunk') back to client(s)
                    │
9. After last chunk → sio.emit('tts_sentence_complete')
                    │
10. On final=true → sio.emit('tts_response_complete')
```
