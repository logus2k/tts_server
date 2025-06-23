#!/usr/bin/env python3

"""
Minimal Kokoro TTS Server with FlagEmbedding API Integration
Essential functionality only - no stats, HTML pages, or debug endpoints
"""

"""
pip install kokoro==0.9.4 fastapi uvicorn python-socketio soundfile numpy torch aiohttp
"""

import asyncio
import base64
import io
import logging
from typing import Dict, Optional, AsyncGenerator
from datetime import datetime
import numpy as np
import soundfile as sf
from fastapi import FastAPI
import socketio
from kokoro import KPipeline
import uvicorn
from contextlib import asynccontextmanager
import urllib.parse
import json
from pathlib import Path


# FlagEmbedding API Integration
from bge_api_boundary_detector import APIStreamingSentenceBuffer, process_tts_with_api_bge

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
tts_pipeline: Optional[KPipeline] = None
flagembedding_settings = None

async def initialize_flagembedding_settings(settings: dict):
    global flagembedding_settings
    flagembedding_settings = {
        "FlagEmbedding": settings.get("FlagEmbedding", {
            "ServerAPIAddress": "http://localhost:8000",
            "BatchSize": 8,
            "Dimension": 1024,
            "Debug": False
        })
    }

    logger.info(f"FlagEmbedding configured: {flagembedding_settings['FlagEmbedding']['ServerAPIAddress']}")

    try:
        from bge_api_boundary_detector import FlagEmbeddingAPIClient
        api_client = FlagEmbeddingAPIClient(flagembedding_settings)
        health_ok = await api_client.check_health()
        await api_client.close()

        if health_ok:
            logger.info("✅ FlagEmbedding API available")
        else:
            logger.warning("⚠️ FlagEmbedding API unavailable - using traditional detection")

    except Exception as e:
        logger.warning(f"FlagEmbedding API test failed: {e}")


async def initialize_pipeline():
    """Initialize Kokoro TTS pipeline"""
    global tts_pipeline
    try:
        logger.info("Initializing Kokoro TTS pipeline...")
        loop = asyncio.get_event_loop()
        tts_pipeline = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: KPipeline(lang_code='a')),
            timeout=300.0
        )
        logger.info("✅ TTS pipeline initialized")
    except Exception as e:
        logger.error(f"❌ TTS pipeline failed: {e}")
        tts_pipeline = None
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    try:
        await initialize_pipeline()
        logger.info("✅ TTS pipeline ready")
    except Exception as e:
        logger.error(f"❌ TTS initialization failed: {e}")
        logger.warning("⚠️ TTS functionality disabled")
    
    yield
    
    logger.info("Shutting down...")
    try:
        for buffer in client_sentence_buffers.values():
            await buffer.close()
    except Exception:
        pass

# Create Socket.IO server
sio = socketio.AsyncServer(
    cors_allowed_origins="*",
    logger=False,
    engineio_logger=False,
    async_mode='asgi'
)

# FastAPI app
app = FastAPI(
    title="Minimal Kokoro TTS Server",
    version="1.3.0-minimal",
    lifespan=lifespan
)

# Client management
client_sentence_buffers: Dict[str, APIStreamingSentenceBuffer] = {}
client_tts_sessions: Dict[str, dict] = {}
audio_client_mapping: Dict[str, str] = {}

def audio_to_base64(audio_data: np.ndarray, sample_rate: int = 24000) -> str:
    """Convert audio to base64 WAV"""
    try:
        if audio_data is None or len(audio_data) == 0:
            return ""
        
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sample_rate, format='WAV', subtype='PCM_16')
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Base64 conversion error: {e}")
        return ""

def audio_to_wav_bytes(audio_data: np.ndarray, sample_rate: int = 24000) -> bytes:
    """Convert audio to WAV bytes"""
    try:
        if audio_data is None or len(audio_data) == 0:
            return b""
        
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sample_rate, format='WAV', subtype='PCM_16')
        buffer.seek(0)
        return buffer.read()
    except Exception as e:
        logger.error(f"WAV conversion error: {e}")
        return b""

async def generate_tts_binary(sentence: str, voice: str, client_id: str):
    """Generate TTS audio in binary format"""
    try:
        if not tts_pipeline:
            error_data = {
                'type': 'tts_error',
                'error': 'TTS pipeline not initialized',
                'sentence_text': sentence[:100],
                'client_id': client_id,
                'format': 'binary',
                'timestamp': datetime.now().isoformat()
            }
            yield error_data, None
            return
        
        if len(sentence) > 180:
            sentence = sentence[:180] + "..."
        
        cleaned_sentence = sentence.strip()
        logger.info(f"🎵 Generating TTS: {cleaned_sentence[:50]}...")
        
        loop = asyncio.get_event_loop()
        generator = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: tts_pipeline(cleaned_sentence, voice=voice, speed=1.25, split_pattern=r"\n+")),
            timeout=25.0
        )
        
        chunk_count = 0
        total_duration = 0
        
        for i, chunk_data in enumerate(generator):
            if i >= 40:  # Limit chunks
                break
                
            try:
                if hasattr(chunk_data, 'output') and hasattr(chunk_data.output, 'audio'):
                    audio = chunk_data.output.audio
                    
                    if hasattr(audio, 'cpu'):
                        audio_np = audio.cpu().detach().numpy().astype(np.float32)
                    else:
                        audio_np = np.array(audio, dtype=np.float32)
                    
                    if audio_np.ndim > 1:
                        audio_np = audio_np.flatten()
                    
                    if len(audio_np) == 0:
                        continue
                    
                    wav_bytes = await asyncio.wait_for(
                        loop.run_in_executor(None, audio_to_wav_bytes, audio_np, 24000),
                        timeout=3.0
                    )
                    
                    if not wav_bytes:
                        continue
                    
                    chunk_duration = len(audio_np) / 24000
                    total_duration += chunk_duration
                    
                    metadata = {
                        'type': 'tts_audio_chunk',
                        'sentence_text': cleaned_sentence[:100],
                        'chunk_id': i,
                        'sample_rate': 24000,
                        'duration': chunk_duration,
                        'sentence_duration': total_duration,
                        'client_id': client_id,
                        'format': 'binary',
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    yield metadata, wav_bytes
                    chunk_count += 1
                    await asyncio.sleep(0.005)
                    
            except Exception as e:
                logger.error(f"Chunk processing error: {e}")
                continue
        
        # Completion
        completion = {
            'type': 'tts_sentence_complete',
            'sentence_text': cleaned_sentence,
            'total_chunks': chunk_count,
            'total_duration': total_duration,
            'client_id': client_id,
            'format': 'binary',
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"🎵 TTS completed: {chunk_count} chunks, {total_duration:.2f}s")
        yield completion, None
        
    except Exception as e:
        logger.error(f"TTS generation error: {e}")
        error_data = {
            'type': 'tts_error',
            'error': str(e),
            'sentence_text': sentence,
            'client_id': client_id,
            'format': 'binary',
            'timestamp': datetime.now().isoformat()
        }
        yield error_data, None

async def generate_tts_base64(sentence: str, voice: str, client_id: str) -> AsyncGenerator[dict, None]:
    """Generate TTS audio in base64 format"""
    try:
        if not tts_pipeline:
            yield {
                'type': 'tts_error',
                'error': 'TTS pipeline not initialized',
                'sentence_text': sentence,
                'client_id': client_id,
                'format': 'base64',
                'timestamp': datetime.now().isoformat()
            }
            return
        
        logger.info(f"Generating base64 TTS: {sentence[:50]}...")
        
        loop = asyncio.get_event_loop()
        generator = await loop.run_in_executor(None, lambda: tts_pipeline(sentence, voice=voice, speed=1.25))
        
        chunk_count = 0
        total_duration = 0
        
        for i, chunk_data in enumerate(generator):
            try:
                if hasattr(chunk_data, 'output') and hasattr(chunk_data.output, 'audio'):
                    audio = chunk_data.output.audio
                    
                    if hasattr(audio, 'cpu'):
                        audio_np = audio.cpu().detach().numpy().astype(np.float32)
                    else:
                        audio_np = np.array(audio, dtype=np.float32)
                    
                    if audio_np.ndim > 1:
                        audio_np = audio_np.flatten()
                        
                    if len(audio_np) == 0:
                        continue
                    
                    audio_b64 = await loop.run_in_executor(None, audio_to_base64, audio_np)
                    
                    if audio_b64:
                        chunk_duration = len(audio_np) / 24000
                        total_duration += chunk_duration
                        
                        yield {
                            'type': 'tts_audio_chunk',
                            'sentence_text': sentence[:100],
                            'chunk_id': i,
                            'audio_data': audio_b64,
                            'sample_rate': 24000,
                            'duration': chunk_duration,
                            'sentence_duration': total_duration,
                            'client_id': client_id,
                            'format': 'base64',
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        chunk_count += 1
                        await asyncio.sleep(0.005)
                        
            except Exception as e:
                logger.error(f"Base64 chunk error: {e}")
                continue
        
        # Completion
        yield {
            'type': 'tts_sentence_complete',
            'sentence_text': sentence,
            'total_chunks': chunk_count,
            'total_duration': total_duration,
            'client_id': client_id,
            'format': 'base64',
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Base64 TTS error: {e}")
        yield {
            'type': 'tts_error',
            'error': str(e),
            'sentence_text': sentence,
            'client_id': client_id,
            'format': 'base64',
            'timestamp': datetime.now().isoformat()
        }

# Socket.IO Event Handlers
@sio.event
async def connect(sid, environ):
    """Handle client connection"""
    query_params = environ.get('QUERY_STRING', '')
    main_client_id = None
    connection_type = 'unknown'
    format_type = 'base64'
    
    if query_params:
        parsed_params = urllib.parse.parse_qs(query_params)
        main_client_id = parsed_params.get('main_client_id', [None])[0]
        connection_type = parsed_params.get('type', ['browser'])[0]
        format_type = parsed_params.get('format', ['base64'])[0]
    
    logger.info(f"🎵 Client connected: {sid} (type: {connection_type}, format: {format_type})")
    
    # Initialize client
    client_sentence_buffers[sid] = APIStreamingSentenceBuffer(sid, flagembedding_settings)
    client_tts_sessions[sid] = {
        'voice': 'af_heart',
        'enabled': True,
        'format': format_type,
        'connected_time': datetime.now(),
        'main_client_id': main_client_id,
        'connection_type': connection_type
    }
    
    # Send confirmation
    await sio.emit('tts_connected', {
        'status': 'Connected to TTS server',
        'client_id': sid,
        'format': format_type,
        'version': '1.3.0-minimal',
        'timestamp': datetime.now().isoformat()
    }, room=sid)

@sio.event
async def disconnect(sid):
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {sid}")
    
    # Cleanup mappings
    main_client_to_remove = None
    for main_client_id, audio_sid in audio_client_mapping.items():
        if audio_sid == sid:
            main_client_to_remove = main_client_id
            break
    
    if main_client_to_remove:
        del audio_client_mapping[main_client_to_remove]
    
    # Cleanup resources
    if sid in client_sentence_buffers:
        try:
            await client_sentence_buffers[sid].close()
        except Exception:
            pass
        del client_sentence_buffers[sid]
    
    if sid in client_tts_sessions:
        del client_tts_sessions[sid]

@sio.event
async def register_audio_client(sid, data):
    """Register audio client mapping"""
    main_client_id = data.get('main_client_id')
    
    if main_client_id and main_client_id != 'unknown':
        audio_client_mapping[main_client_id] = sid
        logger.info(f"🎵 Audio mapping: {main_client_id} -> {sid}")
        
        if sid in client_tts_sessions:
            client_tts_sessions[sid]['main_client_id'] = main_client_id
            client_tts_sessions[sid]['connection_type'] = 'browser_audio'
        
        await sio.emit('audio_client_registered', {
            'status': 'registered',
            'main_client_id': main_client_id,
            'audio_client_id': sid,
            'timestamp': datetime.now().isoformat()
        }, room=sid)

@sio.event
async def tts_text_chunk(sid, data):
    """Handle streaming text chunks with BGE-M3 processing"""
    try:
        text_chunk = data.get('chunk', '')
        is_final = data.get('final', False)
        target_client_id = data.get('target_client_id', data.get('client_id', sid))
        
        if not text_chunk and not is_final:
            return
        
        # Find audio connection
        audio_sid = audio_client_mapping.get(target_client_id, target_client_id)
        
        # Get session settings
        session = client_tts_sessions.get(audio_sid, {
            'voice': 'af_heart',
            'enabled': True,
            'format': 'base64'
        })
        
        if not session.get('enabled', True):
            return
        
        voice = session.get('voice', 'af_heart')
        format_type = session.get('format', 'base64')
        
        # TTS generation callback
        async def tts_generation_callback(sentence: str, client_id: str, analysis: dict):
            logger.info(f"🎵 TTS: {sentence[:40]}... (method: {analysis['method']}, confidence: {analysis['confidence']:.3f})")
            
            # Generate and stream
            if format_type == 'binary':
                async for metadata, binary_data in generate_tts_binary(sentence, voice, client_id):
                    if binary_data is not None:
                        metadata_with_binary = metadata.copy()
                        metadata_with_binary['audio_data'] = binary_data
                        await sio.emit('tts_audio_chunk', metadata_with_binary, room=audio_sid)
                    else:
                        await sio.emit('tts_sentence_complete', metadata, room=audio_sid)
            else:
                async for chunk_data in generate_tts_base64(sentence, voice, client_id):
                    await sio.emit('tts_audio_chunk', chunk_data, room=audio_sid)
        
        # Process with BGE-M3
        await process_tts_with_api_bge(
            target_client_id, 
            text_chunk, 
            is_final, 
            client_sentence_buffers, 
            tts_generation_callback,
            flagembedding_settings
        )
        
        # Final completion
        if is_final:
            await sio.emit('tts_response_complete', {
                'timestamp': datetime.now().isoformat()
            }, room=audio_sid)
        
    except Exception as e:
        logger.error(f"Text chunk processing error: {e}")
        audio_sid = audio_client_mapping.get(data.get('target_client_id', data.get('client_id', sid)), sid)
        await sio.emit('tts_error', {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }, room=audio_sid)

@sio.event
async def tts_configure(sid, data):
    """Configure TTS settings"""
    session = client_tts_sessions.get(sid, {})
    
    if 'voice' in data:
        session['voice'] = data['voice']
    if 'enabled' in data:
        session['enabled'] = data['enabled']
    
    client_tts_sessions[sid] = session
    
    await sio.emit('tts_configured', {
        'voice': session.get('voice'),
        'enabled': session.get('enabled'),
        'format': session.get('format'),
        'timestamp': datetime.now().isoformat()
    }, room=sid)

@sio.event
async def tts_configure_client(sid, data):
    """Configure TTS for relay client"""
    client_id = data.get('client_id')
    audio_sid = audio_client_mapping.get(client_id, client_id)
    
    session = client_tts_sessions.get(audio_sid, {})
    
    if 'voice' in data:
        session['voice'] = data['voice']
    if 'enabled' in data:
        session['enabled'] = data['enabled']
    
    client_tts_sessions[audio_sid] = session
    
    await sio.emit('tts_client_configured', {
        'client_id': client_id,
        'voice': session.get('voice'),
        'enabled': session.get('enabled'),
        'timestamp': datetime.now().isoformat()
    }, room=sid)

@sio.event
async def tts_get_voices(sid, data):
    """Get available voices"""
    voices = [
        'af_heart', 'af_sky', 'af_bella', 'af_sarah',
        'am_adam', 'am_michael', 'bf_emma', 'bf_isabella'
    ]
    
    await sio.emit('tts_voices_response', {
        'voices': voices,
        'requesting_client': data.get('requesting_client'),
        'timestamp': datetime.now().isoformat()
    }, room=sid)

@sio.event
async def tts_client_disconnect(sid, data):
    """Handle relay client disconnect"""
    client_id = data.get('client_id')
    audio_sid = audio_client_mapping.get(client_id)
    
    if audio_sid:
        if client_id in audio_client_mapping:
            del audio_client_mapping[client_id]
        if audio_sid in client_sentence_buffers:
            try:
                await client_sentence_buffers[audio_sid].close()
            except Exception:
                pass
            del client_sentence_buffers[audio_sid]
        if audio_sid in client_tts_sessions:
            del client_tts_sessions[audio_sid]

# Cleanup task
async def cleanup_stale_buffers():
    """Clean up stale buffers"""
    while True:
        try:
            stale_clients = []
            for client_id, buffer in client_sentence_buffers.items():
                if hasattr(buffer, 'is_stale') and buffer.is_stale(timeout_seconds=45):
                    stale_clients.append(client_id)
            
            for client_id in stale_clients:
                try:
                    await client_sentence_buffers[client_id].close()
                except Exception:
                    pass
                del client_sentence_buffers[client_id]
                if client_id in client_tts_sessions:
                    del client_tts_sessions[client_id]
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
        
        await asyncio.sleep(20)

# Basic health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.3.0-minimal",
        "pipeline_ready": tts_pipeline is not None,
        "active_clients": len(client_sentence_buffers),
        "timestamp": datetime.now().isoformat()
    }


def load_config(path: str = "/tts_server/data/configuration/tts.server.settings.json") -> dict:
    config_file = Path(path)
    if not config_file.exists():
        logger.warning(f"⚠️ Config file {path} not found. Using defaults.")
        return {}
    
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"❌ Failed to load config: {e}")
        return {}



# Socket.IO ASGI app
sio_asgi_app = socketio.ASGIApp(sio, other_asgi_app=app)

async def main():
    settings = load_config()  # 👈 Load the settings dict from JSON
    await initialize_flagembedding_settings(settings)  # 👈 Pass it here
    asyncio.create_task(cleanup_stale_buffers())

    config = uvicorn.Config(
        sio_asgi_app,
        host=settings.get("host", "0.0.0.0"),
        port=settings.get("port", 7700),
        log_level=settings.get("log_level", "info"),
        access_log=settings.get("access_log", False),
        reload=settings.get("reload", False)
    )

    server = uvicorn.Server(config)
    logger.info("🎵 Starting Minimal Kokoro TTS Server...")
    logger.info(f"📡 Server: http://{config.host}:{config.port}")
    logger.info(f"🔌 Socket.IO: http://{config.host}:{config.port}/socket.io/")
    logger.info(f"💡 Health check: http://{config.host}:{config.port}/health")

    await server.serve()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        raise
