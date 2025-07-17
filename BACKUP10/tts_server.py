#!/usr/bin/env python3

"""
Kokoro TTS Server with FlagEmbedding API Integration
"""

"""
pip install kokoro==0.9.4 fastapi uvicorn python-socketio soundfile numpy torch aiohttp
"""


import asyncio
import base64
import io
import logging
from typing import Dict, List, Optional, AsyncGenerator
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
tts_pipelines: Dict[str, Optional[KPipeline]] = {}
flagembedding_settings = None
tts_settings = None

# Voice-to-Language mapping (based on official Kokoro documentation)
VOICE_LANGUAGE_MAP = {
    # American English (lang_code='a') - 19 voices
    'af_heart': 'a',        # 🚺❤️ Grade A
    'af_alloy': 'a',        # 🚺 Grade C  
    'af_aoede': 'a',        # 🚺 Grade C+
    'af_bella': 'a',        # 🚺🔥 Grade A-
    'af_jessica': 'a',      # 🚺 Grade D
    'af_kore': 'a',         # 🚺 Grade C+
    'af_nicole': 'a',       # 🚺🎧 Grade B-
    'af_nova': 'a',         # 🚺 Grade C
    'af_river': 'a',        # 🚺 Grade D
    'af_sarah': 'a',        # 🚺 Grade C+
    'af_sky': 'a',          # 🚺 Grade C-
    'am_adam': 'a',         # 🚹 Grade F+
    'am_echo': 'a',         # 🚹 Grade D
    'am_eric': 'a',         # 🚹 Grade D
    'am_fenrir': 'a',       # 🚹 Grade C+
    'am_liam': 'a',         # 🚹 Grade D
    'am_michael': 'a',      # 🚹 Grade C+
    'am_onyx': 'a',         # 🚹 Grade D
    'am_puck': 'a',         # 🚹 Grade C+
    'am_santa': 'a',        # 🚹 Grade D-
    
    # British English (lang_code='b') - 8 voices
    'bf_alice': 'b',        # 🚺 Grade D
    'bf_emma': 'b',         # 🚺 Grade B-
    'bf_isabella': 'b',     # 🚺 Grade C
    'bf_lily': 'b',         # 🚺 Grade D
    'bm_daniel': 'b',       # 🚹 Grade D
    'bm_fable': 'b',        # 🚹 Grade C
    'bm_george': 'b',       # 🚹 Grade C
    'bm_lewis': 'b',        # 🚹 Grade D+
    
    # Japanese (lang_code='j') - 5 voices
    'jf_alpha': 'j',        # 🚺 Grade C+
    'jf_gongitsune': 'j',   # 🚺 Grade C
    'jf_nezumi': 'j',       # 🚺 Grade C-
    'jf_tebukuro': 'j',     # 🚺 Grade C
    'jm_kumo': 'j',         # 🚹 Grade C-
    
    # Mandarin Chinese (lang_code='z') - 8 voices
    'zf_xiaobei': 'z',      # 🚺 Grade D
    'zf_xiaoni': 'z',       # 🚺 Grade D
    'zf_xiaoxiao': 'z',     # 🚺 Grade D
    'zf_xiaoyi': 'z',       # 🚺 Grade D
    'zm_yunjian': 'z',      # 🚹 Grade D
    'zm_yunxi': 'z',        # 🚹 Grade D
    'zm_yunxia': 'z',       # 🚹 Grade D
    'zm_yunyang': 'z',      # 🚹 Grade D
    
    # Spanish (lang_code='e') - 3 voices
    'ef_dora': 'e',         # 🚺
    'em_alex': 'e',         # 🚹
    'em_santa': 'e',        # 🚹
    
    # French (lang_code='f') - 1 voice
    'ff_siwis': 'f',        # 🚺 Grade B-
    
    # Hindi (lang_code='h') - 4 voices
    'hf_alpha': 'h',        # 🚺 Grade C
    'hf_beta': 'h',         # 🚺 Grade C
    'hm_omega': 'h',        # 🚹 Grade C
    'hm_psi': 'h',          # 🚹 Grade C
    
    # Italian (lang_code='i') - 2 voices
    'if_sara': 'i',         # 🚺 Grade C
    'im_nicola': 'i',       # 🚹 Grade C
    
    # Brazilian Portuguese (lang_code='p') - 3 voices
    'pf_dora': 'p',         # 🚺
    'pm_alex': 'p',         # 🚹
    'pm_santa': 'p',        # 🚹
}

# Language code definitions (based on official Kokoro documentation)
SUPPORTED_LANGUAGES = {
    'a': 'American English',     # 19 voices
    'b': 'British English',      # 8 voices  
    'j': 'Japanese',             # 5 voices
    'z': 'Mandarin Chinese',     # 8 voices
    'e': 'Spanish',              # 3 voices
    'f': 'French',               # 1 voice
    'h': 'Hindi',                # 4 voices
    'i': 'Italian',              # 2 voices
    'p': 'Brazilian Portuguese', # 3 voices
}

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

async def initialize_tts_settings(settings: dict):
    global tts_settings
    tts_settings = settings.get("TTS", {
        "default_voice": "af_heart",
        "default_speed": 1.20,
        "enabled_languages": ["a", "b", "j", "z", "e", "f", "h", "i", "p"],
        "pipeline_timeout": 300,
        "max_sentence_length": 180,
        "chunk_limit": 40,
        "chunk_timeout": 3.0,
        "generation_timeout": 25.0,
        "speed_min": 0.5,
        "speed_max": 2.0
    })
    
    logger.info(f"TTS Settings loaded:")
    logger.info(f"  Default voice: {tts_settings['default_voice']}")
    logger.info(f"  Default speed: {tts_settings['default_speed']}")
    logger.info(f"  Enabled languages: {tts_settings['enabled_languages']}")
    logger.info(f"  Pipeline timeout: {tts_settings['pipeline_timeout']}s")
    logger.info(f"  Max sentence length: {tts_settings['max_sentence_length']} chars")

def get_enabled_languages():
    """Get list of enabled languages from settings"""
    if not tts_settings:
        return list(SUPPORTED_LANGUAGES.keys())
    return tts_settings.get('enabled_languages', list(SUPPORTED_LANGUAGES.keys()))

def get_default_voice():
    """Get default voice from settings"""
    if not tts_settings:
        return 'af_heart'
    return tts_settings.get('default_voice', 'af_heart')

def get_default_speed():
    """Get default speed from settings"""
    if not tts_settings:
        return 1.20
    return tts_settings.get('default_speed', 1.20)

def validate_speed(speed: float) -> float:
    """Validate and clamp speed within allowed range"""
    if not tts_settings:
        return max(0.5, min(2.0, speed))
    
    speed_min = tts_settings.get('speed_min', 0.5)
    speed_max = tts_settings.get('speed_max', 2.0)
    return max(speed_min, min(speed_max, speed))

async def initialize_pipeline_for_language(lang_code: str) -> Optional[KPipeline]:
    """Initialize Kokoro TTS pipeline for specific language"""
    try:
        enabled_languages = get_enabled_languages()
        if lang_code not in enabled_languages:
            logger.info(f"⏭️ Skipping {lang_code} - not enabled in settings")
            return None
            
        logger.info(f"Initializing Kokoro TTS pipeline for language: {lang_code} ({SUPPORTED_LANGUAGES.get(lang_code, 'Unknown')})")
        loop = asyncio.get_event_loop()
        timeout = tts_settings.get('pipeline_timeout', 300) if tts_settings else 300
        
        pipeline = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: KPipeline(lang_code=lang_code)),
            timeout=timeout
        )
        logger.info(f"✅ TTS pipeline initialized for {lang_code}")
        return pipeline
    except Exception as e:
        logger.error(f"❌ TTS pipeline failed for {lang_code}: {e}")
        return None

async def initialize_all_pipelines():
    """Initialize TTS pipelines for all enabled languages"""
    global tts_pipelines
    
    # Get unique language codes from voice mapping, filtered by enabled languages
    enabled_languages = get_enabled_languages()
    required_languages = set(lang for lang in VOICE_LANGUAGE_MAP.values() if lang in enabled_languages)
    
    logger.info(f"Initializing pipelines for enabled languages: {list(required_languages)}")
    
    initialization_tasks = []
    for lang_code in required_languages:
        task = initialize_pipeline_for_language(lang_code)
        initialization_tasks.append((lang_code, task))
    
    # Initialize pipelines concurrently
    for lang_code, task in initialization_tasks:
        try:
            pipeline = await task
            tts_pipelines[lang_code] = pipeline
            if pipeline:
                logger.info(f"✅ Pipeline ready: {lang_code} ({SUPPORTED_LANGUAGES.get(lang_code)})")
            else:
                logger.info(f"⏭️ Pipeline skipped: {lang_code}")
        except Exception as e:
            logger.error(f"❌ Pipeline initialization error for {lang_code}: {e}")
            tts_pipelines[lang_code] = None

def get_pipeline_for_voice(voice: str) -> Optional[KPipeline]:
    """Get the appropriate TTS pipeline for a given voice"""
    lang_code = VOICE_LANGUAGE_MAP.get(voice)
    if not lang_code:
        logger.warning(f"⚠️ Unknown voice: {voice}, falling back to default")
        default_voice = get_default_voice()
        lang_code = VOICE_LANGUAGE_MAP.get(default_voice, 'a')
    
    pipeline = tts_pipelines.get(lang_code)
    if not pipeline:
        logger.warning(f"⚠️ Pipeline not available for {lang_code}, trying fallback")
        # Try American English as fallback
        pipeline = tts_pipelines.get('a')
    
    return pipeline

def get_language_for_voice(voice: str) -> str:
    """Get language code for a given voice"""
    return VOICE_LANGUAGE_MAP.get(voice, 'a')

def is_voice_enabled(voice: str) -> bool:
    """Check if voice is enabled (its language is enabled)"""
    lang_code = get_language_for_voice(voice)
    enabled_languages = get_enabled_languages()
    return lang_code in enabled_languages

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    try:
        await initialize_all_pipelines()
        active_pipelines = sum(1 for p in tts_pipelines.values() if p is not None)
        total_enabled = len(get_enabled_languages())
        logger.info(f"✅ {active_pipelines}/{total_enabled} TTS pipelines ready")
    except Exception as e:
        logger.error(f"❌ TTS initialization failed: {e}")
        logger.warning("⚠️ TTS functionality may be limited")
    
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
    title="Multi-Language Kokoro TTS Server",
    version="2.1.0-settings",
    lifespan=lifespan
)

# Client management
client_sentence_buffers: Dict[str, APIStreamingSentenceBuffer] = {}
client_tts_sessions: Dict[str, dict] = {}
audio_client_mapping: Dict[str, List[str]] = {}  # client_id -> [socket_ids]

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

async def generate_tts_binary(sentence: str, voice: str, speed: float, client_id: str):
    """Generate TTS audio in binary format with language-specific pipeline and dynamic speed"""
    try:
        pipeline = get_pipeline_for_voice(voice)
        lang_code = get_language_for_voice(voice)
        
        if not pipeline:
            error_data = {
                'type': 'tts_error',
                'error': f'TTS pipeline not available for voice {voice} (language: {lang_code})',
                'sentence_text': sentence[:100],
                'client_id': client_id,
                'voice': voice,
                'language': lang_code,
                'speed': speed,
                'format': 'binary',
                'timestamp': datetime.now().isoformat()
            }
            yield error_data, None
            return
        
        # Apply sentence length limit from settings
        max_length = tts_settings.get('max_sentence_length', 180) if tts_settings else 180
        if len(sentence) > max_length:
            sentence = sentence[:max_length] + "..."
        
        # Validate speed
        validated_speed = validate_speed(speed)
        if validated_speed != speed:
            logger.warning(f"Speed {speed} clamped to {validated_speed}")
        
        cleaned_sentence = sentence.strip()
        logger.info(f"🎵 Generating TTS: {cleaned_sentence[:50]}... (voice: {voice}, lang: {lang_code}, speed: {validated_speed})")
        
        loop = asyncio.get_event_loop()
        generation_timeout = tts_settings.get('generation_timeout', 25.0) if tts_settings else 25.0
        
        generator = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: pipeline(cleaned_sentence, voice=voice, speed=validated_speed, split_pattern=r"\n+")),
            timeout=generation_timeout
        )
        
        chunk_count = 0
        total_duration = 0
        chunk_limit = tts_settings.get('chunk_limit', 40) if tts_settings else 40
        chunk_timeout = tts_settings.get('chunk_timeout', 3.0) if tts_settings else 3.0
        
        for i, chunk_data in enumerate(generator):
            if i >= chunk_limit:  # Limit chunks
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
                        timeout=chunk_timeout
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
                        'voice': voice,
                        'language': lang_code,
                        'speed': validated_speed,
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
            'voice': voice,
            'language': lang_code,
            'speed': validated_speed,
            'format': 'binary',
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"🎵 TTS completed: {chunk_count} chunks, {total_duration:.2f}s (voice: {voice}, lang: {lang_code}, speed: {validated_speed})")
        yield completion, None
        
    except Exception as e:
        logger.error(f"TTS generation error: {e}")
        error_data = {
            'type': 'tts_error',
            'error': str(e),
            'sentence_text': sentence,
            'client_id': client_id,
            'voice': voice,
            'language': get_language_for_voice(voice),
            'speed': speed,
            'format': 'binary',
            'timestamp': datetime.now().isoformat()
        }
        yield error_data, None

async def generate_tts_base64(sentence: str, voice: str, speed: float, client_id: str) -> AsyncGenerator[dict, None]:
    """Generate TTS audio in base64 format with language-specific pipeline and dynamic speed"""
    try:
        pipeline = get_pipeline_for_voice(voice)
        lang_code = get_language_for_voice(voice)
        
        if not pipeline:
            yield {
                'type': 'tts_error',
                'error': f'TTS pipeline not available for voice {voice} (language: {lang_code})',
                'sentence_text': sentence,
                'client_id': client_id,
                'voice': voice,
                'language': lang_code,
                'speed': speed,
                'format': 'base64',
                'timestamp': datetime.now().isoformat()
            }
            return
        
        # Apply sentence length limit from settings
        max_length = tts_settings.get('max_sentence_length', 180) if tts_settings else 180
        if len(sentence) > max_length:
            sentence = sentence[:max_length] + "..."
        
        # Validate speed
        validated_speed = validate_speed(speed)
        if validated_speed != speed:
            logger.warning(f"Speed {speed} clamped to {validated_speed}")
        
        logger.info(f"Generating base64 TTS: {sentence[:50]}... (voice: {voice}, lang: {lang_code}, speed: {validated_speed})")
        
        loop = asyncio.get_event_loop()
        generation_timeout = tts_settings.get('generation_timeout', 25.0) if tts_settings else 25.0
        
        generator = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: pipeline(sentence, voice=voice, speed=validated_speed)),
            timeout=generation_timeout
        )
        
        chunk_count = 0
        total_duration = 0
        chunk_limit = tts_settings.get('chunk_limit', 40) if tts_settings else 40
        chunk_timeout = tts_settings.get('chunk_timeout', 3.0) if tts_settings else 3.0
        
        for i, chunk_data in enumerate(generator):
            if i >= chunk_limit:
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
                    
                    audio_b64 = await asyncio.wait_for(
                        loop.run_in_executor(None, audio_to_base64, audio_np),
                        timeout=chunk_timeout
                    )
                    
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
                            'voice': voice,
                            'language': lang_code,
                            'speed': validated_speed,
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
            'voice': voice,
            'language': lang_code,
            'speed': validated_speed,
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
            'voice': voice,
            'language': get_language_for_voice(voice),
            'speed': speed,
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
    
    # Initialize client with default settings
    default_voice = get_default_voice()
    default_speed = get_default_speed()
    
    client_sentence_buffers[sid] = APIStreamingSentenceBuffer(sid, flagembedding_settings)
    client_tts_sessions[sid] = {
        'voice': default_voice,
        'speed': default_speed,
        'enabled': True,
        'format': format_type,
        'connected_time': datetime.now(),
        'main_client_id': main_client_id,
        'connection_type': connection_type
    }
    
    # Send confirmation with language and settings info
    active_languages = [lang for lang, pipeline in tts_pipelines.items() if pipeline is not None]
    await sio.emit('tts_connected', {
        'status': 'Connected to Multi-Language TTS server',
        'client_id': sid,
        'format': format_type,
        'version': '2.1.0-settings',
        'default_voice': default_voice,
        'default_speed': default_speed,
        'supported_languages': active_languages,
        'language_names': {lang: SUPPORTED_LANGUAGES.get(lang, 'Unknown') for lang in active_languages},
        'speed_range': {
            'min': tts_settings.get('speed_min', 0.5) if tts_settings else 0.5,
            'max': tts_settings.get('speed_max', 2.0) if tts_settings else 2.0
        },
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

    main_clients_to_update = []
    for main_client_id, audio_sids in audio_client_mapping.items():
        if isinstance(audio_sids, list):
            if sid in audio_sids:
                audio_sids.remove(sid)
                if not audio_sids:  # Remove empty lists
                    main_clients_to_update.append(main_client_id)
        elif audio_sids == sid:  # Legacy single client
            main_clients_to_update.append(main_client_id)
    
    for client_id in main_clients_to_update:
        del audio_client_mapping[client_id]        
    
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
    main_client_id = data.get('main_client_id')
    
    if main_client_id and main_client_id != 'unknown':
        # Support multiple audio clients per main client
        if main_client_id not in audio_client_mapping:
            audio_client_mapping[main_client_id] = []
        
        if sid not in audio_client_mapping[main_client_id]:
            audio_client_mapping[main_client_id].append(sid)
            logger.info(f"🎵 Audio mapping: {main_client_id} -> {audio_client_mapping[main_client_id]}")

@sio.event
async def tts_text_chunk(sid, data):
    """Handle streaming text chunks with BGE-M3 processing"""
    try:
        text_chunk = data.get('chunk', '')
        is_final = data.get('final', False)
        target_client_id = data.get('target_client_id', data.get('client_id', sid))
        
        if not text_chunk and not is_final:
            return
        
        # Find audio connections - FIXED to handle list
        audio_sids = audio_client_mapping.get(target_client_id, [target_client_id])
        
        # Ensure it's always a list
        if isinstance(audio_sids, str):
            audio_sids = [audio_sids]
        elif not isinstance(audio_sids, list):
            audio_sids = [target_client_id]
        
        # Get session settings from the FIRST audio client
        primary_audio_sid = audio_sids[0] if audio_sids else target_client_id
        session = client_tts_sessions.get(primary_audio_sid, {
            'voice': get_default_voice(),
            'speed': get_default_speed(),
            'enabled': True,
            'format': 'base64'
        })
        
        if not session.get('enabled', True):
            return
        
        voice = session.get('voice', get_default_voice())
        speed = session.get('speed', get_default_speed())
        format_type = session.get('format', 'base64')
        lang_code = get_language_for_voice(voice)
        
        # Check if voice is enabled
        if not is_voice_enabled(voice):
            logger.warning(f"Voice {voice} not enabled (language {lang_code} not in enabled languages)")
            return
        
        # TTS generation callback - FIXED to broadcast to all clients
        async def tts_generation_callback(sentence: str, client_id: str, analysis: dict):
            logger.info(f"🎵 TTS: {sentence[:40]}... broadcasting to {len(audio_sids)} clients (voice: {voice}, lang: {lang_code}, speed: {speed}, method: {analysis['method']}, confidence: {analysis['confidence']:.3f})")
            
            # Generate and stream to ALL audio clients
            for audio_sid in audio_sids:
                try:
                    if format_type == 'binary':
                        async for metadata, binary_data in generate_tts_binary(sentence, voice, speed, client_id):
                            if binary_data is not None:
                                metadata_with_binary = metadata.copy()
                                metadata_with_binary['audio_data'] = binary_data
                                await sio.emit('tts_audio_chunk', metadata_with_binary, room=audio_sid)
                            else:
                                await sio.emit('tts_sentence_complete', metadata, room=audio_sid)
                    else:
                        async for chunk_data in generate_tts_base64(sentence, voice, speed, client_id):
                            await sio.emit('tts_audio_chunk', chunk_data, room=audio_sid)
                except Exception as e:
                    logger.error(f"Error sending TTS to audio client {audio_sid}: {e}")
        
        # Process with BGE-M3
        await process_tts_with_api_bge(
            target_client_id, 
            text_chunk, 
            is_final, 
            client_sentence_buffers, 
            tts_generation_callback,
            flagembedding_settings
        )
        
        # Final completion - send to all clients
        if is_final:
            for audio_sid in audio_sids:
                try:
                    await sio.emit('tts_response_complete', {
                        'timestamp': datetime.now().isoformat()
                    }, room=audio_sid)
                except Exception as e:
                    logger.error(f"Error sending completion to {audio_sid}: {e}")
        
    except Exception as e:
        logger.error(f"Text chunk processing error: {e}")
        # Send error to all possible audio clients
        target_client_id = data.get('target_client_id', data.get('client_id', sid))
        audio_sids = audio_client_mapping.get(target_client_id, [target_client_id])
        if isinstance(audio_sids, str):
            audio_sids = [audio_sids]
            
        for audio_sid in audio_sids:
            try:
                await sio.emit('tts_error', {
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }, room=audio_sid)
            except Exception as send_error:
                logger.error(f"Error sending error to {audio_sid}: {send_error}")

@sio.event
async def tts_configure(sid, data):
    """Configure TTS settings with voice and speed validation"""
    session = client_tts_sessions.get(sid, {})
    
    if 'voice' in data:
        new_voice = data['voice']
        if new_voice in VOICE_LANGUAGE_MAP:
            if is_voice_enabled(new_voice):
                lang_code = get_language_for_voice(new_voice)
                pipeline = get_pipeline_for_voice(new_voice)
                
                if pipeline:
                    session['voice'] = new_voice
                    logger.info(f"Voice set to {new_voice} (language: {lang_code}) for client {sid}")
                else:
                    logger.warning(f"Pipeline not available for voice {new_voice} (language: {lang_code})")
            else:
                logger.warning(f"Voice {new_voice} not enabled (language not in enabled languages)")
        else:
            logger.warning(f"Unknown voice: {new_voice}")
    
    if 'speed' in data:
        new_speed = float(data['speed'])
        validated_speed = validate_speed(new_speed)
        session['speed'] = validated_speed
        if validated_speed != new_speed:
            logger.warning(f"Speed {new_speed} clamped to {validated_speed} for client {sid}")
        else:
            logger.info(f"Speed set to {validated_speed} for client {sid}")
    
    if 'enabled' in data:
        session['enabled'] = data['enabled']
    
    client_tts_sessions[sid] = session
    
    current_voice = session.get('voice', get_default_voice())
    current_speed = session.get('speed', get_default_speed())
    
    await sio.emit('tts_configured', {
        'voice': current_voice,
        'speed': current_speed,
        'language': get_language_for_voice(current_voice),
        'language_name': SUPPORTED_LANGUAGES.get(get_language_for_voice(current_voice), 'Unknown'),
        'enabled': session.get('enabled'),
        'format': session.get('format'),
        'pipeline_available': get_pipeline_for_voice(current_voice) is not None,
        'voice_enabled': is_voice_enabled(current_voice),
        'speed_range': {
            'min': tts_settings.get('speed_min', 0.5) if tts_settings else 0.5,
            'max': tts_settings.get('speed_max', 2.0) if tts_settings else 2.0
        },
        'timestamp': datetime.now().isoformat()
    }, room=sid)

@sio.event
async def tts_configure_client(sid, data):
    """Configure TTS for relay client with voice and speed validation"""
    client_id = data.get('client_id')
    audio_sid = audio_client_mapping.get(client_id, client_id)
    
    # Ensure it's a list and get primary client for settings
    if isinstance(audio_sids, str):
        audio_sids = [audio_sids]
    
    primary_audio_sid = audio_sids[0] if audio_sids else client_id
    session = client_tts_sessions.get(primary_audio_sid, {})
    
    if 'voice' in data:
        new_voice = data['voice']
        if new_voice in VOICE_LANGUAGE_MAP:
            if is_voice_enabled(new_voice):
                lang_code = get_language_for_voice(new_voice)
                pipeline = get_pipeline_for_voice(new_voice)
                
                if pipeline:
                    session['voice'] = new_voice
                    logger.info(f"Voice set to {new_voice} (language: {lang_code}) for client {client_id}")
                else:
                    logger.warning(f"Pipeline not available for voice {new_voice} (language: {lang_code})")
            else:
                logger.warning(f"Voice {new_voice} not enabled (language not in enabled languages)")
        else:
            logger.warning(f"Unknown voice: {new_voice}")
    
    if 'speed' in data:
        new_speed = float(data['speed'])
        validated_speed = validate_speed(new_speed)
        session['speed'] = validated_speed
        if validated_speed != new_speed:
            logger.warning(f"Speed {new_speed} clamped to {validated_speed} for client {client_id}")
        else:
            logger.info(f"Speed set to {validated_speed} for client {client_id}")
    
    if 'enabled' in data:
        session['enabled'] = data['enabled']
    
    client_tts_sessions[primary_audio_sid] = session
    
    current_voice = session.get('voice', get_default_voice())
    current_speed = session.get('speed', get_default_speed())
    
    await sio.emit('tts_client_configured', {
        'client_id': client_id,
        'voice': current_voice,
        'speed': current_speed,
        'language': get_language_for_voice(current_voice),
        'language_name': SUPPORTED_LANGUAGES.get(get_language_for_voice(current_voice), 'Unknown'),
        'enabled': session.get('enabled'),
        'pipeline_available': get_pipeline_for_voice(current_voice) is not None,
        'voice_enabled': is_voice_enabled(current_voice),
        'speed_range': {
            'min': tts_settings.get('speed_min', 0.5) if tts_settings else 0.5,
            'max': tts_settings.get('speed_max', 2.0) if tts_settings else 2.0
        },
        'timestamp': datetime.now().isoformat()
    }, room=sid)

@sio.event
async def tts_get_voices(sid, data):
    """Get available voices with language information"""
    voices_with_languages = []
    
    for voice, lang_code in VOICE_LANGUAGE_MAP.items():
        pipeline = get_pipeline_for_voice(voice)
        voice_enabled = is_voice_enabled(voice)
        voices_with_languages.append({
            'voice': voice,
            'language_code': lang_code,
            'language_name': SUPPORTED_LANGUAGES.get(lang_code, 'Unknown'),
            'available': pipeline is not None,
            'enabled': voice_enabled
        })
    
    # Group by language
    languages = {}
    for voice_info in voices_with_languages:
        lang_code = voice_info['language_code']
        if lang_code not in languages:
            languages[lang_code] = {
                'code': lang_code,
                'name': voice_info['language_name'],
                'enabled': voice_info['enabled'],
                'voices': []
            }
        languages[lang_code]['voices'].append(voice_info)
    
    await sio.emit('tts_voices_response', {
        'voices': [voice for voice in VOICE_LANGUAGE_MAP.keys() if is_voice_enabled(voice)],
        'voices_detailed': voices_with_languages,
        'languages': languages,
        'default_voice': get_default_voice(),
        'default_speed': get_default_speed(),
        'speed_range': {
            'min': tts_settings.get('speed_min', 0.5) if tts_settings else 0.5,
            'max': tts_settings.get('speed_max', 2.0) if tts_settings else 2.0
        },
        'requesting_client': data.get('requesting_client'),
        'timestamp': datetime.now().isoformat()
    }, room=sid)

@sio.event
async def tts_get_languages(sid, data):
    """Get supported languages and their status"""
    language_status = []
    enabled_languages = get_enabled_languages()
    
    for lang_code, lang_name in SUPPORTED_LANGUAGES.items():
        pipeline = tts_pipelines.get(lang_code)
        voices = [voice for voice, voice_lang in VOICE_LANGUAGE_MAP.items() if voice_lang == lang_code]
        is_enabled = lang_code in enabled_languages
        
        language_status.append({
            'code': lang_code,
            'name': lang_name,
            'available': pipeline is not None,
            'enabled': is_enabled,
            'voices': voices,
            'voice_count': len(voices)
        })
    
    await sio.emit('tts_languages_response', {
        'languages': language_status,
        'total_languages': len(SUPPORTED_LANGUAGES),
        'enabled_languages': len(enabled_languages),
        'active_languages': sum(1 for p in tts_pipelines.values() if p is not None),
        'default_voice': get_default_voice(),
        'default_speed': get_default_speed(),
        'speed_range': {
            'min': tts_settings.get('speed_min', 0.5) if tts_settings else 0.5,
            'max': tts_settings.get('speed_max', 2.0) if tts_settings else 2.0
        },
        'requesting_client': data.get('requesting_client'),
        'timestamp': datetime.now().isoformat()
    }, room=sid)

@sio.event
async def tts_get_settings(sid, data):
    """Get current TTS settings"""
    await sio.emit('tts_settings_response', {
        'default_voice': get_default_voice(),
        'default_speed': get_default_speed(),
        'enabled_languages': get_enabled_languages(),
        'speed_range': {
            'min': tts_settings.get('speed_min', 0.5) if tts_settings else 0.5,
            'max': tts_settings.get('speed_max', 2.0) if tts_settings else 2.0
        },
        'max_sentence_length': tts_settings.get('max_sentence_length', 180) if tts_settings else 180,
        'pipeline_timeout': tts_settings.get('pipeline_timeout', 300) if tts_settings else 300,
        'generation_timeout': tts_settings.get('generation_timeout', 25.0) if tts_settings else 25.0,
        'requesting_client': data.get('requesting_client'),
        'timestamp': datetime.now().isoformat()
    }, room=sid)

@sio.event
async def tts_client_disconnect(sid, data):
    """Handle relay client disconnect"""
    client_id = data.get('client_id')
    audio_sids = audio_client_mapping.get(client_id, [])

    # Make sure it's a list
    if isinstance(audio_sids, str):
        audio_sids = [audio_sids]

    # Remove from mapping
    if client_id in audio_client_mapping:
        del audio_client_mapping[client_id]

    # Cleanup all audio session buffers and sessions
    for audio_sid in audio_sids:
        if audio_sid in client_sentence_buffers:
            try:
                await client_sentence_buffers[audio_sid].close()
            except Exception:
                pass
            del client_sentence_buffers[audio_sid]
        if audio_sid in client_tts_sessions:
            del client_tts_sessions[audio_sid]


@sio.event
async def stop_generation(sid, data):
    """Handle stop of TTS generation - IMPROVED VERSION"""
    client_id = data.get('client_id')
    reason = data.get('reason', 'unknown')
    
    logger.info(f"🚨 STOP GENERATION RECEIVED: {client_id} (reason: {reason})")
    
    # Find the audio connection
    audio_sid = audio_client_mapping.get(client_id, client_id)
    
    if audio_sid in client_sentence_buffers:
        try:
            # SURGICAL FIX: Completely recreate the buffer instead of just stopping
            old_buffer = client_sentence_buffers[audio_sid]
            await old_buffer.close()
            
            # Create fresh buffer to ensure complete stop
            client_sentence_buffers[audio_sid] = APIStreamingSentenceBuffer(audio_sid, flagembedding_settings)
            
            logger.info(f"🚨 RECREATED sentence buffer for complete stop: {audio_sid}")
        except Exception as e:
            logger.error(f"Error recreating buffer: {e}")
    
    # CRITICAL: Send immediate stop signal to client to stop any playing audio
    await sio.emit('tts_stop_immediate', {
        'client_id': client_id,
        'reason': reason,
        'timestamp': datetime.now().isoformat()
    }, room=audio_sid)
    
    logger.info(f"🚨 Stop generation completed for {client_id}")

@sio.event
async def stop_current_generation(sid, data):
    """Handle immediate stop of current TTS generation"""
    client_id = data.get('client_id')
    immediate = data.get('immediate', False)
    
    logger.info(f"🚨 STOP CURRENT GENERATION: {client_id} (immediate: {immediate})")
    
    # Use the same logic as stop_generation
    await stop_generation(sid, data)




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

# Health check with language status
@app.get("/health")
async def health_check():
    pipeline_status = {}
    enabled_languages = get_enabled_languages()
    
    for lang_code, pipeline in tts_pipelines.items():
        pipeline_status[lang_code] = {
            'available': pipeline is not None,
            'enabled': lang_code in enabled_languages,
            'language_name': SUPPORTED_LANGUAGES.get(lang_code, 'Unknown'),
            'voice_count': len([v for v, l in VOICE_LANGUAGE_MAP.items() if l == lang_code])
        }
    
    return {
        "status": "healthy",
        "version": "2.1.0-settings",
        "pipelines": pipeline_status,
        "total_languages": len(SUPPORTED_LANGUAGES),
        "enabled_languages": len(enabled_languages),
        "active_pipelines": sum(1 for p in tts_pipelines.values() if p is not None),
        "active_clients": len(client_sentence_buffers),
        "supported_voices": len(VOICE_LANGUAGE_MAP),
        "enabled_voices": len([v for v in VOICE_LANGUAGE_MAP.keys() if is_voice_enabled(v)]),
        "default_voice": get_default_voice(),
        "default_speed": get_default_speed(),
        "speed_range": {
            "min": tts_settings.get('speed_min', 0.5) if tts_settings else 0.5,
            "max": tts_settings.get('speed_max', 2.0) if tts_settings else 2.0
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/languages")
async def get_languages():
    """REST endpoint for language information"""
    language_info = []
    enabled_languages = get_enabled_languages()
    
    for lang_code, lang_name in SUPPORTED_LANGUAGES.items():
        pipeline = tts_pipelines.get(lang_code)
        voices = [voice for voice, voice_lang in VOICE_LANGUAGE_MAP.items() if voice_lang == lang_code]
        is_enabled = lang_code in enabled_languages
        
        language_info.append({
            'code': lang_code,
            'name': lang_name,
            'available': pipeline is not None,
            'enabled': is_enabled,
            'voices': voices,
            'voice_count': len(voices)
        })
    
    return {
        "languages": language_info,
        "total_languages": len(SUPPORTED_LANGUAGES),
        "enabled_languages": len(enabled_languages),
        "active_languages": sum(1 for p in tts_pipelines.values() if p is not None),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/voices")
async def get_voices():
    """REST endpoint for voice information"""
    voices_with_languages = []
    
    for voice, lang_code in VOICE_LANGUAGE_MAP.items():
        pipeline = get_pipeline_for_voice(voice)
        voice_enabled = is_voice_enabled(voice)
        voices_with_languages.append({
            'voice': voice,
            'language_code': lang_code,
            'language_name': SUPPORTED_LANGUAGES.get(lang_code, 'Unknown'),
            'available': pipeline is not None,
            'enabled': voice_enabled
        })
    
    return {
        "voices": voices_with_languages,
        "total_voices": len(VOICE_LANGUAGE_MAP),
        "enabled_voices": len([v for v in voices_with_languages if v['enabled']]),
        "available_voices": sum(1 for v in voices_with_languages if v['available']),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/settings")
async def get_settings():
    """REST endpoint for TTS settings"""
    return {
        "default_voice": get_default_voice(),
        "default_speed": get_default_speed(),
        "enabled_languages": get_enabled_languages(),
        "speed_range": {
            "min": tts_settings.get('speed_min', 0.5) if tts_settings else 0.5,
            "max": tts_settings.get('speed_max', 2.0) if tts_settings else 2.0
        },
        "max_sentence_length": tts_settings.get('max_sentence_length', 180) if tts_settings else 180,
        "pipeline_timeout": tts_settings.get('pipeline_timeout', 300) if tts_settings else 300,
        "generation_timeout": tts_settings.get('generation_timeout', 25.0) if tts_settings else 25.0,
        "chunk_limit": tts_settings.get('chunk_limit', 40) if tts_settings else 40,
        "chunk_timeout": tts_settings.get('chunk_timeout', 3.0) if tts_settings else 3.0,
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
    settings = load_config()
    await initialize_flagembedding_settings(settings)
    await initialize_tts_settings(settings)
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
    logger.info("🌍 Starting Multi-Language Kokoro TTS Server with Dynamic Settings...")
    logger.info(f"📡 Server: http://{config.host}:{config.port}")
    logger.info(f"🔌 Socket.IO: http://{config.host}:{config.port}/socket.io/")
    logger.info(f"💡 Health check: http://{config.host}:{config.port}/health")
    logger.info(f"🌐 Languages: http://{config.host}:{config.port}/languages")
    logger.info(f"🎤 Voices: http://{config.host}:{config.port}/voices")
    logger.info(f"⚙️ Settings: http://{config.host}:{config.port}/settings")
    
    # Log TTS configuration
    logger.info(f"🎵 Default voice: {get_default_voice()}")
    logger.info(f"⚡ Default speed: {get_default_speed()}")
    enabled_langs = get_enabled_languages()
    enabled_lang_names = [f"{code} ({SUPPORTED_LANGUAGES.get(code, 'Unknown')})" for code in enabled_langs]
    logger.info(f"🗣️ Enabled languages: {', '.join(enabled_lang_names)}")

    await server.serve()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        raise
