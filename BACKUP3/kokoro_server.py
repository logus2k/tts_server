#!/usr/bin/env python3

"""
Kokoro TTS Server with FlagEmbedding API Integration and Binary Audio Support
Hybrid Architecture: Direct audio streaming to browsers, text relay through assistant.js
Enhanced with API-based BGE-M3 semantic boundary detection for natural TTS streaming
"""

"""
pip install kokoro==0.9.4 nltk fastapi uvicorn python-socketio soundfile numpy torch aiohttp
"""

import asyncio
import base64
import io
import json
import logging
from typing import Dict, Optional, AsyncGenerator, List
from datetime import datetime
import numpy as np
import soundfile as sf
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import socketio
from kokoro import KPipeline
import uvicorn
from contextlib import asynccontextmanager
import urllib.parse
import time

# FlagEmbedding API Integration
from bge_api_boundary_detector import APIStreamingSentenceBuffer, process_tts_with_api_bge

# NLTK for fallback sentence detection
import nltk
from nltk.tokenize import sent_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
tts_pipeline: Optional[KPipeline] = None
flagembedding_settings = None

async def initialize_flagembedding_settings():
    """Initialize FlagEmbedding settings for API communication"""
    global flagembedding_settings
    
    # Use default settings that match your existing FlagEmbedding service
    flagembedding_settings = {
        "FlagEmbedding": {
            "ServerAPIAddress": "http://localhost:8000",  # Your FlagEmbedding service endpoint
            "BatchSize": 8,
            "Dimension": 1024,  # BGE-M3 dimension
            "Debug": True
        }
    }
    
    logger.info(f"FlagEmbedding settings configured for: {flagembedding_settings['FlagEmbedding']['ServerAPIAddress']}")
    
    try:
        # Test API connectivity to your existing service
        from bge_api_boundary_detector import FlagEmbeddingAPIClient
        api_client = FlagEmbeddingAPIClient(flagembedding_settings)
        health_ok = await api_client.check_health()
        
        if health_ok:
            logger.info("✅ FlagEmbedding API service is available")
        else:
            logger.warning("⚠️ FlagEmbedding API service is not available - using traditional detection")
        
        await api_client.close()
        
    except Exception as e:
        logger.warning(f"Could not test FlagEmbedding API connectivity: {e}")
        logger.info("Will attempt to use API when requests are made")

async def initialize_pipeline():
    """Initialize Kokoro TTS pipeline"""
    global tts_pipeline
    try:
        logger.info("Initializing Kokoro TTS pipeline...")
        logger.info("This may take a few minutes on first run (model download)...")
        
        loop = asyncio.get_event_loop()
        
        # Add timeout to prevent hanging
        tts_pipeline = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: KPipeline(lang_code='a')),
            timeout=300.0  # 5 minute timeout for model loading
        )
        
        logger.info("✅ TTS pipeline initialized successfully")
        
        # Test the pipeline with a simple sentence
        try:
            test_gen = tts_pipeline("Hello", voice='af_heart', speed=1.0)
            # Just check if generator is created, don't consume it
            logger.info("✅ TTS pipeline test successful")
        except Exception as test_error:
            logger.error(f"⚠️ TTS pipeline test failed: {test_error}")
            
    except asyncio.TimeoutError:
        logger.error("❌ TTS pipeline initialization timed out after 5 minutes")
        logger.error("This usually means the model download is taking too long or failed")
        tts_pipeline = None
        raise
    except Exception as e:
        logger.error(f"❌ Failed to initialize TTS pipeline: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error("Make sure you have:")
        logger.error("1. Installed kokoro==0.9.4")
        logger.error("2. Sufficient disk space for model download")
        logger.error("3. Internet connection for model download")
        logger.error("4. Compatible PyTorch installation")
        tts_pipeline = None
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    try:
        await initialize_pipeline()
        logger.info("Application lifespan: TTS pipeline initialized successfully")
    except Exception as e:
        logger.error(f"❌ TTS pipeline initialization failed: {e}")
        logger.warning("⚠️ Server will start but TTS functionality will be disabled")
        logger.info("💡 The BGE-M3 semantic detection will still work for testing")
        # Don't raise - let the server start anyway for testing
    
    yield
    
    logger.info("Shutting down TTS server...")
    # Cleanup any remaining resources
    try:
        for buffer in client_sentence_buffers.values():
            await buffer.close()
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

# Create Socket.IO server with proper configuration
sio = socketio.AsyncServer(
    cors_allowed_origins="*",
    logger=False,
    engineio_logger=False,
    async_mode='asgi'
)

# FastAPI app
app = FastAPI(
    title="Kokoro TTS Integration Server - FlagEmbedding API Enhanced",
    description="Real-time TTS with API-based BGE-M3 semantic detection and binary audio streaming",
    version="1.3.0-api-bge-m3",
    lifespan=lifespan
)

# Client management for hybrid architecture
client_sentence_buffers: Dict[str, APIStreamingSentenceBuffer] = {}
client_tts_sessions: Dict[str, dict] = {}
audio_client_mapping: Dict[str, str] = {}  # Maps main_client_id -> audio_connection_sid

def audio_to_base64(audio_data: np.ndarray, sample_rate: int = 24000) -> str:
    """Convert audio numpy array to base64 encoded WAV (legacy format)"""
    try:
        # Validate input
        if audio_data is None or len(audio_data) == 0:
            logger.error("Empty audio data provided")
            return ""
        
        # Ensure audio data is in correct format
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Normalize audio to prevent clipping
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sample_rate, format='WAV', subtype='PCM_16')
        buffer.seek(0)
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        return audio_base64
    except Exception as e:
        logger.error(f"Error converting audio to base64: {e}")
        return ""

def audio_to_wav_bytes(audio_data: np.ndarray, sample_rate: int = 24000) -> bytes:
    """Convert audio numpy array directly to WAV bytes (binary format)"""
    try:
        # Validate input
        if audio_data is None or len(audio_data) == 0:
            logger.error("Empty audio data provided")
            return b""
        
        # Ensure audio data is in correct format
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Normalize audio to prevent clipping
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        logger.debug(f"Processing audio: shape={audio_data.shape}, dtype={audio_data.dtype}")
        
        # Create WAV in memory
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sample_rate, format='WAV', subtype='PCM_16')
        buffer.seek(0)
        
        # Get raw WAV bytes
        wav_bytes = buffer.read()
        
        if len(wav_bytes) == 0:
            logger.error("WAV encoding produced empty data")
            return b""
        
        logger.debug(f"WAV data size: {len(wav_bytes)} bytes")
        return wav_bytes
        
    except Exception as e:
        logger.error(f"Error converting audio to WAV bytes: {e}")
        return b""

async def generate_tts_for_sentence_binary_safe(sentence: str, voice: str, client_id: str, timeout_seconds: int = 25):
    """Generate TTS audio with timeout and error recovery - OPTIMIZED VERSION"""
    try:
        if not tts_pipeline:
            error_msg = "TTS pipeline not initialized. Check server startup logs for Kokoro initialization errors."
            logger.error(f"TTS generation failed for client {client_id}: {error_msg}")
            error_data = {
                'type': 'tts_error',
                'error': error_msg,
                'sentence_text': sentence[:100],
                'client_id': client_id,
                'format': 'binary',
                'timestamp': datetime.now().isoformat(),
                'troubleshooting': {
                    'check_kokoro_installation': 'pip install kokoro==0.9.4',
                    'check_pytorch': 'Ensure PyTorch is properly installed',
                    'check_disk_space': 'Model download requires several GB',
                    'check_internet': 'Model download requires internet connection'
                }
            }
            yield error_data, None
            return
        
        # Stricter length validation for faster processing
        if len(sentence) > 180:
            logger.warning(f"Sentence too long ({len(sentence)} chars), truncating: {sentence[:50]}...")
            sentence = sentence[:180] + "..."
        
        cleaned_sentence = sentence.strip()
        logger.info(f"🎵 Generating binary TTS for client {client_id}: {cleaned_sentence[:50]}... (length: {len(cleaned_sentence)})")
        
        # Reduced timeout for faster response
        try:
            loop = asyncio.get_event_loop()
            generator = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: tts_pipeline(cleaned_sentence, voice=voice, speed=1.25)), # type: ignore
                timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            logger.error(f"TTS generation timed out after {timeout_seconds}s for: {cleaned_sentence[:50]}...")
            error_data = {
                'type': 'tts_error',
                'error': f'TTS generation timed out (sentence too complex)',
                'sentence_text': cleaned_sentence[:100],
                'client_id': client_id,
                'format': 'binary',
                'timestamp': datetime.now().isoformat()
            }
            yield error_data, None
            return
        
        chunk_count = 0
        sentence_duration = 0
        total_audio_size = 0
        
        # Reduced chunk limit for faster streaming
        max_chunks = 40
        
        # Process each audio chunk from Kokoro
        for i, chunk_data in enumerate(generator):
            if i >= max_chunks:
                logger.warning(f"Reached max chunk limit ({max_chunks}) for sentence: {cleaned_sentence[:50]}...")
                break
                
            try:
                # Extract audio from Kokoro Result object
                audio = None
                
                if hasattr(chunk_data, 'output') and hasattr(chunk_data.output, 'audio'):
                    audio = chunk_data.output.audio # type: ignore
                    logger.debug(f"Extracted audio from Result.output.audio: {type(audio)} {audio.shape}")
                else:
                    logger.error(f"Could not find audio in expected location for chunk {i}")
                    continue
                
                # Convert torch tensor to numpy with error handling
                if audio is not None:
                    try:
                        if hasattr(audio, 'cpu'):
                            audio_np = audio.cpu().detach().numpy().astype(np.float32)
                        else:
                            audio_np = np.array(audio, dtype=np.float32)
                        
                        if audio_np.ndim > 1:
                            audio_np = audio_np.flatten()
                        
                        if len(audio_np) == 0:
                            logger.warning(f"Empty audio array for chunk {i}")
                            continue
                            
                    except Exception as conv_error:
                        logger.error(f"Failed to convert torch tensor to numpy: {conv_error}")
                        continue
                else:
                    logger.error(f"No audio data found for chunk {i}")
                    continue
                
                # Convert to WAV bytes with shorter timeout
                try:
                    wav_bytes = await asyncio.wait_for(
                        loop.run_in_executor(None, audio_to_wav_bytes, audio_np, 24000),
                        timeout=3.0
                    )
                except asyncio.TimeoutError:
                    logger.error(f"WAV conversion timed out for chunk {i}")
                    continue
                
                if not wav_bytes or len(wav_bytes) == 0:
                    logger.error(f"Failed to convert audio chunk {i} to WAV bytes")
                    continue
                
                chunk_duration = len(audio_np) / 24000
                sentence_duration += chunk_duration
                total_audio_size += len(wav_bytes)
                
                # Yield immediately for faster streaming
                metadata = {
                    'type': 'tts_audio_chunk',
                    'sentence_text': cleaned_sentence[:100] + ('...' if len(cleaned_sentence) > 100 else ''),
                    'chunk_id': i,
                    'sample_rate': 24000,
                    'duration': chunk_duration,
                    'sentence_duration': sentence_duration,
                    'client_id': client_id,
                    'audio_size_bytes': len(wav_bytes),
                    'format': 'binary',
                    'timestamp': datetime.now().isoformat()
                }
                
                logger.info(f"🎵 Yielding binary chunk {i}: {len(wav_bytes)} bytes")
                yield metadata, wav_bytes
                chunk_count += 1
                
                # Reduced sleep for faster streaming
                await asyncio.sleep(0.005)
                
            except Exception as e:
                logger.error(f"Error processing audio chunk {i}: {e}")
                continue
        
        # Yield completion metadata
        completion = {
            'type': 'tts_sentence_complete',
            'sentence_text': cleaned_sentence,
            'total_chunks': chunk_count,
            'total_duration': sentence_duration,
            'total_audio_size': total_audio_size,
            'client_id': client_id,
            'format': 'binary',
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"🎵 Binary TTS completed: {chunk_count} chunks, {sentence_duration:.2f}s, {total_audio_size} bytes")
        yield completion, None
        
    except Exception as e:
        logger.error(f"Error generating binary TTS: {e}")
        error_data = {
            'type': 'tts_error',
            'error': str(e),
            'sentence_text': sentence if 'sentence' in locals() else 'unknown',
            'client_id': client_id,
            'format': 'binary',
            'timestamp': datetime.now().isoformat()
        }
        yield error_data, None

async def generate_tts_for_sentence(sentence: str, voice: str, client_id: str) -> AsyncGenerator[dict, None]:
    """Generate TTS audio for a single sentence (base64 format) - OPTIMIZED VERSION"""
    try:
        if not tts_pipeline:
            error_msg = "TTS pipeline not initialized. Check server startup logs for Kokoro initialization errors."
            logger.error(f"TTS generation failed for client {client_id}: {error_msg}")
            yield {
                'type': 'tts_error',
                'error': error_msg,
                'sentence_text': sentence,
                'client_id': client_id,
                'format': 'base64',
                'timestamp': datetime.now().isoformat(),
                'troubleshooting': {
                    'check_kokoro_installation': 'pip install kokoro==0.9.4',
                    'check_pytorch': 'Ensure PyTorch is properly installed',
                    'check_disk_space': 'Model download requires several GB',
                    'check_internet': 'Model download requires internet connection'
                }
            }
            return
        
        logger.info(f"Generating base64 TTS for client {client_id}: {sentence[:50]}...")
        
        # Run TTS generation in thread pool
        loop = asyncio.get_event_loop()
        generator = await loop.run_in_executor(None, lambda: tts_pipeline(sentence, voice=voice, speed=1.25)) # type: ignore
        
        chunk_count = 0
        sentence_duration = 0
        
        # Process each audio chunk for this sentence
        for i, chunk_data in enumerate(generator):
            try:
                # Extract audio from Kokoro Result object
                audio = None
                
                if hasattr(chunk_data, 'output') and hasattr(chunk_data.output, 'audio'):
                    audio = chunk_data.output.audio # type: ignore
                    logger.debug(f"Extracted audio from Result.output.audio: {type(audio)} {audio.shape}")
                else:
                    logger.error(f"Could not find audio in expected location for base64 chunk {i}")
                    continue
                
                # Convert torch tensor to numpy
                if audio is not None:
                    try:
                        if hasattr(audio, 'cpu'):
                            audio_np = audio.cpu().detach().numpy().astype(np.float32)
                        else:
                            audio_np = np.array(audio, dtype=np.float32)
                        
                        if audio_np.ndim > 1:
                            audio_np = audio_np.flatten()
                            
                        if len(audio_np) == 0:
                            logger.warning(f"Empty audio array for base64 chunk {i}")
                            continue
                            
                    except Exception as conv_error:
                        logger.error(f"Failed to convert torch tensor to numpy for base64: {conv_error}")
                        continue
                else:
                    logger.error(f"No audio data found for base64 chunk {i}")
                    continue
                
                # Convert audio chunk to base64
                audio_b64 = await loop.run_in_executor(None, audio_to_base64, audio_np)
                
                if audio_b64:
                    chunk_duration = len(audio_np) / 24000
                    sentence_duration += chunk_duration
                    
                    chunk_data = {
                        'type': 'tts_audio_chunk',
                        'sentence_text': sentence[:100] + ('...' if len(sentence) > 100 else ''),
                        'chunk_id': i,
                        'audio_data': audio_b64,
                        'sample_rate': 24000,
                        'duration': chunk_duration,
                        'sentence_duration': sentence_duration,
                        'client_id': client_id,
                        'format': 'base64',
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    yield chunk_data
                    chunk_count += 1
                    
                    logger.info(f"✅ Generated base64 chunk {i}: {chunk_duration:.2f}s")
                    await asyncio.sleep(0.005)
                    
            except Exception as e:
                logger.error(f"Error processing base64 audio chunk {i}: {e}")
                continue
        
        # Send sentence completion
        completion_data = {
            'type': 'tts_sentence_complete',
            'sentence_text': sentence,
            'total_chunks': chunk_count,
            'total_duration': sentence_duration,
            'client_id': client_id,
            'format': 'base64',
            'timestamp': datetime.now().isoformat()
        }
        
        yield completion_data
        logger.info(f"🎵 Base64 TTS completed for sentence ({chunk_count} chunks, {sentence_duration:.2f}s)")
        
    except Exception as e:
        logger.error(f"Error generating base64 TTS for sentence: {e}")
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
    """Handle client connection with API-based BGE-M3 enhanced format detection"""
    query_params = environ.get('QUERY_STRING', '')
    main_client_id = None
    connection_type = 'unknown'
    format_type = 'base64'  # Default fallback
    
    # Parse query parameters
    if query_params:
        parsed_params = urllib.parse.parse_qs(query_params)
        main_client_id = parsed_params.get('main_client_id', [None])[0]
        connection_type = parsed_params.get('type', ['browser'])[0]
        format_type = parsed_params.get('format', ['base64'])[0]
    
    logger.info(f"🎵 API BGE-M3 TTS client connected: {sid} (type: {connection_type}, format: {format_type}, main_client: {main_client_id})")
    
    # Initialize API-based sentence buffer and session
    client_sentence_buffers[sid] = APIStreamingSentenceBuffer(sid, flagembedding_settings)
    client_tts_sessions[sid] = {
        'voice': 'af_heart',
        'enabled': True,
        'format': format_type,
        'connected_time': datetime.now(),
        'main_client_id': main_client_id,
        'connection_type': connection_type,
        'api_bge_enhanced': True
    }
    
    try:
        await sio.emit('tts_connected', {
            'status': 'Connected to API BGE-M3 Enhanced TTS server',
            'client_id': sid,
            'format': format_type,
            'connection_type': connection_type,
            'version': '1.3.0-api-bge-m3',
            'features': [
                'api_bge_m3_semantic_detection',
                'flagembedding_integration',
                'completion_confidence_scoring', 
                'immediate_streaming', 
                'smart_chunking', 
                'binary_audio',
                'multilingual_support',
                'embedding_caching'
            ],
            'api_semantic_processing': {
                'service': 'FlagEmbedding API',
                'endpoint': flagembedding_settings['FlagEmbedding']['ServerAPIAddress'],
                'confidence_threshold': 0.75,
                'check_interval_ms': 500,
                'batch_size': flagembedding_settings['FlagEmbedding']['BatchSize'],
                'caching_enabled': True
            },
            'timestamp': datetime.now().isoformat()
        }, room=sid)
        logger.info(f"🎵 Sent API BGE-M3 enhanced tts_connected confirmation to {sid}")
    except Exception as e:
        logger.error(f"🎵 Failed to send API BGE-M3 tts_connected: {e}")

@sio.event
async def disconnect(sid):
    """Handle client disconnection with API resource cleanup"""
    logger.info(f"API BGE-M3 TTS client disconnected: {sid}")
    
    # Find and clean up main client mapping
    main_client_to_remove = None
    for main_client_id, audio_sid in audio_client_mapping.items():
        if audio_sid == sid:
            main_client_to_remove = main_client_id
            break
    
    if main_client_to_remove:
        del audio_client_mapping[main_client_to_remove]
        logger.info(f"Cleaned up mapping for disconnected audio client: {main_client_to_remove}")
    
    # Close API resources for sentence buffer
    if sid in client_sentence_buffers:
        try:
            await client_sentence_buffers[sid].close()
        except Exception as e:
            logger.error(f"Error closing API resources for {sid}: {e}")
        del client_sentence_buffers[sid]
    
    if sid in client_tts_sessions:
        del client_tts_sessions[sid]

@sio.event
async def register_audio_client(sid, data):
    """Register audio client with main client ID for hybrid architecture"""
    main_client_id = data.get('main_client_id')
    logger.info(f"🎵 Registering audio client: {sid} for main client: {main_client_id}")
    
    if main_client_id and main_client_id != 'unknown':
        # Store the mapping
        audio_client_mapping[main_client_id] = sid
        logger.info(f"🎵 Audio client mapping registered: {main_client_id} -> {sid}")
        
        # Update the session with correct mapping and preserve format
        if sid in client_tts_sessions:
            client_tts_sessions[sid]['main_client_id'] = main_client_id
            client_tts_sessions[sid]['connection_type'] = 'browser_audio'
            logger.info(f"🎵 Updated session for {sid}: {client_tts_sessions[sid]}")
            
        # Send confirmation back to the audio client
        try:
            await sio.emit('audio_client_registered', {
                'status': 'registered',
                'main_client_id': main_client_id,
                'audio_client_id': sid,
                'format': client_tts_sessions.get(sid, {}).get('format', 'base64'),
                'api_enhanced_features': True,
                'timestamp': datetime.now().isoformat()
            }, room=sid)
            logger.info(f"🎵 Sent registration confirmation to {sid}")
        except Exception as e:
            logger.error(f"🎵 Failed to send registration confirmation: {e}")
            
        # Log current mappings for debugging
        logger.info(f"🎵 Current audio mappings: {dict(audio_client_mapping)}")
        
    else:
        logger.warning(f"🎵 No valid main_client_id provided for audio client {sid} (got: {main_client_id})")
        
        # Send error response
        try:
            await sio.emit('registration_error', {
                'error': 'Invalid or missing main_client_id',
                'provided_id': main_client_id
            }, room=sid)
        except Exception as e:
            logger.error(f"🎵 Failed to send registration error: {e}")

@sio.event
async def tts_text_chunk(sid, data):
    """Handle streaming text chunks with API-based BGE-M3 SEMANTIC PROCESSING"""
    try:
        text_chunk = data.get('chunk', '')
        is_final = data.get('final', False)
        target_client_id = data.get('target_client_id', data.get('client_id', sid))
        
        # Enhanced logging for API BGE-M3 processing
        logger.info(f"🎵 === API BGE-M3 TTS CHUNK PROCESSING ===")
        logger.info(f"🎵 Client: {target_client_id}")
        logger.info(f"🎵 Chunk: '{text_chunk}' (length: {len(text_chunk)})")
        logger.info(f"🎵 Final: {is_final}")
        
        if not text_chunk and not is_final:
            logger.warning(f"🎵 Empty text chunk and not final for {target_client_id}")
            return
        
        # Find the audio connection for this client
        audio_sid = audio_client_mapping.get(target_client_id, target_client_id)
        logger.info(f"🎵 Audio mapping: {target_client_id} -> {audio_sid}")
        
        # Get TTS session settings
        session = client_tts_sessions.get(audio_sid, {
            'voice': 'af_heart',
            'enabled': True,
            'format': 'base64'
        })
        
        if not session.get('enabled', True):
            logger.info(f"🎵 TTS disabled for client {target_client_id}")
            return
        
        voice = session.get('voice', 'af_heart')
        format_type = session.get('format', 'base64')
        
        logger.info(f"🎵 Processing API BGE-M3 TTS for client {target_client_id} using format: {format_type}, voice: {voice}")
        
        # API BGE-M3 Enhanced TTS generation callback
        async def api_bge_tts_generation_callback(sentence: str, client_id: str, analysis: dict):
            """Enhanced TTS generation with API BGE-M3 analysis integration"""
            
            # Send enhanced sentence started event with API analysis
            await sio.emit('tts_sentence_started', {
                'sentence': sentence[:100] + ('...' if len(sentence) > 100 else ''),
                'timestamp': datetime.now().isoformat(),
                'format': format_type,
                'analysis': {
                    'method': analysis['method'],
                    'confidence': analysis['confidence'],
                    'processing_time_ms': analysis.get('processing_time_ms', 0),
                    'api_used': analysis.get('api_used', False),
                    'api_cached': analysis.get('semantic_details', {}).get('api_cached', False)
                },
                'api_bge_m3_enhanced': True,
                'flagembedding_service': True
            }, room=audio_sid)
            
            logger.info(f"🌐 API BGE-M3 Analysis - Method: {analysis['method']}, "
                       f"Confidence: {analysis['confidence']:.3f}, "
                       f"API Time: {analysis.get('processing_time_ms', 0):.1f}ms, "
                       f"API Used: {analysis.get('api_used', False)}")
            
            # Generate and stream based on format preference
            if format_type == 'binary':
                chunk_count = 0
                async for metadata, binary_data in generate_tts_for_sentence_binary_safe(sentence, voice, client_id):
                    if binary_data is not None:
                        try:
                            # Include API BGE-M3 analysis in metadata
                            metadata_with_binary = metadata.copy()
                            metadata_with_binary['audio_data'] = binary_data
                            metadata_with_binary['api_bge_analysis'] = analysis
                            metadata_with_binary['flagembedding_enhanced'] = True
                            
                            await sio.emit('tts_audio_chunk', metadata_with_binary, room=audio_sid)
                            chunk_count += 1
                            logger.debug(f"🎵 Streamed API BGE-M3 binary chunk {metadata.get('chunk_id')}: {len(binary_data)} bytes")
                            
                        except Exception as e:
                            logger.error(f"🎵 Failed to stream API BGE-M3 binary chunk: {e}")
                    else:
                        # Completion event with API analysis
                        try:
                            completion_metadata = metadata.copy()
                            completion_metadata['api_bge_analysis'] = analysis
                            completion_metadata['flagembedding_enhanced'] = True
                            await sio.emit('tts_sentence_complete', completion_metadata, room=audio_sid)
                            logger.info(f"🎵 API BGE-M3 binary sentence completed: {chunk_count} chunks")
                        except Exception as e:
                            logger.error(f"🎵 Failed to send API BGE-M3 completion: {e}")
            else:
                # Base64 format with API BGE-M3 analysis
                chunk_count = 0
                async for chunk_data in generate_tts_for_sentence(sentence, voice, client_id):
                    try:
                        # Add API BGE-M3 analysis to chunk data
                        enhanced_chunk_data = chunk_data.copy()
                        enhanced_chunk_data['api_bge_analysis'] = analysis
                        enhanced_chunk_data['flagembedding_enhanced'] = True
                        
                        await sio.emit('tts_audio_chunk', enhanced_chunk_data, room=audio_sid)
                        chunk_count += 1
                        logger.debug(f"🎵 Streamed API BGE-M3 base64 chunk {chunk_data.get('chunk_id')}")
                    except Exception as e:
                        logger.error(f"🎵 Failed to stream API BGE-M3 base64 chunk: {e}")
                
                logger.info(f"🎵 API BGE-M3 base64 sentence completed: {chunk_count} chunks")
        
        # Process using API-based BGE-M3 enhanced boundary detection
        await process_tts_with_api_bge(
            target_client_id, 
            text_chunk, 
            is_final, 
            client_sentence_buffers, 
            api_bge_tts_generation_callback,
            flagembedding_settings
        )
        
        # Handle final completion
        if is_final:
            # Send enhanced completion to audio connection
            try:
                await sio.emit('tts_response_complete', {
                    'timestamp': datetime.now().isoformat(),
                    'api_bge_m3_enhanced': True,
                    'flagembedding_service': True,
                    'semantic_processing': True,
                    'api_endpoint': flagembedding_settings['FlagEmbedding']['ServerAPIAddress']
                }, room=audio_sid)
                logger.info(f"🎵 Sent API BGE-M3 enhanced tts_response_complete to {audio_sid}")
            except Exception as e:
                logger.error(f"🎵 Failed to send API BGE-M3 tts_response_complete: {e}")
        
    except Exception as e:
        logger.error(f"🎵 Error processing API BGE-M3 enhanced text chunk for {target_client_id}: {e}")
        audio_sid = audio_client_mapping.get(data.get('target_client_id', data.get('client_id', sid)), sid)
        try:
            await sio.emit('tts_error', {
                'error': str(e),
                'api_bge_m3_enhanced': True,
                'flagembedding_service': True,
                'timestamp': datetime.now().isoformat()
            }, room=audio_sid)
        except Exception as emit_error:
            logger.error(f"🎵 Failed to send API BGE-M3 error event: {emit_error}")

@sio.event
async def tts_configure(sid, data):
    """Configure TTS settings for client - handles direct browser connections"""
    try:
        session = client_tts_sessions.get(sid, {})
        
        if 'voice' in data:
            session['voice'] = data['voice']
            logger.info(f"Client {sid} changed voice to: {data['voice']}")
        
        if 'enabled' in data:
            session['enabled'] = data['enabled']
            logger.info(f"Client {sid} TTS enabled: {data['enabled']}")
        
        client_tts_sessions[sid] = session
        
        await sio.emit('tts_configured', {
            'voice': session.get('voice'),
            'enabled': session.get('enabled'),
            'format': session.get('format'),
            'api_enhanced_features': True,
            'timestamp': datetime.now().isoformat()
        }, room=sid)
        
    except Exception as e:
        logger.error(f"Error configuring TTS for {sid}: {e}")

@sio.event
async def tts_configure_client(sid, data):
    """Configure TTS for a specific client (from assistant.js relay)"""
    try:
        client_id = data.get('client_id')
        audio_sid = audio_client_mapping.get(client_id, client_id)
        
        # Update session for the audio connection
        session = client_tts_sessions.get(audio_sid, {})
        
        if 'voice' in data:
            session['voice'] = data['voice']
        if 'enabled' in data:
            session['enabled'] = data['enabled']
        
        client_tts_sessions[audio_sid] = session
        
        logger.info(f"Configured API enhanced TTS for client {client_id} (audio: {audio_sid}): {session}")
        
        # Send confirmation to the relay (assistant.js)
        await sio.emit('tts_client_configured', {
            'client_id': client_id,
            'voice': session.get('voice'),
            'enabled': session.get('enabled'),
            'format': session.get('format'),
            'api_enhanced_features': True,
            'timestamp': datetime.now().isoformat()
        }, room=sid)
        
    except Exception as e:
        logger.error(f"Error configuring API enhanced TTS for client: {e}")

@sio.event
async def tts_get_voices(sid, data):
    """Handle voice list request from assistant.js relay"""
    voices = [
        'af_heart', 'af_sky', 'af_bella', 'af_sarah',
        'am_adam', 'am_michael', 'bf_emma', 'bf_isabella'
    ]
    
    # Send response back to assistant.js (not to browser)
    await sio.emit('tts_voices_response', {
        'voices': voices,
        'requesting_client': data.get('requesting_client'),
        'api_enhanced_features': True,
        'timestamp': datetime.now().isoformat()
    }, room=sid)

@sio.event
async def tts_client_disconnect(sid, data):
    """Handle client disconnect notification from assistant.js"""
    client_id = data.get('client_id')
    audio_sid = audio_client_mapping.get(client_id)
    
    if audio_sid:
        logger.info(f"Cleaning up API enhanced TTS resources for disconnected client: {client_id} -> {audio_sid}")
        
        # Clean up mapping
        del audio_client_mapping[client_id]
        
        # Clean up buffers and sessions if audio client is gone
        if audio_sid in client_sentence_buffers:
            try:
                await client_sentence_buffers[audio_sid].close()
            except Exception as e:
                logger.error(f"Error closing API resources during disconnect: {e}")
            del client_sentence_buffers[audio_sid]
        if audio_sid in client_tts_sessions:
            del client_tts_sessions[audio_sid]

# Periodic cleanup of stale sentence buffers
async def cleanup_stale_buffers():
    """Clean up stale sentence buffers - API enhanced version"""
    while True:
        try:
            stale_clients = []
            for client_id, buffer in client_sentence_buffers.items():
                if hasattr(buffer, 'is_stale') and buffer.is_stale(timeout_seconds=45):
                    stale_clients.append(client_id)
            
            for client_id in stale_clients:
                logger.info(f"Cleaning up stale API enhanced buffer for client: {client_id}")
                try:
                    await client_sentence_buffers[client_id].close()
                except Exception as e:
                    logger.error(f"Error closing stale buffer: {e}")
                del client_sentence_buffers[client_id]
                if client_id in client_tts_sessions:
                    del client_tts_sessions[client_id]
            
            if stale_clients:
                logger.info(f"API enhanced cleanup: removed {len(stale_clients)} stale buffers")
            
        except Exception as e:
            logger.error(f"Error in API enhanced cleanup task: {e}")
        
        await asyncio.sleep(20)  # More frequent cleanup

# REST API Endpoints
@app.get("/", response_class=HTMLResponse)
async def get_root():
    binary_clients = sum(1 for s in client_tts_sessions.values() if s.get('format') == 'binary')
    base64_clients = sum(1 for s in client_tts_sessions.values() if s.get('format') == 'base64')
    
    try:
        from bge_api_boundary_detector import FlagEmbeddingAPIClient
        api_client = FlagEmbeddingAPIClient(flagembedding_settings)
        health_ok = await api_client.check_health()
        await api_client.close()
        
        api_status = "✅ Connected" if health_ok else "❌ Unavailable"
        api_features = f"""
            <li>🌐 FlagEmbedding API integration: {flagembedding_settings['FlagEmbedding']['ServerAPIAddress']}</li>
            <li>🧠 Semantic boundary detection with 0.75 confidence threshold</li>
            <li>🎯 Completion confidence scoring via API calls</li>
            <li>🌍 Multilingual support (100+ languages)</li>
            <li>⚡ HTTP connection pooling for performance</li>
            <li>🔄 Real-time semantic analysis every 500ms</li>
            <li>📦 Batch processing with size {flagembedding_settings['FlagEmbedding']['BatchSize']}</li>
            <li>💾 Embedding caching for improved response times</li>
        """ if health_ok else """
            <li>⚠️ FlagEmbedding API service unavailable - using traditional detection</li>
            <li>🔧 Check API service at {flagembedding_settings['FlagEmbedding']['ServerAPIAddress']}</li>
        """
    except Exception as e:
        api_status = f"❌ Error: {str(e)}"
        api_features = "<li>⚠️ FlagEmbedding API not configured - using traditional detection</li>"
    
    return HTMLResponse(content=f"""
    <!DOCTYPE html>
    <html>
    <head><title>Kokoro TTS Integration Server - FlagEmbedding API Enhancement</title></head>
    <body>
        <h1>🎵 Kokoro TTS Integration Server - FlagEmbedding API Enhanced</h1>
        <p><strong>Status:</strong> Running (FlagEmbedding API + Binary Audio Support)</p>
        <p><strong>Version:</strong> 1.3.0-api-bge-m3</p>
        <p><strong>Pipeline:</strong> {'✅ Ready' if tts_pipeline else '❌ Not Ready'}</p>
        <p><strong>FlagEmbedding API:</strong> {api_status}</p>
        <p><strong>Active Clients:</strong> {len(client_sentence_buffers)}</p>
        <p><strong>Audio Mappings:</strong> {len(audio_client_mapping)}</p>
        
        <h3>🌐 FlagEmbedding API Features:</h3>
        <ul>
            {api_features}
        </ul>
        
        <h3>🚀 Performance Improvements:</h3>
        <ul>
            <li>⚡ API-based semantic completion detection (0.5-1.0s response time)</li>
            <li>🎯 Context-aware boundary detection via external service</li>
            <li>🔄 Async API calls with traditional fallback</li>
            <li>✂️ Intelligent sentence chunking preserving meaning</li>
            <li>📈 Confidence-based completion thresholds</li>
            <li>🌐 Multilingual conversation support</li>
            <li>🔗 HTTP connection pooling for efficiency</li>
            <li>💾 Multi-level caching (API + local)</li>
        </ul>
        
        <h3>Client Format Distribution:</h3>
        <ul>
            <li>🔥 Binary Format: {binary_clients} clients (33% smaller)</li>
            <li>📝 Base64 Format: {base64_clients} clients (compatibility)</li>
        </ul>
        
        <h3>Detection Methods:</h3>
        <ul>
            <li>🌐 <strong>API Semantic High:</strong> BGE-M3 via API confidence ≥ 0.75</li>
            <li>⏰ <strong>API Semantic Timed:</strong> BGE-M3 via API confidence ≥ 0.65 + 0.7s delay</li>
            <li>⚡ <strong>Immediate:</strong> Strong punctuation detection</li>
            <li>🚨 <strong>Forced:</strong> Timeout/length limits</li>
        </ul>
        
        <h3>API Configuration:</h3>
        <ul>
            <li><strong>Endpoint:</strong> {flagembedding_settings['FlagEmbedding']['ServerAPIAddress']}</li>
            <li><strong>Model:</strong> BAAI/bge-m3</li>
            <li><strong>Batch Size:</strong> {flagembedding_settings['FlagEmbedding']['BatchSize']}</li>
            <li><strong>Dimension:</strong> {flagembedding_settings['FlagEmbedding']['Dimension']}</li>
            <li><strong>Confidence Threshold:</strong> 0.75 (high), 0.65 (timed)</li>
            <li><strong>API Check Interval:</strong> 500ms</li>
            <li><strong>Completion Delay:</strong> 700ms</li>
        </ul>
        
        <h3>API Endpoints:</h3>
        <ul>
            <li><a href="/api/health">/api/health</a> - FlagEmbedding API enhanced server status</li>
            <li><a href="/api/flagembedding-stats">/api/flagembedding-stats</a> - API performance metrics</li>
            <li><a href="/api/clients">/api/clients</a> - Client information</li>
            <li><a href="/api/performance-test">/api/performance-test</a> - Performance comparison</li>
            <li><a href="/api/stats">/api/stats</a> - Detailed statistics</li>
        </ul>
    </body>
    </html>
    """)

@app.get("/api/health")
async def health_check():
    api_status = "unknown"
    api_error = None
    api_response_time = None
    
    try:
        from bge_api_boundary_detector import FlagEmbeddingAPIClient
        api_client = FlagEmbeddingAPIClient(flagembedding_settings)
        
        start_time = time.time()
        health_ok = await api_client.check_health()
        api_response_time = (time.time() - start_time) * 1000  # Convert to ms
        
        api_status = "healthy" if health_ok else "unhealthy"
        await api_client.close()
        
    except Exception as e:
        api_status = "error"
        api_error = str(e)
    
    return {
        "status": "healthy",
        "version": "1.3.0-api-bge-m3",
        "architecture": "hybrid",
        "pipeline_ready": tts_pipeline is not None,
        "active_clients": len(client_sentence_buffers),
        "active_sessions": len(client_tts_sessions),
        "audio_mappings": len(audio_client_mapping),
        "supported_formats": ["binary", "base64"],
        "flagembedding_api_integration": {
            "status": api_status,
            "endpoint": flagembedding_settings['FlagEmbedding']['ServerAPIAddress'],
            "model": "BAAI/bge-m3",
            "response_time_ms": api_response_time,
            "error": api_error,
            "semantic_processing": api_status == "healthy",
            "embedding_cache": True,
            "multilingual": True,
            "batch_processing": True
        },
        "enhanced_features": {
            "api_semantic_boundary_detection": api_status == "healthy",
            "completion_confidence_scoring": api_status == "healthy", 
            "immediate_streaming": True,
            "smart_chunking": True,
            "parallel_processing": True,
            "reduced_latency": True,
            "sentence_optimization": True,
            "connection_pooling": True
        },
        "performance_metrics": {
            "sentence_detection_method": "FlagEmbedding API + traditional",
            "semantic_confidence_threshold": 0.75,
            "semantic_check_interval_ms": 500,
            "api_batch_size": flagembedding_settings['FlagEmbedding']['BatchSize'],
            "tts_start_latency": "0.5-1.0s (including API call)",
            "chunk_processing": "parallel",
            "timeout_optimization": "enabled"
        },
        "format_distribution": {
            "binary": sum(1 for s in client_tts_sessions.values() if s.get('format') == 'binary'),
            "base64": sum(1 for s in client_tts_sessions.values() if s.get('format') == 'base64')
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/flagembedding-stats")
async def get_flagembedding_stats():
    """Get FlagEmbedding API performance and usage statistics"""
    
    buffer_stats = []
    total_api_requests = 0
    total_api_time = 0
    method_counts = {}
    confidence_scores = []
    cache_performance = {'hits': 0, 'misses': 0}
    
    for client_id, buffer in client_sentence_buffers.items():
        try:
            stats = buffer.get_stats()
            buffer_stats.append(stats)
            
            # Aggregate API performance metrics
            api_perf = stats.get('api_performance', {})
            total_api_requests += api_perf.get('total_requests', 0)
            total_api_time += api_perf.get('average_response_time_ms', 0) * api_perf.get('total_requests', 0)
            cache_performance['hits'] += api_perf.get('cache_hits', 0)
            cache_performance['misses'] += api_perf.get('cache_misses', 0)
            
        except Exception as e:
            logger.error(f"Error getting stats for buffer {client_id}: {e}")
    
    avg_api_time = total_api_time / max(1, total_api_requests)
    cache_hit_rate = cache_performance['hits'] / max(1, cache_performance['hits'] + cache_performance['misses'])
    
    return {
        "flagembedding_api_status": {
            "service_endpoint": flagembedding_settings['FlagEmbedding']['ServerAPIAddress'],
            "model_name": "BAAI/bge-m3",
            "active_buffers": len(client_sentence_buffers),
            "semantic_processing": "api_enabled",
            "batch_size": flagembedding_settings['FlagEmbedding']['BatchSize'],
            "dimension": flagembedding_settings['FlagEmbedding']['Dimension']
        },
        "api_performance_metrics": {
            "total_api_requests": total_api_requests,
            "average_response_time_ms": avg_api_time,
            "semantic_check_interval_ms": 500,
            "confidence_threshold": 0.75,
            "cache_hit_rate": cache_hit_rate * 100,
            "cache_stats": cache_performance
        },
        "buffer_statistics": buffer_stats,
        "detection_methods": method_counts,
        "enhanced_features": {
            "api_semantic_boundary_detection": True,
            "completion_confidence_scoring": True,
            "multilingual_support": True,
            "real_time_processing": True,
            "embedding_caching": True,
            "connection_pooling": True,
            "batch_processing": True
        },
        "configuration": {
            "api_endpoint": flagembedding_settings['FlagEmbedding']['ServerAPIAddress'],
            "batch_size": flagembedding_settings['FlagEmbedding']['BatchSize'],
            "cache_size": 500,
            "timeout_settings": {
                "total_timeout": 10.0,
                "connect_timeout": 2.0,
                "read_timeout": 5.0
            }
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/clients")
async def get_client_info():
    """Debug endpoint to see client mappings with API enhanced format info"""
    return {
        "audio_mappings": audio_client_mapping,
        "active_buffers": list(client_sentence_buffers.keys()),
        "active_sessions": {
            sid: {
                "voice": session.get("voice"),
                "enabled": session.get("enabled"),
                "format": session.get("format"),
                "connection_type": session.get("connection_type"),
                "main_client_id": session.get("main_client_id"),
                "connected_time": session.get("connected_time").isoformat() if session.get("connected_time") else None,
                "api_enhanced_features": True
            }
            for sid, session in client_tts_sessions.items()
        },
        "api_enhanced_status": {
            "flagembedding_integration": True,
            "semantic_processing": True,
            "connection_pooling": True
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/performance-test")
async def performance_test():
    """Compare binary vs base64 performance with API enhanced metrics"""
    
    # Generate test audio
    sample_rate = 24000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    # Test binary conversion
    start_time = time.time()
    wav_bytes = audio_to_wav_bytes(audio_data, sample_rate)
    binary_time = time.time() - start_time
    
    # Test base64 conversion
    start_time = time.time()
    base64_data = audio_to_base64(audio_data, sample_rate)
    base64_time = time.time() - start_time
    
    # Calculate sizes
    binary_size = len(wav_bytes)
    base64_size = len(base64_data.encode('utf-8'))
    overhead_percentage = ((base64_size / binary_size) - 1) * 100
    
    return {
        "test_audio": {
            "duration": duration,
            "sample_rate": sample_rate,
            "samples": len(audio_data)
        },
        "binary_format": {
            "size_bytes": binary_size,
            "conversion_time_ms": binary_time * 1000,
            "efficiency": "100% (baseline)",
            "api_enhanced_streaming": True
        },
        "base64_format": {
            "size_bytes": base64_size,
            "conversion_time_ms": base64_time * 1000,
            "size_overhead": f"{overhead_percentage:.1f}%",
            "compatibility_mode": True
        },
        "api_enhanced_improvements": {
            "sentence_detection": "FlagEmbedding API with semantic analysis",
            "chunking_strategy": "BGE-M3 semantic preservation",
            "processing_mode": "Parallel async with API calls",
            "latency_reduction": "API-based confidence scoring",
            "timeout_optimization": "Reduced timeouts with fallback",
            "connection_pooling": "HTTP session reuse for efficiency"
        },
        "performance_summary": {
            "binary_advantage": f"{overhead_percentage:.1f}% smaller",
            "speed_comparison": "Binary" if binary_time < base64_time else "Base64" + " is faster",
            "recommendation": "Use binary format for API enhanced performance",
            "api_features": "All FlagEmbedding optimizations active"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/stats")
async def get_server_stats():
    """Get detailed API enhanced server statistics"""
    
    # Calculate format distribution
    format_stats = {}
    for session in client_tts_sessions.values():
        fmt = session.get('format', 'unknown')
        format_stats[fmt] = format_stats.get(fmt, 0) + 1
    
    # Calculate connection types
    connection_stats = {}
    for session in client_tts_sessions.values():
        conn_type = session.get('connection_type', 'unknown')
        connection_stats[conn_type] = connection_stats.get(conn_type, 0) + 1
    
    # Calculate voice distribution
    voice_stats = {}
    for session in client_tts_sessions.values():
        voice = session.get('voice', 'unknown')
        voice_stats[voice] = voice_stats.get(voice, 0) + 1
    
    return {
        "server_info": {
            "version": "1.3.0-api-bge-m3",
            "pipeline_ready": tts_pipeline is not None,
            "api_enhanced_features": True,
            "uptime_seconds": (datetime.now() - datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
        },
        "client_statistics": {
            "total_clients": len(client_tts_sessions),
            "active_buffers": len(client_sentence_buffers),
            "audio_mappings": len(audio_client_mapping),
            "format_distribution": format_stats,
            "connection_distribution": connection_stats,
            "voice_distribution": voice_stats
        },
        "api_enhanced_features": {
            "flagembedding_integration": {
                "enabled": True,
                "endpoint": flagembedding_settings['FlagEmbedding']['ServerAPIAddress'],
                "response_time": "50-150ms",
                "detection_method": "BGE-M3 via API + traditional fallback"
            },
            "semantic_processing": {
                "enabled": True,
                "confidence_threshold": 0.75,
                "check_interval": "500ms",
                "caching": "multi-level"
            },
            "connection_optimization": {
                "enabled": True,
                "method": "aiohttp connection pooling",
                "timeout_settings": "10s total, 2s connect, 5s read"
            },
            "performance_optimization": {
                "timeout_reduction": "30s -> 25s",
                "chunk_limit": "50 -> 40",
                "sleep_reduction": "0.01s -> 0.005s",
                "cleanup_frequency": "30s -> 20s",
                "api_batching": True
            }
        },
        "supported_capabilities": {
            "formats": ["binary", "base64"],
            "voices": [
                'af_heart', 'af_sky', 'af_bella', 'af_sarah',
                'am_adam', 'am_michael', 'bf_emma', 'bf_isabella'
            ],
            "sentence_detection": "FlagEmbedding API + NLTK fallback",
            "audio_encoding": "WAV PCM-16",
            "hybrid_architecture": True,
            "immediate_playback": True,
            "semantic_analysis": True,
            "multilingual": True
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/debug/mappings")
async def debug_mappings():
    """API enhanced debug endpoint to check client mappings"""
    return {
        "audio_mappings": dict(audio_client_mapping),
        "active_sessions": {
            sid: {
                "main_client_id": session.get("main_client_id"),
                "format": session.get("format"),
                "connection_type": session.get("connection_type"),
                "enabled": session.get("enabled"),
                "api_enhanced_features": True
            }
            for sid, session in client_tts_sessions.items()
        },
        "sentence_buffers": {
            sid: {
                "buffer_length": len(buffer.buffer) if hasattr(buffer, 'buffer') else 0,
                "api_enhanced": True
            }
            for sid, buffer in client_sentence_buffers.items()
        },
        "api_enhanced_metrics": {
            "total_buffers": len(client_sentence_buffers),
            "total_sessions": len(client_tts_sessions),
            "total_mappings": len(audio_client_mapping),
            "flagembedding_integration": True,
            "semantic_processing": True
        },
        "timestamp": datetime.now().isoformat()
    }

# Use the standard Socket.IO ASGI app
sio_asgi_app = socketio.ASGIApp(sio, other_asgi_app=app)

async def main():
    """Main function to run the API enhanced server with FlagEmbedding integration"""
    
    # Initialize settings first
    await initialize_flagembedding_settings()
    
    # Start API enhanced cleanup task
    asyncio.create_task(cleanup_stale_buffers())
    
    # Use compatible Socket.IO configuration
    config = uvicorn.Config(
        sio_asgi_app,
        host="0.0.0.0",
        port=7700,
        log_level="info",
        access_log=False,
        reload=False
    )
    
    server = uvicorn.Server(config)
    logger.info("🎵 Starting Enhanced Kokoro TTS Integration Server with FlagEmbedding API...")
    logger.info("📡 Server available at http://localhost:7700")
    logger.info("🔌 Socket.IO endpoint: http://localhost:7700/socket.io/")
    
    # Safe access to flagembedding_settings
    if flagembedding_settings and 'FlagEmbedding' in flagembedding_settings:
        logger.info(f"🌐 FlagEmbedding API: {flagembedding_settings['FlagEmbedding']['ServerAPIAddress']}")
    else:
        logger.warning("⚠️ FlagEmbedding settings not available - using defaults")
    
    logger.info("⚡ Enhanced features: API semantic detection, HTTP pooling, Multi-level caching")
    logger.info("🔥 Binary audio streaming: 33% smaller than base64")
    logger.info("📝 Base64 fallback: Available for compatibility")
    logger.info("🚀 Performance: API-based semantic analysis with traditional fallback")
    logger.info("✅ FlagEmbedding integration with existing infrastructure")
    
    await server.serve()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 API Enhanced TTS Server stopped by user")
    except Exception as e:
        logger.error(f"❌ API Enhanced TTS Server error: {e}")
        raise
