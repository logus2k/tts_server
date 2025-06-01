#!/usr/bin/env python3

"""
Kokoro TTS Server with NLTK Sentence Detection and Binary Audio Support
Hybrid Architecture: Direct audio streaming to browsers, text relay through assistant.js
Supports both binary and base64 audio formats for maximum compatibility and performance
"""

"""
pip install kokoro==0.9.4 nltk fastapi uvicorn python-socketio soundfile numpy torch
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

# NLTK for sentence detection
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


class SentenceBuffer:
    """Intelligent sentence buffer with length limits and smart chunking for TTS"""
    
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.buffer = ""
        self.complete_sentences = []
        self.last_chunk_time = datetime.now()
        self.max_sentence_length = 150  # Safe limit for Kokoro TTS
        
    def add_chunk(self, chunk: str) -> List[str]:
        """
        Add text chunk and return any complete sentences found
        Returns list of complete sentences ready for TTS
        """
        self.buffer += chunk
        self.last_chunk_time = datetime.now()
        
        # Use NLTK to detect sentence boundaries
        try:
            sentences = sent_tokenize(self.buffer)
            
            new_complete = []
            
            if len(sentences) == 1:
                # Single sentence - check if it looks complete
                sentence = sentences[0].strip()
                if self._looks_complete(sentence):
                    # ADDED: Check length and split if necessary
                    chunked_sentences = self._safe_chunk_sentence(sentence)
                    new_complete.extend(chunked_sentences)
                    self.buffer = ""
                else:
                    # ADDED: Check if buffer is getting too long
                    if len(sentence) > self.max_sentence_length * 1.5:
                        logger.warning(f"Client {self.client_id}: Buffer getting very long ({len(sentence)}), force-splitting")
                        # Force split at a safe point
                        chunked_sentences = self._safe_chunk_sentence(sentence, force=True)
                        new_complete.extend(chunked_sentences[:-1])  # Process all but last chunk
                        self.buffer = chunked_sentences[-1] if chunked_sentences else ""
            else:
                # Multiple sentences - process all but last
                for i in range(len(sentences) - 1):
                    sentence = sentences[i].strip()
                    if sentence and len(sentence) > 2:
                        # ADDED: Check length and split if necessary
                        chunked_sentences = self._safe_chunk_sentence(sentence)
                        new_complete.extend(chunked_sentences)
                
                # Keep the last (potentially incomplete) sentence in buffer
                self.buffer = sentences[-1].strip()
            
            if new_complete:
                logger.info(f"Client {self.client_id}: Found {len(new_complete)} complete sentences")
                for i, sent in enumerate(new_complete):
                    logger.info(f"  Sentence {i+1}: {sent[:60]}... (length: {len(sent)})")
                    
            return new_complete
            
        except Exception as e:
            logger.error(f"Error in sentence detection for {self.client_id}: {e}")
            return []
    
    def _safe_chunk_sentence(self, sentence: str, force: bool = False) -> List[str]:
        """
        Safely chunk a sentence if it's too long for TTS
        Returns list of chunks that are safe for TTS processing
        """
        if len(sentence) <= self.max_sentence_length and not force:
            return [sentence]
        
        logger.info(f"Client {self.client_id}: Chunking long sentence ({len(sentence)} chars)")
        
        chunks = []
        remaining = sentence
        
        while len(remaining) > self.max_sentence_length:
            # Find a good split point (prefer punctuation, then spaces)
            split_point = self._find_split_point(remaining, self.max_sentence_length)
            
            if split_point == -1:
                # No good split point found, force split at max length
                split_point = self.max_sentence_length
            
            chunk = remaining[:split_point].strip()
            if chunk:
                chunks.append(chunk)
            
            remaining = remaining[split_point:].strip()
        
        # Add the remaining part
        if remaining:
            chunks.append(remaining)
        
        logger.info(f"Client {self.client_id}: Split into {len(chunks)} chunks")
        return chunks
    
    def _find_split_point(self, text: str, max_length: int) -> int:
        """
        Find the best point to split a long sentence
        Returns the index to split at, or -1 if no good point found
        """
        # Look for split points in order of preference
        split_chars = ['.', ';', ',', ' and ', ' or ', ' but ', ' that ', ' which ', ' ']
        
        for split_char in split_chars:
            # Find the last occurrence of split_char within max_length
            search_text = text[:max_length]
            last_pos = search_text.rfind(split_char)
            
            if last_pos > max_length * 0.5:  # Don't split too early
                return last_pos + len(split_char)
        
        return -1  # No good split point found
    
    def _looks_complete(self, sentence: str) -> bool:
        """
        Check if a single sentence looks complete enough to process
        """
        if len(sentence) < 10:  # Too short to be a meaningful sentence
            return False
            
        # Check for sentence-ending punctuation
        if sentence.endswith(('.', '!', '?', '."', '!"', '?"')):
            return True
            
        # Check for colon (often complete thoughts)
        if sentence.endswith(':'):
            return True
            
        # ADDED: Force completion if sentence is getting very long
        if len(sentence) > self.max_sentence_length * 1.2:
            logger.info(f"Client {self.client_id}: Force-completing long sentence ({len(sentence)} chars)")
            return True
            
        # Check for typical sentence patterns that might be complete
        if len(sentence) > 40 and any(sentence.endswith(pattern) for pattern in [
            ' you', ' me', ' it', ' that', ' this', ' them', ' us',
            ' here', ' there', ' now', ' then', ' help', ' know'
        ]):
            return True
            
        # Check for common complete phrases
        complete_patterns = [
            "I'm here to help",
            "let me know",
            "feel free to ask",
            "here's what I can tell you",
            "would you like to know"
        ]
        
        sentence_lower = sentence.lower()
        for pattern in complete_patterns:
            if pattern in sentence_lower:
                return True
                
        return False
    
    def flush_remaining(self) -> List[str]:
        """Get any remaining text as final sentence when stream ends"""
        if self.buffer.strip() and len(self.buffer.strip()) > 2:
            final_sentence = self.buffer.strip()
            self.buffer = ""
            logger.info(f"Client {self.client_id}: Flushing final sentence: {final_sentence[:50]}... (length: {len(final_sentence)})")
            
            # ADDED: Chunk the final sentence if it's too long
            return self._safe_chunk_sentence(final_sentence)
        return []
    
    def is_stale(self, timeout_seconds: int = 30) -> bool:
        """Check if buffer is stale (no new chunks for timeout_seconds)"""
        return (datetime.now() - self.last_chunk_time).total_seconds() > timeout_seconds


async def initialize_pipeline():
    """Initialize Kokoro TTS pipeline"""
    global tts_pipeline
    try:
        logger.info("Initializing Kokoro TTS pipeline...")
        loop = asyncio.get_event_loop()
        tts_pipeline = await loop.run_in_executor(None, lambda: KPipeline(lang_code='a'))
        logger.info("TTS pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize TTS pipeline: {e}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    await initialize_pipeline()
    yield
    logger.info("Shutting down TTS server...")

# Create Socket.IO server with proper configuration
sio = socketio.AsyncServer(
    cors_allowed_origins="*",
    logger=False,
    engineio_logger=False,
    async_mode='asgi'
)

# FastAPI app
app = FastAPI(
    title="Kokoro TTS Integration Server - Binary Audio Support",
    description="Real-time TTS with NLTK sentence detection, binary and base64 audio formats",
    version="1.0.2-binary",
    lifespan=lifespan
)

# Client management for hybrid architecture
client_sentence_buffers: Dict[str, SentenceBuffer] = {}
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



async def generate_tts_for_sentence_binary_safe(sentence: str, voice: str, client_id: str, timeout_seconds: int = 30):
    """Generate TTS audio with timeout and error recovery - SAFE VERSION"""
    try:
        if not tts_pipeline:
            raise RuntimeError("TTS pipeline not initialized")
        
        # ADDED: Length validation
        if len(sentence) > 200:
            logger.warning(f"Sentence too long ({len(sentence)} chars), truncating: {sentence[:50]}...")
            sentence = sentence[:200] + "..."
        
        cleaned_sentence = sentence.strip()
        logger.info(f"Generating binary TTS for client {client_id}: {cleaned_sentence[:50]}... (length: {len(cleaned_sentence)})")
        
        # ADDED: Timeout wrapper for TTS generation
        try:
            # Run TTS generation with timeout
            loop = asyncio.get_event_loop()
            generator = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: tts_pipeline(cleaned_sentence, voice=voice, speed=1.25)),
                timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            logger.error(f"TTS generation timed out after {timeout_seconds}s for: {cleaned_sentence[:50]}...")
            # Yield error instead of hanging
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
        
        # Collect all audio chunks first
        audio_chunks = []
        
        # ADDED: Process with chunk limit to prevent memory issues
        max_chunks = 50  # Reasonable limit
        
        # Process each audio chunk from Kokoro
        for i, chunk_data in enumerate(generator):
            if i >= max_chunks:
                logger.warning(f"Reached max chunk limit ({max_chunks}) for sentence: {cleaned_sentence[:50]}...")
                break
                
            try:
                # Extract audio from Kokoro Result object
                audio = None
                
                if hasattr(chunk_data, 'output') and hasattr(chunk_data.output, 'audio'):
                    audio = chunk_data.output.audio
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
                
                # Convert to WAV bytes with timeout
                try:
                    wav_bytes = await asyncio.wait_for(
                        loop.run_in_executor(None, audio_to_wav_bytes, audio_np, 24000),
                        timeout=5.0  # 5 second timeout for conversion
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
                
                # Store the chunk data
                audio_chunks.append({
                    'chunk_id': i,
                    'wav_bytes': wav_bytes,
                    'duration': chunk_duration,
                    'sentence_duration': sentence_duration
                })
                
                logger.info(f"✅ Processed binary chunk {i}: {chunk_duration:.2f}s, {len(wav_bytes)} bytes")
                
            except Exception as e:
                logger.error(f"Error processing audio chunk {i}: {e}")
                continue
        
        # Yield audio chunks
        for i, chunk_info in enumerate(audio_chunks):
            metadata = {
                'type': 'tts_audio_chunk',
                'sentence_text': cleaned_sentence[:100] + ('...' if len(cleaned_sentence) > 100 else ''),
                'chunk_id': chunk_info['chunk_id'],
                'sample_rate': 24000,
                'duration': chunk_info['duration'],
                'sentence_duration': chunk_info['sentence_duration'],
                'client_id': client_id,
                'audio_size_bytes': len(chunk_info['wav_bytes']),
                'format': 'binary',
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"🎵 Yielding binary chunk {chunk_info['chunk_id']}: {len(chunk_info['wav_bytes'])} bytes")
            yield metadata, chunk_info['wav_bytes']
            chunk_count += 1
            
            await asyncio.sleep(0.01)
        
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
    """Generate TTS audio for a single sentence (base64 format) - FINAL FIX"""
    try:
        if not tts_pipeline:
            raise RuntimeError("TTS pipeline not initialized")
        
        logger.info(f"Generating base64 TTS for client {client_id}: {sentence[:50]}...")
        
        # Run TTS generation in thread pool
        loop = asyncio.get_event_loop()
        generator = await loop.run_in_executor(None, lambda: tts_pipeline(sentence, voice=voice, speed=1.25))
        
        chunk_count = 0
        sentence_duration = 0
        
        # Process each audio chunk for this sentence
        for i, chunk_data in enumerate(generator):
            try:
                # Extract audio from Kokoro Result object
                audio = None
                
                # Based on debug output: chunk_data.output.audio contains the torch.Tensor
                if hasattr(chunk_data, 'output') and hasattr(chunk_data.output, 'audio'):
                    audio = chunk_data.output.audio
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
                    await asyncio.sleep(0.01)
                    
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
    """Handle client connection with format detection - WITH DEBUGGING"""
    query_params = environ.get('QUERY_STRING', '')
    main_client_id = None
    connection_type = 'unknown'
    format_type = 'base64'  # Default fallback
    
    # Parse query parameters to determine connection type and format
    if query_params:
        parsed_params = urllib.parse.parse_qs(query_params)
        main_client_id = parsed_params.get('main_client_id', [None])[0]
        connection_type = parsed_params.get('type', ['browser'])[0]
        format_type = parsed_params.get('format', ['base64'])[0]  # Check for binary format request
    
    logger.info(f"🎵 TTS client connected: {sid} (type: {connection_type}, format: {format_type}, main_client: {main_client_id})")
    
    # Initialize sentence buffer and session with format preference
    client_sentence_buffers[sid] = SentenceBuffer(sid)
    client_tts_sessions[sid] = {
        'voice': 'af_heart',
        'enabled': True,
        'format': format_type,  # Store format preference
        'connected_time': datetime.now(),
        'main_client_id': main_client_id,
        'connection_type': connection_type
    }
    
    try:
        await sio.emit('tts_connected', {
            'status': 'Connected to TTS server',
            'client_id': sid,
            'format': format_type,
            'connection_type': connection_type,
            'timestamp': datetime.now().isoformat()
        }, room=sid)
        logger.info(f"🎵 Sent tts_connected confirmation to {sid}")
    except Exception as e:
        logger.error(f"🎵 Failed to send tts_connected: {e}")

@sio.event
async def disconnect(sid):
    """Handle client disconnection"""
    logger.info(f"TTS client disconnected: {sid}")
    
    # Find and clean up main client mapping
    main_client_to_remove = None
    for main_client_id, audio_sid in audio_client_mapping.items():
        if audio_sid == sid:
            main_client_to_remove = main_client_id
            break
    
    if main_client_to_remove:
        del audio_client_mapping[main_client_to_remove]
        logger.info(f"Cleaned up mapping for disconnected audio client: {main_client_to_remove}")
    
    # Standard cleanup
    if sid in client_sentence_buffers:
        del client_sentence_buffers[sid]
    if sid in client_tts_sessions:
        del client_tts_sessions[sid]



@sio.event
async def register_audio_client(sid, data):
    """Register audio client with main client ID for hybrid architecture - FIXED"""
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
            
        # FIXED: Send confirmation back to the audio client
        try:
            await sio.emit('audio_client_registered', {
                'status': 'registered',
                'main_client_id': main_client_id,
                'audio_client_id': sid,
                'format': client_tts_sessions.get(sid, {}).get('format', 'base64'),
                'timestamp': datetime.now().isoformat()
            }, room=sid)
            logger.info(f"🎵 Sent registration confirmation to {sid}")
        except Exception as e:
            logger.error(f"🎵 Failed to send registration confirmation: {e}")
            
        # FIXED: Log current mappings for debugging
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
    """Handle streaming text chunks with enhanced debugging"""
    try:
        text_chunk = data.get('chunk', '')
        is_final = data.get('final', False)
        target_client_id = data.get('target_client_id', data.get('client_id', sid))
        
        # ENHANCED LOGGING
        logger.info(f"🎵 === TTS TEXT CHUNK DEBUG ===")
        logger.info(f"🎵 Client: {target_client_id}")
        logger.info(f"🎵 Chunk: '{text_chunk}' (length: {len(text_chunk)})")
        logger.info(f"🎵 Final: {is_final}")
        
        if not text_chunk and not is_final:
            logger.warning(f"🎵 Empty text chunk and not final for {target_client_id}")
            return
        
        # Find the audio connection for this client
        audio_sid = audio_client_mapping.get(target_client_id, target_client_id)
        logger.info(f"🎵 Audio mapping: {target_client_id} -> {audio_sid}")
        
        # Get or create sentence buffer for the target client
        buffer = client_sentence_buffers.get(audio_sid)
        if not buffer:
            buffer = SentenceBuffer(target_client_id)
            client_sentence_buffers[audio_sid] = buffer
            logger.info(f"🎵 Created sentence buffer for hybrid client: {target_client_id} -> {audio_sid}")
        
        # ENHANCED BUFFER DEBUGGING
        logger.info(f"🎵 Buffer before processing: '{buffer.buffer}' (length: {len(buffer.buffer)})")
        
        # Get TTS session settings including format preference
        session = client_tts_sessions.get(audio_sid, {
            'voice': 'af_heart',
            'enabled': True,
            'format': 'base64'  # Default fallback
        })
        
        if not session.get('enabled', True):
            logger.info(f"🎵 TTS disabled for client {target_client_id}")
            return
        
        voice = session.get('voice', 'af_heart')
        format_type = session.get('format', 'base64')
        
        logger.info(f"🎵 Processing TTS for client {target_client_id} using format: {format_type}, voice: {voice}")
        
        # Process the chunk with enhanced debugging
        if text_chunk:
            complete_sentences = buffer.add_chunk(text_chunk)
            
            # ENHANCED SENTENCE DEBUGGING
            logger.info(f"🎵 Buffer after processing: '{buffer.buffer}' (length: {len(buffer.buffer)})")
            logger.info(f"🎵 Complete sentences found: {len(complete_sentences)}")
            
            for i, sentence in enumerate(complete_sentences):
                logger.info(f"🎵   Sentence {i+1}: '{sentence}' (length: {len(sentence)})")
            
            # Generate TTS for each complete sentence
            for sentence in complete_sentences:
                logger.info(f"🎵 🔥 GENERATING TTS for sentence: {sentence[:50]}...")
                
                # Send TTS events to the AUDIO connection
                try:
                    await sio.emit('tts_sentence_started', {
                        'sentence': sentence[:100] + ('...' if len(sentence) > 100 else ''),
                        'timestamp': datetime.now().isoformat()
                    }, room=audio_sid)
                    logger.info(f"🎵 Sent tts_sentence_started to {audio_sid}")
                except Exception as e:
                    logger.error(f"🎵 Failed to send tts_sentence_started: {e}")
                
                # Choose generation method based on format preference
                if format_type == 'binary':
                    # BINARY FORMAT: Use Method 2 (binary in data) - WORKING METHOD
                    logger.info(f"🎵 Starting binary TTS generation for sentence: {sentence[:30]}...")
                    chunk_count = 0
                    
                    try:
                        generator = generate_tts_for_sentence_binary_safe(sentence, voice, target_client_id)
                        
                        async for metadata, binary_data in generator:
                            if binary_data is not None:
                                # METHOD 2: Include binary data directly in the message payload
                                try:
                                    # Add the binary data to the metadata payload
                                    metadata_with_binary = metadata.copy()
                                    metadata_with_binary['audio_data'] = binary_data  # Include binary directly
                                    
                                    logger.info(f"🎵 Sending binary chunk {metadata.get('chunk_id')} via method 2: {len(binary_data)} bytes")
                                    await sio.emit('tts_audio_chunk', metadata_with_binary, room=audio_sid)
                                    chunk_count += 1
                                    logger.info(f"🎵 ✅ Successfully sent binary chunk {metadata.get('chunk_id')}")
                                    
                                except Exception as e:
                                    logger.error(f"🎵 ❌ Failed to send binary chunk: {e}")
                            else:
                                # Completion event
                                try:
                                    logger.info(f"🎵 Sending completion event to {audio_sid}")
                                    await sio.emit('tts_sentence_complete', metadata, room=audio_sid)
                                    logger.info(f"🎵 ✅ Successfully sent completion event")
                                except Exception as e:
                                    logger.error(f"🎵 Failed to send completion: {e}")
                        
                        logger.info(f"🎵 Binary TTS completed: {chunk_count} chunks sent")
                        
                    except Exception as gen_error:
                        logger.error(f"🎵 Error in binary generation: {gen_error}")
                    
                else:
                    # BASE64 FORMAT: Fallback for compatibility
                    logger.info(f"🎵 Generating base64 TTS for sentence...")
                    chunk_count = 0
                    async for chunk_data in generate_tts_for_sentence(sentence, voice, target_client_id):
                        try:
                            await sio.emit('tts_audio_chunk', chunk_data, room=audio_sid)
                            chunk_count += 1
                            logger.info(f"🎵 Sent base64 chunk {chunk_data.get('chunk_id')} to {audio_sid}")
                        except Exception as e:
                            logger.error(f"🎵 Failed to send base64 chunk: {e}")
                    
                    logger.info(f"🎵 Base64 TTS completed: {chunk_count} chunks sent")
        
        # Handle final chunk (end of LLM response)
        if is_final:
            final_sentences = buffer.flush_remaining()
            logger.info(f"🎵 Processing {len(final_sentences)} final sentences")
            
            for sentence in final_sentences:
                logger.info(f"🎵 Processing final sentence for {format_type} TTS: {sentence[:50]}...")
                
                try:
                    await sio.emit('tts_sentence_started', {
                        'sentence': sentence[:100] + ('...' if len(sentence) > 100 else ''),
                        'final': True,
                        'timestamp': datetime.now().isoformat()
                    }, room=audio_sid)
                except Exception as e:
                    logger.error(f"🎵 Failed to send final tts_sentence_started: {e}")
                
                # Choose generation method based on format preference
                if format_type == 'binary':
                    async for metadata, binary_data in generate_tts_for_sentence_binary_safe(sentence, voice, target_client_id):
                        if binary_data is not None:
                            try:
                                await sio.emit('tts_audio_chunk', metadata, [binary_data], room=audio_sid)
                            except Exception as e:
                                logger.error(f"🎵 Failed to send final binary chunk: {e}")
                        else:
                            try:
                                await sio.emit('tts_sentence_complete', metadata, room=audio_sid)
                            except Exception as e:
                                logger.error(f"🎵 Failed to send final completion: {e}")
                else:
                    async for chunk_data in generate_tts_for_sentence(sentence, voice, target_client_id):
                        try:
                            await sio.emit('tts_audio_chunk', chunk_data, room=audio_sid)
                        except Exception as e:
                            logger.error(f"🎵 Failed to send final base64 chunk: {e}")
            
            # Send completion to AUDIO connection
            try:
                await sio.emit('tts_response_complete', {
                    'timestamp': datetime.now().isoformat()
                }, room=audio_sid)
                logger.info(f"🎵 Sent tts_response_complete to {audio_sid}")
            except Exception as e:
                logger.error(f"🎵 Failed to send tts_response_complete: {e}")
        
    except Exception as e:
        logger.error(f"🎵 Error processing text chunk for {target_client_id}: {e}")
        audio_sid = audio_client_mapping.get(data.get('target_client_id', data.get('client_id', sid)), sid)
        try:
            await sio.emit('tts_error', {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }, room=audio_sid)
        except Exception as emit_error:
            logger.error(f"🎵 Failed to send error event: {emit_error}")


async def test_binary_emission(sid, data):
    """Test handler using working binary method"""
    logger.info(f"🧪 Test binary emission called by {sid}")
    
    try:
        # Create a small test WAV file
        import numpy as np
        sample_rate = 24000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        # Convert to WAV bytes
        test_wav = audio_to_wav_bytes(audio_data, sample_rate)
        
        if test_wav:
            test_metadata = {
                'type': 'test_audio_chunk',
                'chunk_id': 0,
                'sample_rate': sample_rate,
                'duration': duration,
                'audio_size_bytes': len(test_wav),
                'format': 'binary',
                'timestamp': datetime.now().isoformat(),
                'audio_data': test_wav  # METHOD 2: Include binary in data
            }
            
            logger.info(f"🧪 Emitting test binary chunk via method 2: {len(test_wav)} bytes")
            await sio.emit('test_audio_chunk', test_metadata, room=sid)
            logger.info(f"🧪 ✅ Test emission successful")
            
        else:
            logger.error(f"🧪 ❌ Failed to create test WAV")
            
    except Exception as e:
        logger.error(f"🧪 ❌ Test emission failed: {e}")
        import traceback
        logger.error(f"🧪 Traceback:\n{traceback.format_exc()}")


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
        
        logger.info(f"Configured TTS for client {client_id} (audio: {audio_sid}): {session}")
        
        # Send confirmation to the relay (assistant.js)
        await sio.emit('tts_client_configured', {
            'client_id': client_id,
            'voice': session.get('voice'),
            'enabled': session.get('enabled'),
            'format': session.get('format'),
            'timestamp': datetime.now().isoformat()
        }, room=sid)
        
    except Exception as e:
        logger.error(f"Error configuring TTS for client: {e}")

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
        'timestamp': datetime.now().isoformat()
    }, room=sid)

@sio.event
async def tts_client_disconnect(sid, data):
    """Handle client disconnect notification from assistant.js"""
    client_id = data.get('client_id')
    audio_sid = audio_client_mapping.get(client_id)
    
    if audio_sid:
        logger.info(f"Cleaning up TTS resources for disconnected client: {client_id} -> {audio_sid}")
        
        # Clean up mapping
        del audio_client_mapping[client_id]
        
        # Clean up buffers and sessions if audio client is gone
        if audio_sid in client_sentence_buffers:
            del client_sentence_buffers[audio_sid]
        if audio_sid in client_tts_sessions:
            del client_tts_sessions[audio_sid]

# Periodic cleanup of stale sentence buffers
async def cleanup_stale_buffers():
    """Clean up stale sentence buffers"""
    while True:
        try:
            stale_clients = []
            for client_id, buffer in client_sentence_buffers.items():
                if buffer.is_stale(timeout_seconds=60):
                    stale_clients.append(client_id)
            
            for client_id in stale_clients:
                logger.info(f"Cleaning up stale buffer for client: {client_id}")
                del client_sentence_buffers[client_id]
                if client_id in client_tts_sessions:
                    del client_tts_sessions[client_id]
            
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")
        
        await asyncio.sleep(30)  # Run every 30 seconds

# REST API Endpoints
@app.get("/", response_class=HTMLResponse)
async def get_root():
    binary_clients = sum(1 for s in client_tts_sessions.values() if s.get('format') == 'binary')
    base64_clients = sum(1 for s in client_tts_sessions.values() if s.get('format') == 'base64')
    
    return HTMLResponse(content=f"""
    <!DOCTYPE html>
    <html>
    <head><title>Kokoro TTS Integration Server - Binary Audio Support</title></head>
    <body>
        <h1>🎵 Kokoro TTS Integration Server</h1>
        <p><strong>Status:</strong> Running (Binary + Base64 Support)</p>
        <p><strong>Pipeline:</strong> {'✅ Ready' if tts_pipeline else '❌ Not Ready'}</p>
        <p><strong>Active Clients:</strong> {len(client_sentence_buffers)}</p>
        <p><strong>Audio Mappings:</strong> {len(audio_client_mapping)}</p>
        
        <h3>Client Format Distribution:</h3>
        <ul>
            <li>🔥 Binary Format: {binary_clients} clients</li>
            <li>📝 Base64 Format: {base64_clients} clients</li>
        </ul>
        
        <h3>Features:</h3>
        <ul>
            <li>✅ Binary audio streaming (33% smaller)</li>
            <li>✅ Base64 fallback compatibility</li>
            <li>✅ Hybrid architecture support</li>
            <li>✅ NLTK sentence detection</li>
            <li>✅ Real-time audio generation</li>
        </ul>
        
        <h3>API Endpoints:</h3>
        <ul>
            <li><a href="/api/health">/api/health</a> - Server status</li>
            <li><a href="/api/clients">/api/clients</a> - Client information</li>
            <li><a href="/api/performance-test">/api/performance-test</a> - Performance comparison</li>
        </ul>
    </body>
    </html>
    """)



@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.2-binary",
        "architecture": "hybrid",
        "pipeline_ready": tts_pipeline is not None,
        "active_clients": len(client_sentence_buffers),
        "active_sessions": len(client_tts_sessions),
        "audio_mappings": len(audio_client_mapping),
        "supported_formats": ["binary", "base64"],
        "format_distribution": {
            "binary": sum(1 for s in client_tts_sessions.values() if s.get('format') == 'binary'),
            "base64": sum(1 for s in client_tts_sessions.values() if s.get('format') == 'base64')
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/clients")
async def get_client_info():
    """Debug endpoint to see client mappings with format info"""
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
                "connected_time": session.get("connected_time").isoformat() if session.get("connected_time") else None
            }
            for sid, session in client_tts_sessions.items()
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/performance-test")
async def performance_test():
    """Compare binary vs base64 performance"""
    
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
            "efficiency": "100% (baseline)"
        },
        "base64_format": {
            "size_bytes": base64_size,
            "conversion_time_ms": base64_time * 1000,
            "size_overhead": f"{overhead_percentage:.1f}%"
        },
        "performance_summary": {
            "binary_advantage": f"{overhead_percentage:.1f}% smaller",
            "speed_comparison": "Binary" if binary_time < base64_time else "Base64" + " is faster",
            "recommendation": "Use binary format for better performance"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/stats")
async def get_server_stats():
    """Get detailed server statistics"""
    
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
            "version": "1.0.2-binary",
            "pipeline_ready": tts_pipeline is not None,
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
        "supported_features": {
            "formats": ["binary", "base64"],
            "voices": [
                'af_heart', 'af_sky', 'af_bella', 'af_sarah',
                'am_adam', 'am_michael', 'bf_emma', 'bf_isabella'
            ],
            "sentence_detection": "NLTK",
            "audio_encoding": "WAV PCM-16",
            "hybrid_architecture": True
        },
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/debug/mappings")
async def debug_mappings():
    """Debug endpoint to check client mappings"""
    return {
        "audio_mappings": dict(audio_client_mapping),
        "active_sessions": {
            sid: {
                "main_client_id": session.get("main_client_id"),
                "format": session.get("format"),
                "connection_type": session.get("connection_type"),
                "enabled": session.get("enabled")
            }
            for sid, session in client_tts_sessions.items()
        },
        "sentence_buffers": list(client_sentence_buffers.keys()),
        "timestamp": datetime.now().isoformat()
    }


# Use the standard Socket.IO ASGI app
sio_asgi_app = socketio.ASGIApp(sio, other_asgi_app=app)

async def main():
    """Main function to run the server"""
    # Start cleanup task
    asyncio.create_task(cleanup_stale_buffers())
    
    # Use compatible Socket.IO configuration
    config = uvicorn.Config(
        sio_asgi_app,  # Use the Socket.IO ASGI app directly
        host="0.0.0.0",
        port=7700,
        log_level="info",
        access_log=False,  # Reduce noise
        reload=False
    )
    
    server = uvicorn.Server(config)
    logger.info("🎵 Starting Kokoro TTS Integration Server (Binary Audio Support)...")
    logger.info("📡 Server available at http://localhost:7700")
    logger.info("🔌 Socket.IO endpoint: http://localhost:7700/socket.io/")
    logger.info("🔥 Binary audio streaming enabled (33% smaller than base64)")
    logger.info("📝 Base64 fallback available for compatibility")
    logger.info("✅ Hybrid architecture with format auto-detection")
    
    await server.serve()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 TTS Server stopped by user")
    except Exception as e:
        logger.error(f"❌ TTS Server error: {e}")
        raise
