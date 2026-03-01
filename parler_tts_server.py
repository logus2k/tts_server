#!/usr/bin/env python3

"""
Parler TTS Server
Functionally equivalent to Kokoro TTS Server with same Socket.IO and REST APIs
"""

"""
pip install parler-tts transformers torch soundfile numpy fastapi uvicorn python-socketio scipy spacy sacremoses
pip install git+https://github.com/huggingface/parler-tts.git
python -m spacy download en_core_web_sm
python -m spacy download pt_core_news_sm
"""

import asyncio
import base64
import io
import logging
from typing import Dict, List, Optional, AsyncGenerator
from datetime import datetime
import time
import numpy as np
import soundfile as sf
from fastapi import FastAPI
import socketio
import uvicorn
from contextlib import asynccontextmanager
import urllib.parse
import json
from pathlib import Path
import re
import spacy
from sacremoses import MosesTokenizer, MosesDetokenizer
import torch
from scipy import signal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
tts_model = None
tts_tokenizer = None
tts_description_tokenizer = None
tts_settings = None
model_device = None
model_dtype = None
model_sample_rate = 44100  # Parler outputs at 44.1kHz

# Parler TTS Voices (34 speakers from mini model, ranked by similarity score)
VOICE_LIST = [
    # Top performers for mini model
    'Jon', 'Lea', 'Gary', 'Jenna', 'Mike', 'Laura', 'Lauren', 'Eileen',
    'Alisa', 'Karen', 'Barbara', 'Carol', 'Emily', 'Rose', 'Will', 'Patrick',
    'Eric', 'Rick', 'Anna', 'Tina', 'Jordan', 'Brenda', 'David', 'Yann',
    'Joy', 'James', 'Jason', 'Aaron', 'Naomie', 'Jerry', 'Bill', 'Tom',
    'Rebecca', 'Bruce'
]

# Voice-to-Language mapping (Parler supports 8 European languages)
# All voices can speak all languages, but we map default language preferences
VOICE_LANGUAGE_MAP = {voice: 'en' for voice in VOICE_LIST}

# Language code definitions (Parler multilingual v1.1 supports these)
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'fr': 'French',
    'es': 'Spanish',
    'pt': 'Portuguese',
    'pl': 'Polish',
    'de': 'German',
    'it': 'Italian',
    'nl': 'Dutch',
}

# Voice characteristics for description generation
VOICE_CHARACTERISTICS = {
    'Jon': {'gender': 'male', 'style': 'clear and calm'},
    'Lea': {'gender': 'female', 'style': 'warm and friendly'},
    'Gary': {'gender': 'male', 'style': 'professional and steady'},
    'Jenna': {'gender': 'female', 'style': 'bright and engaging'},
    'Mike': {'gender': 'male', 'style': 'natural and conversational'},
    'Laura': {'gender': 'female', 'style': 'clear and articulate'},
    'Lauren': {'gender': 'female', 'style': 'smooth and pleasant'},
    'Eileen': {'gender': 'female', 'style': 'gentle and soothing'},
    'Alisa': {'gender': 'female', 'style': 'expressive and dynamic'},
    'Karen': {'gender': 'female', 'style': 'confident and clear'},
    'Barbara': {'gender': 'female', 'style': 'warm and mature'},
    'Carol': {'gender': 'female', 'style': 'friendly and approachable'},
    'Emily': {'gender': 'female', 'style': 'youthful and energetic'},
    'Rose': {'gender': 'female', 'style': 'elegant and refined'},
    'Will': {'gender': 'male', 'style': 'strong and assured'},
    'Patrick': {'gender': 'male', 'style': 'friendly and relaxed'},
    'Eric': {'gender': 'male', 'style': 'clear and measured'},
    'Rick': {'gender': 'male', 'style': 'casual and natural'},
    'Anna': {'gender': 'female', 'style': 'soft and gentle'},
    'Tina': {'gender': 'female', 'style': 'cheerful and bright'},
    'Jordan': {'gender': 'male', 'style': 'modern and casual'},
    'Brenda': {'gender': 'female', 'style': 'professional and warm'},
    'David': {'gender': 'male', 'style': 'authoritative and clear'},
    'Yann': {'gender': 'male', 'style': 'calm and thoughtful'},
    'Joy': {'gender': 'female', 'style': 'happy and uplifting'},
    'James': {'gender': 'male', 'style': 'deep and resonant'},
    'Jason': {'gender': 'male', 'style': 'energetic and engaging'},
    'Aaron': {'gender': 'male', 'style': 'smooth and confident'},
    'Naomie': {'gender': 'female', 'style': 'graceful and clear'},
    'Jerry': {'gender': 'male', 'style': 'friendly and warm'},
    'Bill': {'gender': 'male', 'style': 'straightforward and clear'},
    'Tom': {'gender': 'male', 'style': 'natural and easy'},
    'Rebecca': {'gender': 'female', 'style': 'pleasant and articulate'},
    'Bruce': {'gender': 'male', 'style': 'steady and reliable'},
}

# Default voice characteristics for unknown voices
DEFAULT_CHARACTERISTICS = {'gender': 'male', 'style': 'clear and natural'}


# ---------- Sacremoses (per-sentence) ----------
_DETOK_CACHE = {}
_TOK_CACHE = {}
_SUPPORTED_MOSES_LANGS = {"en", "pt", "fr", "es", "de", "it", "nl", "pl"}

def _norm_lang_code(lang_code: str) -> str:
    base = (lang_code or "en").split("-")[0].lower()
    return base if base in _SUPPORTED_MOSES_LANGS else "en"

def _get_detok(lang_code: str) -> MosesDetokenizer:
    lang = _norm_lang_code(lang_code)
    if lang not in _DETOK_CACHE:
        _DETOK_CACHE[lang] = MosesDetokenizer(lang=lang)
    return _DETOK_CACHE[lang]

def _get_tok(lang_code: str) -> MosesTokenizer:
    lang = _norm_lang_code(lang_code)
    if lang not in _TOK_CACHE:
        _TOK_CACHE[lang] = MosesTokenizer(lang=lang)
    return _TOK_CACHE[lang]

def _detok_sentence(text: str, lang_code: str) -> str:
    if not text:
        return text
    tok = _get_tok(lang_code)
    detok = _get_detok(lang_code)
    return detok.detokenize(tok.tokenize(text, escape=False))


# ---------- spaCy sentencizer (fast, rule-based) ----------
_SPACY_CACHE = {}

def _get_sentencizer(lang_code: str):
    lang = (lang_code or "en").split("-")[0].lower()
    if lang not in ("en", "pt", "fr", "es", "de", "it"):
        lang = "en"
    if lang in _SPACY_CACHE:
        return _SPACY_CACHE[lang]

    try:
        model_map = {
            "en": "en_core_web_sm",
            "pt": "pt_core_news_sm",
            "fr": "fr_core_news_sm",
            "es": "es_core_news_sm",
            "de": "de_core_news_sm",
            "it": "it_core_news_sm",
        }
        model_name = model_map.get(lang, "en_core_web_sm")
        nlp = spacy.load(model_name, exclude=["tagger", "parser", "ner", "lemmatizer", "attribute_ruler"])
    except Exception:
        nlp = spacy.blank(lang)

    if "senter" in nlp.pipe_names:
        pass
    elif "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer", config={"punct_chars": [".", "!", "?", "…", "。", "？", "！"]})
    _SPACY_CACHE[lang] = nlp
    return nlp

def _split_sentences_spacy(text: str, lang_code: str, treat_nl_as_boundary: bool) -> list[str]:
    if not text:
        return []
    nlp = _get_sentencizer(lang_code)
    if treat_nl_as_boundary and "\n" in text:
        out = []
        for seg in re.split(r"\n+", text):
            seg = seg.strip()
            if not seg:
                continue
            doc = nlp(seg)
            out.extend(s.text.strip() for s in doc.sents if s.text.strip())
        return out
    doc = nlp(text)
    return [s.text.strip() for s in doc.sents if s.text.strip()]

# ---------- streaming join (no auto-spaces) ----------
def _append_stream(prev: str, nxt: str) -> str:
    """Append chunk without inserting spaces; collapse double whitespace at seam."""
    if not prev:
        return (nxt or "").replace("'", "'")
    if not nxt:
        return prev
    nxt = nxt.replace("'", "'")
    if prev[-1].isspace() and nxt[:1].isspace():
        return prev + nxt.lstrip()
    return prev + nxt


def build_voice_description(voice: str, speed: float = 1.0) -> str:
    """Build a natural language description for Parler TTS based on voice and speed."""
    chars = VOICE_CHARACTERISTICS.get(voice, DEFAULT_CHARACTERISTICS)
    gender = chars['gender']
    style = chars['style']
    
    # Map speed to description
    if speed < 0.8:
        speed_desc = "very slow"
    elif speed < 0.95:
        speed_desc = "slow"
    elif speed <= 1.05:
        speed_desc = "moderate"
    elif speed <= 1.2:
        speed_desc = "slightly fast"
    else:
        speed_desc = "fast"
    
    description = f"{voice}'s voice is {style} with a {speed_desc} delivery. The recording is high quality with no background noise."
    return description


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate."""
    if orig_sr == target_sr:
        return audio
    
    # Calculate number of samples in resampled audio
    num_samples = int(len(audio) * target_sr / orig_sr)
    resampled = signal.resample(audio, num_samples)
    return resampled.astype(np.float32)


def adjust_speed(audio: np.ndarray, speed: float, sample_rate: int) -> np.ndarray:
    """Adjust audio speed by resampling (speed > 1 = faster, speed < 1 = slower)."""
    if abs(speed - 1.0) < 0.01:
        return audio
    
    # To speed up: resample to lower rate, then back to original
    # To slow down: resample to higher rate, then back to original
    intermediate_sr = int(sample_rate / speed)
    
    # Resample to intermediate rate (changes duration)
    num_samples = int(len(audio) * intermediate_sr / sample_rate)
    resampled = signal.resample(audio, num_samples)
    
    return resampled.astype(np.float32)


async def initialize_tts_settings(settings: dict):
    global tts_settings
    tts_settings = settings.get("TTS", {
        "default_voice": "Jon",
        "default_speed": 1.0,
        "enabled_languages": ["en", "fr", "es", "pt", "pl", "de", "it", "nl"],
        "pipeline_timeout": 300,
        "max_sentence_length": 1500,
        "chunk_limit": 500,
        "chunk_timeout": 10.0,
        "generation_timeout": 60.0,
        "speed_min": 0.5,
        "speed_max": 1.5,
        "output_sample_rate": 24000,  # Resample to 24kHz for compatibility
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
        return 'Jon'
    return tts_settings.get('default_voice', 'Jon')

def get_default_speed():
    """Get default speed from settings"""
    if not tts_settings:
        return 1.0
    return tts_settings.get('default_speed', 1.0)

def get_output_sample_rate():
    """Get output sample rate (for client compatibility)"""
    if not tts_settings:
        return 24000
    return tts_settings.get('output_sample_rate', 24000)

def validate_speed(speed: float) -> float:
    """Validate and clamp speed within allowed range"""
    if not tts_settings:
        return max(0.5, min(1.5, speed))
    
    speed_min = tts_settings.get('speed_min', 0.5)
    speed_max = tts_settings.get('speed_max', 1.5)
    return max(speed_min, min(speed_max, speed))


async def initialize_tts_model(model_path: str = "models/parler-tts", t5_path: str = "models/flan-t5-large"):
    """Initialize Parler TTS model"""
    global tts_model, tts_tokenizer, tts_description_tokenizer, model_device, model_dtype, model_sample_rate
    
    try:
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
        
        model_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        logger.info(f"Loading Parler TTS model from {model_path}...")
        logger.info(f"Device: {model_device}, dtype: {model_dtype}")
        
        # Load model
        tts_model = ParlerTTSForConditionalGeneration.from_pretrained(
            model_path,
            attn_implementation="eager"
        ).to(model_device, dtype=model_dtype)
        
        # Load tokenizers
        tts_tokenizer = AutoTokenizer.from_pretrained(model_path)
        tts_description_tokenizer = AutoTokenizer.from_pretrained(t5_path)
        
        # Get sample rate from model config
        model_sample_rate = tts_model.config.sampling_rate
        logger.info(f"Model sample rate: {model_sample_rate} Hz")
        
        logger.info("✅ Parler TTS model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load Parler TTS model: {e}")
        return False


def get_pipeline_for_voice(voice: str):
    """Get the TTS model (Parler uses single model for all voices)"""
    if voice not in VOICE_LIST:
        logger.warning(f"⚠️ Unknown voice: {voice}, falling back to default")
        voice = get_default_voice()
    
    return tts_model

def get_language_for_voice(voice: str) -> str:
    """Get language code for a given voice (default to English)"""
    return VOICE_LANGUAGE_MAP.get(voice, 'en')

def is_voice_enabled(voice: str) -> bool:
    """Check if voice is available"""
    return voice in VOICE_LIST and tts_model is not None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    try:
        # Get model paths from settings or use defaults
        settings = load_config()
        model_path = settings.get("model_path", "models/parler-tts")
        t5_path = settings.get("t5_path", "models/flan-t5-large")
        
        success = await initialize_tts_model(model_path, t5_path)
        if success:
            logger.info("✅ Parler TTS ready")
        else:
            logger.warning("⚠️ TTS functionality may be limited")
    except Exception as e:
        logger.error(f"❌ TTS initialization failed: {e}")
        logger.warning("⚠️ TTS functionality may be limited")
    
    yield
    
    logger.info("Shutting down...")


# Create Socket.IO server
sio = socketio.AsyncServer(
    cors_allowed_origins=[
        "https://logus2k.com",
        "https://www.logus2k.com",
        "http://localhost:7676",
        "http://127.0.0.1:7676",
        "http://localhost:7800",
        "http://127.0.0.1:7800",
        "http://localhost:7701",
        "http://127.0.0.1:7701",
        "http://localhost:6678",
        "http://127.0.0.1:6678",
        "http://localhost:6677",
        "http://127.0.0.1:6677",
    ],
    logger=False,
    engineio_logger=False,
    async_mode="asgi",
)

# FastAPI app
app = FastAPI(
    title="Parler TTS Server",
    version="2.1.0-parler",
    lifespan=lifespan
)

# Client management
client_tts_sessions: Dict[str, dict] = {}
audio_client_mapping: Dict[str, List[str]] = {}


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


def generate_tts_sync(text: str, voice: str, speed: float) -> np.ndarray:
    """Synchronous TTS generation (to be run in executor)"""
    global tts_model, tts_tokenizer, tts_description_tokenizer, model_device, model_sample_rate
    
    if tts_model is None:
        raise RuntimeError("TTS model not loaded")
    
    # Build voice description
    description = build_voice_description(voice, speed)
    
    # Tokenize
    input_ids = tts_description_tokenizer(description, return_tensors="pt").input_ids.to(model_device)
    prompt_input_ids = tts_tokenizer(text, return_tensors="pt").input_ids.to(model_device)
    
    # Generate
    with torch.no_grad():
        generation = tts_model.generate(
            input_ids=input_ids,
            prompt_input_ids=prompt_input_ids,
        )
    
    # Convert to numpy
    audio_arr = generation.cpu().float().numpy().squeeze()
    
    # Apply speed adjustment
    if abs(speed - 1.0) >= 0.01:
        audio_arr = adjust_speed(audio_arr, speed, model_sample_rate)
    
    return audio_arr


async def generate_tts_binary(sentence: str, voice: str, speed: float, client_id: str):
    """Generate TTS audio in binary format"""
    try:
        if tts_model is None:
            error_data = {
                'type': 'tts_error',
                'error': 'TTS model not available',
                'sentence_text': sentence[:100],
                'client_id': client_id,
                'voice': voice,
                'language': get_language_for_voice(voice),
                'speed': speed,
                'format': 'binary',
                'timestamp': datetime.now().isoformat()
            }
            yield error_data, None
            return
        
        # Apply sentence length limit
        max_length = tts_settings.get('max_sentence_length', 500) if tts_settings else 500
        if len(sentence) > max_length:
            sentence = sentence[:max_length] + "..."
        
        validated_speed = validate_speed(speed)
        if validated_speed != speed:
            logger.warning(f"Speed {speed} clamped to {validated_speed}")
        
        cleaned_sentence = sentence.strip()
        lang_code = get_language_for_voice(voice)
        logger.info(f"🎵 Generating TTS: {cleaned_sentence[:50]}... (voice: {voice}, speed: {validated_speed})")
        
        loop = asyncio.get_event_loop()
        generation_timeout = tts_settings.get('generation_timeout', 60.0) if tts_settings else 60.0
        
        # Generate audio
        audio_arr = await asyncio.wait_for(
            loop.run_in_executor(None, generate_tts_sync, cleaned_sentence, voice, validated_speed),
            timeout=generation_timeout
        )
        
        # Resample to output rate
        output_sr = get_output_sample_rate()
        if model_sample_rate != output_sr:
            audio_arr = resample_audio(audio_arr, model_sample_rate, output_sr)
        
        # Convert to WAV bytes
        wav_bytes = audio_to_wav_bytes(audio_arr, output_sr)
        
        if not wav_bytes:
            error_data = {
                'type': 'tts_error',
                'error': 'Failed to convert audio',
                'sentence_text': cleaned_sentence[:100],
                'client_id': client_id,
                'voice': voice,
                'language': lang_code,
                'speed': validated_speed,
                'format': 'binary',
                'timestamp': datetime.now().isoformat()
            }
            yield error_data, None
            return
        
        duration = len(audio_arr) / output_sr
        
        # Emit single chunk (Parler generates full sentence at once)
        metadata = {
            'type': 'tts_audio_chunk',
            'sentence_text': cleaned_sentence[:100],
            'chunk_id': 0,
            'sample_rate': output_sr,
            'duration': duration,
            'sentence_duration': duration,
            'client_id': client_id,
            'voice': voice,
            'language': lang_code,
            'speed': validated_speed,
            'format': 'binary',
            'timestamp': datetime.now().isoformat()
        }
        
        yield metadata, wav_bytes
        
        # Completion
        completion = {
            'type': 'tts_sentence_complete',
            'sentence_text': cleaned_sentence,
            'total_chunks': 1,
            'total_duration': duration,
            'client_id': client_id,
            'voice': voice,
            'language': lang_code,
            'speed': validated_speed,
            'format': 'binary',
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"🎵 TTS completed: 1 chunk, {duration:.2f}s (voice: {voice}, speed: {validated_speed})")
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
    """Generate TTS audio in base64 format"""
    try:
        if tts_model is None:
            yield {
                'type': 'tts_error',
                'error': 'TTS model not available',
                'sentence_text': sentence,
                'client_id': client_id,
                'voice': voice,
                'language': get_language_for_voice(voice),
                'speed': speed,
                'format': 'base64',
                'timestamp': datetime.now().isoformat()
            }
            return
        
        # Apply sentence length limit
        max_length = tts_settings.get('max_sentence_length', 500) if tts_settings else 500
        if len(sentence) > max_length:
            sentence = sentence[:max_length] + "..."
        
        validated_speed = validate_speed(speed)
        if validated_speed != speed:
            logger.warning(f"Speed {speed} clamped to {validated_speed}")
        
        lang_code = get_language_for_voice(voice)
        logger.info(f"Generating base64 TTS: {sentence[:50]}... (voice: {voice}, speed: {validated_speed})")
        
        loop = asyncio.get_event_loop()
        generation_timeout = tts_settings.get('generation_timeout', 60.0) if tts_settings else 60.0
        
        # Generate audio
        audio_arr = await asyncio.wait_for(
            loop.run_in_executor(None, generate_tts_sync, sentence, voice, validated_speed),
            timeout=generation_timeout
        )
        
        # Resample to output rate
        output_sr = get_output_sample_rate()
        if model_sample_rate != output_sr:
            audio_arr = resample_audio(audio_arr, model_sample_rate, output_sr)
        
        # Convert to base64
        audio_b64 = audio_to_base64(audio_arr, output_sr)
        
        if not audio_b64:
            yield {
                'type': 'tts_error',
                'error': 'Failed to convert audio to base64',
                'sentence_text': sentence,
                'client_id': client_id,
                'voice': voice,
                'language': lang_code,
                'speed': validated_speed,
                'format': 'base64',
                'timestamp': datetime.now().isoformat()
            }
            return
        
        duration = len(audio_arr) / output_sr
        
        # Emit single chunk
        yield {
            'type': 'tts_audio_chunk',
            'sentence_text': sentence[:100],
            'chunk_id': 0,
            'audio_data': audio_b64,
            'sample_rate': output_sr,
            'duration': duration,
            'sentence_duration': duration,
            'client_id': client_id,
            'voice': voice,
            'language': lang_code,
            'speed': validated_speed,
            'format': 'base64',
            'timestamp': datetime.now().isoformat()
        }
        
        # Completion
        yield {
            'type': 'tts_sentence_complete',
            'sentence_text': sentence,
            'total_chunks': 1,
            'total_duration': duration,
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
    logger.info(f"TTS CONNECT: {sid}")

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
    
    default_voice = get_default_voice()
    default_speed = get_default_speed()
    
    client_tts_sessions[sid] = {
        'voice': default_voice,
        'speed': default_speed,
        'enabled': True,
        'mode': 'tts',
        'format': format_type,
        'connected_time': datetime.now(),
        'main_client_id': main_client_id,
        'connection_type': connection_type
    }

    logger.info(f"SESSION CREATED: {sid}")    
    
    active_languages = get_enabled_languages()
    await sio.emit('tts_connected', {
        'status': 'Connected to Parler TTS server',
        'client_id': sid,
        'format': format_type,
        'version': '2.1.0-parler',
        'default_voice': default_voice,
        'default_speed': default_speed,
        'current_mode': 'tts',
        'supported_languages': active_languages,
        'language_names': {lang: SUPPORTED_LANGUAGES.get(lang, 'Unknown') for lang in active_languages},
        'speed_range': {
            'min': tts_settings.get('speed_min', 0.5) if tts_settings else 0.5,
            'max': tts_settings.get('speed_max', 1.5) if tts_settings else 1.5
        },
        'timestamp': datetime.now().isoformat()
    }, room=sid)


@sio.event
async def set_client_mode(sid, data):
    """Handle client mode changes (tts/avatar)"""
    client_id = data.get('client_id')
    mode = data.get('mode')
    
    if mode not in ['tts', 'avatar']:
        logger.warning(f"Invalid mode '{mode}' for client {client_id}")
        return
    
    audio_sids = audio_client_mapping.get(client_id, [client_id])
    if isinstance(audio_sids, str):
        audio_sids = [audio_sids]
    
    for audio_sid in audio_sids:
        if audio_sid in client_tts_sessions:
            old_mode = client_tts_sessions[audio_sid].get('mode', 'tts')
            client_tts_sessions[audio_sid]['mode'] = mode
            logger.info(f"🎛️ Client {client_id} mode: {old_mode} → {mode}")
    
    await sio.emit('client_mode_set', {
        'client_id': client_id,
        'mode': mode,
        'timestamp': datetime.now().isoformat()
    }, room=sid)


@sio.event
async def disconnect(sid):
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {sid}")
    
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
                if not audio_sids:
                    main_clients_to_update.append(main_client_id)
        elif audio_sids == sid:
            main_clients_to_update.append(main_client_id)
    
    for client_id in main_clients_to_update:
        del audio_client_mapping[client_id]        
    
    if sid in client_tts_sessions:
        del client_tts_sessions[sid]


@sio.event
async def register_audio_client(sid, data):
    """Register audio client"""
    logger.info(f"REGISTER CALLED: {sid} with {data}")

    main_client_id = data.get('main_client_id')
    
    if main_client_id and main_client_id != 'unknown':
        if main_client_id not in audio_client_mapping:
            audio_client_mapping[main_client_id] = []

        if sid not in audio_client_mapping[main_client_id]:
            audio_client_mapping[main_client_id].append(sid)
            logger.info(f"🎵 Audio mapping: {main_client_id} -> {audio_client_mapping[main_client_id]}")
    
    existing_session = client_tts_sessions.get(sid, {})
    existing_format = existing_session.get('format', 'base64')
    
    client_tts_sessions[sid] = {
        "connection_type": data.get("connection_type", "browser"),
        "mode": data.get("mode", "tts"),
        "format": data.get("format", existing_format),
        "voice": data.get("voice", get_default_voice()),
        "speed": data.get("speed", get_default_speed()),
        "enabled": data.get("enabled", True)
    }

    logger.info(f"FINAL SESSION: {sid} -> {client_tts_sessions.get(sid, 'NOT_FOUND')}")


@sio.event
async def tts_text_chunk(sid, data):
    """Handle streaming text chunks for TTS generation"""
    try:
        max_len = (tts_settings or {}).get("max_sentence_length", 500)
        idle_timeout_s = (tts_settings or {}).get("buffer_idle_timeout", 0.25)
        treat_nl_as_boundary = (tts_settings or {}).get("treat_newline_as_boundary", True)

        global _tts_buffers, _tts_last_touch
        if "_tts_buffers" not in globals():
            _tts_buffers = {}
        if "_tts_last_touch" not in globals():
            _tts_last_touch = {}

        text_chunk = data.get("chunk", "") or ""
        is_final = bool(data.get("final", False))
        target_client_id = data.get("target_client_id", data.get("client_id", sid))

        if not text_chunk and not is_final:
            return

        audio_sids = audio_client_mapping.get(target_client_id, [target_client_id])
        if isinstance(audio_sids, str):
            audio_sids = [audio_sids]
        elif not isinstance(audio_sids, list):
            audio_sids = [target_client_id]

        primary_audio_sid = audio_sids[0] if audio_sids else target_client_id
        session = client_tts_sessions.get(primary_audio_sid, {
            "voice": get_default_voice(),
            "speed": get_default_speed(),
            "enabled": True,
            "mode": "tts",
            "format": "base64"
        })
        if not session.get("enabled", True):
            return

        voice = session.get("voice", get_default_voice())
        speed = session.get("speed", get_default_speed())
        lang_code = get_language_for_voice(voice) or "en"
        norm_lang = _norm_lang_code(lang_code)

        if not is_voice_enabled(voice):
            logger.warning(f"Voice {voice} not enabled")
            return

        buf = _tts_buffers.get(target_client_id, "")
        if text_chunk:
            buf = _append_stream(buf, text_chunk)
            _tts_buffers[target_client_id] = buf
            _tts_last_touch[target_client_id] = time.time()

        buffered = _tts_buffers.get(target_client_id, "")
        parts = _split_sentences_spacy(buffered, norm_lang, treat_nl_as_boundary)

        to_speak, keep_tail = [], ""
        if parts:
            for i, p in enumerate(parts):
                if i == len(parts) - 1:
                    if is_final:
                        to_speak.append(p)
                    else:
                        keep_tail = p
                else:
                    to_speak.append(p)

        now = time.time()
        last = _tts_last_touch.get(target_client_id, now)
        if (not is_final) and (not to_speak) and keep_tail and (now - last) >= idle_timeout_s:
            if re.search(r"[0-9A-Za-zÀ-ÖØ-öø-ÿ]$", keep_tail):
                pass
            else:
                to_speak.append(keep_tail)
                keep_tail = ""

        _tts_buffers[target_client_id] = keep_tail

        for sentence in to_speak:
            s_raw = sentence.strip()
            if not s_raw:
                continue
            if len(s_raw) > max_len:
                s_raw = s_raw[:max_len] + "..."

            try:
                s = _detok_sentence(s_raw, norm_lang)
            except Exception as detok_err:
                logger.warning(f"Detokenize failed for lang={norm_lang}: {detok_err}")
                s = s_raw

            logger.info(f"TTS: {s[:60]}... (voice: {voice}, lang: {norm_lang}, speed: {speed})")

            for audio_sid in audio_sids:
                try:
                    sess = client_tts_sessions.get(audio_sid, {})
                    format_type = sess.get("format", "base64")
                    mode = sess.get("mode", "tts")
                    conn_type = sess.get("connection_type", "browser")

                    if conn_type != "avatar_server" and mode != "tts":
                        continue

                    if format_type == "binary":
                        async for metadata, binary_data in generate_tts_binary(s, voice, speed, target_client_id):
                            if binary_data is not None:
                                md = metadata.copy()
                                md["audio_buffer"] = binary_data
                                await sio.emit("tts_audio_chunk", md, room=audio_sid)
                            else:
                                await sio.emit("tts_sentence_complete", metadata, room=audio_sid)
                    else:
                        async for chunk_data in generate_tts_base64(s, voice, speed, target_client_id):
                            await sio.emit("tts_audio_chunk", chunk_data, room=audio_sid)

                except Exception as e:
                    logger.error(f"Error sending TTS to {audio_sid}: {e}")

        if is_final:
            _tts_buffers.pop(target_client_id, None)
            _tts_last_touch.pop(target_client_id, None)
            for audio_sid in audio_sids:
                try:
                    await sio.emit("tts_response_complete", {
                        "timestamp": datetime.now().isoformat(),
                        "client_id": target_client_id
                    }, room=audio_sid)
                except Exception as e:
                    logger.error(f"Error sending completion to {audio_sid}: {e}")

    except Exception as e:
        logger.error(f"Text chunk processing error: {e}")
        try:
            target_client_id = data.get("target_client_id", data.get("client_id", sid))
            audio_sids = audio_client_mapping.get(target_client_id, [target_client_id])
            if isinstance(audio_sids, str):
                audio_sids = [audio_sids]
            for audio_sid in audio_sids:
                await sio.emit("tts_error", {
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                    "client_id": target_client_id
                }, room=audio_sid)
        except Exception as send_error:
            logger.error(f"Error sending error to clients: {send_error}")


@sio.event
async def tts_configure(sid, data):
    """Configure TTS settings"""
    session = client_tts_sessions.get(sid, {})
    
    if 'voice' in data:
        new_voice = data['voice']
        if new_voice in VOICE_LIST:
            session['voice'] = new_voice
            logger.info(f"Voice set to {new_voice} for client {sid}")
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
        'pipeline_available': tts_model is not None,
        'voice_enabled': is_voice_enabled(current_voice),
        'speed_range': {
            'min': tts_settings.get('speed_min', 0.5) if tts_settings else 0.5,
            'max': tts_settings.get('speed_max', 1.5) if tts_settings else 1.5
        },
        'timestamp': datetime.now().isoformat()
    }, room=sid)


@sio.event
async def tts_configure_client(sid, data):
    """Configure TTS for relay client"""
    client_id = data.get('client_id')
    audio_sids = audio_client_mapping.get(client_id, client_id)
    
    if isinstance(audio_sids, str):
        audio_sids = [audio_sids]
    
    primary_audio_sid = audio_sids[0] if audio_sids else client_id
    session = client_tts_sessions.get(primary_audio_sid, {})
    
    if 'voice' in data:
        new_voice = data['voice']
        if new_voice in VOICE_LIST:
            session['voice'] = new_voice
            logger.info(f"Voice set to {new_voice} for client {client_id}")
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
        'pipeline_available': tts_model is not None,
        'voice_enabled': is_voice_enabled(current_voice),
        'speed_range': {
            'min': tts_settings.get('speed_min', 0.5) if tts_settings else 0.5,
            'max': tts_settings.get('speed_max', 1.5) if tts_settings else 1.5
        },
        'timestamp': datetime.now().isoformat()
    }, room=sid)


@sio.event
async def tts_get_voices(sid, data):
    """Get available voices"""
    voices_with_languages = []
    
    for voice in VOICE_LIST:
        lang_code = get_language_for_voice(voice)
        voices_with_languages.append({
            'voice': voice,
            'language_code': lang_code,
            'language_name': SUPPORTED_LANGUAGES.get(lang_code, 'Unknown'),
            'available': tts_model is not None,
            'enabled': is_voice_enabled(voice),
            'characteristics': VOICE_CHARACTERISTICS.get(voice, DEFAULT_CHARACTERISTICS)
        })
    
    languages = {}
    for voice_info in voices_with_languages:
        lang_code = voice_info['language_code']
        if lang_code not in languages:
            languages[lang_code] = {
                'code': lang_code,
                'name': voice_info['language_name'],
                'enabled': True,
                'voices': []
            }
        languages[lang_code]['voices'].append(voice_info)
    
    await sio.emit('tts_voices_response', {
        'voices': [voice for voice in VOICE_LIST if is_voice_enabled(voice)],
        'voices_detailed': voices_with_languages,
        'languages': languages,
        'default_voice': get_default_voice(),
        'default_speed': get_default_speed(),
        'speed_range': {
            'min': tts_settings.get('speed_min', 0.5) if tts_settings else 0.5,
            'max': tts_settings.get('speed_max', 1.5) if tts_settings else 1.5
        },
        'requesting_client': data.get('requesting_client'),
        'timestamp': datetime.now().isoformat()
    }, room=sid)


@sio.event
async def tts_get_languages(sid, data):
    """Get supported languages"""
    language_status = []
    enabled_languages = get_enabled_languages()
    
    for lang_code, lang_name in SUPPORTED_LANGUAGES.items():
        voices = VOICE_LIST  # All voices can speak all languages
        is_enabled = lang_code in enabled_languages
        
        language_status.append({
            'code': lang_code,
            'name': lang_name,
            'available': tts_model is not None,
            'enabled': is_enabled,
            'voices': voices,
            'voice_count': len(voices)
        })
    
    await sio.emit('tts_languages_response', {
        'languages': language_status,
        'total_languages': len(SUPPORTED_LANGUAGES),
        'enabled_languages': len(enabled_languages),
        'active_languages': len(enabled_languages) if tts_model else 0,
        'default_voice': get_default_voice(),
        'default_speed': get_default_speed(),
        'speed_range': {
            'min': tts_settings.get('speed_min', 0.5) if tts_settings else 0.5,
            'max': tts_settings.get('speed_max', 1.5) if tts_settings else 1.5
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
            'max': tts_settings.get('speed_max', 1.5) if tts_settings else 1.5
        },
        'max_sentence_length': tts_settings.get('max_sentence_length', 180) if tts_settings else 180,
        'pipeline_timeout': tts_settings.get('pipeline_timeout', 300) if tts_settings else 300,
        'generation_timeout': tts_settings.get('generation_timeout', 60.0) if tts_settings else 60.0,
        'requesting_client': data.get('requesting_client'),
        'timestamp': datetime.now().isoformat()
    }, room=sid)


@sio.event
async def tts_client_disconnect(sid, data):
    """Handle relay client disconnect"""
    client_id = data.get('client_id')
    audio_sids = audio_client_mapping.get(client_id, [])

    if isinstance(audio_sids, str):
        audio_sids = [audio_sids]

    if client_id in audio_client_mapping:
        del audio_client_mapping[client_id]

    for audio_sid in audio_sids:
        if audio_sid in client_tts_sessions:
            del client_tts_sessions[audio_sid]


@sio.event
async def stop_generation(sid, data):
    """Handle stop of TTS generation"""
    client_id = data.get('client_id')
    reason = data.get('reason', 'unknown')
    
    logger.info(f"🚨 STOP GENERATION RECEIVED: {client_id} (reason: {reason})")
    
    audio_sid = audio_client_mapping.get(client_id, client_id)
    
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
    
    await stop_generation(sid, data)


# Cleanup task
async def cleanup_stale_buffers():
    """Clean up stale buffers"""
    while True:
        try:
            stale_clients = []
            
            for client_id in stale_clients:
                if client_id in client_tts_sessions:
                    del client_tts_sessions[client_id]
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
        
        await asyncio.sleep(20)


# REST Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    enabled_languages = get_enabled_languages()
    
    mode_stats = {'tts': 0, 'avatar': 0, 'unknown': 0}
    for session in client_tts_sessions.values():
        mode = session.get('mode', 'unknown')
        mode_stats[mode] = mode_stats.get(mode, 0) + 1
    
    return {
        "status": "healthy" if tts_model else "degraded",
        "version": "2.1.0-parler",
        "model_loaded": tts_model is not None,
        "model_device": str(model_device) if model_device else None,
        "model_sample_rate": model_sample_rate,
        "output_sample_rate": get_output_sample_rate(),
        "total_languages": len(SUPPORTED_LANGUAGES),
        "enabled_languages": len(enabled_languages),
        "active_clients": len(client_tts_sessions),
        "client_modes": mode_stats,
        "supported_voices": len(VOICE_LIST),
        "default_voice": get_default_voice(),
        "default_speed": get_default_speed(),
        "speed_range": {
            "min": tts_settings.get('speed_min', 0.5) if tts_settings else 0.5,
            "max": tts_settings.get('speed_max', 1.5) if tts_settings else 1.5
        },
        "timestamp": datetime.now().isoformat()
    }


@app.get("/languages")
async def get_languages():
    """REST endpoint for language information"""
    language_info = []
    enabled_languages = get_enabled_languages()
    
    for lang_code, lang_name in SUPPORTED_LANGUAGES.items():
        is_enabled = lang_code in enabled_languages
        
        language_info.append({
            'code': lang_code,
            'name': lang_name,
            'available': tts_model is not None,
            'enabled': is_enabled,
            'voices': VOICE_LIST,
            'voice_count': len(VOICE_LIST)
        })
    
    return {
        "languages": language_info,
        "total_languages": len(SUPPORTED_LANGUAGES),
        "enabled_languages": len(enabled_languages),
        "active_languages": len(enabled_languages) if tts_model else 0,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/voices")
async def get_voices():
    """REST endpoint for voice information"""
    voices_with_info = []
    
    for voice in VOICE_LIST:
        lang_code = get_language_for_voice(voice)
        voices_with_info.append({
            'voice': voice,
            'language_code': lang_code,
            'language_name': SUPPORTED_LANGUAGES.get(lang_code, 'Unknown'),
            'available': tts_model is not None,
            'enabled': is_voice_enabled(voice),
            'characteristics': VOICE_CHARACTERISTICS.get(voice, DEFAULT_CHARACTERISTICS)
        })
    
    return {
        "voices": voices_with_info,
        "total_voices": len(VOICE_LIST),
        "enabled_voices": len([v for v in voices_with_info if v['enabled']]),
        "available_voices": len(voices_with_info) if tts_model else 0,
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
            "max": tts_settings.get('speed_max', 1.5) if tts_settings else 1.5
        },
        "max_sentence_length": tts_settings.get('max_sentence_length', 180) if tts_settings else 180,
        "pipeline_timeout": tts_settings.get('pipeline_timeout', 300) if tts_settings else 300,
        "generation_timeout": tts_settings.get('generation_timeout', 60.0) if tts_settings else 60.0,
        "output_sample_rate": get_output_sample_rate(),
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
    logger.info("🌍 Starting Parler TTS Server...")
    logger.info(f"📡 Server: http://{config.host}:{config.port}")
    logger.info(f"🔌 Socket.IO: http://{config.host}:{config.port}/socket.io/")
    logger.info(f"💡 Health check: http://{config.host}:{config.port}/health")
    logger.info(f"🌐 Languages: http://{config.host}:{config.port}/languages")
    logger.info(f"🎤 Voices: http://{config.host}:{config.port}/voices")
    logger.info(f"⚙️ Settings: http://{config.host}:{config.port}/settings")
    
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
