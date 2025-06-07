#!/usr/bin/env python3

"""
BGE-M3 API-Based Streaming Boundary Detector
Uses existing FlagEmbedding service instead of loading model directly
Optimized for real-time TTS streaming with HTTP API calls
Enhanced with conservative settings and incomplete detection
"""

import asyncio
import aiohttp
import json
import logging
import time
import numpy as np
from datetime import datetime
from typing import List, Tuple, Optional, Dict
from functools import lru_cache
import threading

logger = logging.getLogger(__name__)


class FlagEmbeddingAPIClient:
    """
    Client for BGE-M3 FlagEmbedding service
    Adapted from your flag.embedding.js implementation
    """
    
    def __init__(self, settings: Dict = None):
        self.settings = settings
        self.model_endpoint = None
        self.reranker_endpoint = None
        self.health_endpoint = None
        self.batch_size = 8  # Default batch size
        self.dimension = 1024  # BGE-M3 dimension
        self.debug = False
        
        # HTTP session for connection pooling
        self.session = None
        self._session_lock = threading.Lock()
        
        # Performance tracking
        self.request_count = 0
        self.total_response_time = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
        
        if settings:
            self._configure_from_settings(settings)
        
        logger.info("FlagEmbedding API Client initialized")
    
    def _configure_from_settings(self, settings: Dict):
        """Configure client from settings dictionary"""
        flag_config = settings.get('FlagEmbedding', {})
        
        server_address = flag_config.get('ServerAPIAddress', 'http://localhost:8000')
        self.model_endpoint = f"{server_address}/embed"
        self.reranker_endpoint = f"{server_address}/rerank"
        self.health_endpoint = f"{server_address}/health"
        
        self.batch_size = flag_config.get('BatchSize', 8)
        self.dimension = flag_config.get('Dimension', 1024)
        self.debug = flag_config.get('Debug', False)
        
        logger.info(f"Configured API client: {server_address}, batch_size: {self.batch_size}")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session with connection pooling"""
        if self.session is None or self.session.closed:
            with self._session_lock:
                if self.session is None or self.session.closed:
                    connector = aiohttp.TCPConnector(
                        limit=100,  # Connection pool limit
                        limit_per_host=20,
                        keepalive_timeout=30,
                        enable_cleanup_closed=True
                    )
                    
                    timeout = aiohttp.ClientTimeout(
                        total=10.0,  # Total request timeout
                        connect=2.0,  # Connection timeout
                        sock_read=5.0  # Socket read timeout
                    )
                    
                    self.session = aiohttp.ClientSession(
                        connector=connector,
                        timeout=timeout,
                        headers={'Content-Type': 'application/json'}
                    )
        
        return self.session
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        if not texts:
            logger.warning("No texts provided for embedding generation")
            return []
        
        # Process in batches
        batches = self._create_batches(texts, self.batch_size)
        all_embeddings = []
        
        for batch in batches:
            try:
                batch_embeddings = await self._process_batch(batch)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                raise
        
        return all_embeddings
    
    async def generate_single_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single text"""
        if not text or not isinstance(text, str):
            logger.warning("Invalid text provided for embedding generation")
            return None
        
        try:
            embeddings = await self._process_batch([text])
            return embeddings[0] if embeddings else None
        except Exception as e:
            logger.error(f"Error generating single embedding: {e}")
            raise
    
    async def _process_batch(self, texts: List[str]) -> List[List[float]]:
        """Process a batch of texts through the API"""
        if not self.model_endpoint:
            raise ValueError("Model endpoint not configured")
        
        start_time = time.time() if self.debug else None
        session = await self._get_session()
        
        try:
            payload = {"texts": texts}
            
            async with session.post(self.model_endpoint, json=payload) as response:
                if not response.status == 200:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")
                
                data = await response.json()
                
                if 'embeddings' not in data or not isinstance(data['embeddings'], list):
                    raise Exception("Invalid response format: embeddings array not found")
                
                embeddings = data['embeddings']
                
                # Update performance metrics
                self.request_count += 1
                if start_time:
                    duration = time.time() - start_time
                    self.total_response_time += duration
                    logger.debug(f"Processed batch of {len(texts)} texts in {duration*1000:.2f}ms")
                
                return embeddings
                
        except Exception as e:
            logger.error(f"Error in _process_batch: {e}. URL: {self.model_endpoint}")
            raise
    
    def _create_batches(self, texts: List[str], batch_size: int) -> List[List[str]]:
        """Split texts into batches"""
        batches = []
        for i in range(0, len(texts), batch_size):
            batches.append(texts[i:i + batch_size])
        return batches
    
    async def check_health(self) -> bool:
        """Check if the FlagEmbedding service is healthy"""
        if not self.health_endpoint:
            return False
        
        try:
            session = await self._get_session()
            async with session.get(self.health_endpoint) as response:
                is_healthy = response.status == 200
                response_text = await response.text()
                logger.debug(f"Health check: {response.status} {response_text}")
                return is_healthy
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        avg_response_time = (
            self.total_response_time / self.request_count 
            if self.request_count > 0 else 0
        )
        
        return {
            'total_requests': self.request_count,
            'average_response_time_ms': avg_response_time * 1000,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'session_active': self.session is not None and not self.session.closed
        }
    
    async def close(self):
        """Close the HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()


class APIBasedBGEBoundaryDetector:
    """
    BGE-M3 boundary detection using your existing FlagEmbedding API service
    Enhanced with incomplete detection and conservative settings
    """
    
    def __init__(self, settings: Dict = None):
        self.api_client = FlagEmbeddingAPIClient(settings)
        
        # Pre-defined completion patterns for reference
        self.completion_patterns = {
            'explanatory': [
                "That's exactly how this process works in practice.",
                "This explanation should clarify the main concept completely.",
                "I hope this detailed breakdown helps you understand.",
                "The key point is now clear and well-defined."
            ],
            'conversational': [
                "I'm here to help with any questions you have.",
                "Let me know if you need further assistance.",
                "Feel free to ask if anything needs clarification.",
                "I hope this information is helpful to you."
            ],
            'definitive': [
                "The answer to your question is clear and definitive.",
                "This is the correct solution to the problem.",
                "The result is exactly what you were looking for.",
                "This conclusion addresses your concern completely."
            ]
        }
        
        self.incomplete_patterns = [
            "What I'm trying to explain is that",
            "The reason this is important is because",
            "Let me walk you through how this",
            "This is something that requires careful",
            "The key thing to understand about this"
        ]
        
        # Cache for reference embeddings
        self.reference_embeddings = {}
        self.embeddings_initialized = False
        
        # Cache for text embeddings (LRU-style)
        self.text_embedding_cache = {}
        self.max_cache_size = 500
        
        logger.info("API-based BGE Boundary Detector initialized")
    
    async def initialize_reference_embeddings(self):
        """Pre-compute embeddings for reference patterns"""
        if self.embeddings_initialized:
            return
        
        logger.info("Initializing reference embeddings via API...")
        
        try:
            # Flatten all patterns
            all_complete = []
            for category, patterns in self.completion_patterns.items():
                all_complete.extend(patterns)
            
            # Generate embeddings via API
            complete_embeddings = await self.api_client.generate_embeddings(all_complete)
            incomplete_embeddings = await self.api_client.generate_embeddings(self.incomplete_patterns)
            
            self.reference_embeddings = {
                'complete': complete_embeddings,
                'incomplete': incomplete_embeddings
            }
            
            self.embeddings_initialized = True
            logger.info(f"Reference embeddings initialized: {len(complete_embeddings)} complete, {len(incomplete_embeddings)} incomplete")
            
        except Exception as e:
            logger.error(f"Failed to initialize reference embeddings: {e}")
            # Continue without reference embeddings
            self.reference_embeddings = {'complete': [], 'incomplete': []}
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors"""
        try:
            # Convert to numpy arrays for calculation
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            
            if norm_v1 == 0 or norm_v2 == 0:
                return 0.0
                
            return float(dot_product / (norm_v1 * norm_v2))
        except Exception as e:
            logger.error(f"Error computing cosine similarity: {e}")
            return 0.0
    
    async def _get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding with caching"""
        # Check cache first
        if text in self.text_embedding_cache:
            self.api_client.cache_hits += 1
            return self.text_embedding_cache[text]
        
        # Generate via API
        try:
            embedding = await self.api_client.generate_single_embedding(text)
            
            if embedding:
                # Add to cache, manage size
                if len(self.text_embedding_cache) >= self.max_cache_size:
                    # Remove oldest entry (simple FIFO)
                    oldest_key = next(iter(self.text_embedding_cache))
                    del self.text_embedding_cache[oldest_key]
                
                self.text_embedding_cache[text] = embedding
                self.api_client.cache_misses += 1
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error getting embedding for text: {e}")
            return None
    
    def _check_obvious_incomplete(self, text: str) -> Tuple[bool, str]:
        """Check for obvious incomplete patterns that BGE-M3 might miss"""
        text_lower = text.strip().lower()
        
        # Obvious incomplete endings
        incomplete_endings = [
            ' of', ' and', ' the', ' which', ' that', ' where', ' when', 
            ' because', ' since', ' while', ' during', ' through', ' with',
            ' for', ' from', ' into', ' onto', ' upon', ' about', ' above',
            ' below', ' under', ' over', ' between', ' among', ' within',
            ' without', ' against', ' towards', ' across', ' behind',
            ' beside', ' beyond', ' beneath', ' inside', ' outside',
            ' including', ' excluding', ' regarding', ' concerning',
            ' considering', ' following', ' preceding', ' during',
            ' throughout', ' underneath', ' alongside', ' amidst'
        ]
        
        for ending in incomplete_endings:
            if text_lower.endswith(ending):
                return True, f"incomplete_ending_{ending.strip()}"
        
        # Check for incomplete conjunctions and transitions
        incomplete_conjunctions = [
            ' and then', ' but also', ' not only', ' either', ' neither',
            ' both', ' whether', ' although', ' though', ' unless',
            ' whereas', ' whereby', ' wherein', ' whereupon', ' therefore',
            ' however', ' moreover', ' furthermore', ' nevertheless',
            ' nonetheless', ' meanwhile', ' consequently', ' accordingly',
            ' thus', ' hence', ' otherwise', ' instead', ' besides',
            ' additionally', ' similarly', ' likewise', ' conversely',
            ' in contrast', ' on the other hand', ' for example',
            ' for instance', ' such as', ' in particular', ' specifically',
            ' especially', ' namely', ' that is', ' in other words'
        ]
        
        for conjunction in incomplete_conjunctions:
            if text_lower.endswith(conjunction):
                return True, f"incomplete_conjunction_{conjunction.strip()}"
        
        # Check for incomplete scientific/technical phrases
        incomplete_technical = [
            ' theory of', ' principle of', ' law of', ' concept of',
            ' definition of', ' explanation of', ' description of',
            ' analysis of', ' study of', ' research on', ' investigation into',
            ' examination of', ' evaluation of', ' assessment of',
            ' measurement of', ' calculation of', ' determination of',
            ' application of', ' implementation of', ' development of',
            ' creation of', ' formation of', ' construction of',
            ' production of', ' generation of', ' synthesis of'
        ]
        
        for technical in incomplete_technical:
            if text_lower.endswith(technical):
                return True, f"incomplete_technical_{technical.strip()}"
        
        return False, "complete"
    
    async def completion_confidence(self, text_buffer: str) -> Tuple[float, Dict]:
        """
        Analyze text buffer and return completion confidence using API
        Enhanced with incomplete detection
        """
        if len(text_buffer.strip()) < 5:
            return 0.0, {'reason': 'too_short', 'length': len(text_buffer)}
        
        cleaned_text = text_buffer.strip()
        
        # First check for obvious incomplete patterns
        is_obviously_incomplete, incomplete_reason = self._check_obvious_incomplete(cleaned_text)
        if is_obviously_incomplete:
            return 0.0, {
                'reason': 'obviously_incomplete',
                'incomplete_type': incomplete_reason,
                'text_ending': cleaned_text[-20:],
                'length': len(cleaned_text)
            }
        
        # Ensure reference embeddings are loaded
        if not self.embeddings_initialized:
            await self.initialize_reference_embeddings()
        
        try:
            # Get embedding for current buffer
            buffer_embedding = await self._get_cached_embedding(cleaned_text)
            
            if not buffer_embedding:
                return 0.0, {'error': 'failed_to_generate_embedding'}
            
            # Compare with reference embeddings
            complete_similarities = []
            incomplete_similarities = []
            
            if self.reference_embeddings.get('complete'):
                complete_similarities = [
                    self._cosine_similarity(buffer_embedding, ref_emb)
                    for ref_emb in self.reference_embeddings['complete']
                ]
            
            if self.reference_embeddings.get('incomplete'):
                incomplete_similarities = [
                    self._cosine_similarity(buffer_embedding, ref_emb)
                    for ref_emb in self.reference_embeddings['incomplete']
                ]
            
            # Calculate confidence
            max_complete = max(complete_similarities) if complete_similarities else 0.0
            max_incomplete = max(incomplete_similarities) if incomplete_similarities else 0.0
            avg_complete = np.mean(complete_similarities) if complete_similarities else 0.0
            avg_incomplete = np.mean(incomplete_similarities) if incomplete_similarities else 0.0
            
            # Confidence calculation
            confidence_raw = max_complete - max_incomplete
            confidence_avg = avg_complete - avg_incomplete
            
            # Weighted combination and normalization
            confidence = 0.7 * confidence_raw + 0.3 * confidence_avg
            confidence = max(0, min(1, (confidence + 1) / 2))
            
            analysis = {
                'max_complete_sim': max_complete,
                'max_incomplete_sim': max_incomplete,
                'avg_complete_sim': avg_complete,
                'avg_incomplete_sim': avg_incomplete,
                'raw_confidence': confidence_raw,
                'final_confidence': confidence,
                'text_length': len(cleaned_text),
                'api_cached': cleaned_text in self.text_embedding_cache,
                'incomplete_check_passed': True
            }
            
            return confidence, analysis
            
        except Exception as e:
            logger.error(f"Error in completion_confidence: {e}")
            return 0.0, {'error': str(e)}
    
    async def close(self):
        """Close API client resources"""
        await self.api_client.close()


class APIStreamingSentenceBuffer:
    """
    Streaming sentence buffer using BGE-M3 API service
    Enhanced with conservative settings and better incomplete detection
    """
    
    def __init__(self, client_id: str, settings: Dict = None):
        self.client_id = client_id
        self.buffer = ""
        self.last_chunk_time = datetime.now()
        
        # Configuration
        self.min_length = 8
        self.ideal_length = 75
        self.max_length = 180
        
        # Conservative timing configuration to prevent premature breaks
        self.completion_delay = 1.0  # Increased from 0.7s
        self.force_completion_delay = 2.5  # Increased from 2.0s
        self.semantic_check_interval = 0.8  # Increased from 0.5s (less frequent)
        
        # BGE-M3 API integration with conservative thresholds
        self.semantic_detector = APIBasedBGEBoundaryDetector(settings)
        self.last_semantic_check = 0
        self.semantic_confidence_threshold = 0.85  # Increased from 0.75
        self.semantic_confidence_threshold_timed = 0.75  # Increased from 0.65
        
        # State tracking
        self.last_apparent_completion = None
        
        logger.info(f"Conservative API-based buffer initialized for {client_id}: "
                   f"thresholds={self.semantic_confidence_threshold}/{self.semantic_confidence_threshold_timed}, "
                   f"delays={self.completion_delay}s/{self.force_completion_delay}s")
    
    async def add_chunk(self, chunk: str) -> List[Tuple[str, Dict]]:
        """Add text chunk and return complete sentences with analysis"""
        self.buffer += chunk
        self.last_chunk_time = datetime.now()
        current_time = time.time()
        
        ready_sentences = []
        
        # Skip processing if buffer too small
        if len(self.buffer.strip()) < self.min_length:
            return ready_sentences
        
        # 1. IMMEDIATE COMPLETION: Traditional indicators
        immediate_complete, immediate_reason = self._check_immediate_completion()
        if immediate_complete:
            sentence = self.buffer.strip()
            analysis = {
                'method': 'immediate',
                'reason': immediate_reason,
                'confidence': 1.0,
                'length': len(sentence),
                'processing_time_ms': 0,
                'api_used': False
            }
            ready_sentences.append((sentence, analysis))
            self.buffer = ""
            return ready_sentences
        
        # 2. SEMANTIC ANALYSIS: Use BGE-M3 API (less frequent due to latency)
        should_run_semantic = (
            (current_time - self.last_semantic_check) > self.semantic_check_interval
            and len(self.buffer.strip()) >= 20  # Increased minimum length
        )
        
        if should_run_semantic:
            semantic_start = time.time()
            
            try:
                confidence, semantic_analysis = await self.semantic_detector.completion_confidence(self.buffer)
                semantic_time = (time.time() - semantic_start) * 1000
                
                self.last_semantic_check = current_time
                
                # High confidence completion (more conservative threshold)
                if confidence >= self.semantic_confidence_threshold:
                    sentence = self.buffer.strip()
                    analysis = {
                        'method': 'semantic_high',
                        'reason': 'high_confidence_api',
                        'confidence': confidence,
                        'length': len(sentence),
                        'processing_time_ms': semantic_time,
                        'semantic_details': semantic_analysis,
                        'api_used': True
                    }
                    ready_sentences.append((sentence, analysis))
                    self.buffer = ""
                    return ready_sentences
                
                # Medium confidence with timing (more conservative)
                time_since_last_update = (datetime.now() - self.last_chunk_time).total_seconds()
                if (confidence >= self.semantic_confidence_threshold_timed 
                    and time_since_last_update >= self.completion_delay):
                    sentence = self.buffer.strip()
                    analysis = {
                        'method': 'semantic_timed',
                        'reason': 'medium_confidence_api_with_delay',
                        'confidence': confidence,
                        'length': len(sentence),
                        'processing_time_ms': semantic_time,
                        'delay_seconds': time_since_last_update,
                        'semantic_details': semantic_analysis,
                        'api_used': True
                    }
                    ready_sentences.append((sentence, analysis))
                    self.buffer = ""
                    return ready_sentences
                    
            except Exception as e:
                logger.error(f"Semantic analysis failed, falling back to traditional: {e}")
                # Continue with traditional methods
        
        # 3. FORCE COMPLETION: Prevent hanging (more conservative)
        if self._should_force_complete():
            sentence = self.buffer.strip()
            time_since_update = (datetime.now() - self.last_chunk_time).total_seconds()
            analysis = {
                'method': 'forced',
                'reason': 'timeout_or_length',
                'confidence': 0.5,
                'length': len(sentence),
                'processing_time_ms': 0,
                'delay_seconds': time_since_update,
                'api_used': False
            }
            ready_sentences.append((sentence, analysis))
            self.buffer = ""
        
        return ready_sentences
    
    def _check_immediate_completion(self) -> Tuple[bool, str]:
        """Check for definitive completion indicators"""
        text = self.buffer.strip()
        
        # Strong punctuation endings
        strong_endings = ['.', '!', '?', '."', '!"', '?"']
        for ending in strong_endings:
            if text.endswith(ending):
                return True, f"punctuation_{ending.replace('.', 'period').replace('!', 'exclamation').replace('?', 'question')}"
        
        # Colon endings (more conservative - require longer text)
        if text.endswith(':') and len(text) > 30:  # Increased from 20
            return True, "colon_ending"
        
        return False, "none"
    
    def _should_force_complete(self) -> bool:
        """Check if we should force completion (more conservative)"""
        text = self.buffer.strip()
        
        if len(text) > self.max_length:
            return True
        
        time_since_update = (datetime.now() - self.last_chunk_time).total_seconds()
        if time_since_update >= self.force_completion_delay and len(text) >= self.min_length:
            return True
        
        return False
    
    async def flush_remaining(self, force: bool = True) -> List[Tuple[str, Dict]]:
        """Get any remaining text when stream ends"""
        if not self.buffer.strip() or len(self.buffer.strip()) < 3:
            return []
        
        sentence = self.buffer.strip()
        self.buffer = ""
        
        analysis = {
            'method': 'flush',
            'reason': 'stream_end',
            'confidence': 0.8 if force else 0.6,
            'length': len(sentence),
            'processing_time_ms': 0,
            'forced': force,
            'api_used': False
        }
        
        logger.info(f"Client {self.client_id}: Flushing remaining text: {sentence[:50]}...")
        return [(sentence, analysis)]
    
    async def close(self):
        """Close resources"""
        await self.semantic_detector.close()
    
    def get_stats(self) -> Dict:
        """Get performance and usage statistics"""
        api_stats = self.semantic_detector.api_client.get_performance_stats()
        
        return {
            'client_id': self.client_id,
            'current_buffer_length': len(self.buffer),
            'last_chunk_time': self.last_chunk_time.isoformat(),
            'semantic_check_interval': self.semantic_check_interval,
            'confidence_threshold': self.semantic_confidence_threshold,
            'api_performance': api_stats,
            'conservative_settings': {
                'confidence_threshold': self.semantic_confidence_threshold,
                'timed_threshold': self.semantic_confidence_threshold_timed,
                'completion_delay': self.completion_delay,
                'force_delay': self.force_completion_delay,
                'check_interval': self.semantic_check_interval
            },
            'configuration': {
                'min_length': self.min_length,
                'ideal_length': self.ideal_length,
                'max_length': self.max_length,
                'completion_delay': self.completion_delay,
                'force_completion_delay': self.force_completion_delay
            }
        }


# Integration helper for existing TTS server
async def process_tts_with_api_bge(client_id: str, text_chunk: str, is_final: bool, 
                                   client_buffers: Dict, tts_generation_callback, settings: Dict = None):
    """
    Process text chunk using BGE-M3 API-based boundary detection
    """
    
    # Get or create buffer for this client
    if client_id not in client_buffers:
        client_buffers[client_id] = APIStreamingSentenceBuffer(client_id, settings)
    
    buffer = client_buffers[client_id]
    
    # Process chunk and get ready sentences
    if text_chunk:
        ready_sentences = await buffer.add_chunk(text_chunk)
        
        # Start TTS for each ready sentence immediately
        for sentence, analysis in ready_sentences:
            logger.info(f"🎵 API BGE-M3 TTS START: {sentence[:40]}... "
                       f"(method: {analysis['method']}, confidence: {analysis['confidence']:.3f}, "
                       f"api_time: {analysis.get('processing_time_ms', 0):.1f}ms)")
            
            # Start TTS generation asynchronously
            asyncio.create_task(
                tts_generation_callback(sentence, client_id, analysis)
            )
    
    # Handle final flush
    if is_final:
        final_sentences = await buffer.flush_remaining(force=True)
        for sentence, analysis in final_sentences:
            logger.info(f"🎵 API BGE-M3 FINAL TTS: {sentence[:40]}... "
                       f"(method: {analysis['method']})")
            
            asyncio.create_task(
                tts_generation_callback(sentence, client_id, analysis)
            )


# Example configuration loader (keeping it simple since FlagEmbedding handles its own settings)
async def load_flagembedding_settings(settings_path: str = "./data/configuration/flagembedding.settings.json") -> Dict:
    """Load FlagEmbedding settings from configuration file"""
    try:
        with open(settings_path, 'r') as f:
            settings = json.load(f)
        return settings
    except Exception as e:
        logger.error(f"Failed to load FlagEmbedding settings: {e}")
        # Return default configuration
        return {
            "FlagEmbedding": {
                "ServerAPIAddress": "http://flagembedding:8000",
                "BatchSize": 8,
                "Dimension": 1024,
                "Debug": True
            }
        }
