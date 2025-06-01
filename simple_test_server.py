#!/usr/bin/env python3
"""
simple_test_server.py - Minimal Binary Socket.IO Test Server
"""

import asyncio
import socketio
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn
import numpy as np
import io
import soundfile as sf

# Create Socket.IO server
sio = socketio.AsyncServer(
    cors_allowed_origins="*",
    logger=True,
    engineio_logger=True,
    async_mode='asgi'
)

# Create FastAPI app
app = FastAPI()

def create_test_wav(duration=1.0, frequency=440):
    """Create a simple test WAV file"""
    sample_rate = 24000
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = 0.3 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    # Convert to WAV bytes
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sample_rate, format='WAV', subtype='PCM_16')
    buffer.seek(0)
    return buffer.read()

@sio.event
async def connect(sid, environ):
    print(f"✅ Client connected: {sid}")
    await sio.emit('connected', {'message': 'Connected to test server'}, room=sid)

@sio.event
async def disconnect(sid):
    print(f"❌ Client disconnected: {sid}")

@sio.event
async def test_text(sid, data):
    """Test simple text communication"""
    print(f"📝 Received text test from {sid}: {data}")
    await sio.emit('text_response', {'received': data, 'timestamp': 'now'}, room=sid)

@sio.event
async def test_binary_method1(sid, data):
    """Test binary emission - Method 1: Binary attachments"""
    print(f"🔥 Testing binary method 1 for {sid}")
    
    try:
        wav_data = create_test_wav(duration=0.5, frequency=880)  # Short beep
        metadata = {
            'method': 'binary_attachments',
            'size': len(wav_data),
            'type': 'test_audio'
        }
        
        print(f"🔥 Emitting with binary attachments: {len(wav_data)} bytes")
        await sio.emit('binary_result', metadata, [wav_data], room=sid)
        print(f"✅ Method 1 emission successful")
        
    except Exception as e:
        print(f"❌ Method 1 failed: {e}")
        await sio.emit('test_error', {'method': 1, 'error': str(e)}, room=sid)

@sio.event
async def test_binary_method2(sid, data):
    """Test binary emission - Method 2: Binary in data"""
    print(f"🔥 Testing binary method 2 for {sid}")
    
    try:
        wav_data = create_test_wav(duration=0.5, frequency=660)  # Different tone
        metadata = {
            'method': 'binary_in_data',
            'size': len(wav_data),
            'type': 'test_audio',
            'audio_data': wav_data  # Include binary directly
        }
        
        print(f"🔥 Emitting with binary in data: {len(wav_data)} bytes")
        await sio.emit('binary_result', metadata, room=sid)
        print(f"✅ Method 2 emission successful")
        
    except Exception as e:
        print(f"❌ Method 2 failed: {e}")
        await sio.emit('test_error', {'method': 2, 'error': str(e)}, room=sid)

@sio.event
async def test_binary_method3(sid, data):
    """Test binary emission - Method 3: Base64 encoded"""
    print(f"🔥 Testing binary method 3 for {sid}")
    
    try:
        import base64
        wav_data = create_test_wav(duration=0.5, frequency=440)  # Original tone
        b64_data = base64.b64encode(wav_data).decode('utf-8')
        
        metadata = {
            'method': 'base64_encoded',
            'size': len(wav_data),
            'encoded_size': len(b64_data),
            'type': 'test_audio',
            'audio_data': b64_data
        }
        
        print(f"🔥 Emitting base64 encoded: {len(wav_data)} -> {len(b64_data)} chars")
        await sio.emit('binary_result', metadata, room=sid)
        print(f"✅ Method 3 emission successful")
        
    except Exception as e:
        print(f"❌ Method 3 failed: {e}")
        await sio.emit('test_error', {'method': 3, 'error': str(e)}, room=sid)

# Combine with FastAPI
sio_asgi_app = socketio.ASGIApp(sio, other_asgi_app=app)

@app.get("/")
async def get_test_page():
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Binary Socket.IO Test</title>
        <script src="https://cdn.socket.io/4.7.2/socket.io.min.js"></script>
    </head>
    <body>
        <h1>Binary Socket.IO Test</h1>
        <div id="status">Connecting...</div>
        <div id="results"></div>
        <br>
        <button onclick="testText()">Test Text</button>
        <button onclick="testBinary1()">Test Binary Method 1</button>
        <button onclick="testBinary2()">Test Binary Method 2</button>
        <button onclick="testBinary3()">Test Binary Method 3</button>
        <br><br>
        <div id="logs"></div>
        
        <script>
            const socket = io();
            
            function log(message) {
                const logs = document.getElementById('logs');
                logs.innerHTML += '<div>' + new Date().toLocaleTimeString() + ': ' + message + '</div>';
                logs.scrollTop = logs.scrollHeight;
            }
            
            socket.on('connect', () => {
                document.getElementById('status').textContent = 'Connected!';
                log('✅ Connected to server');
            });
            
            socket.on('disconnect', () => {
                document.getElementById('status').textContent = 'Disconnected';
                log('❌ Disconnected from server');
            });
            
            socket.on('connected', (data) => {
                log('📡 Server says: ' + data.message);
            });
            
            socket.on('text_response', (data) => {
                log('📝 Text response: ' + JSON.stringify(data));
            });
            
            socket.on('binary_result', (data, binaryData) => {
                log('🔥 Binary result received!');
                log('  Method: ' + data.method);
                log('  Size: ' + data.size + ' bytes');
                log('  Has binary attachments: ' + (binaryData ? 'YES (' + binaryData.length + ')' : 'NO'));
                log('  Has audio_data in data: ' + (data.audio_data ? 'YES' : 'NO'));
                
                if (binaryData && binaryData.length > 0) {
                    log('  Binary attachment size: ' + binaryData[0].byteLength + ' bytes');
                    playAudio(binaryData[0], 'Binary attachment');
                } else if (data.audio_data) {
                    if (typeof data.audio_data === 'string') {
                        // Base64 decode
                        try {
                            const binaryString = atob(data.audio_data);
                            const bytes = new Uint8Array(binaryString.length);
                            for (let i = 0; i < binaryString.length; i++) {
                                bytes[i] = binaryString.charCodeAt(i);
                            }
                            playAudio(bytes.buffer, 'Base64 decoded');
                        } catch (e) {
                            log('  ❌ Failed to decode base64: ' + e.message);
                        }
                    } else if (data.audio_data instanceof ArrayBuffer) {
                        playAudio(data.audio_data, 'Binary in data');
                    }
                }
            });
            
            socket.on('test_error', (data) => {
                log('❌ Test error (method ' + data.method + '): ' + data.error);
            });
            
            function playAudio(arrayBuffer, source) {
                try {
                    const blob = new Blob([arrayBuffer], {type: 'audio/wav'});
                    const url = URL.createObjectURL(blob);
                    const audio = new Audio(url);
                    
                    audio.onloadeddata = () => {
                        log('🎵 Audio loaded from ' + source + ' (duration: ' + audio.duration.toFixed(2) + 's)');
                    };
                    
                    audio.onended = () => {
                        URL.revokeObjectURL(url);
                        log('🎵 Audio finished playing');
                    };
                    
                    audio.onerror = (e) => {
                        log('❌ Audio error: ' + e.message);
                        URL.revokeObjectURL(url);
                    };
                    
                    audio.play().then(() => {
                        log('🎵 Playing audio from ' + source);
                    }).catch(e => {
                        log('❌ Failed to play audio: ' + e.message);
                    });
                    
                } catch (e) {
                    log('❌ Failed to create audio: ' + e.message);
                }
            }
            
            function testText() {
                log('📝 Testing text communication...');
                socket.emit('test_text', {message: 'Hello from browser!', timestamp: Date.now()});
            }
            
            function testBinary1() {
                log('🔥 Testing binary method 1 (attachments)...');
                socket.emit('test_binary_method1', {});
            }
            
            function testBinary2() {
                log('🔥 Testing binary method 2 (in data)...');
                socket.emit('test_binary_method2', {});
            }
            
            function testBinary3() {
                log('🔥 Testing binary method 3 (base64)...');
                socket.emit('test_binary_method3', {});
            }
        </script>
    </body>
    </html>
    """)

if __name__ == '__main__':
    print("🧪 Starting Binary Socket.IO Test Server...")
    print("📡 Open http://localhost:8888 in your browser")
    print("🔥 Test different binary transmission methods")
    
    config = uvicorn.Config(
        sio_asgi_app,
        host="0.0.0.0",
        port=8888,
        log_level="info"
    )
    
    server = uvicorn.Server(config)
    asyncio.run(server.serve())
