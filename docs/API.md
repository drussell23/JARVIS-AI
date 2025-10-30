# JARVIS AI Agent API Documentation

## Overview

The JARVIS AI Agent provides a comprehensive REST and WebSocket API for voice-controlled AI assistance with advanced capabilities including screen vision, voice authentication, workflow automation, and autonomous operation.

## Base URL

```
http://localhost:8000
```

## Authentication

Most endpoints require no authentication. Voice unlock endpoints use voice biometric authentication when configured.

## Core Endpoints

### Health Check

#### GET /health
Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "2.0"
}
```

**Example:**
```bash
curl -X GET http://localhost:8000/health
```

---

## Voice API

### Voice Command Processing

#### POST /voice/jarvis/command
Process natural language voice commands.

**Request Body:**
```json
{
  "text": "unlock my screen",
  "context": {
    "user_id": "user123",
    "session_id": "session456"
  }
}
```

**Response:**
```json
{
  "success": true,
  "response": "I'll unlock your screen right away, Sir.",
  "command_type": "unlock",
  "executed": true,
  "execution_time_ms": 1250
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/voice/jarvis/command \
  -H "Content-Type: application/json" \
  -d '{"text": "open Safari and search for weather"}'
```

#### POST /voice/jarvis/activate
Activate JARVIS voice system.

**Response:**
```json
{
  "status": "activated",
  "message": "JARVIS voice system online",
  "capabilities": ["voice_recognition", "tts", "command_processing"]
}
```

#### GET /voice/jarvis/status
Get voice system status.

**Response:**
```json
{
  "active": true,
  "voice_engine": "CoreML",
  "tts_engine": "edge_tts",
  "wake_word_enabled": true,
  "last_command": "2024-01-15T10:25:00Z"
}
```

### CoreML Voice Engine

#### POST /voice/detect-coreml
Hardware-accelerated voice detection using Apple Neural Engine.

**Request Body:**
```json
{
  "audio_data": "base64_encoded_audio_data",
  "priority": 1
}
```

**Response:**
```json
{
  "is_user_voice": true,
  "vad_confidence": 0.89,
  "speaker_confidence": 0.75,
  "inference_time_ms": 8.2,
  "metrics": {
    "model_size_kb": 232,
    "memory_usage_mb": 7.3
  }
}
```

**Example:**
```bash
# Encode audio to base64 first
audio_b64=$(base64 -i audio_sample.wav)
curl -X POST http://localhost:8000/voice/detect-coreml \
  -H "Content-Type: application/json" \
  -d "{\"audio_data\": \"$audio_b64\", \"priority\": 1}"
```

#### POST /voice/detect-vad-coreml
Voice activity detection only (faster).

**Request Body:**
```json
{
  "audio_data": "base64_encoded_audio_data"
}
```

**Response:**
```json
{
  "voice_detected": true,
  "confidence": 0.92,
  "inference_time_ms": 4.1
}
```

---

## Vision API

### Screen Analysis

#### POST /vision/analyze
Analyze current screen content using Claude Vision.

**Request Body:**
```json
{
  "prompt": "What applications are currently open?",
  "multi_space": true,
  "include_windows": true
}
```

**Response:**
```json
{
  "success": true,
  "analysis": "I can see Safari with 3 tabs open, Terminal running a Python script, and Finder showing your Documents folder.",
  "confidence": 0.95,
  "detected_elements": [
    {
      "type": "application",
      "name": "Safari",
      "windows": 1,
      "tabs": 3
    }
  ]
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/vision/analyze \
  -H "Content-Type: application/json" \
  -d '{"prompt": "How many browser tabs are open?"}'
```

#### GET /vision/capture
Capture current screen.

**Query Parameters:**
- `multi_space` (boolean): Capture all desktop spaces
- `space_number` (integer): Specific space to capture
- `format` (string): Image format (png, jpg)

**Response:**
```json
{
  "success": true,
  "image_data": "base64_encoded_image",
  "format": "png",
  "resolution": [1920, 1080],
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Example:**
```bash
curl -X GET "http://localhost:8000/vision/capture?multi_space=true&format=png"
```

#### GET /vision/displays
Get information about connected displays.

**Response:**
```json
{
  "total_displays": 2,
  "displays": [
    {
      "id": 1,
      "name": "Built-in Display",
      "resolution": [1920, 1080],
      "position": [0, 0],
      "is_primary": true,
      "spaces": [1, 2, 3]
    }
  ],
  "space_mappings": {
    "1": 1,
    "2": 1,
    "3": 2
  }
}
```

### Vision Monitoring

#### POST /monitor/control
Start or stop screen monitoring.

**Request Body:**
```json
{
  "action": "start"
}
```

**Response:**
```json
{
  "success": true,
  "action": "start",
  "response": "Screen monitoring activated",
  "monitoring_active": true
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/monitor/control \
  -H "Content-Type: application/json" \
  -d '{"action": "start"}'
```

---

## Screen Control API

### Screen Lock/Unlock

#### POST /api/screen/unlock
Unlock the screen with voice verification.

**Request Body:**
```json
{
  "action": "unlock",
  "context": {
    "reason": "user_request",
    "authenticated": true
  },
  "audio_data": "base64_encoded_voice_sample"
}
```

**Response:**
```json
{
  "success": true,
  "action": "unlock",
  "method": "voice_biometric",
  "latency_ms": 850.2,
  "verified_speaker": "primary_user",
  "message": "Screen unlocked successfully"
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/screen/unlock \
  -H "Content-Type: application/json" \
  -d '{"action": "unlock", "context": {"reason": "user_request"}}'
```

---

## Workflow API

### Multi-Command Workflows

#### POST /workflow/execute
Execute complex multi-step workflows.

**Request Body:**
```json
{
  "command": "unlock my screen then open Safari and search for weather",
  "context": {
    "user_preferences": {
      "search_engine": "google",
      "browser": "safari"
    }
  }
}
```

**Response:**
```json
{
  "success": true,
  "workflow_id": "wf_123456",
  "steps_completed": 3,
  "total_steps": 3,
  "execution_time_ms": 2150,
  "results": [
    {
      "step": 1,
      "action": "unlock_screen",
      "status": "completed",
      "duration_ms": 850
    },
    {
      "step": 2,
      "action": "open_application",
      "target": "Safari",
      "status": "completed",
      "duration_ms": 650
    },
    {
      "step": 3,
      "action": "web_search",
      "query": "weather",
      "status": "completed",
      "duration_ms": 650
    }
  ]
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/workflow/execute \
  -H "Content-Type: application/json" \
  -d '{"command": "open Terminal and run ls command"}'
```

---

## Rust Acceleration API

### Performance Status

#### GET /rust/status
Get Rust acceleration status and performance metrics.

**Response:**
```json
{
  "enabled": true,
  "built": true,
  "components": {
    "voice_processing": true,
    "image_analysis": true,
    "pattern_matching": true
  },
  "performance_boost": {
    "voice_processing": 3.2,
    "image_analysis": 2.8,
    "overall": 2.95
  },
  "memory_savings": {
    "enabled": true,
    "reduction_percent": 35.2,
    "current_usage_mb": 145.8
  }
}
```

#### POST /rust/build
Build or rebuild Rust components.

**Request Body:**
```json
{
  "force_rebuild": false,
  "optimize_for_system": true
}
```

**Response:**
```json
{
  "success": true,
  "build_time_seconds": 45.2,
  "components_built": ["voice_engine", "vision_processor"],
  "optimizations_applied": ["apple_silicon", "vectorization"]
}
```

---

## Audio ML API

### Audio Processing

#### POST /audio/ml/predict
ML-enhanced audio processing and error recovery.

**Request Body:**
```json
{
  "audio_data": "base64_encoded_audio",
  "format": "wav",
  "sample_rate": 16000,
  "enhance_quality": true
}
```

**Response:**
```json
{
  "success": true,
  "transcription": "unlock my screen",
  "confidence": 0.94,
  "audio_quality_score": 0.87,
  "enhancements_applied": ["noise_reduction", "volume_normalization"],
  "processing_time_ms": 125.3
}
```

#### GET /audio/ml/status
Get ML audio system status.

**Response:**
```json
{
  "ml_available": true,
  "models_loaded": ["vad_model", "enhancement_model"],
  "processing_queue_size": 2,
  "average_latency_ms": 98.5,
  "error_rate": 0.02
}
```

---

## Autonomous Services API

### Service Management

#### GET /services/status
Get autonomous service orchestration status.

**Response:**
```json
{
  "orchestrator_active": true,
  "total_services": 12,
  "healthy_services": 11,
  "degraded_services": 1,
  "services": [
    {
      "name": "voice_recognition",
      "status": "healthy",
      "health_score": 0.98,
      "last_check": "2024-01-15T10:29:45Z"
    }
  ],
  "system_load": {
    "cpu_percent": 15.2,
    "memory_percent": 32.1,
    "disk_io": "low"
  }
}
```

#### POST /services/discover
Trigger service discovery.

**Response:**
```json
{
  "discovered_services": 8,
  "new_services": 2,
  "updated_services": 1,
  "discovery_time_ms": 1250
}
```

---

## Model Management API

### Model Selection

#### POST /models/select
Intelligent model selection for queries.

**Request Body:**
```json
{
  "query": "What's the weather like?",
  "intent": "information_request",
  "required_capabilities": ["web_search", "data_analysis"],
  "context": {
    "user_location": "San Francisco",
    "time_sensitive": true
  }
}
```

**Response:**
```json
{
  "selected_model": "claude-3-sonnet",
  "confidence": 0.92,
  "reasoning": "Query requires real-time data and analysis capabilities",
  "fallback_models": ["gpt-4", "claude-3-haiku"],
  "estimated_cost": 0.0015,
  "estimated_latency_ms": 850
}
```

#### GET /models/status
Get model loading and availability status.

**Response:**
```json
{
  "total_models": 5,
  "loaded_models": 3,
  "loading_models": 1,
  "failed_models": 0,
  "models": [
    {
      "name": "claude-3-sonnet",
      "status": "loaded",
      "capabilities": ["vision", "reasoning", "code"],
      "memory_usage_mb": 2048,
      "load_time_seconds": 12.5
    }
  ]
}
```

---

## WebSocket Endpoints

### Main WebSocket Connection

#### WS /ws
Main WebSocket endpoint for real-time communication.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
```

**Message Format:**
```json
{
  "type": "command",
  "data": {
    "text": "show me what's on my screen",
    "context": {}
  },
  "id": "msg_123"
}
```

**Response Format:**
```json
{
  "type": "response",
  "data": {
    "text": "I can see Safari with 3 tabs open...",
    "speak": true,
    "command_type": "vision_query"
  },
  "id": "msg_123"
}
```

### Vision WebSocket

#### WS /vision/ws
Real-time vision analysis and monitoring.

**Connection:**
```javascript
const visionWs = new WebSocket('ws://localhost:8000/vision/ws');
```

**Start Monitoring:**
```json
{
  "type": "start_monitoring",
  "interval_ms": 2000,
  "capabilities": ["screen_capture", "change_detection"]
}
```

**Monitoring Update:**
```json
{
  "type": "screen_update",
  "timestamp": "2024-01-15T10:30:00Z",
  "changes_detected": true,
  "analysis": "New notification appeared in the top-right corner",
  "confidence": 0.89
}
```

### Audio ML WebSocket

#### WS /audio/ml/stream
Real-time audio processing stream.

**Connection:**
```javascript
const audioWs = new WebSocket('ws://localhost:8000/audio/ml/stream');
```

**Audio Data:**
```json
{
  "type": "audio_chunk",
  "data": "base64_encoded_audio",
  "sequence": 1,
  "sample_rate": 16000
}
```

**Processing Result:**
```json
{
  "type": "transcription",
  "text": "hello jarvis",
  "confidence": 0.95,
  "is_wake_word": true,
  "sequence": 1
}
```

---

## Error Handling

### Standard Error Response

```json
{
  "error": {
    "code": "VISION_UNAVAILABLE",
    "message": "Vision system is currently initializing",
    "details": {
      "retry_after_seconds": 5,
      "fallback_available": true
    }
  },
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req_123456"
}
```

### Common Error Codes

| Code | Description | HTTP Status |
|------|-------------|-------------|
| `VOICE_UNAVAILABLE` | Voice system not ready | 503 |
| `VISION_UNAVAILABLE` | Vision system not ready | 503 |
| `AUTHENTICATION_FAILED` | Voice authentication failed | 401 |
| `COMMAND_NOT_UNDERSTOOD` | Could not parse command | 400 |
| `SYSTEM_OVERLOADED` | Too many concurrent requests | 429 |
| `HARDWARE_UNAVAILABLE` | Required hardware not available | 503 |

---

## Rate Limiting

### Limits by Endpoint Category

| Category | Requests per Minute | Burst Limit |
|----------|-------------------|-------------|
| Voice Commands | 60 | 10 |
| Vision Analysis | 30 | 5 |
| Screen Capture | 120 | 20 |
| Workflow Execution | 20 | 3 |
| Model Selection | 100 | 15 |

### Rate Limit Headers

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1642248600
```

---

## WebSocket Events

### Connection Events

- `connection_established` - WebSocket connected
- `authentication_required` - Authentication needed
- `capability_negotiation` - Negotiate client capabilities
- `heartbeat` - Keep-alive ping/pong

### Voice Events

- `wake_word_detected` - Wake word activation
- `voice_command_received` - Command transcribed
- `voice_command_processed` - Command executed
- `tts_started` - Text-to-speech began
- `tts_completed` - Speech synthesis finished

### Vision Events

- `screen_captured` - Screenshot taken
- `analysis_completed` - Vision analysis done
- `change_detected` - Screen content changed
- `monitoring_started` - Screen monitoring active
- `monitoring_stopped` - Screen monitoring inactive

### System Events

- `service_status_changed` - Service health update
- `model_loaded` - ML model ready
- `performance_alert` - System performance issue
- `autonomous_action` - Autonomous behavior triggered

---

## SDK Examples

### Python SDK

```python
import asyncio
import aiohttp
import json

class JARVISClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    async def send_command(self, text, context=None):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/voice/jarvis/command",
                json={"text": text, "context": context or {}}
            ) as response:
                return await response.json()
    
    async def analyze_screen(self, prompt):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/vision/analyze",
                json={"prompt": prompt}
            ) as response:
                return await response.json()

# Usage
async def main():
    client = JARVISClient()
    
    # Send voice command
    result = await client.send_command("unlock my screen")
    print(f"Command result: {result}")
    
    # Analyze screen
    analysis = await client.analyze_screen("What apps are open?")
    print(f"Screen analysis: {analysis}")

asyncio.run(main())
```

### JavaScript SDK

```javascript
class JARVISClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
        this.ws = null;
    }
    
    async sendCommand(text, context = {}) {
        const response = await fetch(`${this.baseUrl}/voice/jarvis/command`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, context })
        });
        return response.json();
    }
    
    connectWebSocket() {
        this.ws = new WebSocket(`ws://localhost:8000/ws`);
        
        this.ws.onmessage = (event) => {
            const message = JSON.parse(event.data);
            console.log('Received:', message);
        };
        
        return this.ws;
    }
    
    async analyzeScreen(prompt) {
        const response = await fetch(`${this.baseUrl}/vision/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt })
        });
        return response.json();
    }
}

// Usage
const client = new JARVISClient();

// Send command
client.sendCommand('open Safari').then(result => {
    console.log('Command result:', result);
});

// Connect WebSocket
const ws = client.connectWebSocket();
ws.onopen = () => {
    ws.send(JSON.stringify({
        type: 'command',
        data: { text: 'what do you see on my screen?' }
    }));
};
```

---

## Configuration

### Environment Variables

```bash
# API Configuration
BACKEND_PORT=8000
ANTHROPIC_API_KEY=your_claude_api_key

# Voice Configuration
VOICE_ENGINE=CoreML
TTS_ENGINE=edge_tts
WAKE_WORD_ENABLED=true

# Vision Configuration
VISION_QUALITY=high
MULTI_SPACE_ENABLED=true
SCREENSHOT_FORMAT=png

# Performance Configuration
RUST_ACCELERATION=true
MEMORY_OPTIMIZED_MODE=true
MAX_CONCURRENT_REQUESTS=50

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### Configuration File

```yaml
# config.yaml
jarvis:
  voice:
    engine: "CoreML"
    wake_word: "Hey JARVIS"
    tts_voice: "Daniel"
    language: "en-US"
  
  vision:
    quality: "high"
    multi_space: true
    monitoring_interval: 2.0
    auto_capture: true
  
  performance:
    rust_acceleration: true
    memory_limit_mb: 512
    cpu_limit_percent: 80
  
  security:
    voice_authentication: true
    require_proximity: false
    session_timeout: 3600
```

This comprehensive API documentation covers all major endpoints, WebSocket connections, error handling, rate limiting, and provides practical examples for integration with the JARVIS AI Agent system.