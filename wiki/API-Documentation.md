# API Documentation

Complete API reference for JARVIS AI Agent including REST endpoints, WebSocket API, Voice commands, and Intelligence system APIs.

---

## Table of Contents

1. [Overview](#overview)
2. [REST API](#rest-api)
3. [WebSocket API](#websocket-api)
4. [Voice Commands](#voice-commands)
5. [Intelligence System APIs](#intelligence-system-apis)
6. [Error Handling](#error-handling)
7. [Rate Limiting](#rate-limiting)
8. [Examples](#examples)

---

## Overview

### Base URLs

**Local (Development):**
```
http://localhost:8010
ws://localhost:8010/ws
```

**GCP Cloud (Production):**
```
http://34.10.137.70:8010
ws://34.10.137.70:8010/ws
```

### Authentication

Most endpoints require no authentication. Voice unlock and biometric features use voice authentication when configured.

### Response Format

All API responses follow this structure:

```json
{
  "success": true,
  "data": { },
  "error": null,
  "timestamp": "2025-10-30T12:00:00Z",
  "execution_time_ms": 150
}
```

---

## REST API

### Health & Status

#### GET /health

Check system health and component status.

**Response:**
```json
{
  "status": "healthy",
  "version": "17.4.0",
  "components": {
    "uae": "running",
    "sai": "running",
    "cai": "running",
    "database": "connected",
    "voice": "ready",
    "cloud_sql_proxy": "running",
    "gcp_vm": "available"
  },
  "memory": {
    "local_usage_percent": 65,
    "gcp_shift_recommended": false
  },
  "uptime_seconds": 3600
}
```

**Example:**
```bash
curl http://localhost:8010/health
```

---

### Voice API

#### POST /voice/jarvis/command

Process natural language voice command.

**Request:**
```json
{
  "text": "unlock my screen",
  "context": {
    "user_id": "derek",
    "session_id": "session_123"
  }
}
```

**Response:**
```json
{
  "success": true,
  "response": "Of course, Derek. Unlocking your screen now.",
  "command_type": "unlock",
  "executed": true,
  "speaker_verified": true,
  "confidence": 0.95,
  "execution_time_ms": 2800
}
```

**Example:**
```bash
curl -X POST http://localhost:8010/voice/jarvis/command \
  -H "Content-Type: application/json" \
  -d '{"text": "what time is it?"}'
```

#### POST /voice/detect-coreml

Hardware-accelerated voice detection (Apple Neural Engine).

**Request:**
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
  "speaker_confidence": 0.95,
  "speaker_name": "Derek",
  "inference_time_ms": 8.2,
  "model": "ecapa-tdnn"
}
```

#### GET /voice/jarvis/status

Get voice system status.

**Response:**
```json
{
  "active": true,
  "voice_engine": "SpeechBrain",
  "tts_engine": "gTTS",
  "wake_word_enabled": true,
  "speaker_profiles": 2,
  "last_command": "2025-10-30T11:30:00Z"
}
```

---

### Vision API

#### POST /vision/analyze

Analyze screen content with Claude Vision API.

**Request:**
```json
{
  "prompt": "What applications are currently open?",
  "multi_space": true,
  "include_windows": true,
  "use_cloud": true
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
      "tabs": 3,
      "active": true
    },
    {
      "type": "application",
      "name": "Terminal",
      "windows": 1,
      "script": "start_system.py"
    }
  ],
  "desktop_space": 3,
  "displays": 2,
  "processed_on": "cloud",
  "execution_time_ms": 1800
}
```

**Example:**
```bash
curl -X POST http://localhost:8010/vision/analyze \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Describe what you see"}'
```

#### GET /vision/capture

Capture current screen(s).

**Query Parameters:**
- `display` (optional): Display number (default: all)
- `space` (optional): Desktop space number

**Response:**
```json
{
  "success": true,
  "images": [
    {
      "display": 1,
      "space": 3,
      "data": "base64_encoded_image",
      "resolution": "3024x1964",
      "timestamp": "2025-10-30T12:00:00Z"
    }
  ]
}
```

---

### Hybrid Orchestrator API

#### POST /hybrid/execute

Execute command with intelligent routing (local vs cloud).

**Request:**
```json
{
  "command": "analyze my screen",
  "command_type": "vision_analyze",
  "metadata": {
    "client_id": "web_001",
    "context": "user_initiated"
  }
}
```

**Response:**
```json
{
  "success": true,
  "result": "Analysis complete",
  "routed_to": "cloud",
  "reason": "memory_pressure_85_percent",
  "vm_created": true,
  "vm_ip": "34.xxx.xxx.xxx",
  "execution_time_ms": 3200,
  "cost_usd": 0.0001
}
```

#### GET /hybrid/status

Get hybrid orchestrator status.

**Response:**
```json
{
  "local": {
    "ram_percent": 78,
    "active_components": ["uae_light", "cai_light", "voice_basic"],
    "status": "high_pressure"
  },
  "cloud": {
    "vm_running": true,
    "vm_ip": "34.xxx.xxx.xxx",
    "ram_percent": 45,
    "active_components": ["vision_full", "chatbot_full", "ml_models"],
    "cost_today_usd": 0.12
  },
  "routing_mode": "hybrid",
  "auto_scaling": true
}
```

---

### Intelligence APIs

#### POST /intelligence/uae/analyze

Get UAE context analysis.

**Request:**
```json
{
  "query": "what am I working on?",
  "depth": "full"
}
```

**Response:**
```json
{
  "context": {
    "active_application": "Terminal",
    "current_task": "Python development",
    "recent_commands": ["git commit", "python start_system.py"],
    "desktop_space": 3,
    "time_of_day": "afternoon",
    "user_state": "focused"
  },
  "confidence": 0.92,
  "processed_by": "uae_full"
}
```

#### POST /intelligence/cai/predict_intent

Predict user intent from command.

**Request:**
```json
{
  "command": "unlock my screen"
}
```

**Response:**
```json
{
  "intent": "screen_unlock",
  "confidence": 0.95,
  "requires_auth": true,
  "suggested_flow": "voice_biometric",
  "alternative_intents": [
    {"intent": "password_entry", "confidence": 0.05}
  ]
}
```

#### GET /intelligence/sai/health

Get SAI self-healing status.

**Response:**
```json
{
  "status": "healthy",
  "self_healing_active": true,
  "recent_recoveries": 3,
  "patterns_learned": 127,
  "optimization_suggestions": [
    "shift_vision_to_cloud_at_70_percent",
    "cache_common_tts_phrases"
  ]
}
```

---

### Database API

#### GET /database/patterns

Get learned command patterns.

**Response:**
```json
{
  "patterns": [
    {
      "command": "unlock my screen",
      "count": 45,
      "success_rate": 0.98,
      "avg_execution_time_ms": 2800,
      "last_used": "2025-10-30T11:30:00Z"
    }
  ],
  "total_patterns": 127
}
```

#### GET /database/stats

Get database statistics.

**Response:**
```json
{
  "local": {
    "db_size_mb": 45.3,
    "total_commands": 1234,
    "cache_hit_rate": 0.67
  },
  "cloud": {
    "db_size_mb": 234.7,
    "total_commands": 8976,
    "sync_status": "synced",
    "last_sync": "2025-10-30T10:00:00Z"
  }
}
```

---

## WebSocket API

### Connection

Connect to WebSocket for real-time communication:

```javascript
const ws = new WebSocket('ws://localhost:8010/ws');

ws.onopen = () => {
  console.log('Connected to JARVIS');
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  console.log('Received:', message);
};
```

### Message Types

#### Voice Command

**Client → Server:**
```json
{
  "type": "voice_command",
  "text": "unlock my screen",
  "audio_data": "base64_encoded_audio",
  "metadata": {
    "client_id": "web_001"
  }
}
```

**Server → Client:**
```json
{
  "type": "voice_response",
  "success": true,
  "response": "Unlocking your screen now.",
  "executed": true,
  "execution_time_ms": 2800
}
```

#### System Status

**Server → Client (periodic):**
```json
{
  "type": "status_update",
  "memory": {
    "local_percent": 78,
    "cloud_percent": 45
  },
  "components": {
    "uae": "running",
    "sai": "running",
    "cai": "running"
  },
  "timestamp": "2025-10-30T12:00:00Z"
}
```

#### Component Lifecycle

**Server → Client:**
```json
{
  "type": "component_lifecycle",
  "event": "component_offloaded",
  "component": "vision_full",
  "from": "local",
  "to": "cloud",
  "reason": "memory_pressure_85_percent"
}
```

---

## Voice Commands

### Natural Language Commands

JARVIS supports natural language voice commands:

#### Screen Unlock
```
"Hey JARVIS, unlock my screen"
"JARVIS, unlock"
"Unlock my Mac"
```

#### Vision Analysis
```
"What's on my screen?"
"Analyze my screen"
"What applications are open?"
"How many browser tabs do I have?"
```

#### System Control
```
"Open Safari"
"Close all windows"
"Switch to desktop 3"
"Show me all displays"
```

#### Information Queries
```
"What time is it?"
"What's the weather?"
"Tell me about..."
```

#### Conversation
```
"Hey JARVIS, let's chat"
"What were we talking about?"
"Remember this for later"
```

### Voice Command Response Format

All voice commands return:

```json
{
  "success": true,
  "response": "Natural language response",
  "command_type": "category",
  "speaker_verified": true,
  "confidence": 0.95,
  "executed": true,
  "execution_time_ms": 1500
}
```

---

## Intelligence System APIs

### UAE (Unified Awareness Engine)

**Context Aggregation:**
```python
from intelligence.unified_awareness_engine import get_uae

uae = get_uae()
context = await uae.get_current_context()
# Returns: {
#   'active_apps': [...],
#   'desktop_space': 3,
#   'user_state': 'focused',
#   'recent_activity': [...]
# }
```

### SAI (Self-Aware Intelligence)

**Self-Healing:**
```python
from intelligence.self_aware_intelligence import get_sai

sai = get_sai()
heal_result = await sai.attempt_self_heal(
    error=Exception("API rate limit"),
    context={"retries": 0}
)
# Returns: {
#   'success': True,
#   'fix': 'implement_exponential_backoff',
#   'confidence': 0.92
# }
```

### CAI (Context Awareness Intelligence)

**Intent Prediction:**
```python
from intelligence.context_awareness_intelligence import get_cai

cai = get_cai()
intent = await cai.predict_intent("unlock my screen")
# Returns: {
#   'intent': 'screen_unlock',
#   'confidence': 0.95,
#   'requires_auth': True
# }
```

### Learning Database

**Pattern Learning:**
```python
from database.learning_database import JARVISLearningDatabase

db = JARVISLearningDatabase()
patterns = db.get_command_patterns(limit=10)
# Returns: List of most common command patterns
```

---

## Error Handling

### Error Response Format

```json
{
  "success": false,
  "error": {
    "code": "VOICE_AUTH_FAILED",
    "message": "Voice authentication failed. Confidence: 32%",
    "details": {
      "required_confidence": 0.75,
      "actual_confidence": 0.32,
      "speaker_detected": false
    }
  },
  "timestamp": "2025-10-30T12:00:00Z"
}
```

### Common Error Codes

| Code | Description | HTTP Status |
|------|-------------|-------------|
| `VOICE_AUTH_FAILED` | Voice biometric authentication failed | 401 |
| `CLOUD_SQL_UNAVAILABLE` | Cloud SQL connection failed | 503 |
| `GCP_VM_QUOTA_EXCEEDED` | Maximum VMs reached | 429 |
| `MEMORY_PRESSURE_CRITICAL` | Out of memory | 507 |
| `COMPONENT_NOT_FOUND` | Requested component unavailable | 404 |
| `INVALID_REQUEST` | Malformed request | 400 |
| `RATE_LIMIT_EXCEEDED` | Too many requests | 429 |

---

## Rate Limiting

### Limits

- **Voice Commands:** 60 per minute
- **Vision Analysis:** 10 per minute (cloud), 30 per minute (local)
- **WebSocket Messages:** 120 per minute
- **General API:** 1000 per hour

### Rate Limit Response

```json
{
  "error": "RATE_LIMIT_EXCEEDED",
  "message": "Rate limit exceeded. Try again in 45 seconds.",
  "retry_after_seconds": 45,
  "limit": 60,
  "window_seconds": 60
}
```

---

## Examples

### Complete Voice Unlock Flow

```python
import requests
import json

# 1. Check voice system status
status = requests.get('http://localhost:8010/voice/jarvis/status')
print(f"Voice system: {status.json()['active']}")

# 2. Send voice command
command = {
    "text": "unlock my screen",
    "context": {"user_id": "derek"}
}
response = requests.post(
    'http://localhost:8010/voice/jarvis/command',
    json=command
)

result = response.json()
if result['success']:
    print(f"✅ {result['response']}")
    print(f"Speaker verified: {result['speaker_verified']}")
    print(f"Executed in: {result['execution_time_ms']}ms")
else:
    print(f"❌ Error: {result['error']}")
```

### WebSocket Real-Time Communication

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8010/ws');

ws.onopen = () => {
  console.log('✅ Connected to JARVIS');

  // Send voice command
  ws.send(JSON.stringify({
    type: 'voice_command',
    text: "what's on my screen?",
    metadata: { client_id: 'web_001' }
  }));
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);

  switch (message.type) {
    case 'voice_response':
      console.log(`JARVIS: ${message.response}`);
      break;

    case 'status_update':
      console.log(`Memory: ${message.memory.local_percent}%`);
      break;

    case 'component_lifecycle':
      console.log(`${message.component} → ${message.to}`);
      break;
  }
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};
```

### Hybrid Routing Example

```python
import asyncio
from core.hybrid_orchestrator import get_orchestrator

async def main():
    orchestrator = get_orchestrator()

    # Execute command with intelligent routing
    result = await orchestrator.execute_command(
        command="analyze my screen in detail",
        command_type="vision_analyze",
        metadata={
            "client_id": "python_001",
            "priority": "high"
        }
    )

    print(f"Routed to: {result['routed_to']}")
    print(f"Reason: {result['reason']}")
    if result.get('vm_created'):
        print(f"VM IP: {result['vm_ip']}")
    print(f"Cost: ${result['cost_usd']:.4f}")
    print(f"Time: {result['execution_time_ms']}ms")

asyncio.run(main())
```

---

**Related Documentation:**
- [Architecture & Design](Architecture-&-Design.md) - System architecture
- [Setup & Installation](Setup-&-Installation.md) - Setup guide
- [Troubleshooting Guide](Troubleshooting-Guide.md) - Common issues

---

**Last Updated:** 2025-10-30
**Version:** 17.4.0
