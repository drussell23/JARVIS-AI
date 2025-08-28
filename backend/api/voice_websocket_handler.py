"""
Voice WebSocket Handler
Handles voice-related WebSocket messages for real-time audio processing
"""

import asyncio
import json
import logging
import base64
from typing import Dict, Any, Optional
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports
_voice_system = None
_jarvis_voice = None


def get_voice_system():
    """Get voice system instance lazily"""
    global _voice_system
    if _voice_system is None:
        try:
            from voice.ml_enhanced_voice_system import MLEnhancedVoiceSystem
            _voice_system = MLEnhancedVoiceSystem()
        except ImportError:
            logger.warning("ML Enhanced Voice System not available")
    return _voice_system


def get_jarvis_voice():
    """Get JARVIS voice instance lazily"""
    global _jarvis_voice
    if _jarvis_voice is None:
        try:
            from voice.jarvis_agent_voice import JARVISAgentVoice
            _jarvis_voice = JARVISAgentVoice()
        except ImportError:
            logger.warning("JARVIS Agent Voice not available")
    return _jarvis_voice


# Active audio streams
audio_streams: Dict[str, Dict[str, Any]] = {}


async def handle_websocket_message(message: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Main WebSocket message handler for voice messages
    """
    try:
        message_type = message.get('type', '')
        
        handlers = {
            'start_listening': handle_start_listening,
            'stop_listening': handle_stop_listening,
            'process_audio': handle_process_audio,
            'voice_command': handle_voice_command,
            'set_voice': handle_set_voice,
            'get_voices': handle_get_voices,
            'start_stream': handle_start_stream,
            'audio_chunk': handle_audio_chunk,
            'end_stream': handle_end_stream
        }
        
        handler = handlers.get(message_type, handle_unknown)
        result = await handler(message, context)
        
        # Add metadata
        result['timestamp'] = datetime.now().isoformat()
        if 'correlation_id' in message:
            result['correlation_id'] = message['correlation_id']
            
        return result
        
    except Exception as e:
        logger.error(f"Error handling voice message: {e}", exc_info=True)
        return {
            'type': 'error',
            'error': str(e),
            'original_message_type': message.get('type', 'unknown')
        }


async def handle_start_listening(message: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Start listening for voice input"""
    voice_system = get_voice_system()
    
    if not voice_system:
        return {
            'type': 'listening_status',
            'listening': False,
            'error': 'Voice system not available'
        }
    
    try:
        # Start listening
        await voice_system.start_listening()
        
        return {
            'type': 'listening_status',
            'listening': True,
            'message': 'Listening for voice input...'
        }
        
    except Exception as e:
        logger.error(f"Start listening error: {e}")
        return {
            'type': 'listening_status',
            'listening': False,
            'error': str(e)
        }


async def handle_stop_listening(message: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Stop listening for voice input"""
    voice_system = get_voice_system()
    
    if voice_system:
        try:
            await voice_system.stop_listening()
        except Exception as e:
            logger.error(f"Stop listening error: {e}")
    
    return {
        'type': 'listening_status',
        'listening': False,
        'message': 'Stopped listening'
    }


async def handle_process_audio(message: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Process audio data"""
    voice_system = get_voice_system()
    
    if not voice_system:
        return {
            'type': 'audio_result',
            'success': False,
            'error': 'Voice system not available'
        }
    
    try:
        # Get audio data (base64 encoded)
        audio_data = message.get('audio', '')
        
        if not audio_data:
            return {
                'type': 'audio_result',
                'success': False,
                'error': 'No audio data provided'
            }
        
        # Decode base64 audio
        audio_bytes = base64.b64decode(audio_data)
        
        # Process audio
        result = await voice_system.process_audio(audio_bytes)
        
        return {
            'type': 'audio_result',
            'success': True,
            'transcription': result.get('transcription', ''),
            'confidence': result.get('confidence', 0.0),
            'language': result.get('language', 'en')
        }
        
    except Exception as e:
        logger.error(f"Process audio error: {e}")
        return {
            'type': 'audio_result',
            'success': False,
            'error': str(e)
        }


async def handle_voice_command(message: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Handle voice command"""
    jarvis = get_jarvis_voice()
    
    if not jarvis:
        return {
            'type': 'command_result',
            'success': False,
            'error': 'JARVIS voice system not available'
        }
    
    try:
        command = message.get('command', '')
        
        # Process command through JARVIS
        response = await jarvis.process_voice_input(command)
        
        return {
            'type': 'command_result',
            'success': True,
            'command': command,
            'response': response,
            'action_taken': jarvis.last_action if hasattr(jarvis, 'last_action') else None
        }
        
    except Exception as e:
        logger.error(f"Voice command error: {e}")
        return {
            'type': 'command_result',
            'success': False,
            'error': str(e)
        }


async def handle_set_voice(message: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Set voice settings"""
    voice_system = get_voice_system()
    
    if not voice_system:
        return {
            'type': 'voice_settings',
            'success': False,
            'error': 'Voice system not available'
        }
    
    try:
        settings = message.get('settings', {})
        
        # Apply settings
        if 'voice' in settings:
            voice_system.set_voice(settings['voice'])
        if 'speed' in settings:
            voice_system.set_speed(settings['speed'])
        if 'pitch' in settings:
            voice_system.set_pitch(settings['pitch'])
        
        return {
            'type': 'voice_settings',
            'success': True,
            'current_settings': {
                'voice': getattr(voice_system, 'current_voice', 'default'),
                'speed': getattr(voice_system, 'current_speed', 1.0),
                'pitch': getattr(voice_system, 'current_pitch', 1.0)
            }
        }
        
    except Exception as e:
        logger.error(f"Set voice error: {e}")
        return {
            'type': 'voice_settings',
            'success': False,
            'error': str(e)
        }


async def handle_get_voices(message: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Get available voices"""
    voice_system = get_voice_system()
    
    if not voice_system:
        return {
            'type': 'available_voices',
            'voices': [],
            'error': 'Voice system not available'
        }
    
    try:
        voices = voice_system.get_available_voices() if hasattr(voice_system, 'get_available_voices') else []
        
        return {
            'type': 'available_voices',
            'voices': voices,
            'current_voice': getattr(voice_system, 'current_voice', 'default')
        }
        
    except Exception as e:
        logger.error(f"Get voices error: {e}")
        return {
            'type': 'available_voices',
            'voices': [],
            'error': str(e)
        }


async def handle_start_stream(message: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Start audio streaming session"""
    stream_id = message.get('stream_id', f"stream_{datetime.now().timestamp()}")
    
    # Initialize stream
    audio_streams[stream_id] = {
        'chunks': [],
        'started': datetime.now(),
        'sample_rate': message.get('sample_rate', 16000),
        'channels': message.get('channels', 1),
        'format': message.get('format', 'int16')
    }
    
    return {
        'type': 'stream_started',
        'stream_id': stream_id,
        'success': True
    }


async def handle_audio_chunk(message: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Handle audio chunk in stream"""
    stream_id = message.get('stream_id')
    
    if not stream_id or stream_id not in audio_streams:
        return {
            'type': 'chunk_result',
            'success': False,
            'error': 'Invalid stream ID'
        }
    
    try:
        # Add chunk to stream
        chunk_data = base64.b64decode(message.get('chunk', ''))
        audio_streams[stream_id]['chunks'].append(chunk_data)
        
        # Optional: Process partial results
        if message.get('process_partial', False):
            voice_system = get_voice_system()
            if voice_system:
                # Combine chunks for processing
                audio_data = b''.join(audio_streams[stream_id]['chunks'])
                partial_result = await voice_system.process_audio(audio_data, partial=True)
                
                return {
                    'type': 'chunk_result',
                    'success': True,
                    'partial_transcription': partial_result.get('transcription', '')
                }
        
        return {
            'type': 'chunk_result',
            'success': True,
            'chunks_received': len(audio_streams[stream_id]['chunks'])
        }
        
    except Exception as e:
        logger.error(f"Audio chunk error: {e}")
        return {
            'type': 'chunk_result',
            'success': False,
            'error': str(e)
        }


async def handle_end_stream(message: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
    """End audio streaming session and process complete audio"""
    stream_id = message.get('stream_id')
    
    if not stream_id or stream_id not in audio_streams:
        return {
            'type': 'stream_result',
            'success': False,
            'error': 'Invalid stream ID'
        }
    
    try:
        voice_system = get_voice_system()
        
        if not voice_system:
            # Clean up stream
            del audio_streams[stream_id]
            return {
                'type': 'stream_result',
                'success': False,
                'error': 'Voice system not available'
            }
        
        # Combine all chunks
        stream_data = audio_streams[stream_id]
        audio_data = b''.join(stream_data['chunks'])
        
        # Process complete audio
        result = await voice_system.process_audio(audio_data)
        
        # Clean up stream
        del audio_streams[stream_id]
        
        return {
            'type': 'stream_result',
            'success': True,
            'transcription': result.get('transcription', ''),
            'confidence': result.get('confidence', 0.0),
            'duration': (datetime.now() - stream_data['started']).total_seconds()
        }
        
    except Exception as e:
        logger.error(f"End stream error: {e}")
        # Clean up stream on error
        if stream_id in audio_streams:
            del audio_streams[stream_id]
        return {
            'type': 'stream_result',
            'success': False,
            'error': str(e)
        }


async def handle_unknown(message: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Handle unknown message types"""
    logger.warning(f"Unknown voice message type: {message.get('type')}")
    return {
        'type': 'error',
        'error': f"Unknown message type: {message.get('type')}",
        'supported_types': [
            'start_listening', 'stop_listening', 'process_audio',
            'voice_command', 'set_voice', 'get_voices',
            'start_stream', 'audio_chunk', 'end_stream'
        ],
        'success': False
    }


# Export for WebSocket integration
__all__ = ['handle_websocket_message', 'audio_streams']