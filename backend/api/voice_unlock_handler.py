"""
Voice Unlock Command Handler
===========================

Handles natural language commands for voice unlock functionality.
"""

import logging
from typing import Dict, Any, Optional
import asyncio

from ..voice_unlock import VoiceEnrollmentManager, VoiceAuthenticator
from ..voice_unlock.services.mac_unlock_service import MacUnlockService
from ..voice_unlock.services.keychain_service import KeychainService
from ..voice_unlock.config import get_config

logger = logging.getLogger(__name__)

# Global instances (will be initialized from main app)
enrollment_manager = None
authenticator = None
unlock_service = None
keychain_service = None


def initialize_handlers():
    """Initialize voice unlock handlers"""
    global enrollment_manager, authenticator, unlock_service, keychain_service
    
    try:
        enrollment_manager = VoiceEnrollmentManager()
        authenticator = VoiceAuthenticator()
        keychain_service = KeychainService()
        unlock_service = MacUnlockService(authenticator)
        logger.info("Voice unlock handlers initialized")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize voice unlock handlers: {e}")
        return False


async def handle_voice_unlock_command(command: str, websocket=None) -> Dict[str, Any]:
    """Handle voice unlock related commands"""
    
    command_lower = command.lower()
    
    # Initialize if needed
    if not unlock_service:
        if not initialize_handlers():
            return {
                'type': 'error',
                'message': 'Voice unlock system not available. Please install dependencies.',
                'command': command
            }
    
    # Enable voice unlock monitoring
    if any(phrase in command_lower for phrase in ['enable voice unlock', 'start voice unlock', 'activate voice unlock']):
        try:
            await unlock_service.start_service()
            return {
                'type': 'voice_unlock',
                'action': 'enabled',
                'message': 'Voice unlock monitoring enabled, Sir. Your Mac will now respond to your voice.',
                'status': unlock_service.get_service_status()
            }
        except Exception as e:
            return {
                'type': 'error',
                'message': f'Failed to enable voice unlock: {str(e)}',
                'command': command
            }
    
    # Disable voice unlock monitoring
    elif any(phrase in command_lower for phrase in ['disable voice unlock', 'stop voice unlock', 'deactivate voice unlock']):
        try:
            await unlock_service.stop_service()
            return {
                'type': 'voice_unlock',
                'action': 'disabled',
                'message': 'Voice unlock monitoring disabled, Sir.',
                'status': unlock_service.get_service_status()
            }
        except Exception as e:
            return {
                'type': 'error',
                'message': f'Failed to disable voice unlock: {str(e)}',
                'command': command
            }
    
    # Start enrollment
    elif any(phrase in command_lower for phrase in ['enroll my voice', 'set up voice', 'voice enrollment', 'register my voice']):
        # Extract user name if provided
        user_id = "default_user"  # Could extract from command
        
        # For WebSocket, we'll return enrollment instructions
        if websocket:
            session_id = enrollment_manager.start_enrollment(user_id)
            return {
                'type': 'voice_unlock',
                'action': 'enrollment_started',
                'message': 'Voice enrollment started, Sir. Please follow the instructions.',
                'session_id': session_id,
                'websocket_url': f'/api/voice-unlock/enrollment/ws/{session_id}',
                'instructions': 'Connect to the WebSocket URL to complete enrollment with real-time feedback.'
            }
        else:
            # For non-WebSocket, start auto-enrollment
            session_id = enrollment_manager.start_enrollment(user_id)
            
            # Create progress callback
            enrollment_progress = {'completed': False, 'progress': 0}
            
            def progress_callback(session, progress):
                enrollment_progress['progress'] = progress
                logger.info(f"Enrollment progress: {progress * 100:.0f}%")
            
            # Run enrollment
            success, message = await enrollment_manager.auto_enroll(user_id, progress_callback)
            
            return {
                'type': 'voice_unlock',
                'action': 'enrollment_completed' if success else 'enrollment_failed',
                'message': message,
                'success': success
            }
    
    # Test voice unlock
    elif any(phrase in command_lower for phrase in ['test voice unlock', 'try voice unlock', 'unlock now']):
        try:
            success, message = await unlock_service.manual_unlock()
            return {
                'type': 'voice_unlock',
                'action': 'unlock_test',
                'success': success,
                'message': message
            }
        except Exception as e:
            return {
                'type': 'error',
                'message': f'Voice unlock test failed: {str(e)}',
                'command': command
            }
    
    # Check status
    elif any(phrase in command_lower for phrase in ['voice unlock status', 'unlock status', 'voice security status']):
        try:
            status = unlock_service.get_service_status()
            enrolled_users = keychain_service.list_voiceprints()
            
            return {
                'type': 'voice_unlock',
                'action': 'status',
                'message': f'Voice unlock is {"enabled" if status["enabled"] else "disabled"}, Sir. {len(enrolled_users)} user(s) enrolled.',
                'status': status,
                'enrolled_users': len(enrolled_users)
            }
        except Exception as e:
            return {
                'type': 'error',
                'message': f'Failed to get status: {str(e)}',
                'command': command
            }
    
    # Delete voiceprint
    elif any(phrase in command_lower for phrase in ['delete my voiceprint', 'remove my voice', 'clear voice data']):
        try:
            # Extract user or use default
            user_id = "default_user"
            success = keychain_service.delete_voiceprint(user_id)
            
            if success:
                return {
                    'type': 'voice_unlock',
                    'action': 'voiceprint_deleted',
                    'message': 'Your voiceprint has been deleted, Sir.',
                    'success': True
                }
            else:
                return {
                    'type': 'error',
                    'message': 'No voiceprint found to delete.',
                    'command': command
                }
        except Exception as e:
            return {
                'type': 'error',
                'message': f'Failed to delete voiceprint: {str(e)}',
                'command': command
            }
    
    # Test audio system
    elif any(phrase in command_lower for phrase in ['test audio', 'test microphone', 'audio test']):
        try:
            result = await unlock_service.test_audio_system()
            
            if result['success'] and result.get('audio_detected'):
                return {
                    'type': 'voice_unlock',
                    'action': 'audio_test',
                    'message': f'Audio system working, Sir. Detected {result["duration"]:.1f} seconds of audio with noise level {result["noise_level"]:.3f}.',
                    'result': result
                }
            else:
                return {
                    'type': 'error',
                    'message': 'No audio detected. Please check your microphone.',
                    'result': result
                }
        except Exception as e:
            return {
                'type': 'error',
                'message': f'Audio test failed: {str(e)}',
                'command': command
            }
    
    # Default response
    return {
        'type': 'voice_unlock',
        'action': 'help',
        'message': 'Voice unlock commands: "enable voice unlock", "enroll my voice", "test voice unlock", "voice unlock status", "disable voice unlock"',
        'available_commands': [
            'enable voice unlock',
            'disable voice unlock',
            'enroll my voice',
            'test voice unlock',
            'voice unlock status',
            'delete my voiceprint',
            'test audio'
        ]
    }


# For backward compatibility with the unified command processor
async def process_command(command: str, websocket=None) -> Dict[str, Any]:
    """Process voice unlock command (called by UnifiedCommandProcessor)"""
    return await handle_voice_unlock_command(command, websocket)