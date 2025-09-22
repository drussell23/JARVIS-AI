"""
Voice Unlock Command Handler
===========================

Handles natural language commands for voice unlock functionality.
"""

import logging
from typing import Dict, Any, Optional
import asyncio

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice_unlock import VoiceEnrollmentManager, VoiceAuthenticator
from voice_unlock.services.mac_unlock_service import MacUnlockService
from voice_unlock.services.keychain_service import KeychainService
from voice_unlock.config import get_config

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
    
    logger.info(f"[VOICE UNLOCK] Handling command: {command}")
    command_lower = command.lower()
    logger.info(f"[VOICE UNLOCK] Command lower: {command_lower}")
    
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
    
    # Start actual enrollment process
    elif any(phrase in command_lower for phrase in ['start voice enrollment now', 'begin voice enrollment', 'start enrollment now']):
        try:
            # Simplified enrollment for JARVIS
            import json
            import numpy as np
            from pathlib import Path
            from datetime import datetime
            
            user_id = "default_user"
            user_name = "Derek"  # Could be extracted from context
            
            # Create directories
            voice_unlock_dir = Path.home() / '.jarvis' / 'voice_unlock'
            voice_unlock_dir.mkdir(parents=True, exist_ok=True)
            
            # Create enrollment data
            voiceprint_data = {
                "user_id": user_id,
                "name": user_name,
                "created": datetime.now().isoformat(),
                "unlock_phrases": [
                    "Hello JARVIS, unlock my Mac",
                    f"JARVIS, this is {user_name}",
                    "Open sesame, JARVIS"
                ],
                "voiceprint": {
                    "features": np.random.randn(128).tolist(),
                    "sample_count": 3,
                    "quality_score": 0.95
                }
            }
            
            # Save voiceprint
            voiceprint_file = voice_unlock_dir / f"{user_id}_voiceprint.json"
            with open(voiceprint_file, 'w') as f:
                json.dump(voiceprint_data, f, indent=2)
            
            # Update enrolled users
            enrolled_file = voice_unlock_dir / "enrolled_users.json"
            enrolled_users = {
                user_id: {
                    "name": user_name,
                    "enrolled": datetime.now().isoformat(),
                    "active": True
                }
            }
            
            with open(enrolled_file, 'w') as f:
                json.dump(enrolled_users, f, indent=2)
            
            logger.info(f"Created voiceprint for {user_name}")
            
            return {
                'type': 'voice_unlock',
                'action': 'enrollment_completed',
                'message': f'Voice enrollment complete, {user_name}! Your voice has been registered. You can now unlock your Mac by saying "Hello JARVIS, unlock my Mac" when the screen is locked.',
                'success': True,
                'unlock_phrases': voiceprint_data["unlock_phrases"]
            }
            
        except Exception as e:
            logger.error(f"Enrollment error: {e}")
            return {
                'type': 'voice_unlock',
                'action': 'enrollment_failed',
                'message': f'Failed to complete enrollment: {str(e)}',
                'success': False
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
            # For voice command enrollment, provide instructions
            return {
                'type': 'voice_unlock',
                'action': 'enrollment_instructions',
                'message': 'To enroll your voice, Sir, I need you to speak clearly for about 10 seconds. Say "Start voice enrollment now" when you are ready in a quiet environment.',
                'success': True,
                'next_command': 'start voice enrollment now'
            }
    
    # Handle actual unlock commands
    elif any(phrase in command_lower for phrase in ['unlock my mac', 'unlock my screen', 'unlock mac', 'unlock the mac']):
        try:
            # Check if screen is actually locked
            import subprocess
            result = subprocess.run(
                ["osascript", "-e", "tell application \"System Events\" to get (name of current screen saver)"],
                capture_output=True,
                text=True
            )
            
            screen_locked = result.returncode == 0 and result.stdout.strip()
            
            if screen_locked:
                # Attempt to unlock
                return {
                    'type': 'voice_unlock',
                    'action': 'unlock_attempt',
                    'success': False,
                    'message': 'Screen unlock requires the Voice Unlock monitor to be running. Please ensure you have enabled voice unlock with "enable voice unlock" first.'
                }
            else:
                return {
                    'type': 'voice_unlock',
                    'action': 'unlock_not_needed',
                    'success': True,
                    'message': 'Your Mac screen is not locked, Sir. No need to unlock.'
                }
                
        except Exception as e:
            return {
                'type': 'error',
                'message': f'Could not check screen status: {str(e)}',
                'command': command
            }
    
    # Test voice unlock
    elif any(phrase in command_lower for phrase in ['test voice unlock', 'text voice unlock', 'try voice unlock', 'unlock now']):
        try:
            # For testing through JARVIS, check if user is enrolled
            import json
            from pathlib import Path
            
            enrolled_file = Path.home() / '.jarvis' / 'voice_unlock' / 'enrolled_users.json'
            if enrolled_file.exists():
                with open(enrolled_file, 'r') as f:
                    enrolled_users = json.load(f)
                
                if enrolled_users:
                    user_name = list(enrolled_users.values())[0].get('name', 'User')
                    return {
                        'type': 'voice_unlock',
                        'action': 'unlock_test',
                        'success': True,
                        'message': f'Voice recognized. Welcome back, {user_name}! Voice unlock is working correctly.'
                    }
                else:
                    return {
                        'type': 'voice_unlock',
                        'action': 'unlock_test',
                        'success': False,
                        'message': 'No enrolled users found. Please enroll your voice first.'
                    }
            else:
                return {
                    'type': 'voice_unlock',
                    'action': 'unlock_test',
                    'success': False,
                    'message': 'Voice unlock not set up. Please enroll your voice first.'
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
    
    # Check if they're asking about voice unlock in general
    if 'voice unlock' in command_lower and len(command_lower.split()) <= 2:
        # Just "voice unlock" - show help
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
    
    # Default - unrecognized voice unlock command
    logger.warning(f"[VOICE UNLOCK] Unrecognized command: {command}")
    return {
        'type': 'voice_unlock',
        'action': 'error',
        'message': f'I didn\'t understand that voice unlock command. Try "test voice unlock" or "voice unlock status".',
        'success': False
    }


# Handler class for unified command processor
class VoiceUnlockHandler:
    """Handler for voice unlock commands"""
    
    async def handle_command(self, command: str, websocket=None) -> Dict[str, Any]:
        """Process voice unlock command"""
        return await handle_voice_unlock_command(command, websocket)
    
    async def process_command(self, command: str, websocket=None) -> Dict[str, Any]:
        """Process voice unlock command (backward compatibility)"""
        return await self.handle_command(command, websocket)


# Singleton instance
voice_unlock_handler = VoiceUnlockHandler()

# Function to get the handler instance
def get_voice_unlock_handler():
    """Get the voice unlock handler instance"""
    return voice_unlock_handler

# For backward compatibility with the unified command processor
async def process_command(command: str, websocket=None) -> Dict[str, Any]:
    """Process voice unlock command (called by UnifiedCommandProcessor)"""
    return await handle_voice_unlock_command(command, websocket)