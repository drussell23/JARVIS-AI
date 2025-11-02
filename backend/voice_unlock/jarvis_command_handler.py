"""
JARVIS Voice Command Handler for Voice Unlock
============================================

Handles JARVIS-specific voice commands for the unlock system.
"""

import re
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class VoiceCommand:
    """Parsed voice command"""
    command_type: str
    user_name: Optional[str]
    parameters: Dict[str, Any]
    confidence: float
    raw_text: str


class JARVISCommandHandler:
    """
    Handles JARVIS voice commands for authentication
    """
    
    # Command patterns
    UNLOCK_PATTERNS = [
        (r"(?:hey |hi |hello )?jarvis[,.]? (?:please )?unlock (?:my |the )?mac", None),
        (r"jarvis[,.]? (?:this is |it'?s) (\w+)", 'user'),
        (r"(?:hey )?jarvis[,.]? (\w+) (?:is )?here", 'user'),
        (r"jarvis[,.]? authenticate (?:me|user)?\s*(\w+)?", 'user'),
        (r"open sesame[,.]? jarvis", None),
    ]
    
    LOCK_PATTERNS = [
        (r"(?:hey )?jarvis[,.]? (?:please )?lock (?:my |the )?(?:mac|computer|system)", None),
        (r"jarvis[,.]? (?:activate |enable )?(?:security|lock)", None),
        (r"jarvis[,.]? (?:i'?m |i am )?(?:leaving|going away|done)", None),
    ]
    
    STATUS_PATTERNS = [
        (r"jarvis[,.]? (?:what'?s |what is )?(?:the )?(?:status|state)", None),
        (r"jarvis[,.]? (?:am i |is user )?authenticated", None),
        (r"jarvis[,.]? (?:who'?s |who is )?(?:logged in|authenticated)", None),
    ]
    
    ENROLLMENT_PATTERNS = [
        (r"jarvis[,.]? (?:please )?(?:enroll|register|add) (?:me|user)?\s*(\w+)?", 'user'),
        (r"jarvis[,.]? (?:create|setup) (?:voice )?profile (?:for )?\s*(\w+)?", 'user'),
    ]

    SECURITY_TEST_PATTERNS = [
        (r"jarvis[,.]? (?:please )?(?:test|check|verify) (?:my )?voice (?:security|authentication|biometric)", None),
        (r"jarvis[,.]? (?:run|start|perform) (?:a )?(?:voice )?security (?:test|check)", None),
        (r"jarvis[,.]? (?:verify|validate) (?:voice )?(?:authentication|security)", None),
        (r"jarvis[,.]? (?:test|check) (?:if )?(?:my )?voice (?:unlock|authentication) (?:is )?(?:secure|working)", None),
    ]

    def __init__(self):
        self.last_command = None
        self.command_history = []
        
    def parse_command(self, text: str) -> Optional[VoiceCommand]:
        """
        Parse voice command from transcribed text
        
        Args:
            text: Transcribed voice text
            
        Returns:
            Parsed command or None
        """
        # Normalize text
        text = text.lower().strip()
        
        # Check unlock commands
        for pattern, capture_group in self.UNLOCK_PATTERNS:
            match = re.search(pattern, text)
            if match:
                user_name = None
                if capture_group == 'user' and match.lastindex:
                    user_name = match.group(1)
                    
                command = VoiceCommand(
                    command_type='unlock',
                    user_name=user_name,
                    parameters={},
                    confidence=0.9 if 'please' in text else 0.8,
                    raw_text=text
                )
                
                self._record_command(command)
                return command
                
        # Check lock commands
        for pattern, _ in self.LOCK_PATTERNS:
            if re.search(pattern, text):
                command = VoiceCommand(
                    command_type='lock',
                    user_name=None,
                    parameters={},
                    confidence=0.9,
                    raw_text=text
                )
                
                self._record_command(command)
                return command
                
        # Check status commands
        for pattern, _ in self.STATUS_PATTERNS:
            if re.search(pattern, text):
                command = VoiceCommand(
                    command_type='status',
                    user_name=None,
                    parameters={},
                    confidence=0.9,
                    raw_text=text
                )
                
                self._record_command(command)
                return command
                
        # Check enrollment commands
        for pattern, capture_group in self.ENROLLMENT_PATTERNS:
            match = re.search(pattern, text)
            if match:
                user_name = None
                if capture_group == 'user' and match.lastindex:
                    user_name = match.group(1)
                    
                command = VoiceCommand(
                    command_type='enroll',
                    user_name=user_name,
                    parameters={},
                    confidence=0.8,
                    raw_text=text
                )
                
                self._record_command(command)
                return command

        # Check security test commands
        for pattern, _ in self.SECURITY_TEST_PATTERNS:
            if re.search(pattern, text):
                command = VoiceCommand(
                    command_type='security_test',
                    user_name=None,
                    parameters={},
                    confidence=0.9,
                    raw_text=text
                )

                self._record_command(command)
                return command

        # No matching command
        logger.debug(f"No command matched for: {text}")
        return None
        
    def _record_command(self, command: VoiceCommand):
        """Record command in history"""
        self.last_command = command
        self.command_history.append({
            'command': command,
            'timestamp': datetime.now()
        })
        
        # Keep only last 100 commands
        if len(self.command_history) > 100:
            self.command_history = self.command_history[-100:]
            
    def generate_response(self, command: VoiceCommand, result: Dict[str, Any]) -> str:
        """
        Generate JARVIS response based on command and result
        
        Args:
            command: The voice command
            result: Command execution result
            
        Returns:
            Response text for TTS
        """
        if command.command_type == 'unlock':
            if result.get('authenticated'):
                user = result.get('user_id', command.user_name or 'User')
                responses = [
                    f"Welcome back, {user}. System unlocked.",
                    f"Authentication successful. Good to see you, {user}.",
                    f"Access granted. How may I assist you today, {user}?",
                    f"Voice and proximity verified. Welcome, {user}."
                ]
                
                # Add watch status if available
                if result.get('watch_nearby'):
                    distance = result.get('watch_distance', 0)
                    if distance < 1:
                        responses.append(f"Your Apple Watch is very close. Welcome back, {user}.")
                    else:
                        responses.append(f"Apple Watch detected at {distance:.1f} meters. Access granted, {user}.")
                        
            else:
                error = result.get('error', 'Unknown error')
                responses = [
                    f"Authentication failed. {error}",
                    f"I'm unable to verify your identity. {error}",
                    f"Access denied. {error}",
                ]
                
                if 'Apple Watch' in error:
                    responses.append("Please ensure your Apple Watch is nearby and unlocked.")
                elif 'voice' in error.lower():
                    responses.append("I didn't recognize your voice. Please try again.")
                    
        elif command.command_type == 'lock':
            if result.get('success'):
                responses = [
                    "System locked. Have a good day.",
                    "Security activated. System is now locked.",
                    "Locking system. See you later.",
                ]
            else:
                responses = [
                    "Unable to lock the system.",
                    "Lock command failed.",
                ]
                
        elif command.command_type == 'status':
            if result.get('is_locked'):
                responses = [
                    "The system is currently locked.",
                    "Security is active. Authentication required.",
                ]
            else:
                user = result.get('current_user', 'Unknown')
                responses = [
                    f"System is unlocked. Current user: {user}.",
                    f"{user} is currently authenticated.",
                ]
                
                if result.get('watch_nearby'):
                    responses[0] += " Apple Watch is in range."
                    
        elif command.command_type == 'enroll':
            if result.get('success'):
                user = command.user_name or 'User'
                responses = [
                    f"Voice profile created for {user}.",
                    f"Enrollment successful. {user} can now use voice authentication.",
                    f"I've registered your voice, {user}.",
                ]
            else:
                responses = [
                    "Enrollment failed. Please try again.",
                    "I was unable to create a voice profile.",
                ]

        elif command.command_type == 'security_test':
            if result.get('success'):
                summary = result.get('summary', {})
                total = summary.get('total', 0)
                passed = summary.get('passed', 0)
                breaches = summary.get('security_breaches', 0)
                false_rejects = summary.get('false_rejections', 0)

                if breaches == 0 and false_rejects == 0:
                    responses = [
                        f"Voice security test complete. All {total} tests passed. Your voice authentication is fully secure.",
                        f"Security test successful. {passed} of {total} tests passed with no security breaches detected.",
                        f"Voice biometric security verified. {total} tests completed successfully. No unauthorized access possible.",
                    ]
                elif breaches > 0:
                    responses = [
                        f"Security alert: {breaches} unauthorized voices were accepted. Your voice authentication needs attention.",
                        f"Security test found {breaches} breach{'es' if breaches > 1 else ''}. Please review your voice security settings.",
                        f"Warning: {breaches} security breach{'es' if breaches > 1 else ''} detected in voice authentication.",
                    ]
                elif false_rejects > 0:
                    responses = [
                        f"Voice authentication is secure, but {false_rejects} false rejection{'s' if false_rejects > 1 else ''} occurred. You may need to re-enroll.",
                        f"Security test passed, but your voice was rejected {false_rejects} time{'s' if false_rejects > 1 else ''}. Consider updating your voice profile.",
                    ]
            else:
                error = result.get('error', 'Unknown error')
                responses = [
                    f"Voice security test failed: {error}",
                    f"I was unable to complete the security test. {error}",
                ]

        else:
            responses = ["Command processed."]
            
        # Return random response from options
        import random
        return random.choice(responses)
        
    def get_command_suggestions(self, context: str = 'unlock') -> List[str]:
        """Get command suggestions based on context"""
        if context == 'unlock':
            return [
                "Hey JARVIS, unlock my Mac",
                "JARVIS, this is John",
                "JARVIS, authenticate me",
                "Open sesame, JARVIS"
            ]
        elif context == 'lock':
            return [
                "JARVIS, lock my Mac",
                "JARVIS, activate security",
                "JARVIS, I'm leaving"
            ]
        elif context == 'enroll':
            return [
                "JARVIS, enroll me",
                "JARVIS, create voice profile for John",
                "JARVIS, register user Sarah"
            ]
        else:
            return [
                "JARVIS, what's the status?",
                "JARVIS, who's logged in?",
                "JARVIS, am I authenticated?"
            ]
            
    def validate_command_context(self, command: VoiceCommand, 
                               system_state: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate if command is appropriate for current context
        
        Returns:
            (is_valid, error_message)
        """
        if command.command_type == 'unlock':
            if not system_state.get('is_locked'):
                return False, "System is already unlocked"
                
        elif command.command_type == 'lock':
            if system_state.get('is_locked'):
                return False, "System is already locked"
                
        elif command.command_type == 'enroll':
            if system_state.get('is_locked'):
                return False, "Please unlock the system first"
                
        return True, None


# Test function
def test_jarvis_commands():
    """Test JARVIS command parsing"""
    handler = JARVISCommandHandler()
    
    test_phrases = [
        "Hey JARVIS, unlock my Mac",
        "JARVIS, this is John",
        "JARVIS authenticate me",
        "JARVIS, lock the computer",
        "JARVIS, what's the status?",
        "JARVIS, enroll user Sarah",
        "Hello JARVIS, please unlock the Mac",
        "Open sesame, JARVIS",
        "JARVIS, I'm leaving",
    ]
    
    print("Testing JARVIS command parsing:\n")
    
    for phrase in test_phrases:
        print(f"Input: '{phrase}'")
        command = handler.parse_command(phrase)
        
        if command:
            print(f"  Command: {command.command_type}")
            print(f"  User: {command.user_name}")
            print(f"  Confidence: {command.confidence}")
        else:
            print("  No command detected")
            
        print()
        
    # Test response generation
    print("\nTesting response generation:")
    
    mock_command = VoiceCommand(
        command_type='unlock',
        user_name='John',
        parameters={},
        confidence=0.9,
        raw_text='JARVIS, this is John'
    )
    
    # Successful auth
    result = {
        'authenticated': True,
        'user_id': 'John',
        'watch_nearby': True,
        'watch_distance': 2.5
    }
    
    response = handler.generate_response(mock_command, result)
    print(f"\nSuccess response: {response}")
    
    # Failed auth
    result = {
        'authenticated': False,
        'error': 'Apple Watch not detected nearby'
    }
    
    response = handler.generate_response(mock_command, result)
    print(f"Failure response: {response}")


if __name__ == "__main__":
    test_jarvis_commands()