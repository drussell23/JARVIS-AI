"""
Unified Command Processor - Single point of command interpretation
Fixes the multi-interpreter chaos by providing one unified system
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)


class CommandType(Enum):
    """Types of commands JARVIS can handle"""
    VISION = "vision"
    SYSTEM = "system"
    WEATHER = "weather"
    COMMUNICATION = "communication"
    AUTONOMY = "autonomy"
    QUERY = "query"
    COMPOUND = "compound"
    META = "meta"
    UNKNOWN = "unknown"


@dataclass
class UnifiedContext:
    """Single context shared across all command processing"""
    conversation_history: List[Dict[str, Any]]
    current_visual: Optional[Dict[str, Any]] = None
    last_entity: Optional[Dict[str, Any]] = None  # For "it/that" resolution
    active_monitoring: bool = False
    user_preferences: Dict[str, Any] = None
    system_state: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.user_preferences is None:
            self.user_preferences = {}
        if self.system_state is None:
            self.system_state = {}
            
    def resolve_reference(self, text: str) -> Tuple[Optional[str], float]:
        """Resolve 'it', 'that', 'this' to actual entities"""
        reference_words = ['it', 'that', 'this', 'them']
        
        for word in reference_words:
            if word in text.lower():
                if self.last_entity:
                    # Check how recent the entity is
                    if 'timestamp' in self.last_entity:
                        age = (datetime.now() - self.last_entity['timestamp']).seconds
                        confidence = 0.9 if age < 30 else 0.7 if age < 60 else 0.5
                    else:
                        confidence = 0.8
                    return self.last_entity.get('value', ''), confidence
                    
                # Check visual context
                if self.current_visual:
                    return self.current_visual.get('focused_element', ''), 0.7
                    
        return None, 0.0
        
    def update_from_command(self, command_type: CommandType, result: Dict[str, Any]):
        """Update context based on command execution"""
        self.conversation_history.append({
            'type': command_type.value,
            'result': result,
            'timestamp': datetime.now()
        })
        
        # Extract entities for future reference
        if command_type == CommandType.VISION and 'elements' in result:
            if result['elements']:
                self.last_entity = {
                    'value': result['elements'][0],
                    'timestamp': datetime.now(),
                    'type': 'visual_element'
                }
                
        # Update visual context
        if command_type == CommandType.VISION:
            self.current_visual = result.get('visual_context', {})


class UnifiedCommandProcessor:
    """Single command processor that handles all interpretation"""
    
    def __init__(self, claude_api_key: Optional[str] = None):
        self.context = UnifiedContext(conversation_history=[])
        self.handlers = {}
        self._initialize_handlers()
        self.claude_api_key = claude_api_key
        
    def _initialize_handlers(self):
        """Initialize command handlers lazily"""
        # We'll import handlers only when needed to avoid circular imports
        self.handler_modules = {
            CommandType.VISION: 'api.vision_command_handler',
            CommandType.SYSTEM: 'system_control.macos_controller',
            CommandType.WEATHER: 'system_control.weather_system_config',
            CommandType.AUTONOMY: 'api.autonomy_handler'
        }
        
    async def process_command(self, command_text: str, websocket=None) -> Dict[str, Any]:
        """Process any command through unified pipeline"""
        logger.info(f"[UNIFIED] Processing: '{command_text}'")
        
        # Step 1: Classify command intent
        command_type, confidence = await self._classify_command(command_text)
        logger.info(f"[UNIFIED] Classified as {command_type.value} (confidence: {confidence})")
        
        # Step 2: Resolve references if needed
        resolved_text = command_text
        reference, ref_confidence = self.context.resolve_reference(command_text)
        if reference and ref_confidence > 0.5:
            # Replace reference with resolved entity
            for word in ['it', 'that', 'this']:
                if word in command_text.lower():
                    resolved_text = command_text.lower().replace(word, reference)
                    logger.info(f"[UNIFIED] Resolved '{word}' to '{reference}'")
                    break
                    
        # Step 3: Handle compound commands
        if command_type == CommandType.COMPOUND:
            return await self._handle_compound_command(resolved_text)
            
        # Step 4: Route to appropriate handler
        result = await self._execute_command(command_type, resolved_text, websocket)
        
        # Step 5: Update context
        self.context.update_from_command(command_type, result)
        
        return result
        
    async def _classify_command(self, command_text: str) -> Tuple[CommandType, float]:
        """Classify command using intelligent pattern matching"""
        command_lower = command_text.lower()
        
        # System commands - check first to avoid confusion with vision
        system_patterns = [
            'open', 'close', 'launch', 'quit', 'start', 'restart', 'shutdown',
            'volume', 'brightness', 'settings', 'wifi', 'wi-fi', 'screenshot',
            'mute', 'unmute', 'sleep display'
        ]
        
        # App names that might be mentioned
        common_apps = [
            'safari', 'chrome', 'firefox', 'spotify', 'music', 'photos', 
            'mail', 'messages', 'calendar', 'notes', 'finder', 'slack',
            'zoom', 'whatsapp', 'vscode', 'code', 'terminal'
        ]
        
        # Check if it's a system command with an app name
        if any(pattern in command_lower for pattern in system_patterns):
            # Special case: "start monitoring" is vision, not system
            if 'monitor' in command_lower and any(word in command_lower for word in ['start', 'begin', 'activate']):
                return CommandType.VISION, 0.95
            # Otherwise it's a system command
            return CommandType.SYSTEM, 0.9
            
        # Also check if any app name is mentioned with action verbs
        if any(app in command_lower for app in common_apps):
            action_verbs = ['open', 'launch', 'start', 'close', 'quit', 'switch to']
            if any(verb in command_lower for verb in action_verbs):
                return CommandType.SYSTEM, 0.9
        
        # Vision commands
        vision_patterns = [
            'see', 'look', 'monitor', 'screen', 'watching', 'vision',
            'what\'s on', 'what is on', 'show me', 'analyze', 'what do you see',
            'describe the screen', 'read the screen'
        ]
        if any(pattern in command_lower for pattern in vision_patterns):
            return CommandType.VISION, 0.85
            
        # Weather commands
        if 'weather' in command_lower:
            return CommandType.WEATHER, 0.95
            
        # Autonomy commands
        autonomy_patterns = [
            'autonomy', 'autonomous', 'auto mode', 'full control',
            'take over', 'activate yourself'
        ]
        if any(pattern in command_lower for pattern in autonomy_patterns):
            return CommandType.AUTONOMY, 0.9
            
        # Compound commands (multiple actions)
        if ' and ' in command_lower or ' then ' in command_lower:
            return CommandType.COMPOUND, 0.8
            
        # Meta commands (about previous commands)
        meta_patterns = ['cancel', 'stop that', 'undo', 'never mind', 'not that']
        if any(pattern in command_lower for pattern in meta_patterns):
            return CommandType.META, 0.85
            
        # Default to query
        return CommandType.QUERY, 0.5
        
    async def _execute_command(self, command_type: CommandType, command_text: str, websocket=None) -> Dict[str, Any]:
        """Execute command using appropriate handler"""
        
        # Get or initialize handler
        if command_type not in self.handlers:
            handler = await self._get_handler(command_type)
            if handler:
                self.handlers[command_type] = handler
                
        handler = self.handlers.get(command_type)
        
        if not handler and command_type != CommandType.SYSTEM:
            return {
                'success': False,
                'response': f"I don't have a handler for {command_type.value} commands yet.",
                'command_type': command_type.value
            }
            
        # Execute with unified context
        try:
            # Different handlers have different interfaces, normalize them
            if command_type == CommandType.VISION:
                # For vision commands, check if it's a monitoring command or analysis
                if any(word in command_text.lower() for word in ['start', 'stop', 'monitor']):
                    result = await handler.handle_command(command_text)
                else:
                    # It's a vision query - analyze the screen with the specific query
                    result = await handler.analyze_screen(command_text)
                    
                return {
                    'success': result.get('handled', False),
                    'response': result.get('response', ''),
                    'command_type': command_type.value,
                    **result
                }
            elif command_type == CommandType.WEATHER:
                result = await handler.get_weather(command_text)
                return {
                    'success': result.get('success', False),
                    'response': result.get('formatted_response', result.get('message', '')),
                    'command_type': command_type.value,
                    **result
                }
            elif command_type == CommandType.SYSTEM:
                # Handle system commands (app control, system settings, etc.)
                result = await self._execute_system_command(command_text)
                return {
                    'success': result.get('success', False),
                    'response': result.get('response', ''),
                    'command_type': command_type.value,
                    **result
                }
            else:
                # Generic handler interface
                return {
                    'success': True,
                    'response': f"Executing {command_type.value} command",
                    'command_type': command_type.value
                }
                
        except Exception as e:
            logger.error(f"Error executing {command_type.value} command: {e}", exc_info=True)
            return {
                'success': False,
                'response': f"I encountered an error with that {command_type.value} command.",
                'command_type': command_type.value,
                'error': str(e)
            }
            
    async def _get_handler(self, command_type: CommandType):
        """Dynamically import and get handler for command type"""
        # System commands are handled directly in _execute_command
        if command_type == CommandType.SYSTEM:
            return True  # Return True to indicate system handler is available
            
        module_name = self.handler_modules.get(command_type)
        if not module_name:
            return None
            
        try:
            if command_type == CommandType.VISION:
                from api.vision_command_handler import vision_command_handler
                return vision_command_handler
            elif command_type == CommandType.WEATHER:
                from system_control.weather_system_config import get_weather_system
                return get_weather_system()
            elif command_type == CommandType.AUTONOMY:
                from api.autonomy_handler import get_autonomy_handler
                return get_autonomy_handler()
            # Add other handlers as needed
            
        except ImportError as e:
            logger.error(f"Failed to import handler for {command_type.value}: {e}")
            return None
            
    async def _handle_compound_command(self, command_text: str) -> Dict[str, Any]:
        """Handle commands with multiple parts"""
        # Split by 'and' or 'then'
        parts = command_text.replace(' then ', ' and ').split(' and ')
        
        results = []
        all_success = True
        
        for part in parts:
            part = part.strip()
            if part:
                result = await self.process_command(part)
                results.append(result)
                if not result.get('success', False):
                    all_success = False
                    
        return {
            'success': all_success,
            'response': ' '.join(r.get('response', '') for r in results),
            'command_type': CommandType.COMPOUND.value,
            'sub_results': results
        }
    
    async def _execute_system_command(self, command_text: str) -> Dict[str, Any]:
        """Execute system control commands"""
        try:
            from system_control.macos_controller import MacOSController
            from system_control.dynamic_app_controller import get_dynamic_app_controller
            
            macos_controller = MacOSController()
            dynamic_controller = get_dynamic_app_controller()
            
            command_lower = command_text.lower()
            
            # Parse app control commands
            if 'open' in command_lower or 'launch' in command_lower or 'start' in command_lower:
                # Extract app name after 'open', 'launch', or 'start'
                app_name = None
                for keyword in ['open', 'launch', 'start']:
                    if keyword in command_lower:
                        parts = command_text.split(keyword, 1)
                        if len(parts) > 1:
                            app_name = parts[1].strip()
                            break
                
                if app_name:
                    # Use dynamic controller for intelligent app discovery
                    success, message = await dynamic_controller.open_app_intelligently(app_name)
                    return {'success': success, 'response': message}
                else:
                    return {'success': False, 'response': "Please specify which app to open"}
                    
            elif 'close' in command_lower or 'quit' in command_lower:
                # Extract app name after 'close' or 'quit'
                app_name = None
                for keyword in ['close', 'quit']:
                    if keyword in command_lower:
                        parts = command_text.split(keyword, 1)
                        if len(parts) > 1:
                            app_name = parts[1].strip()
                            break
                
                if app_name:
                    success, message = await dynamic_controller.close_app_intelligently(app_name)
                    return {'success': success, 'response': message}
                else:
                    return {'success': False, 'response': "Please specify which app to close"}
                    
            elif 'volume' in command_lower:
                # Handle volume commands
                if 'mute' in command_lower:
                    success, message = macos_controller.mute_volume(True)
                elif 'unmute' in command_lower:
                    success, message = macos_controller.mute_volume(False)
                else:
                    # Try to extract volume level
                    import re
                    match = re.search(r'(\d+)', command_text)
                    if match:
                        level = int(match.group(1))
                        success, message = macos_controller.set_volume(level)
                    else:
                        return {'success': False, 'response': "Please specify a volume level (0-100)"}
                return {'success': success, 'response': message}
                
            elif 'brightness' in command_lower:
                # Handle brightness commands
                import re
                match = re.search(r'(\d+)', command_text)
                if match:
                    level = int(match.group(1)) / 100.0  # Convert to 0.0-1.0
                    success, message = macos_controller.adjust_brightness(level)
                else:
                    return {'success': False, 'response': "Please specify brightness level (0-100)"}
                return {'success': success, 'response': message}
                
            elif 'screenshot' in command_lower:
                success, message = macos_controller.take_screenshot()
                return {'success': success, 'response': message}
                
            elif 'wifi' in command_lower or 'wi-fi' in command_lower:
                if 'off' in command_lower or 'disable' in command_lower:
                    success, message = macos_controller.toggle_wifi(False)
                elif 'on' in command_lower or 'enable' in command_lower:
                    success, message = macos_controller.toggle_wifi(True)
                else:
                    return {'success': False, 'response': "Please specify whether to turn WiFi on or off"}
                return {'success': success, 'response': message}
                
            else:
                # Try to handle as generic system command
                return {
                    'success': False,
                    'response': "I'm not sure how to handle that system command. Try 'open [app name]', 'close [app name]', or other system controls."
                }
                
        except Exception as e:
            logger.error(f"Error executing system command: {e}", exc_info=True)
            return {
                'success': False,
                'response': f"Failed to execute system command: {str(e)}"
            }
        

# Singleton instance
_unified_processor = None

def get_unified_processor(api_key: Optional[str] = None) -> UnifiedCommandProcessor:
    """Get or create the unified command processor"""
    global _unified_processor
    if _unified_processor is None:
        _unified_processor = UnifiedCommandProcessor(api_key)
    return _unified_processor