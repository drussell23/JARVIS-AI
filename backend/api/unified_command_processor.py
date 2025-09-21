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
    VOICE_UNLOCK = "voice_unlock"
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
            CommandType.AUTONOMY: 'api.autonomy_handler',
            CommandType.VOICE_UNLOCK: 'api.voice_unlock_handler',
            CommandType.QUERY: 'api.query_handler'  # Add basic query handler
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
        
        # Check compound commands FIRST (highest priority)
        if ' and ' in command_lower or ' then ' in command_lower or ', and ' in command_lower:
            # Special case: "open X and search for Y" should be compound
            if 'open' in command_lower and 'search' in command_lower:
                return CommandType.COMPOUND, 0.95
            # Make sure it's not just part of a search query or typing command
            if not any(pattern in command_lower for pattern in ['search for', 'look up', 'type', 'and press enter', 'and enter']):
                return CommandType.COMPOUND, 0.95
        
        # System commands - check after compound
        system_patterns = [
            'open', 'close', 'launch', 'quit', 'start', 'restart', 'shutdown',
            'volume', 'brightness', 'settings', 'wifi', 'wi-fi', 'screenshot',
            'mute', 'unmute', 'sleep display', 'go to', 'navigate to', 'visit',
            'browse to', 'search for', 'search in', 'google', 'new tab', 'open tab', 'type',
            'enter', 'search bar', 'click', 'another tab', 'open another'
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
            
        # Voice Unlock commands
        voice_unlock_patterns = [
            'unlock my mac', 'voice unlock', 'unlock with voice', 'enroll my voice',
            'set up voice', 'voice authentication', 'voice security', 'unlock mac',
            'enable voice unlock', 'disable voice unlock', 'test voice unlock',
            'delete my voiceprint', 'voice enrollment', 'unlock screen'
        ]
        if any(pattern in command_lower for pattern in voice_unlock_patterns):
            return CommandType.VOICE_UNLOCK, 0.95
            
        # Autonomy commands
        autonomy_patterns = [
            'autonomy', 'autonomous', 'auto mode', 'full control',
            'take over', 'activate yourself'
        ]
        if any(pattern in command_lower for pattern in autonomy_patterns):
            return CommandType.AUTONOMY, 0.9
            
            
        # Meta commands (about previous commands)
        meta_patterns = ['cancel', 'stop that', 'undo', 'never mind', 'not that']
        if any(pattern in command_lower for pattern in meta_patterns):
            return CommandType.META, 0.85
            
        # Wake word/activation commands - should be ignored or treated as META
        wake_patterns = ['activate', 'wake', 'wake up', 'hello', 'hey', 'hi jarvis', 'hey jarvis']
        if command_lower.strip() in wake_patterns:
            return CommandType.META, 0.9  # Treat as meta command to avoid errors
            
        # Default to query only for actual questions
        question_words = ['what', 'who', 'where', 'when', 'why', 'how', 'is', 'are', 'can', 'could', 'would', 'should']
        if any(command_lower.startswith(word) for word in question_words):
            return CommandType.QUERY, 0.7
            
        # For very short commands that don't match anything, treat as system
        if len(command_lower.split()) <= 2:
            return CommandType.SYSTEM, 0.6
            
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
            elif command_type == CommandType.META:
                # Handle meta commands (wake words, cancellations)
                if command_text.lower().strip() in ['activate', 'wake', 'wake up', 'hello', 'hey']:
                    # Silent acknowledgment for wake words
                    return {
                        'success': True,
                        'response': '',
                        'command_type': 'meta',
                        'silent': True
                    }
                else:
                    return {
                        'success': True,
                        'response': 'Understood',
                        'command_type': 'meta'
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
        """Handle commands with multiple parts and maintain context between them"""
        # Parse compound commands more intelligently
        parts = self._parse_compound_parts(command_text)
        
        results = []
        all_success = True
        responses = []
        
        # Track context for dependent commands
        active_app = None
        previous_result = None
        
        # Check if all parts are similar operations that can be parallelized
        can_parallelize = self._can_parallelize_commands(parts)
        
        if can_parallelize:
            # Process similar operations in parallel (e.g., closing multiple apps)
            logger.info(f"[COMPOUND] Processing {len(parts)} similar commands in parallel")
            
            # Create tasks for parallel execution
            tasks = []
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                    
                # Process each part as an independent command
                async def process_part(p):
                    command_type, _ = await self._classify_command(p)
                    if command_type == CommandType.COMPOUND:
                        command_type = CommandType.SYSTEM
                    return await self._execute_command(command_type, p)
                
                tasks.append(process_part(part))
            
            # Execute all tasks in parallel
            results = await asyncio.gather(*tasks)
            
            # Collect responses
            for result in results:
                if result.get('success', False):
                    responses.append(result.get('response', ''))
                else:
                    all_success = False
                    responses.append(f"Failed: {result.get('response', 'Unknown error')}")
        else:
            # Sequential processing for dependent commands
            for i, part in enumerate(parts):
                part = part.strip()
                if not part:
                    continue
                    
                # Provide user feedback for multi-step commands
                if len(parts) > 1:
                    logger.info(f"[COMPOUND] Step {i+1}/{len(parts)}: {part}")
                
                # Check if this is a dependent command that needs context
                enhanced_command = self._enhance_with_context(part, active_app, previous_result)
                
                # Process individual part (not as compound to avoid recursion)
                command_type, _ = await self._classify_command(enhanced_command)
                # Force non-compound to avoid recursion
                if command_type == CommandType.COMPOUND:
                    command_type = CommandType.SYSTEM
                
                result = await self._execute_command(command_type, enhanced_command)
                results.append(result)
                
                # Update context for next command
                if result.get('success', False):
                    # Track opened apps for subsequent commands
                    if any(word in part.lower() for word in ['open', 'launch', 'start']):
                        for app in ['safari', 'chrome', 'firefox']:
                            if app in enhanced_command.lower():
                                active_app = app
                                break
                                
                    responses.append(result.get('response', ''))
                else:
                    all_success = False
                    responses.append(f"Failed: {result.get('response', 'Unknown error')}")
                    # Don't continue if a step fails
                    break
                
                previous_result = result
                
                # Add small delay between commands for reliability
                if i < len(parts) - 1:
                    await asyncio.sleep(0.5)
                
        # Create conversational response
        if len(responses) > 1:
            # Clean up individual responses first
            cleaned_responses = []
            for i, resp in enumerate(responses):
                # Remove trailing "Sir" from all but the last response
                if resp.endswith(", Sir") and i < len(responses) - 1:
                    resp = resp[:-5]
                cleaned_responses.append(resp)
            
            # Combine into natural response
            if len(cleaned_responses) == 2:
                # For 2 steps: "Opening Safari and searching for dogs"
                response = f"{cleaned_responses[0]} and {cleaned_responses[1]}"
            else:
                # For 3+ steps: "Opening Safari, navigating to Google, and taking a screenshot"
                response = ", ".join(cleaned_responses[:-1]) + f" and {cleaned_responses[-1]}"
            
            # Add "Sir" at the end if it's not already there
            if not response.endswith(", Sir"):
                response += ", Sir"
        else:
            response = responses[0] if responses else "I'll help you with that"
            
        return {
            'success': all_success,
            'response': response,
            'command_type': CommandType.COMPOUND.value,
            'sub_results': results,
            'steps_completed': len([r for r in results if r.get('success', False)]),
            'total_steps': len(parts)
        }
    
    def _parse_compound_parts(self, command_text: str) -> List[str]:
        """Parse compound command into logical parts"""
        # Special handling for complex browser commands
        if 'search for' in command_text.lower() and ('separate tabs' in command_text.lower() or 'different tabs' in command_text.lower()):
            # This is a single complex command, don't split it
            return [command_text]
            
        # Special handling for "open X and Y" where both are apps
        if command_text.lower().startswith('open '):
            apps = ['safari', 'chrome', 'firefox', 'whatsapp', 'music', 'messages', 'mail', 'spotify', 'slack', 'zoom', 'terminal', 'finder']
            # Count how many apps are mentioned
            app_count = sum(1 for app in apps if app in command_text.lower())
            if app_count >= 2 and 'search' not in command_text.lower() and 'go to' not in command_text.lower():
                # Split simple app opens: "open safari and whatsapp"
                parts = []
                remaining = command_text
                for app in apps:
                    if app in remaining.lower():
                        # Extract this app command
                        if 'open ' + app in remaining.lower():
                            parts.append('open ' + app)
                            remaining = remaining.lower().replace('open ' + app, '', 1)
                        elif app in remaining.lower():
                            parts.append('open ' + app)
                            remaining = remaining.lower().replace(app, '', 1)
                return [p for p in parts if p.strip()]
        
        # Handle various connectors
        command_text = command_text.replace(' then ', ' and ')
        command_text = command_text.replace(', and ', ' and ')
        command_text = command_text.replace(', ', ' and ')
        
        # Smart split by 'and' - check context on both sides
        parts = []
        
        # Use regex to find 'and' with word boundaries
        import re
        and_positions = []
        for match in re.finditer(r'\band\b', command_text, re.IGNORECASE):
            and_positions.append(match.span())
        
        if not and_positions:
            return [command_text]
        
        last_pos = 0
        for start, end in and_positions:
            # Get text before and after 'and'
            before_and = command_text[last_pos:start].strip()
            after_and = command_text[end:].strip()
            
            # Check if this 'and' is part of a phrase that shouldn't be split
            keep_together = False
            
            # Don't split if 'and' is part of a URL or between words that form a phrase
            if before_and and after_and:
                # Check for patterns where 'and' should not split
                if any(pattern in before_and.lower() for pattern in ['go to', 'navigate to']):
                    if not any(word in after_and.lower().split()[:3] for word in ['open', 'close', 'search', 'type', 'click']):
                        keep_together = True
            
            if keep_together:
                # Don't split here
                continue
            else:
                # This is a splitting point
                if before_and:
                    parts.append(before_and)
                last_pos = end
        
        # Add the remaining part
        remaining = command_text[last_pos:].strip()
        if remaining:
            parts.append(remaining)
            
        # If no valid splits were made, return the whole command
        if not parts:
            return [command_text]
            
        return parts
    
    def _can_parallelize_commands(self, parts: List[str]) -> bool:
        """Check if commands can be run in parallel"""
        if len(parts) < 2:
            return False
            
        # Analyze each part to determine if they're independent
        independent_commands = []
        for part in parts:
            part_lower = part.lower().strip()
            
            # Commands that are typically independent
            is_independent = (
                # App operations without dependencies
                (any(op in part_lower for op in ['open', 'close', 'launch', 'quit']) and 
                 not any(dep in part_lower for dep in ['and search', 'and go to', 'and type', 'then'])) or
                # Simple app operations
                (len(part_lower.split()) <= 3 and any(app in part_lower for app in 
                 ['safari', 'chrome', 'firefox', 'whatsapp', 'music', 'messages', 'mail', 'spotify']))
            )
            
            independent_commands.append(is_independent)
        
        # Can parallelize if all commands are independent
        # Or if they're all similar operations (all opens or all closes)
        all_independent = all(independent_commands)
        
        if all_independent:
            return True
            
        # Check if all parts are similar operations
        operations = []
        for part in parts:
            part_lower = part.lower().strip()
            if any(op in part_lower for op in ['close', 'quit', 'kill']):
                operations.append('close')
            elif any(op in part_lower for op in ['open', 'launch', 'start']):
                operations.append('open')
            else:
                operations.append('other')
        
        # Can parallelize if all operations are the same type
        return len(set(operations)) == 1 and operations[0] in ['close', 'open']
    
    def _enhance_with_context(self, command: str, active_app: Optional[str], previous_result: Optional[Dict]) -> str:
        """Enhance command with context from previous commands"""
        command_lower = command.lower()
        
        # URL navigation patterns
        url_patterns = ['go to', 'navigate to', 'open url', 'visit', 'browse to']
        search_patterns = ['search for', 'google', 'look up', 'find']
        
        # Check if this is a navigation command without app context
        if any(pattern in command_lower for pattern in url_patterns + search_patterns):
            # If no browser is specified but we have an active browser
            if active_app and active_app in ['safari', 'chrome', 'firefox']:
                if not any(browser in command_lower for browser in ['safari', 'chrome', 'firefox']):
                    # Enhance with browser context
                    if 'go to' in command_lower:
                        command = command.replace('go to', f'tell {active_app} to go to')
                    elif 'search for' in command_lower:
                        command = command.replace('search for', f'search in {active_app} for')
                    elif not any(browser in command_lower for browser in ['safari', 'chrome', 'firefox']):
                        command = f"in {active_app} {command}"
        
        return command
    
    async def _execute_system_command(self, command_text: str) -> Dict[str, Any]:
        """Execute system control commands"""
        try:
            from system_control.macos_controller import MacOSController
            from system_control.dynamic_app_controller import get_dynamic_app_controller
            
            macos_controller = MacOSController()
            dynamic_controller = get_dynamic_app_controller()
            
            command_lower = command_text.lower()
            
            # Handle new tab commands first (before app control)
            if any(pattern in command_lower for pattern in ['new tab', 'open tab', 'open a tab', 'open another tab', 'another tab', 'open another']):
                browser = None
                # Check if browser is specified
                for browser_name in ['safari', 'chrome', 'firefox']:
                    if browser_name in command_lower:
                        browser = browser_name
                        break
                
                # If no browser specified, use default Safari
                if not browser:
                    browser = 'safari'
                
                # Check if URL is specified
                url = None
                for pattern in ['and go to', 'and navigate to', 'and open']:
                    if pattern in command_lower:
                        parts = command_text.split(pattern, 1)
                        if len(parts) > 1:
                            url_part = parts[1].strip()
                            # Handle common sites
                            if url_part.lower() == 'google':
                                url = 'https://google.com'
                            elif '.' not in url_part:
                                url = f'https://{url_part}.com'
                            else:
                                url = url_part if url_part.startswith('http') else f'https://{url_part}'
                            break
                
                success, message = macos_controller.open_new_tab(browser, url)
                return {'success': success, 'response': message}
            
            # Handle complex browser search operations
            elif ('search' in command_lower or 'search for' in command_lower) and any(browser in command_lower for browser in ['safari', 'chrome', 'firefox']):
                # Handle searches like "open safari and search for dogs and cats on two separate tabs"
                browser = None
                for browser_name in ['safari', 'chrome', 'firefox']:
                    if browser_name in command_lower:
                        browser = browser_name
                        break
                
                # First open the browser if needed
                if 'open' in command_lower:
                    success, _ = dynamic_controller.open_app_intelligently(browser)
                    if success:
                        # Wait a moment for browser to open
                        await asyncio.sleep(1.5)
                
                # Check if we need multiple tabs
                if 'separate tabs' in command_lower or 'different tabs' in command_lower or 'multiple tabs' in command_lower:
                    # Extract search terms
                    search_pattern = r'search for (.+?)(?:\s+on\s+(?:two|multiple|different)\s+separate\s+tabs?)?$'
                    import re
                    match = re.search(search_pattern, command_text, re.IGNORECASE)
                    if match:
                        search_terms_str = match.group(1)
                        # Split by 'and' to get individual search terms
                        search_terms = [term.strip() for term in search_terms_str.split(' and ')]
                        
                        # Open a tab for each search term
                        results = []
                        for i, term in enumerate(search_terms):
                            if i == 0:
                                # First search in current tab
                                success, msg = macos_controller.web_search(term, browser=browser)
                            else:
                                # Subsequent searches in new tabs
                                await asyncio.sleep(0.5)  # Small delay between tabs
                                success, msg = macos_controller.open_new_tab(browser, f"https://google.com/search?q={term.replace(' ', '+')}")
                            results.append(success)
                        
                        if all(results):
                            return {'success': True, 'response': f"Searching for {' and '.join(search_terms)} in separate tabs, Sir"}
                        else:
                            return {'success': False, 'response': "Had trouble opening some tabs"}
                else:
                    # Single search
                    search_match = re.search(r'search(?:\s+for)?\s+(.+?)(?:\s+in\s+' + browser + r')?$', command_text, re.IGNORECASE)
                    if search_match:
                        query = search_match.group(1).strip()
                        success, message = macos_controller.web_search(query, browser=browser)
                        return {'success': success, 'response': message}
            
            # Parse app control commands
            elif 'open' in command_lower or 'launch' in command_lower or 'start' in command_lower:
                # Extract app name after 'open', 'launch', or 'start'
                app_name = None
                for keyword in ['open', 'launch', 'start']:
                    if keyword in command_lower:
                        # Use case-insensitive split
                        pattern_index = command_lower.find(keyword)
                        if pattern_index != -1:
                            after_keyword = command_text[pattern_index + len(keyword):].strip()
                            # Handle compound commands - stop at common connectors
                            connectors = [' and ', ', and ', ' then ', ', then ', ', ']
                            app_name = after_keyword
                            for connector in connectors:
                                if connector in after_keyword:
                                    app_name = after_keyword.split(connector)[0].strip()
                                    break
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
                        pattern_index = command_lower.find(keyword)
                        if pattern_index != -1:
                            after_keyword = command_text[pattern_index + len(keyword):].strip()
                            # Handle compound commands - stop at common connectors
                            connectors = [' and ', ', and ', ' then ', ', then ', ', ']
                            app_name = after_keyword
                            for connector in connectors:
                                if connector in after_keyword:
                                    app_name = after_keyword.split(connector)[0].strip()
                                    break
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
                
            # Handle URL navigation and web searches
            elif any(pattern in command_lower for pattern in ['go to', 'navigate to', 'browse to', 'visit', 'google.com', '.com', '.org', '.net']):
                # Extract URL or search query
                url = None
                browser = None
                
                # Check if browser is specified
                for browser_name in ['safari', 'chrome', 'firefox']:
                    if browser_name in command_lower:
                        browser = browser_name
                        break
                
                # Extract URL
                for pattern in ['go to', 'navigate to', 'browse to', 'visit']:
                    if pattern in command_lower:
                        # Find the pattern position for case-insensitive split
                        pattern_pos = command_lower.find(pattern)
                        if pattern_pos != -1:
                            # Extract everything after the pattern
                            after_pattern = command_text[pattern_pos + len(pattern):].strip()
                            
                            # Remove browser specification if present
                            if ' in ' in after_pattern.lower():
                                in_pos = after_pattern.lower().find(' in ')
                                url_part = after_pattern[:in_pos].strip()
                            else:
                                url_part = after_pattern
                            
                            url = url_part
                            break
                
                if url:
                    # Handle common website shortcuts
                    website_shortcuts = {
                        'google': 'google.com',
                        'facebook': 'facebook.com',
                        'twitter': 'twitter.com',
                        'x': 'x.com',
                        'youtube': 'youtube.com',
                        'github': 'github.com',
                        'amazon': 'amazon.com',
                        'reddit': 'reddit.com',
                        'linkedin': 'linkedin.com',
                        'gmail': 'gmail.com'
                    }
                    
                    # Check if it's a known website shortcut
                    url_lower = url.lower()
                    if url_lower in website_shortcuts:
                        url = website_shortcuts[url_lower]
                    
                    # Add protocol if missing
                    if not url.startswith(('http://', 'https://')):
                        if '.' in url:  # Looks like a domain
                            url = f'https://{url}'
                        else:  # Only treat as search if it's clearly not a website
                            # Check if user might mean a website
                            if any(word in url_lower for word in ['search', 'find', 'look']):
                                success, message = macos_controller.web_search(url)
                                return {'success': success, 'response': message}
                            else:
                                # Assume they want the .com version
                                url = f'https://{url}.com'
                    
                    success, message = macos_controller.open_url(url, browser)
                    return {'success': success, 'response': message}
                else:
                    return {'success': False, 'response': "Please specify a URL or search term"}
                    
            elif any(pattern in command_lower for pattern in ['search for', 'google', 'look up', 'search in']):
                # Handle web searches
                query = None
                browser = None
                
                # Check if browser is specified
                for browser_name in ['safari', 'chrome', 'firefox']:
                    if browser_name in command_lower or f'in {browser_name}' in command_lower:
                        browser = browser_name
                        break
                
                # Handle "search in X for Y" pattern
                if 'search in' in command_lower and ' for ' in command_lower:
                    # Extract from "search in safari for dogs"
                    search_start = command_lower.find('search in')
                    for_index = command_lower.find(' for ', search_start)
                    if for_index != -1:
                        query = command_text[for_index + 5:].strip()
                else:
                    # Handle regular patterns
                    for pattern in ['search for', 'google', 'look up']:
                        if pattern in command_lower:
                            # Find the pattern position to do case-insensitive split
                            pattern_index = command_lower.find(pattern)
                            if pattern_index != -1:
                                query = command_text[pattern_index + len(pattern):].strip()
                                # Remove browser specification if present
                                for browser_name in ['safari', 'chrome', 'firefox']:
                                    if f'in {browser_name}' in query.lower():
                                        query = query[:query.lower().find(f'in {browser_name}')].strip()
                                    elif f'on {browser_name}' in query.lower():
                                        query = query[:query.lower().find(f'on {browser_name}')].strip()
                                break
                
                if query:
                    success, message = macos_controller.web_search(query, browser=browser)
                    return {'success': success, 'response': message}
                else:
                    return {'success': False, 'response': "Please specify what to search for"}
                    
            
            # Handle typing/search commands
            elif any(pattern in command_lower for pattern in ['type', 'search for', 'enter']):
                browser = None
                for browser_name in ['safari', 'chrome', 'firefox']:
                    if browser_name in command_lower or f'in {browser_name}' in command_lower:
                        browser = browser_name
                        break
                
                # Extract what to type
                text_to_type = None
                press_enter = 'enter' in command_lower or 'search' in command_lower
                
                # Special handling for "type X and press enter" format
                if 'type' in command_lower:
                    # Find where "type" appears
                    type_index = command_lower.find('type')
                    if type_index != -1:
                        # Get everything after "type"
                        after_type = command_text[type_index + 4:].strip()
                        # Remove "and press enter" or "and enter"
                        after_type = after_type.replace(' and press enter', '').replace(' and enter', '').strip()
                        # Remove "in browser" specifications
                        for browser_name in ['in safari', 'in chrome', 'in firefox']:
                            if after_type.lower().endswith(browser_name):
                                after_type = after_type[:-len(browser_name)].strip()
                        text_to_type = after_type
                elif 'search for' in command_lower:
                    search_index = command_lower.find('search for')
                    if search_index != -1:
                        after_search = command_text[search_index + 10:].strip()
                        # Remove browser specification if at the end
                        for browser_name in ['in safari', 'in chrome', 'in firefox']:
                            if after_search.lower().endswith(browser_name):
                                after_search = after_search[:-len(browser_name)].strip()
                        text_to_type = after_search
                
                if text_to_type:
                    # If it's a search command, focus on search bar first
                    if 'search' in command_lower:
                        macos_controller.click_search_bar(browser)
                        await asyncio.sleep(0.5)
                    
                    success, message = macos_controller.type_in_browser(text_to_type, browser, press_enter)
                    return {'success': success, 'response': message}
                else:
                    return {'success': False, 'response': "Please specify what to type"}
            
            # Handle browser-specific commands
            elif any(f'tell {browser}' in command_lower or f'in {browser}' in command_lower 
                    for browser in ['safari', 'chrome', 'firefox']):
                # This handles enhanced context commands like "tell safari to go to google.com"
                browser = None
                action = None
                target = None
                
                for browser_name in ['safari', 'chrome', 'firefox']:
                    if browser_name in command_lower:
                        browser = browser_name
                        break
                
                if 'go to' in command_lower:
                    parts = command_text.split('go to', 1)
                    if len(parts) > 1:
                        target = parts[1].strip()
                        if not target.startswith(('http://', 'https://')) and '.' in target:
                            target = f'https://{target}'
                        success, message = macos_controller.open_url(target, browser)
                        return {'success': success, 'response': message}
                        
                return {'success': False, 'response': f"Not sure what to do with {browser}"}
                
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