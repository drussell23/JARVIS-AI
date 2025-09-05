#!/usr/bin/env python3
"""
JARVIS System Control Integration
Integrates enhanced macOS system control with JARVIS AI assistant
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime

# Import system control bridge
from system_control_bridge import (
    SystemControlBridge, 
    SystemOperationType,
    AppOperation,
    PreferenceOperation,
    FileOperation,
    ClipboardOperation,
    SystemControlResult
)

# Import JARVIS components
from jarvis_swift_integrated_assistant import JARVISSwiftIntegratedAssistant
from vision.proactive_vision_assistant import IntegratedResponse

logger = logging.getLogger(__name__)

class JARVISSystemControlAssistant(JARVISSwiftIntegratedAssistant):
    """
    Enhanced JARVIS assistant with comprehensive system control capabilities
    """
    
    def __init__(self, user_name: str = "Sir"):
        super().__init__(user_name)
        
        # Initialize system control bridge
        try:
            self.system_control = SystemControlBridge()
            self.system_control_enabled = True
            logger.info("System control initialized successfully")
        except Exception as e:
            logger.warning(f"System control not available: {e}")
            self.system_control_enabled = False
            self.system_control = None
        
        # Track system operations for safety
        self.operation_history: List[SystemControlResult] = []
        self.max_history = 100
        
    async def process_command(self, command: str) -> IntegratedResponse:
        """
        Process command with system control capabilities
        """
        # First, classify the command using Swift
        if self.swift_enabled:
            classification = await self._classify_with_swift(command)
            
            # Check if this is a system control command
            if classification and classification.get('category') == 'system':
                return await self._handle_system_control_command(command, classification)
        
        # Otherwise, process as normal
        return await super().process_vision_command(command)
    
    async def _handle_system_control_command(
        self, 
        command: str, 
        classification: Dict
    ) -> IntegratedResponse:
        """Handle system control commands"""
        
        if not self.system_control_enabled:
            return IntegratedResponse(
                verbal_response="I apologize, but system control is not available at the moment.",
                visual_context={'error': 'System control disabled'},
                action_taken="none",
                confidence=1.0
            )
        
        # Extract action and entities
        action = classification.get('action', '')
        entities = classification.get('entities', {})
        confidence = classification.get('confidence', 0.5)
        
        # Map to system control operations
        result = await self._execute_system_operation(action, entities, command)
        
        # Track operation
        if result:
            self._track_operation(result)
        
        # Generate response
        return self._generate_system_control_response(result, action, entities)
    
    async def _execute_system_operation(
        self,
        action: str,
        entities: Dict,
        original_command: str
    ) -> Optional[SystemControlResult]:
        """Execute the appropriate system operation"""
        
        try:
            # App operations
            if action in ['open', 'launch']:
                app_name = entities.get('app_name', '')
                bundle_id = self._get_bundle_identifier(app_name)
                return await self.system_control.launch_app(bundle_id)
            
            elif action in ['close', 'quit']:
                app_name = entities.get('app_name', '')
                bundle_id = self._get_bundle_identifier(app_name)
                return await self.system_control.close_app(bundle_id)
            
            elif action in ['switch', 'activate']:
                app_name = entities.get('app_name', '')
                bundle_id = self._get_bundle_identifier(app_name)
                return await self.system_control.execute_operation(
                    SystemOperationType.APP_LIFECYCLE,
                    AppOperation.SWITCH_TO,
                    {'bundle_identifier': bundle_id}
                )
            
            # System preferences
            elif action == 'set_volume':
                level = entities.get('level', 0.5)
                return await self.system_control.set_volume(level)
            
            elif action == 'set_brightness':
                level = entities.get('level', 0.5)
                return await self.system_control.set_brightness(level)
            
            # File operations
            elif action == 'search_files':
                query = entities.get('query', '')
                directory = entities.get('directory', '~')
                return await self.system_control.search_files(directory, query)
            
            # Clipboard operations
            elif action == 'copy':
                text = entities.get('text', original_command)
                return await self.system_control.write_clipboard(text)
            
            elif action == 'paste':
                return await self.system_control.read_clipboard()
            
            # Get system info
            elif action == 'system_info':
                return await self.system_control.get_system_info()
            
            # List running apps
            elif action == 'list_apps':
                return await self.system_control.get_running_apps()
            
        except Exception as e:
            logger.error(f"System operation failed: {e}")
            return None
    
    def _get_bundle_identifier(self, app_name: str) -> str:
        """Get bundle identifier for app name"""
        # Common app mappings
        mappings = {
            'safari': 'com.apple.Safari',
            'mail': 'com.apple.mail',
            'messages': 'com.apple.MobileSMS',
            'slack': 'com.tinyspeck.slackmacgap',
            'chrome': 'com.google.Chrome',
            'firefox': 'org.mozilla.firefox',
            'vscode': 'com.microsoft.VSCode',
            'code': 'com.microsoft.VSCode',
            'terminal': 'com.apple.Terminal',
            'finder': 'com.apple.finder',
            'xcode': 'com.apple.dt.Xcode',
            'spotify': 'com.spotify.client',
            'whatsapp': 'net.whatsapp.WhatsApp',
            'zoom': 'us.zoom.xos',
            'discord': 'com.hnc.Discord',
            'notion': 'notion.id',
            'obsidian': 'md.obsidian',
        }
        
        # Check mapping first
        normalized = app_name.lower().strip()
        if normalized in mappings:
            return mappings[normalized]
        
        # Try variations
        if f"com.apple.{normalized}" in mappings.values():
            return f"com.apple.{normalized}"
        
        # Default pattern
        return f"com.{normalized}.{normalized}"
    
    def _generate_system_control_response(
        self,
        result: Optional[SystemControlResult],
        action: str,
        entities: Dict
    ) -> IntegratedResponse:
        """Generate response for system control operation"""
        
        if not result:
            return IntegratedResponse(
                verbal_response="I encountered an error executing that command.",
                visual_context={'error': 'Operation failed'},
                action_taken="error",
                confidence=0.0
            )
        
        if result.success:
            # Generate success response based on action
            if action in ['open', 'launch']:
                app = entities.get('app_name', 'the application')
                response = f"I've launched {app} for you."
            
            elif action in ['close', 'quit']:
                app = entities.get('app_name', 'the application')
                response = f"I've closed {app}."
            
            elif action == 'set_volume':
                level = int(entities.get('level', 0.5) * 100)
                response = f"I've set the volume to {level}%."
            
            elif action == 'set_brightness':
                level = int(entities.get('level', 0.5) * 100)
                response = f"I've set the brightness to {level}%."
            
            elif action == 'search_files':
                if result.result and isinstance(result.result, list):
                    count = len(result.result)
                    response = f"I found {count} files matching your search."
                else:
                    response = "Search completed."
            
            elif action == 'copy':
                response = "I've copied that to the clipboard."
            
            elif action == 'paste':
                if result.result:
                    response = f"Clipboard content: {result.result[:100]}..."
                else:
                    response = "The clipboard is empty."
            
            elif action == 'list_apps':
                if result.result and isinstance(result.result, list):
                    count = len(result.result)
                    apps = [app['name'] for app in result.result[:5]]
                    response = f"You have {count} apps running. Including: {', '.join(apps)}"
                else:
                    response = "I've listed the running applications."
            
            else:
                response = f"I've completed the {action} operation."
            
            return IntegratedResponse(
                verbal_response=response,
                visual_context={
                    'operation': action,
                    'result': result.to_dict(),
                    'entities': entities
                },
                action_taken=action,
                confidence=0.9
            )
        else:
            # Generate error response
            error_msg = result.error or "Unknown error"
            
            if "Permission denied" in error_msg:
                response = (
                    f"I need permission to perform that action. "
                    f"Please grant the necessary permissions in System Preferences."
                )
            elif "not found" in error_msg.lower():
                response = f"I couldn't find what you're looking for."
            else:
                response = f"I couldn't complete that action: {error_msg}"
            
            return IntegratedResponse(
                verbal_response=response,
                visual_context={
                    'operation': action,
                    'error': error_msg,
                    'entities': entities
                },
                action_taken="error",
                confidence=0.3
            )
    
    def _track_operation(self, result: SystemControlResult):
        """Track system operations for safety and auditing"""
        self.operation_history.append(result)
        
        # Trim history
        if len(self.operation_history) > self.max_history:
            self.operation_history = self.operation_history[-self.max_history:]
    
    async def undo_last_operation(self) -> IntegratedResponse:
        """Undo the last system operation if possible"""
        if not self.operation_history:
            return IntegratedResponse(
                verbal_response="There's no recent operation to undo.",
                visual_context={},
                action_taken="none",
                confidence=1.0
            )
        
        last_op = self.operation_history[-1]
        # Implement undo logic based on operation type
        # This is a placeholder - actual implementation would be more complex
        
        return IntegratedResponse(
            verbal_response="I'm sorry, but I can't undo that operation yet.",
            visual_context={'last_operation': last_op.to_dict()},
            action_taken="none",
            confidence=0.5
        )
    
    def get_system_capabilities(self) -> Dict[str, Any]:
        """Get current system capabilities and permissions"""
        capabilities = {
            'system_control_enabled': self.system_control_enabled,
            'swift_classifier_enabled': self.swift_enabled,
            'vision_enabled': True,
            'operations_performed': len(self.operation_history),
            'available_operations': []
        }
        
        if self.system_control_enabled:
            capabilities['available_operations'] = [
                'app_launch', 'app_close', 'app_switch',
                'volume_control', 'brightness_control',
                'file_search', 'clipboard_operations',
                'system_info'
            ]
        
        return capabilities

# Example usage
async def test_system_control_integration():
    """Test JARVIS with system control integration"""
    print("🚀 Testing JARVIS System Control Integration")
    print("=" * 50)
    
    jarvis = JARVISSystemControlAssistant("Sir")
    
    # Get capabilities
    capabilities = jarvis.get_system_capabilities()
    print(f"\n📊 System Capabilities:")
    for key, value in capabilities.items():
        if isinstance(value, list):
            print(f"• {key}: {len(value)} items")
        else:
            print(f"• {key}: {value}")
    
    # Test commands
    test_commands = [
        "What apps are running?",
        "Open Safari",
        "Set volume to 50%",
        "Search for Python files",
        "Copy this text to clipboard",
        "What's on my screen?"
    ]
    
    print("\n🎯 Testing Commands:")
    for cmd in test_commands:
        print(f"\n> {cmd}")
        response = await jarvis.process_command(cmd)
        print(f"< {response.verbal_response}")
        
        if response.action_taken != "none":
            print(f"  Action: {response.action_taken}")
            print(f"  Confidence: {response.confidence:.0%}")
    
    print("\n✅ System control integration test complete!")

if __name__ == "__main__":
    asyncio.run(test_system_control_integration())