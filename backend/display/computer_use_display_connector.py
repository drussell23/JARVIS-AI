#!/usr/bin/env python3
"""
Computer Use Display Connector
================================

Uses Claude Computer Use API for dynamic, vision-based display connection.
Replaces hardcoded workflows with intelligent reasoning.

Features:
- No hardcoded coordinates - Claude finds elements visually
- Dynamic adaptation to UI changes
- Intelligent error recovery
- Voice transparency throughout execution
- Works with macOS Control Center, WiFi, AirPlay

Author: Derek J. Russell
Date: January 2025
Version: 1.0.0
"""

import os
import asyncio
import logging
import base64
import json
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from pathlib import Path
from io import BytesIO

import anthropic
from anthropic.types.beta import BetaMessageParam
from PIL import Image
import pyautogui

logger = logging.getLogger(__name__)


class ComputerUseDisplayConnector:
    """
    Connect to AirPlay devices using Claude Computer Use API
    
    Advantages over coordinate-based approach:
    - Vision-native: Claude sees and understands the UI
    - Adaptive: Works with different macOS versions
    - Intelligent: Handles unexpected dialogs and errors
    - Transparent: Provides voice feedback throughout
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_callback: Optional[Callable[[str], None]] = None,
        display_width: int = 1920,
        display_height: int = 1080,
        display_number: int = 1
    ):
        """
        Initialize Computer Use Display Connector
        
        Args:
            api_key: Anthropic API key (defaults to env var)
            voice_callback: Callback for voice announcements
            display_width: Screen width in pixels
            display_height: Screen height in pixels
            display_number: Display number for multi-monitor setups
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY must be set")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.voice_callback = voice_callback
        
        # Screen configuration
        self.display_width = display_width
        self.display_height = display_height
        self.display_number = display_number
        
        # Scaling factor for screenshots (Computer Use API uses 1024x768 max)
        self.scaling_factor = 1.0
        
        # Statistics
        self.stats = {
            'connections_attempted': 0,
            'connections_successful': 0,
            'tool_calls_made': 0,
            'screenshots_taken': 0,
            'mouse_actions': 0,
            'keyboard_actions': 0,
            'total_tokens_used': 0
        }
        
        logger.info(
            f"[COMPUTER-USE] Initialized Computer Use Display Connector "
            f"(display: {display_width}x{display_height})"
        )
    
    def _speak(self, message: str):
        """Provide voice feedback through JARVIS"""
        if self.voice_callback:
            try:
                self.voice_callback(message)
            except Exception as e:
                logger.error(f"[COMPUTER-USE] Voice callback failed: {e}")
        
        # Always log the message
        logger.info(f"[JARVIS VOICE] {message}")
    
    async def connect_to_device(
        self,
        device_name: str,
        mode: str = "mirror"
    ) -> Dict[str, Any]:
        """
        Connect to AirPlay device using Computer Use API
        
        Args:
            device_name: Name of the AirPlay device (e.g., "Living Room TV")
            mode: Connection mode ("mirror" or "extend")
        
        Returns:
            Dictionary with connection result:
            {
                'success': bool,
                'message': str,
                'device': str,
                'duration': float,
                'tool_calls': int,
                'reasoning': List[str]
            }
        """
        start_time = asyncio.get_event_loop().time()
        self.stats['connections_attempted'] += 1
        
        logger.info(f"[COMPUTER-USE] 🔗 Connecting to '{device_name}' (mode: {mode})")
        self._speak(f"Connecting to {device_name}. Let me locate the control center.")
        
        try:
            # Build initial prompt with clear instructions
            initial_prompt = self._build_connection_prompt(device_name, mode)
            
            # Execute Computer Use workflow
            result = await self._execute_computer_use_workflow(
                initial_prompt,
                device_name
            )
            
            # Calculate duration
            duration = asyncio.get_event_loop().time() - start_time
            
            if result['success']:
                self.stats['connections_successful'] += 1
                logger.info(
                    f"[COMPUTER-USE] ✅ Successfully connected to '{device_name}' "
                    f"in {duration:.2f}s ({result['tool_calls']} tool calls)"
                )
                self._speak(f"Successfully connected to {device_name}.")
            else:
                logger.error(
                    f"[COMPUTER-USE] ❌ Failed to connect to '{device_name}': "
                    f"{result['message']}"
                )
                self._speak(f"I encountered an issue connecting to {device_name}. {result['message']}")
            
            return {
                'success': result['success'],
                'message': result['message'],
                'device': device_name,
                'mode': mode,
                'duration': duration,
                'tool_calls': result['tool_calls'],
                'reasoning': result.get('reasoning', []),
                'stats': {
                    'screenshots_taken': result.get('screenshots_taken', 0),
                    'mouse_actions': result.get('mouse_actions', 0),
                    'keyboard_actions': result.get('keyboard_actions', 0),
                    'tokens_used': result.get('tokens_used', 0)
                }
            }
        
        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(f"[COMPUTER-USE] ❌ {error_msg}", exc_info=True)
            self._speak(f"I encountered an unexpected error: {str(e)}")
            
            return {
                'success': False,
                'message': error_msg,
                'device': device_name,
                'duration': duration,
                'tool_calls': 0,
                'reasoning': []
            }
    
    def _build_connection_prompt(self, device_name: str, mode: str) -> str:
        """Build initial prompt for Claude"""
        return f"""I need you to connect my MacBook to an AirPlay device called "{device_name}".

Here's what you need to do:

1. **Open Control Center**: Located in the top-right corner of the screen (near the clock/battery icons). Click on it to open the Control Center panel.

2. **Find Screen Mirroring**: In Control Center, locate the "Screen Mirroring" button. It usually has an icon showing overlapping rectangles or screens.

3. **Select the device**: Click on Screen Mirroring to open the AirPlay device list, then find and click on "{device_name}".

4. **Verify connection**: Wait for the connection to establish. You should see "{device_name}" marked as connected.

**Important guidelines:**
- Take a screenshot first to see the current state
- Provide clear reasoning for each action you take
- If you encounter any dialogs or errors, explain what you see
- If the device isn't showing up, mention that
- Be patient - UI elements may take a moment to appear after clicking
- If Control Center is already open, skip step 1

Mode: {mode} (mirror the screen)

Please begin by taking a screenshot to see the current state of the screen."""
    
    async def _execute_computer_use_workflow(
        self,
        initial_prompt: str,
        device_name: str
    ) -> Dict[str, Any]:
        """
        Execute Computer Use API workflow with tool calling loop
        
        Args:
            initial_prompt: Initial instructions for Claude
            device_name: Target device name
        
        Returns:
            Result dictionary with success status and metadata
        """
        # Initialize conversation
        messages: List[BetaMessageParam] = [
            {
                "role": "user",
                "content": initial_prompt
            }
        ]
        
        # Track execution
        tool_calls = 0
        screenshots_taken = 0
        mouse_actions = 0
        keyboard_actions = 0
        tokens_used = 0
        reasoning_steps = []
        max_iterations = 30  # Prevent infinite loops
        iteration = 0
        
        # Define tools for Computer Use API
        tools = [
            {
                "type": "computer_20241022",
                "name": "computer",
                "display_width_px": self.display_width,
                "display_height_px": self.display_height,
                "display_number": self.display_number,
            },
            {
                "type": "bash_20241022",
                "name": "bash"
            }
        ]
        
        try:
            while iteration < max_iterations:
                iteration += 1
                logger.debug(f"[COMPUTER-USE] Iteration {iteration}/{max_iterations}")
                
                # Call Claude with Computer Use tools
                response = self.client.beta.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=4096,
                    tools=tools,
                    messages=messages,
                    betas=["computer-use-2024-10-22"]
                )
                
                # Track token usage
                tokens_used += response.usage.input_tokens + response.usage.output_tokens
                self.stats['total_tokens_used'] += response.usage.input_tokens + response.usage.output_tokens
                
                # Extract reasoning from text blocks
                for block in response.content:
                    if hasattr(block, 'text') and block.text:
                        reasoning_steps.append(block.text)
                        logger.info(f"[CLAUDE REASONING] {block.text}")
                        
                        # Provide voice updates for key reasoning
                        if any(keyword in block.text.lower() for keyword in 
                               ['opening', 'clicking', 'found', 'selecting', 'connected']):
                            # Extract concise update
                            self._speak(self._extract_voice_update(block.text))
                
                # Check stop reason
                if response.stop_reason == "end_turn":
                    # Claude has finished - extract final status
                    final_text = " ".join(
                        block.text for block in response.content 
                        if hasattr(block, 'text')
                    )
                    
                    # Check for success indicators
                    success = self._check_connection_success(final_text, device_name)
                    
                    return {
                        'success': success,
                        'message': final_text if not success else f"Connected to {device_name}",
                        'tool_calls': tool_calls,
                        'screenshots_taken': screenshots_taken,
                        'mouse_actions': mouse_actions,
                        'keyboard_actions': keyboard_actions,
                        'tokens_used': tokens_used,
                        'reasoning': reasoning_steps
                    }
                
                elif response.stop_reason == "tool_use":
                    # Claude wants to use tools - execute them
                    tool_results = []
                    
                    for block in response.content:
                        if block.type == "tool_use":
                            tool_calls += 1
                            self.stats['tool_calls_made'] += 1
                            
                            # Execute the tool
                            tool_name = block.name
                            tool_input = block.input
                            
                            logger.info(
                                f"[COMPUTER-USE] Tool call: {tool_name} "
                                f"with input: {json.dumps(tool_input, indent=2)}"
                            )
                            
                            # Execute based on tool type
                            if tool_name == "computer":
                                result = await self._execute_computer_tool(
                                    tool_input,
                                    screenshots_taken,
                                    mouse_actions,
                                    keyboard_actions
                                )
                                
                                # Update counters
                                action = tool_input.get('action', '')
                                if action == 'screenshot':
                                    screenshots_taken += 1
                                    self.stats['screenshots_taken'] += 1
                                elif action in ['mouse_move', 'left_click', 'right_click', 
                                              'middle_click', 'double_click']:
                                    mouse_actions += 1
                                    self.stats['mouse_actions'] += 1
                                elif action in ['type', 'key']:
                                    keyboard_actions += 1
                                    self.stats['keyboard_actions'] += 1
                            
                            elif tool_name == "bash":
                                result = await self._execute_bash_tool(tool_input)
                            
                            else:
                                result = f"Unknown tool: {tool_name}"
                            
                            # Add result to tool_results
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result
                            })
                    
                    # Add assistant response and tool results to conversation
                    messages.append({
                        "role": "assistant",
                        "content": response.content
                    })
                    messages.append({
                        "role": "user",
                        "content": tool_results
                    })
                
                elif response.stop_reason == "max_tokens":
                    logger.warning("[COMPUTER-USE] Hit max_tokens limit")
                    return {
                        'success': False,
                        'message': "Response too long (hit token limit)",
                        'tool_calls': tool_calls,
                        'screenshots_taken': screenshots_taken,
                        'mouse_actions': mouse_actions,
                        'keyboard_actions': keyboard_actions,
                        'tokens_used': tokens_used,
                        'reasoning': reasoning_steps
                    }
                
                else:
                    logger.warning(f"[COMPUTER-USE] Unexpected stop_reason: {response.stop_reason}")
                    return {
                        'success': False,
                        'message': f"Unexpected stop reason: {response.stop_reason}",
                        'tool_calls': tool_calls,
                        'screenshots_taken': screenshots_taken,
                        'mouse_actions': mouse_actions,
                        'keyboard_actions': keyboard_actions,
                        'tokens_used': tokens_used,
                        'reasoning': reasoning_steps
                    }
            
            # Max iterations reached
            logger.warning(f"[COMPUTER-USE] Max iterations ({max_iterations}) reached")
            return {
                'success': False,
                'message': f"Workflow did not complete within {max_iterations} iterations",
                'tool_calls': tool_calls,
                'screenshots_taken': screenshots_taken,
                'mouse_actions': mouse_actions,
                'keyboard_actions': keyboard_actions,
                'tokens_used': tokens_used,
                'reasoning': reasoning_steps
            }
        
        except Exception as e:
            logger.error(f"[COMPUTER-USE] Workflow execution error: {e}", exc_info=True)
            return {
                'success': False,
                'message': f"Workflow error: {str(e)}",
                'tool_calls': tool_calls,
                'screenshots_taken': screenshots_taken,
                'mouse_actions': mouse_actions,
                'keyboard_actions': keyboard_actions,
                'tokens_used': tokens_used,
                'reasoning': reasoning_steps
            }
    
    async def _execute_computer_tool(
        self,
        tool_input: Dict[str, Any],
        screenshots_taken: int,
        mouse_actions: int,
        keyboard_actions: int
    ) -> str:
        """
        Execute computer tool action (screenshot, mouse, keyboard)
        
        Args:
            tool_input: Tool parameters from Claude
            screenshots_taken: Count of screenshots (for tracking)
            mouse_actions: Count of mouse actions
            keyboard_actions: Count of keyboard actions
        
        Returns:
            Tool result as string or base64 image
        """
        action = tool_input.get('action', '')
        
        try:
            if action == 'screenshot':
                # Take screenshot and return as base64
                logger.info("[COMPUTER-USE] 📸 Taking screenshot...")
                screenshot = pyautogui.screenshot()
                
                # Resize if needed (Computer Use API prefers <= 1024x768)
                max_width = 1024
                max_height = 768
                if screenshot.width > max_width or screenshot.height > max_height:
                    # Calculate scaling
                    scale = min(max_width / screenshot.width, max_height / screenshot.height)
                    new_width = int(screenshot.width * scale)
                    new_height = int(screenshot.height * scale)
                    screenshot = screenshot.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    logger.info(f"[COMPUTER-USE] Resized screenshot to {new_width}x{new_height}")
                
                # Convert to base64
                buffer = BytesIO()
                screenshot.save(buffer, format='PNG')
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                return img_base64
            
            elif action == 'mouse_move':
                # Move mouse to coordinate
                x = tool_input.get('coordinate', [0, 0])[0]
                y = tool_input.get('coordinate', [0, 0])[1]
                logger.info(f"[COMPUTER-USE] 🖱️  Moving mouse to ({x}, {y})")
                pyautogui.moveTo(x, y, duration=0.2)
                return f"Moved mouse to ({x}, {y})"
            
            elif action == 'left_click':
                # Left click at current position or specified coordinate
                if 'coordinate' in tool_input:
                    x, y = tool_input['coordinate']
                    logger.info(f"[COMPUTER-USE] 🖱️  Left clicking at ({x}, {y})")
                    pyautogui.click(x, y)
                    return f"Left clicked at ({x}, {y})"
                else:
                    logger.info("[COMPUTER-USE] 🖱️  Left clicking at current position")
                    pyautogui.click()
                    return "Left clicked at current position"
            
            elif action == 'right_click':
                # Right click
                if 'coordinate' in tool_input:
                    x, y = tool_input['coordinate']
                    logger.info(f"[COMPUTER-USE] 🖱️  Right clicking at ({x}, {y})")
                    pyautogui.rightClick(x, y)
                    return f"Right clicked at ({x}, {y})"
                else:
                    logger.info("[COMPUTER-USE] 🖱️  Right clicking at current position")
                    pyautogui.rightClick()
                    return "Right clicked at current position"
            
            elif action == 'double_click':
                # Double click
                if 'coordinate' in tool_input:
                    x, y = tool_input['coordinate']
                    logger.info(f"[COMPUTER-USE] 🖱️  Double clicking at ({x}, {y})")
                    pyautogui.doubleClick(x, y)
                    return f"Double clicked at ({x}, {y})"
                else:
                    logger.info("[COMPUTER-USE] 🖱️  Double clicking at current position")
                    pyautogui.doubleClick()
                    return "Double clicked at current position"
            
            elif action == 'middle_click':
                # Middle click
                if 'coordinate' in tool_input:
                    x, y = tool_input['coordinate']
                    logger.info(f"[COMPUTER-USE] 🖱️  Middle clicking at ({x}, {y})")
                    pyautogui.middleClick(x, y)
                    return f"Middle clicked at ({x}, {y})"
                else:
                    logger.info("[COMPUTER-USE] 🖱️  Middle clicking at current position")
                    pyautogui.middleClick()
                    return "Middle clicked at current position"
            
            elif action == 'type':
                # Type text
                text = tool_input.get('text', '')
                logger.info(f"[COMPUTER-USE] ⌨️  Typing: {text}")
                pyautogui.typewrite(text, interval=0.05)
                return f"Typed: {text}"
            
            elif action == 'key':
                # Press key
                key = tool_input.get('text', '')
                logger.info(f"[COMPUTER-USE] ⌨️  Pressing key: {key}")
                pyautogui.press(key)
                return f"Pressed key: {key}"
            
            elif action == 'cursor_position':
                # Get cursor position
                x, y = pyautogui.position()
                logger.info(f"[COMPUTER-USE] 📍 Cursor position: ({x}, {y})")
                return f"Cursor position: ({x}, {y})"
            
            else:
                logger.warning(f"[COMPUTER-USE] Unknown computer action: {action}")
                return f"Unknown action: {action}"
        
        except Exception as e:
            error_msg = f"Error executing {action}: {str(e)}"
            logger.error(f"[COMPUTER-USE] {error_msg}", exc_info=True)
            return error_msg
    
    async def _execute_bash_tool(self, tool_input: Dict[str, Any]) -> str:
        """
        Execute bash command (if needed for system queries)
        
        Args:
            tool_input: Tool parameters from Claude
        
        Returns:
            Command output
        """
        import subprocess
        
        command = tool_input.get('command', '')
        logger.info(f"[COMPUTER-USE] 🔧 Executing bash: {command}")
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            output = result.stdout if result.returncode == 0 else result.stderr
            logger.info(f"[COMPUTER-USE] Bash output: {output[:200]}")
            return output
        
        except subprocess.TimeoutExpired:
            return "Command timed out after 10 seconds"
        except Exception as e:
            return f"Error executing command: {str(e)}"
    
    def _extract_voice_update(self, reasoning_text: str) -> str:
        """
        Extract concise voice update from Claude's reasoning
        
        Args:
            reasoning_text: Full reasoning text from Claude
        
        Returns:
            Concise voice-friendly update
        """
        # Extract key sentences
        text_lower = reasoning_text.lower()
        
        if 'opening control center' in text_lower:
            return "Opening Control Center"
        elif 'clicking on control center' in text_lower:
            return "Clicking Control Center"
        elif 'found screen mirroring' in text_lower or 'see screen mirroring' in text_lower:
            return "Found Screen Mirroring button"
        elif 'clicking screen mirroring' in text_lower or 'click on screen mirroring' in text_lower:
            return "Opening Screen Mirroring menu"
        elif 'found' in text_lower and 'living room' in text_lower:
            return "Found Living Room TV in the list"
        elif 'selecting' in text_lower or 'clicking on' in text_lower:
            return "Selecting the device"
        elif 'connected' in text_lower or 'connection established' in text_lower:
            return "Connection established"
        elif 'waiting' in text_lower:
            return "Waiting for the menu to appear"
        elif 'not seeing' in text_lower or 'cannot find' in text_lower:
            return "Looking for the control..."
        
        # Default: return first sentence
        sentences = reasoning_text.split('.')
        if sentences:
            return sentences[0].strip()[:80]  # Max 80 chars
        
        return ""
    
    def _check_connection_success(self, final_text: str, device_name: str) -> bool:
        """
        Check if connection was successful based on Claude's final response
        
        Args:
            final_text: Final text from Claude
            device_name: Expected device name
        
        Returns:
            True if connection successful
        """
        text_lower = final_text.lower()
        device_lower = device_name.lower()
        
        # Success indicators
        success_phrases = [
            f"connected to {device_lower}",
            f"{device_lower} is now connected",
            "connection established",
            "successfully connected",
            "screen mirroring is active",
            f"{device_lower} is now mirroring"
        ]
        
        return any(phrase in text_lower for phrase in success_phrases)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connector statistics"""
        return {
            **self.stats,
            'success_rate': (
                self.stats['connections_successful'] / self.stats['connections_attempted']
                if self.stats['connections_attempted'] > 0 else 0.0
            )
        }


# ============================================================================
# Factory Function
# ============================================================================

def get_computer_use_connector(
    api_key: Optional[str] = None,
    voice_callback: Optional[Callable[[str], None]] = None,
    display_width: int = 1920,
    display_height: int = 1080
) -> ComputerUseDisplayConnector:
    """
    Get Computer Use Display Connector instance
    
    Args:
        api_key: Anthropic API key
        voice_callback: JARVIS voice callback
        display_width: Screen width
        display_height: Screen height
    
    Returns:
        ComputerUseDisplayConnector instance
    """
    return ComputerUseDisplayConnector(
        api_key=api_key,
        voice_callback=voice_callback,
        display_width=display_width,
        display_height=display_height
    )


# ============================================================================
# Test/Demo
# ============================================================================

async def main():
    """Demo Computer Use Display Connector"""
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "=" * 80)
    print("🚀 Computer Use Display Connector - Demo")
    print("=" * 80)
    
    # Mock voice callback
    def voice_callback(message: str):
        print(f"\n🔊 [JARVIS]: {message}\n")
    
    # Get screen size
    screen_width, screen_height = pyautogui.size()
    print(f"\n📺 Screen size: {screen_width}x{screen_height}")
    
    # Create connector
    connector = get_computer_use_connector(
        voice_callback=voice_callback,
        display_width=screen_width,
        display_height=screen_height
    )
    
    # Get device name from user or use default
    device_name = "Living Room TV"
    if len(sys.argv) > 1:
        device_name = sys.argv[1]
    
    print(f"\n🎯 Connecting to: {device_name}")
    print("   (Claude will dynamically locate UI elements)\n")
    
    # Execute connection
    result = await connector.connect_to_device(device_name)
    
    # Display results
    print("\n" + "=" * 80)
    print("📊 Connection Result")
    print("=" * 80)
    print(f"Success: {'✅ Yes' if result['success'] else '❌ No'}")
    print(f"Message: {result['message']}")
    print(f"Duration: {result['duration']:.2f}s")
    print(f"Tool Calls: {result['tool_calls']}")
    print(f"\nStats:")
    print(f"  Screenshots: {result['stats']['screenshots_taken']}")
    print(f"  Mouse Actions: {result['stats']['mouse_actions']}")
    print(f"  Keyboard Actions: {result['stats']['keyboard_actions']}")
    print(f"  Tokens Used: {result['stats']['tokens_used']}")
    
    if result.get('reasoning'):
        print(f"\n📝 Reasoning Steps ({len(result['reasoning'])}):")
        for i, step in enumerate(result['reasoning'][:5], 1):  # Show first 5
            print(f"  {i}. {step[:100]}...")
    
    # Show overall stats
    print("\n📈 Overall Stats:")
    stats = connector.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
