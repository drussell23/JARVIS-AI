#!/usr/bin/env python3
"""
WebSocket Server Bridge for JARVIS Voice Unlock Daemon
======================================================

Provides WebSocket API for the Objective-C daemon.
"""

import asyncio
import websockets
import json
import logging
import subprocess
import os
from datetime import datetime
from typing import Optional, Dict, Any, Set
from screen_lock_detector import is_screen_locked

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoiceUnlockWebSocketServer:
    """WebSocket server that interfaces with the Objective-C daemon"""
    
    def __init__(self, port: int = 8765):
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.daemon_process = None
        self.enrolled_users_file = os.path.expanduser("~/.jarvis/voice_unlock/enrolled_users.json")
        
    async def register_client(self, websocket):
        """Register a new WebSocket client"""
        self.clients.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self.clients)}")
        
    async def unregister_client(self, websocket):
        """Unregister a WebSocket client"""
        self.clients.discard(websocket)
        logger.info(f"Client disconnected. Total clients: {len(self.clients)}")
        
    async def handle_message(self, websocket, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming WebSocket message"""
        msg_type = message.get("type")
        command = message.get("command")
        parameters = message.get("parameters", {})
        
        logger.info(f"Received: type={msg_type}, command={command}")
        
        if msg_type == "command":
            if command == "handshake":
                return {
                    "type": "handshake",
                    "success": True,
                    "version": "1.0",
                    "daemon": "JARVIS Voice Unlock"
                }
                
            elif command == "get_status":
                # Get daemon status
                is_running = self.check_daemon_running()
                enrolled_users = self.get_enrolled_users()
                
                # Check actual screen lock status
                screen_locked = is_screen_locked()
                logger.info(f"Screen lock status check: {'LOCKED' if screen_locked else 'UNLOCKED'}")
                
                return {
                    "type": "status",
                    "success": True,
                    "status": {
                        "isMonitoring": is_running,
                        "isScreenLocked": screen_locked,
                        "enrolledUser": enrolled_users[0] if enrolled_users else "none",
                        "failedAttempts": 0,
                        "state": 1 if is_running else 0
                    }
                }
                
            elif command == "start_monitoring":
                # Start daemon if not running
                if not self.check_daemon_running():
                    success = self.start_daemon()
                    return {
                        "type": "command_response",
                        "command": command,
                        "success": success,
                        "message": "Monitoring started" if success else "Failed to start monitoring"
                    }
                else:
                    return {
                        "type": "command_response",
                        "command": command,
                        "success": True,
                        "message": "Monitoring already active"
                    }
                    
            elif command == "stop_monitoring":
                # Stop daemon
                success = self.stop_daemon()
                return {
                    "type": "command_response",
                    "command": command,
                    "success": success,
                    "message": "Monitoring stopped" if success else "Failed to stop monitoring"
                }
                
            elif command == "unlock_screen":
                # Direct unlock command from JARVIS
                logger.info("Received unlock_screen command from JARVIS")
                
                # Check if password is stored
                password = self.retrieve_keychain_password()
                if not password:
                    return {
                        "type": "command_response",
                        "command": command,
                        "success": False,
                        "message": "No password stored. Run enable_screen_unlock.sh first."
                    }
                
                # Perform unlock
                success = await self.perform_screen_unlock(password)
                return {
                    "type": "command_response", 
                    "command": command,
                    "success": success,
                    "message": "Screen unlocked" if success else "Failed to unlock screen"
                }
                
            elif command == "lock_screen":
                # Lock screen command from JARVIS
                logger.info("Received lock_screen command from JARVIS")
                success = await self.perform_screen_lock()
                return {
                    "type": "command_response",
                    "command": command,
                    "success": success,
                    "message": "Screen locked" if success else "Failed to lock screen"
                }
                
            else:
                return {
                    "type": "error",
                    "message": f"Unknown command: {command}",
                    "success": False
                }
        
        return {
            "type": "error",
            "message": f"Unknown message type: {msg_type}",
            "success": False
        }
        
    def check_daemon_running(self) -> bool:
        """Check if the Voice Unlock daemon is running"""
        try:
            result = subprocess.run(
                ["pgrep", "-f", "JARVISVoiceUnlockDaemon"],
                capture_output=True
            )
            return result.returncode == 0
        except:
            return False
            
    def get_enrolled_users(self) -> list:
        """Get list of enrolled user names"""
        try:
            if os.path.exists(self.enrolled_users_file):
                with open(self.enrolled_users_file) as f:
                    users = json.load(f)
                    return [user.get("name", "unknown") for user in users.values()]
        except:
            pass
        return []
        
    def start_daemon(self) -> bool:
        """Start the Voice Unlock daemon"""
        try:
            daemon_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "bin/JARVISVoiceUnlockDaemon"
            )
            
            if not os.path.exists(daemon_path):
                logger.error(f"Daemon not found at {daemon_path}")
                return False
                
            self.daemon_process = subprocess.Popen(
                [daemon_path, "--debug"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Give it time to start
            asyncio.get_event_loop().call_later(1.0, self.check_daemon_started)
            
            return True
        except Exception as e:
            logger.error(f"Failed to start daemon: {e}")
            return False
            
    def check_daemon_started(self):
        """Check if daemon started successfully"""
        if self.daemon_process and self.daemon_process.poll() is None:
            logger.info("Daemon started successfully")
        else:
            logger.error("Daemon failed to start")
            
    def stop_daemon(self) -> bool:
        """Stop the Voice Unlock daemon"""
        try:
            subprocess.run(["pkill", "-f", "JARVISVoiceUnlockDaemon"])
            return True
        except:
            return False
            
    async def websocket_handler(self, websocket, path):
        """Handle WebSocket connections"""
        logger.info(f"New connection on path: {path}")
        await self.register_client(websocket)
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    response = await self.handle_message(websocket, data)
                    await websocket.send(json.dumps(response))
                    
                except json.JSONDecodeError:
                    error_response = {
                        "type": "error",
                        "message": "Invalid JSON",
                        "success": False
                    }
                    await websocket.send(json.dumps(error_response))
                    
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
                    error_response = {
                        "type": "error",
                        "message": str(e),
                        "success": False
                    }
                    await websocket.send(json.dumps(error_response))
                    
        except websockets.exceptions.ConnectionClosed:
            pass
            
        finally:
            await self.unregister_client(websocket)
            
    def retrieve_keychain_password(self) -> Optional[str]:
        """Retrieve the stored password from macOS Keychain"""
        try:
            result = subprocess.run([
                'security', 'find-generic-password',
                '-s', 'com.jarvis.voiceunlock',
                '-a', 'unlock_token',
                '-w'  # Print only the password
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                logger.error(f"Failed to retrieve password: {result.stderr}")
                return None
        except Exception as e:
            logger.error(f"Error retrieving password: {e}")
            return None
    
    def escape_password_for_applescript(self, password: str) -> str:
        """Escape special characters in password for AppleScript"""
        # Escape backslashes first, then quotes, then other special characters
        escaped = password.replace('\\', '\\\\')  # Escape backslashes
        escaped = escaped.replace('"', '\\"')     # Escape double quotes  
        escaped = escaped.replace("'", "\\'")     # Escape single quotes
        escaped = escaped.replace('$', '\\$')     # Escape dollar signs
        escaped = escaped.replace('`', '\\`')     # Escape backticks
        # Add more special characters that might need escaping
        escaped = escaped.replace('!', '\\!')     # Escape exclamation marks
        escaped = escaped.replace('@', '\\@')     # Escape at symbols
        escaped = escaped.replace('#', '\\#')     # Escape hash/pound symbols
        escaped = escaped.replace('&', '\\&')     # Escape ampersands
        escaped = escaped.replace('*', '\\*')     # Escape asterisks
        escaped = escaped.replace('(', '\\(')     # Escape open parentheses
        escaped = escaped.replace(')', '\\)')     # Escape close parentheses
        return escaped

    async def perform_screen_unlock(self, password: str) -> bool:
        """Perform the actual screen unlock using AppleScript"""
        try:
            # Wake the display first
            logger.info("Waking display...")
            subprocess.run(['caffeinate', '-u', '-t', '1'])
            await asyncio.sleep(1)
            
            # Move mouse to wake screen and ensure loginwindow is active
            wake_script = '''
            tell application "System Events"
                -- Wake the display by moving mouse
                do shell script "caffeinate -u -t 2"
                delay 0.5
                
                -- Click on the user profile to show password field
                -- This is more reliable than keyboard navigation
                click at {720, 860}
                delay 1
                
                -- Make sure loginwindow is frontmost
                set frontmost of process "loginwindow" to true
                delay 0.5
                
                -- Sometimes need to click again to ensure password field is active
                click at {720, 500}
                delay 0.5
                
                -- Clear any existing text
                keystroke "a" using command down
                delay 0.1
                key code 51
                delay 0.2
            end tell
            '''
            
            subprocess.run(['osascript', '-e', wake_script])
            await asyncio.sleep(0.5)
            
            # Escape password for AppleScript
            escaped_password = self.escape_password_for_applescript(password)
            logger.info("Password escaped for AppleScript input")
            
            # Type password and press return
            # Type the password using System Events
            logger.info(f"Typing password with {len(password)} characters")
            
            # Clear any existing text first
            subprocess.run([
                'osascript', '-e',
                'tell application "System Events" to keystroke "a" using command down'
            ])
            await asyncio.sleep(0.1)
            subprocess.run([
                'osascript', '-e',
                'tell application "System Events" to key code 51'  # Delete key
            ])
            await asyncio.sleep(0.2)
            
            # Type password character by character with proper special character handling
            logger.info(f"Typing password with {len(password)} characters")
            
            # Map special characters to their key codes with modifiers
            special_char_map = {
                '!': {'keycode': 18, 'modifiers': 'shift down'},  # Shift+1
                '@': {'keycode': 19, 'modifiers': 'shift down'},  # Shift+2  
                '#': {'keycode': 20, 'modifiers': 'shift down'},  # Shift+3
                '$': {'keycode': 21, 'modifiers': 'shift down'},  # Shift+4
                '%': {'keycode': 22, 'modifiers': 'shift down'},  # Shift+5
                '^': {'keycode': 23, 'modifiers': 'shift down'},  # Shift+6
                '&': {'keycode': 24, 'modifiers': 'shift down'},  # Shift+7
                '*': {'keycode': 25, 'modifiers': 'shift down'},  # Shift+8
                '(': {'keycode': 26, 'modifiers': 'shift down'},  # Shift+9
                ')': {'keycode': 27, 'modifiers': 'shift down'},  # Shift+0
            }
            
            # Type each character
            for i, char in enumerate(password):
                if char in special_char_map:
                    # Use key code for special characters
                    info = special_char_map[char]
                    script = f'tell application "System Events" to key code {info["keycode"]} using {{{info["modifiers"]}}}'
                    logger.info(f"Typing special char at position {i+1}: '{char}' using keycode {info['keycode']} with shift")
                else:
                    # Use keystroke for regular characters
                    script = f'tell application "System Events" to keystroke "{char}"'
                    logger.info(f"Typing regular char at position {i+1}: '{char}'")
                
                result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
                if result.returncode != 0:
                    logger.error(f"Failed to type character '{char}': {result.stderr}")
                await asyncio.sleep(0.01)  # Faster typing - reduced from 0.05 to 0.01
            
            # Press return
            await asyncio.sleep(0.2)
            logger.info("Pressing return key...")
            result = subprocess.run([
                'osascript', '-e', 
                'tell application "System Events" to key code 36'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Screen unlock command executed successfully")
                return True
            else:
                logger.error(f"Screen unlock failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error performing screen unlock: {e}")
            return False
    
    async def perform_screen_lock(self) -> bool:
        """Lock the Mac screen using various methods"""
        try:
            logger.info("Locking screen...")
            
            # Method 1: Use CGSession (most reliable)
            try:
                result = subprocess.run([
                    '/System/Library/CoreServices/Menu Extras/User.menu/Contents/Resources/CGSession',
                    '-suspend'
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info("Screen locked successfully using CGSession")
                    return True
            except Exception as e:
                logger.debug(f"CGSession method failed: {e}")
            
            # Method 2: Use loginwindow
            try:
                result = subprocess.run([
                    'osascript', '-e', 
                    'tell application "System Events" to tell process "loginwindow" to keystroke "q" using {command down, control down}'
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info("Screen locked successfully using loginwindow")
                    return True
            except Exception as e:
                logger.debug(f"Loginwindow method failed: {e}")
            
            # Method 3: Use ScreenSaverEngine
            try:
                # Start screensaver which will require password on wake
                subprocess.run([
                    'open', '-a', 'ScreenSaverEngine'
                ])
                logger.info("Started screensaver (will lock if password required)")
                return True
            except Exception as e:
                logger.debug(f"ScreenSaver method failed: {e}")
            
            # Method 4: Fallback to sleep display
            try:
                subprocess.run(['pmset', 'displaysleepnow'])
                logger.info("Put display to sleep")
                return True
            except Exception as e:
                logger.debug(f"Display sleep method failed: {e}")
                
            logger.error("All lock methods failed")
            return False
                
        except Exception as e:
            logger.error(f"Error locking screen: {e}")
            return False
    
    async def start(self):
        """Start the WebSocket server"""
        logger.info(f"Starting Voice Unlock WebSocket server on port {self.port}")
        
        async with websockets.serve(self.websocket_handler, "localhost", self.port):
            logger.info(f"Server listening on ws://localhost:{self.port}")
            await asyncio.Future()  # Run forever


def main():
    """Main entry point"""
    server = VoiceUnlockWebSocketServer(port=8765)
    asyncio.run(server.start())


if __name__ == "__main__":
    main()