"""
Voice Unlock Startup Integration
================================

Integrates the Voice Unlock system into JARVIS's main startup process.
"""

import asyncio
import subprocess
import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class VoiceUnlockStartup:
    """Manages Voice Unlock system startup"""
    
    def __init__(self):
        self.websocket_process: Optional[subprocess.Popen] = None
        self.daemon_process: Optional[subprocess.Popen] = None
        self.voice_unlock_dir = Path(__file__).parent
        self.websocket_port = 8765
        self.initialized = False
        
    async def start(self) -> bool:
        """Start the Voice Unlock system components"""
        try:
            logger.info("🔐 Starting Voice Unlock system...")
            
            # Check if password is stored
            if not self._check_password_stored():
                logger.warning("⚠️  Voice Unlock password not configured")
                logger.info("   Run: backend/voice_unlock/enable_screen_unlock.sh")
                return False
            
            # Start WebSocket server
            if not await self._start_websocket_server():
                logger.error("Failed to start Voice Unlock WebSocket server")
                return False
            
            # Give WebSocket server time to start
            await asyncio.sleep(2)
            
            # Start daemon automatically
            logger.info("✅ Voice Unlock WebSocket server ready on port 8765")
            logger.info("   Voice Unlock is ready to use")
            
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Voice Unlock startup error: {e}")
            return False
    
    def _check_password_stored(self) -> bool:
        """Check if password is stored in Keychain"""
        try:
            result = subprocess.run([
                'security', 'find-generic-password',
                '-s', 'com.jarvis.voiceunlock',
                '-a', 'unlock_token'
            ], capture_output=True, text=True)
            
            return result.returncode == 0
        except Exception:
            return False
    
    async def _start_websocket_server(self) -> bool:
        """Start the Python WebSocket server"""
        try:
            # Kill any existing process on the port
            subprocess.run(
                f"lsof -ti:{self.websocket_port} | xargs kill -9",
                shell=True,
                capture_output=True
            )
            await asyncio.sleep(1)
            
            # Start WebSocket server
            server_script = self.voice_unlock_dir / "objc" / "server" / "websocket_server.py"
            if not server_script.exists():
                logger.error(f"WebSocket server script not found: {server_script}")
                return False
            
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.voice_unlock_dir.parent)
            
            self.websocket_process = subprocess.Popen(
                [sys.executable, str(server_script)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env
            )
            
            logger.info(f"Voice Unlock WebSocket server started (PID: {self.websocket_process.pid})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
            return False
    
    async def start_daemon_if_needed(self) -> bool:
        """Start the Voice Unlock daemon if not running"""
        try:
            # Check if daemon is already running
            result = subprocess.run(
                ["pgrep", "-f", "JARVISVoiceUnlockDaemon"],
                capture_output=True
            )
            
            if result.returncode == 0:
                logger.info("Voice Unlock daemon already running")
                return True
            
            # Start daemon
            daemon_path = self.voice_unlock_dir / "objc" / "bin" / "JARVISVoiceUnlockDaemon"
            if not daemon_path.exists():
                logger.error(f"Voice Unlock daemon not found: {daemon_path}")
                logger.info("Build with: cd backend/voice_unlock/objc && make")
                return False
            
            self.daemon_process = subprocess.Popen(
                [str(daemon_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            logger.info(f"Voice Unlock daemon started (PID: {self.daemon_process.pid})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start daemon: {e}")
            return False
    
    async def stop(self):
        """Stop Voice Unlock components"""
        logger.info("Stopping Voice Unlock system...")
        
        if self.websocket_process:
            self.websocket_process.terminate()
            try:
                self.websocket_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.websocket_process.kill()
            self.websocket_process = None
        
        if self.daemon_process:
            self.daemon_process.terminate()
            try:
                self.daemon_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.daemon_process.kill()
            self.daemon_process = None
        
        # Kill any lingering processes
        subprocess.run("pkill -f websocket_server.py", shell=True, capture_output=True)
        subprocess.run("pkill -f JARVISVoiceUnlockDaemon", shell=True, capture_output=True)
        
        logger.info("Voice Unlock system stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get Voice Unlock system status"""
        return {
            "initialized": self.initialized,
            "websocket_running": self.websocket_process is not None and self.websocket_process.poll() is None,
            "daemon_running": self.daemon_process is not None and self.daemon_process.poll() is None,
            "password_stored": self._check_password_stored(),
            "websocket_port": self.websocket_port
        }


# Global instance
voice_unlock_startup = VoiceUnlockStartup()


async def initialize_voice_unlock_system():
    """Initialize Voice Unlock system for JARVIS integration"""
    global voice_unlock_startup
    return await voice_unlock_startup.start()


async def shutdown_voice_unlock_system():
    """Shutdown Voice Unlock system"""
    global voice_unlock_startup
    await voice_unlock_startup.stop()


# Import for backwards compatibility
import sys