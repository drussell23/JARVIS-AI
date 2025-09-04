"""
Swift Video Bridge for macOS Screen Recording
Provides Python interface to Swift-based screen capture with proper permissions
"""

import os
import json
import subprocess
import asyncio
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import shutil

logger = logging.getLogger(__name__)

@dataclass
class SwiftCaptureConfig:
    """Configuration for Swift video capture"""
    display_id: int = 0
    fps: int = 30
    resolution: str = "1920x1080"
    output_path: Optional[str] = None
    
    @classmethod
    def from_env(cls):
        """Create config from environment variables"""
        return cls(
            display_id=int(os.getenv('VIDEO_CAPTURE_DISPLAY_ID', '0')),
            fps=int(os.getenv('VIDEO_CAPTURE_FPS', '30')),
            resolution=os.getenv('VIDEO_CAPTURE_RESOLUTION', '1920x1080'),
            output_path=os.getenv('VIDEO_CAPTURE_OUTPUT_PATH')
        )

class SwiftVideoBridge:
    """Bridge between Python and Swift video capture module"""
    
    def __init__(self, config: Optional[SwiftCaptureConfig] = None):
        self.config = config or SwiftCaptureConfig.from_env()
        self.swift_module_path = Path(__file__).parent / "SwiftVideoCapture.swift"
        self.compiled_path = Path(__file__).parent / "SwiftVideoCapture"
        self.is_compiled = False
        self._process: Optional[subprocess.Popen] = None
        
        logger.info(f"Swift Video Bridge initialized with config: {self.config}")
        
    def _ensure_swift_available(self) -> bool:
        """Check if Swift is available on the system"""
        try:
            result = subprocess.run(['swift', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"Swift available: {result.stdout.split()[2]}")
                return True
        except Exception as e:
            logger.error(f"Swift not available: {e}")
        return False
    
    def _compile_swift_module(self) -> bool:
        """Compile the Swift module if needed"""
        if self.is_compiled and self.compiled_path.exists():
            return True
            
        if not self._ensure_swift_available():
            return False
            
        try:
            logger.info("Compiling Swift video capture module...")
            
            # Compile with optimization
            compile_cmd = [
                'swiftc',
                '-O',  # Optimize for release
                '-framework', 'AVFoundation',
                '-framework', 'CoreGraphics',
                '-framework', 'CoreMedia',
                '-framework', 'CoreVideo',
                '-framework', 'AppKit',
                str(self.swift_module_path),
                '-o', str(self.compiled_path)
            ]
            
            result = subprocess.run(compile_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Swift module compiled successfully")
                self.is_compiled = True
                return True
            else:
                logger.error(f"Swift compilation failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to compile Swift module: {e}")
            return False
    
    def _prepare_environment(self) -> Dict[str, str]:
        """Prepare environment variables for Swift process"""
        env = os.environ.copy()
        env['VIDEO_CAPTURE_DISPLAY_ID'] = str(self.config.display_id)
        env['VIDEO_CAPTURE_FPS'] = str(self.config.fps)
        env['VIDEO_CAPTURE_RESOLUTION'] = self.config.resolution
        
        if self.config.output_path:
            env['VIDEO_CAPTURE_OUTPUT_PATH'] = self.config.output_path
            
        return env
    
    async def _run_swift_command(self, command: str) -> Dict[str, Any]:
        """Run a Swift video capture command"""
        # Determine command to run
        if self.compiled_path.exists():
            cmd = [str(self.compiled_path), command]
        else:
            if not self._compile_swift_module():
                # Fall back to running with swift interpreter
                cmd = ['swift', str(self.swift_module_path), command]
            else:
                # Use compiled version after successful compilation
                cmd = [str(self.compiled_path), command]
        
        try:
            logger.info(f"Running Swift command: {command} with cmd: {cmd}")
            
            # Run command with proper environment
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._prepare_environment()
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0 and stderr:
                logger.error(f"Swift command error: {stderr.decode()}")
            
            # Parse JSON response
            output = stdout.decode().strip()
            
            # Handle status updates during capture
            lines = output.split('\n')
            for line in lines:
                if line.startswith('STATUS_UPDATE:'):
                    status_json = line.replace('STATUS_UPDATE:', '').strip()
                    logger.debug(f"Status update: {status_json}")
            
            # Get the last line which should be the response
            response_line = lines[-1] if lines else '{}'
            
            try:
                return json.loads(response_line)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse Swift response: {response_line}")
                return {
                    'success': False,
                    'error': f'Invalid response: {response_line}',
                    'message': 'Failed to parse Swift response'
                }
                
        except Exception as e:
            logger.error(f"Failed to run Swift command: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to execute Swift command'
            }
    
    async def check_permission(self) -> Dict[str, Any]:
        """Check screen recording permission status"""
        return await self._run_swift_command('check-permission')
    
    async def request_permission(self) -> Dict[str, Any]:
        """Request screen recording permission"""
        result = await self._run_swift_command('request-permission')
        
        # If permission was just granted, give the system a moment to update
        if result.get('success'):
            await asyncio.sleep(0.5)
            
        return result
    
    async def start_capture(self) -> Dict[str, Any]:
        """Start video capture"""
        logger.info("Starting Swift video capture...")
        
        # First check permission
        permission_check = await self.check_permission()
        
        if permission_check.get('permissionStatus') != 'authorized':
            logger.warning("Screen recording permission not authorized, requesting...")
            
            # Request permission
            permission_result = await self.request_permission()
            
            if not permission_result.get('success'):
                return {
                    'success': False,
                    'error': 'Screen recording permission denied',
                    'message': 'Please grant screen recording permission in System Preferences > Security & Privacy > Privacy > Screen Recording',
                    'needsPermission': True
                }
        
        # Start capture
        result = await self._run_swift_command('start')
        
        if result.get('success'):
            logger.info("Swift video capture started successfully")
        else:
            logger.error(f"Failed to start Swift video capture: {result.get('error')}")
            
        return result
    
    async def stop_capture(self) -> Dict[str, Any]:
        """Stop video capture"""
        logger.info("Stopping Swift video capture...")
        return await self._run_swift_command('stop')
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current capture status"""
        return await self._run_swift_command('status')
    
    async def ensure_permission(self) -> bool:
        """Ensure screen recording permission is granted"""
        # Check current permission
        permission_check = await self.check_permission()
        
        if permission_check.get('permissionStatus') == 'authorized':
            return True
        
        # Request permission
        logger.info("Requesting screen recording permission...")
        permission_result = await self.request_permission()
        
        return permission_result.get('success', False)
    
    def cleanup(self):
        """Clean up resources"""
        if self._process:
            self._process.terminate()
            self._process = None


# Convenience functions for direct usage
async def check_screen_recording_permission() -> bool:
    """Check if screen recording permission is granted"""
    bridge = SwiftVideoBridge()
    result = await bridge.check_permission()
    return result.get('permissionStatus') == 'authorized'

async def request_screen_recording_permission() -> bool:
    """Request screen recording permission"""
    bridge = SwiftVideoBridge()
    result = await bridge.request_permission()
    return result.get('success', False)

async def start_swift_video_capture(config: Optional[SwiftCaptureConfig] = None) -> Dict[str, Any]:
    """Start video capture using Swift"""
    bridge = SwiftVideoBridge(config)
    return await bridge.start_capture()

async def stop_swift_video_capture() -> Dict[str, Any]:
    """Stop video capture using Swift"""
    bridge = SwiftVideoBridge()
    return await bridge.stop_capture()