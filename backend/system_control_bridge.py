#!/usr/bin/env python3
"""
Python bridge for enhanced macOS system control via Swift
Provides secure, robust system operations with comprehensive error handling
"""

import asyncio
import json
import logging
import subprocess
import time
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from datetime import datetime

logger = logging.getLogger(__name__)

class SystemOperationType(Enum):
    """Types of system operations"""
    APP_LIFECYCLE = "app_lifecycle"
    SYSTEM_PREFERENCE = "system_preference"
    FILE_SYSTEM = "file_system"
    CLIPBOARD = "clipboard"

class AppOperation(Enum):
    """Application lifecycle operations"""
    LAUNCH = "launch"
    CLOSE = "close"
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"
    HIDE = "hide"
    SWITCH_TO = "switch_to"
    LIST_RUNNING = "list_running"
    GET_INFO = "get_info"

class PreferenceOperation(Enum):
    """System preference operations"""
    SET_VOLUME = "set_volume"
    SET_BRIGHTNESS = "set_brightness"
    TOGGLE_WIFI = "toggle_wifi"
    TOGGLE_BLUETOOTH = "toggle_bluetooth"
    SET_DO_NOT_DISTURB = "set_do_not_disturb"
    GET_DARK_MODE = "get_dark_mode"
    SET_DARK_MODE = "set_dark_mode"
    GET_SYSTEM_INFO = "get_system_info"

class FileOperation(Enum):
    """File system operations"""
    COPY = "copy"
    MOVE = "move"
    DELETE = "delete"
    CREATE_DIRECTORY = "create_directory"
    SEARCH = "search"
    GET_INFO = "get_info"
    SET_PERMISSIONS = "set_permissions"

class ClipboardOperation(Enum):
    """Clipboard operations"""
    READ = "read"
    WRITE = "write"
    CLEAR = "clear"
    GET_HISTORY = "get_history"
    MANIPULATE = "manipulate"

@dataclass
class SystemControlResult:
    """Result of a system control operation"""
    success: bool
    operation: str
    result: Any
    error: Optional[str]
    execution_time: float
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        return {
            'success': self.success,
            'operation': self.operation,
            'result': self.result,
            'error': self.error,
            'execution_time': self.execution_time,
            'timestamp': self.timestamp.isoformat()
        }

class SystemControlError(Exception):
    """Custom exception for system control errors"""
    pass

class SystemControlBridge:
    """
    Bridge to Swift-based system control functionality
    Provides secure, error-handled system operations
    """
    
    def __init__(self):
        self.swift_executable = self._build_swift_executable()
        self.permission_cache: Dict[str, bool] = {}
        self.retry_policy = {
            'max_retries': 3,
            'base_delay': 0.1,
            'max_delay': 2.0
        }
        
    def _build_swift_executable(self) -> Path:
        """Build the Swift system control executable"""
        swift_dir = Path(__file__).parent / "swift_bridge"
        
        # Build command
        build_cmd = [
            "swift", "build",
            "-c", "release",
            "--package-path", str(swift_dir)
        ]
        
        try:
            result = subprocess.run(
                build_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("Swift system control built successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to build Swift executable: {e.stderr}")
            raise SystemControlError("Failed to build system control module")
        
        # Find executable
        executable = swift_dir / ".build" / "release" / "jarvis-system-control"
        if not executable.exists():
            raise SystemControlError("System control executable not found")
        
        return executable
    
    async def execute_operation(
        self,
        operation_type: SystemOperationType,
        operation: Union[AppOperation, PreferenceOperation, FileOperation, ClipboardOperation],
        parameters: Optional[Dict[str, Any]] = None,
        require_confirmation: bool = True
    ) -> SystemControlResult:
        """
        Execute a system operation with error handling and retry logic
        
        Args:
            operation_type: Type of operation
            operation: Specific operation to perform
            parameters: Operation parameters
            require_confirmation: Whether to require user confirmation
            
        Returns:
            SystemControlResult with operation outcome
        """
        start_time = time.time()
        parameters = parameters or {}
        
        # Check permissions
        permission_key = f"{operation_type.value}:{operation.value}"
        if not await self._check_permission(permission_key):
            return SystemControlResult(
                success=False,
                operation=f"{operation_type.value}.{operation.value}",
                result=None,
                error="Permission denied",
                execution_time=time.time() - start_time,
                timestamp=datetime.now()
            )
        
        # Get user confirmation if required
        if require_confirmation and self._requires_confirmation(operation_type, operation):
            if not await self._get_user_confirmation(operation_type, operation, parameters):
                return SystemControlResult(
                    success=False,
                    operation=f"{operation_type.value}.{operation.value}",
                    result=None,
                    error="User cancelled operation",
                    execution_time=time.time() - start_time,
                    timestamp=datetime.now()
                )
        
        # Execute with retry logic
        for attempt in range(self.retry_policy['max_retries']):
            try:
                result = await self._execute_swift_operation(
                    operation_type,
                    operation,
                    parameters
                )
                
                return SystemControlResult(
                    success=True,
                    operation=f"{operation_type.value}.{operation.value}",
                    result=result,
                    error=None,
                    execution_time=time.time() - start_time,
                    timestamp=datetime.now()
                )
                
            except Exception as e:
                if attempt < self.retry_policy['max_retries'] - 1:
                    # Calculate backoff delay
                    delay = min(
                        self.retry_policy['base_delay'] * (2 ** attempt),
                        self.retry_policy['max_delay']
                    )
                    await asyncio.sleep(delay)
                    logger.warning(f"Retrying operation after error: {e}")
                else:
                    # Final attempt failed
                    return SystemControlResult(
                        success=False,
                        operation=f"{operation_type.value}.{operation.value}",
                        result=None,
                        error=str(e),
                        execution_time=time.time() - start_time,
                        timestamp=datetime.now()
                    )
        
        # Should not reach here
        return SystemControlResult(
            success=False,
            operation=f"{operation_type.value}.{operation.value}",
            result=None,
            error="Unknown error",
            execution_time=time.time() - start_time,
            timestamp=datetime.now()
        )
    
    async def _execute_swift_operation(
        self,
        operation_type: SystemOperationType,
        operation: Union[AppOperation, PreferenceOperation, FileOperation, ClipboardOperation],
        parameters: Dict[str, Any]
    ) -> Any:
        """Execute operation via Swift executable"""
        # Prepare command
        cmd_data = {
            'type': operation_type.value,
            'operation': operation.value,
            'parameters': parameters
        }
        
        cmd = [
            str(self.swift_executable),
            '--json',
            json.dumps(cmd_data)
        ]
        
        # Execute
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            raise SystemControlError(f"Operation failed: {error_msg}")
        
        # Parse result
        try:
            result = json.loads(stdout.decode())
            return result.get('data')
        except json.JSONDecodeError:
            return stdout.decode().strip()
    
    async def _check_permission(self, permission_key: str) -> bool:
        """Check if operation has required permissions"""
        if permission_key in self.permission_cache:
            return self.permission_cache[permission_key]
        
        # Check with Swift security manager
        result = await self._execute_swift_operation(
            SystemOperationType.APP_LIFECYCLE,
            AppOperation.GET_INFO,
            {'permission': permission_key}
        )
        
        has_permission = result.get('has_permission', False)
        self.permission_cache[permission_key] = has_permission
        
        return has_permission
    
    def _requires_confirmation(
        self,
        operation_type: SystemOperationType,
        operation: Union[AppOperation, PreferenceOperation, FileOperation, ClipboardOperation]
    ) -> bool:
        """Check if operation requires user confirmation"""
        # Destructive operations require confirmation
        destructive_operations = {
            (SystemOperationType.APP_LIFECYCLE, AppOperation.CLOSE),
            (SystemOperationType.FILE_SYSTEM, FileOperation.DELETE),
            (SystemOperationType.FILE_SYSTEM, FileOperation.MOVE),
            (SystemOperationType.SYSTEM_PREFERENCE, PreferenceOperation.TOGGLE_WIFI),
            (SystemOperationType.SYSTEM_PREFERENCE, PreferenceOperation.TOGGLE_BLUETOOTH),
        }
        
        return (operation_type, operation) in destructive_operations
    
    async def _get_user_confirmation(
        self,
        operation_type: SystemOperationType,
        operation: Union[AppOperation, PreferenceOperation, FileOperation, ClipboardOperation],
        parameters: Dict[str, Any]
    ) -> bool:
        """Get user confirmation for operation"""
        # In production, this would show a dialog
        # For now, we'll log and return True
        logger.info(f"User confirmation requested for {operation_type.value}.{operation.value}")
        return True
    
    # Convenience methods for common operations
    
    async def launch_app(self, bundle_identifier: str) -> SystemControlResult:
        """Launch an application"""
        return await self.execute_operation(
            SystemOperationType.APP_LIFECYCLE,
            AppOperation.LAUNCH,
            {'bundle_identifier': bundle_identifier},
            require_confirmation=False
        )
    
    async def close_app(self, bundle_identifier: str, force: bool = False) -> SystemControlResult:
        """Close an application"""
        return await self.execute_operation(
            SystemOperationType.APP_LIFECYCLE,
            AppOperation.CLOSE,
            {'bundle_identifier': bundle_identifier, 'force': force}
        )
    
    async def set_volume(self, level: float) -> SystemControlResult:
        """Set system volume (0.0 to 1.0)"""
        return await self.execute_operation(
            SystemOperationType.SYSTEM_PREFERENCE,
            PreferenceOperation.SET_VOLUME,
            {'level': max(0.0, min(1.0, level))},
            require_confirmation=False
        )
    
    async def set_brightness(self, level: float) -> SystemControlResult:
        """Set display brightness (0.0 to 1.0)"""
        return await self.execute_operation(
            SystemOperationType.SYSTEM_PREFERENCE,
            PreferenceOperation.SET_BRIGHTNESS,
            {'level': max(0.0, min(1.0, level))},
            require_confirmation=False
        )
    
    async def copy_file(self, source: str, destination: str) -> SystemControlResult:
        """Copy a file"""
        return await self.execute_operation(
            SystemOperationType.FILE_SYSTEM,
            FileOperation.COPY,
            {'source': source, 'destination': destination}
        )
    
    async def search_files(
        self,
        directory: str,
        query: str,
        recursive: bool = True,
        max_results: int = 100
    ) -> SystemControlResult:
        """Search for files"""
        return await self.execute_operation(
            SystemOperationType.FILE_SYSTEM,
            FileOperation.SEARCH,
            {
                'directory': directory,
                'query': query,
                'recursive': recursive,
                'max_results': max_results
            },
            require_confirmation=False
        )
    
    async def read_clipboard(self) -> SystemControlResult:
        """Read clipboard content"""
        return await self.execute_operation(
            SystemOperationType.CLIPBOARD,
            ClipboardOperation.READ,
            require_confirmation=False
        )
    
    async def write_clipboard(self, text: str) -> SystemControlResult:
        """Write text to clipboard"""
        return await self.execute_operation(
            SystemOperationType.CLIPBOARD,
            ClipboardOperation.WRITE,
            {'text': text},
            require_confirmation=False
        )
    
    async def get_running_apps(self) -> SystemControlResult:
        """Get list of running applications"""
        return await self.execute_operation(
            SystemOperationType.APP_LIFECYCLE,
            AppOperation.LIST_RUNNING,
            require_confirmation=False
        )
    
    async def get_system_info(self) -> SystemControlResult:
        """Get system information"""
        return await self.execute_operation(
            SystemOperationType.SYSTEM_PREFERENCE,
            PreferenceOperation.GET_SYSTEM_INFO,
            require_confirmation=False
        )

# Example usage
async def test_system_control():
    """Test system control functionality"""
    bridge = SystemControlBridge()
    
    # Get running apps
    result = await bridge.get_running_apps()
    if result.success:
        print(f"Running apps: {len(result.result)} apps")
        for app in result.result[:5]:  # Show first 5
            print(f"  - {app['name']} ({app['bundleIdentifier']})")
    
    # Read clipboard
    result = await bridge.read_clipboard()
    if result.success and result.result:
        print(f"\nClipboard content: {result.result[:50]}...")
    
    # Get system info
    result = await bridge.get_system_info()
    if result.success:
        print(f"\nSystem info: {result.result}")

if __name__ == "__main__":
    asyncio.run(test_system_control())