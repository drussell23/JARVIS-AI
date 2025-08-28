"""
Python-TypeScript Bridge for IPC Communication
Uses ZeroMQ for high-performance inter-process communication
"""

import asyncio
import json
import logging
import zmq
import zmq.asyncio
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, asdict
from datetime import datetime
import traceback
from concurrent.futures import ThreadPoolExecutor
import pickle
import base64

logger = logging.getLogger(__name__)


@dataclass
class BridgeMessage:
    """Message structure for bridge communication"""
    id: str
    type: str
    module: str
    function: str
    args: List[Any]
    kwargs: Dict[str, Any]
    timestamp: str
    correlation_id: Optional[str] = None


class PythonTypeScriptBridge:
    """High-performance bridge between Python and TypeScript processes"""
    
    def __init__(self, 
                 python_port: int = 5555,
                 typescript_port: int = 5556,
                 use_pickle: bool = False):
        self.python_port = python_port
        self.typescript_port = typescript_port
        self.use_pickle = use_pickle
        self._port_allocation_attempts = 0
        
        # ZeroMQ context
        self.context = zmq.asyncio.Context()
        
        # Sockets
        self.pull_socket = None  # Receives from TypeScript
        self.push_socket = None  # Sends to TypeScript
        self.pub_socket = None   # Publishes events
        
        # Function registry
        self.functions: Dict[str, Dict[str, Callable]] = {}
        
        # Response handlers
        self.response_handlers: Dict[str, asyncio.Future] = {}
        
        # Thread pool for CPU-bound operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        self._running = False
        
    async def start(self):
        """Start the bridge with dynamic port allocation"""
        max_attempts = 10
        
        while self._port_allocation_attempts < max_attempts:
            try:
                # Setup PULL socket (receive from TypeScript)
                self.pull_socket = self.context.socket(zmq.PULL)
                self.pull_socket.bind(f"tcp://127.0.0.1:{self.python_port}")
                
                # Setup PUSH socket (send to TypeScript)  
                self.push_socket = self.context.socket(zmq.PUSH)
                self.push_socket.connect(f"tcp://127.0.0.1:{self.typescript_port}")
                
                # Setup PUB socket for events
                self.pub_socket = self.context.socket(zmq.PUB)
                self.pub_socket.bind(f"tcp://127.0.0.1:{self.python_port + 1}")
                
                self._running = True
                logger.info(f"Python-TypeScript bridge started on ports {self.python_port}/{self.typescript_port}")
                
                # Start message processing in background
                asyncio.create_task(self._process_messages())
                break
                
            except zmq.error.ZMQError as e:
                if "Address already in use" in str(e):
                    self._port_allocation_attempts += 1
                    logger.warning(f"Port conflict (attempt {self._port_allocation_attempts}), trying next port range...")
                    
                    # Close any partially opened sockets
                    if hasattr(self, 'pull_socket') and self.pull_socket:
                        self.pull_socket.close()
                    if hasattr(self, 'push_socket') and self.push_socket:
                        self.push_socket.close() 
                    if hasattr(self, 'pub_socket') and self.pub_socket:
                        self.pub_socket.close()
                    
                    # Try next port range
                    self.python_port += 10
                    self.typescript_port += 10
                    
                    if self._port_allocation_attempts >= max_attempts:
                        logger.error(f"Failed to allocate ports after {max_attempts} attempts")
                        raise
                else:
                    logger.error(f"Failed to start bridge: {e}")
                    raise
            except Exception as e:
                logger.error(f"Failed to start bridge: {e}")
                raise
            
    async def stop(self):
        """Stop the bridge"""
        self._running = False
        
        # Close sockets
        if self.pull_socket:
            self.pull_socket.close()
        if self.push_socket:
            self.push_socket.close()
        if self.pub_socket:
            self.pub_socket.close()
            
        # Cleanup context
        self.context.term()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Python-TypeScript bridge stopped")
        
    async def _process_messages(self):
        """Main message processing loop"""
        while self._running:
            try:
                # Receive message
                message_data = await self.pull_socket.recv()
                
                # Deserialize message
                if self.use_pickle:
                    message = pickle.loads(message_data)
                else:
                    message = json.loads(message_data.decode('utf-8'))
                    
                # Process message
                asyncio.create_task(self._handle_message(message))
                
            except Exception as e:
                logger.error(f"Message processing error: {e}")
                
    async def _handle_message(self, message: Dict[str, Any]):
        """Handle incoming message"""
        try:
            msg_type = message.get('type')
            
            if msg_type == 'function_call':
                await self._handle_function_call(message)
            elif msg_type == 'response':
                await self._handle_response(message)
            elif msg_type == 'event':
                await self._handle_event(message)
            else:
                logger.warning(f"Unknown message type: {msg_type}")
                
        except Exception as e:
            logger.error(f"Message handling error: {e}\n{traceback.format_exc()}")
            
            # Send error response
            if 'id' in message:
                await self._send_response({
                    'id': message['id'],
                    'type': 'error',
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
                
    async def _handle_function_call(self, message: Dict[str, Any]):
        """Handle function call from TypeScript"""
        module_name = message.get('module')
        function_name = message.get('function')
        args = message.get('args', [])
        kwargs = message.get('kwargs', {})
        msg_id = message.get('id')
        
        try:
            # Get function from registry
            if module_name in self.functions and function_name in self.functions[module_name]:
                func = self.functions[module_name][function_name]
            else:
                # Try dynamic import
                func = self._import_function(module_name, function_name)
                
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                # Run in executor for CPU-bound functions
                result = await asyncio.get_event_loop().run_in_executor(
                    self.executor, func, *args, **kwargs
                )
                
            # Send response
            await self._send_response({
                'id': msg_id,
                'type': 'result',
                'result': self._serialize_result(result)
            })
            
        except Exception as e:
            # Send error response
            await self._send_response({
                'id': msg_id,
                'type': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            
    async def _handle_response(self, message: Dict[str, Any]):
        """Handle response from TypeScript"""
        msg_id = message.get('id')
        
        if msg_id in self.response_handlers:
            future = self.response_handlers.pop(msg_id)
            
            if message.get('type') == 'error':
                future.set_exception(Exception(message.get('error')))
            else:
                future.set_result(message.get('result'))
                
    async def _handle_event(self, message: Dict[str, Any]):
        """Handle event from TypeScript"""
        event_type = message.get('event_type')
        event_data = message.get('data')
        
        # Publish event to subscribers
        await self.publish_event(event_type, event_data)
        
    def _import_function(self, module_name: str, function_name: str) -> Callable:
        """Dynamically import a function"""
        import importlib
        
        try:
            module = importlib.import_module(module_name)
            func = getattr(module, function_name)
            
            # Cache for future use
            if module_name not in self.functions:
                self.functions[module_name] = {}
            self.functions[module_name][function_name] = func
            
            return func
        except Exception as e:
            raise ImportError(f"Failed to import {module_name}.{function_name}: {e}")
            
    def _serialize_result(self, result: Any) -> Any:
        """Serialize Python result for TypeScript"""
        if self.use_pickle:
            return base64.b64encode(pickle.dumps(result)).decode('utf-8')
            
        # Handle common types
        if isinstance(result, (str, int, float, bool, type(None))):
            return result
        elif isinstance(result, (list, tuple)):
            return [self._serialize_result(item) for item in result]
        elif isinstance(result, dict):
            return {k: self._serialize_result(v) for k, v in result.items()}
        elif isinstance(result, datetime):
            return result.isoformat()
        elif hasattr(result, '__dict__'):
            # Convert objects to dict
            return self._serialize_result(result.__dict__)
        else:
            # Fallback to string representation
            return str(result)
            
    async def _send_response(self, response: Dict[str, Any]):
        """Send response to TypeScript"""
        try:
            if self.use_pickle:
                data = pickle.dumps(response)
            else:
                data = json.dumps(response).encode('utf-8')
                
            await self.push_socket.send(data)
        except Exception as e:
            logger.error(f"Failed to send response: {e}")
            
    def register_function(self, module_name: str, function_name: str, func: Callable):
        """Register a Python function for TypeScript to call"""
        if module_name not in self.functions:
            self.functions[module_name] = {}
            
        self.functions[module_name][function_name] = func
        logger.info(f"Registered function: {module_name}.{function_name}")
        
    def register_module(self, module_name: str, module: Any):
        """Register all functions from a module"""
        import inspect
        
        if module_name not in self.functions:
            self.functions[module_name] = {}
            
        for name, func in inspect.getmembers(module, inspect.isfunction):
            if not name.startswith('_'):  # Skip private functions
                self.functions[module_name][name] = func
                logger.info(f"Registered function: {module_name}.{name}")
                
    async def call_typescript(self, 
                            function_name: str,
                            args: List[Any] = None,
                            kwargs: Dict[str, Any] = None,
                            timeout: float = 30.0) -> Any:
        """Call a TypeScript function from Python"""
        import uuid
        
        msg_id = str(uuid.uuid4())
        message = {
            'id': msg_id,
            'type': 'function_call',
            'function': function_name,
            'args': args or [],
            'kwargs': kwargs or {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Create future for response
        future = asyncio.Future()
        self.response_handlers[msg_id] = future
        
        # Send message
        await self._send_response(message)
        
        try:
            # Wait for response
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            self.response_handlers.pop(msg_id, None)
            raise TimeoutError(f"TypeScript function call timed out: {function_name}")
            
    async def publish_event(self, event_type: str, data: Any):
        """Publish an event to TypeScript subscribers"""
        event = {
            'type': 'event',
            'event_type': event_type,
            'data': self._serialize_result(data),
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Publish with topic
            topic = f"{event_type}:".encode('utf-8')
            
            if self.use_pickle:
                message = pickle.dumps(event)
            else:
                message = json.dumps(event).encode('utf-8')
                
            await self.pub_socket.send_multipart([topic, message])
        except Exception as e:
            logger.error(f"Failed to publish event: {e}")


# Global bridge instance
bridge = None


def get_bridge() -> PythonTypeScriptBridge:
    """Get or create bridge instance"""
    global bridge
    if bridge is None:
        bridge = PythonTypeScriptBridge()
    return bridge


async def start_bridge():
    """Start the global bridge instance"""
    bridge = get_bridge()
    await bridge.start()


async def stop_bridge():
    """Stop the global bridge instance"""
    if bridge:
        await bridge.stop()


# Example usage for registering vision handlers
def register_vision_handlers():
    """Register vision-related handlers with the bridge"""
    from ..api import unified_vision_handler
    
    bridge = get_bridge()
    
    # Register the entire module
    bridge.register_module('backend.api.unified_vision_handler', unified_vision_handler)
    
    # Or register specific functions
    bridge.register_function(
        'backend.api.unified_vision_handler',
        'handle_websocket_message',
        unified_vision_handler.unified_handler.handle_websocket_message
    )