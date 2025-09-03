"""
General WebSocket Handler
Handles general WebSocket messages and provides health checks
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import psutil
import platform

logger = logging.getLogger(__name__)

# Lazy imports
_chatbot = None

def get_chatbot():
    """Get chatbot instance lazily"""
    global _chatbot
    if _chatbot is None:
        try:
            # Try to use vision-enabled chatbot first
            from chatbots.claude_vision_chatbot import ClaudeVisionChatbot
            import os
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                _chatbot = ClaudeVisionChatbot(api_key)
                logger.info("Using Claude Vision Chatbot with monitoring capabilities")
            else:
                logger.warning("No API key found for Claude chatbot")
        except ImportError:
            # Fall back to regular chatbot
            try:
                from chatbots.claude_chatbot import ClaudeChatbot
                import os
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if api_key:
                    _chatbot = ClaudeChatbot(api_key)
                    logger.info("Using regular Claude Chatbot (no vision/monitoring)")
                else:
                    logger.warning("No API key found for Claude chatbot")
            except ImportError:
                logger.warning("Claude chatbot not available")
    return _chatbot

async def handle_websocket_message(message: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Main WebSocket message handler for general messages
    """
    try:
        message_type = message.get('type', '')
        
        handlers = {
            'chat': handle_chat,
            'get_status': handle_get_status,
            'echo': handle_echo,
            'health_check': handle_health_check,
            'ping': handle_ping
        }
        
        handler = handlers.get(message_type, handle_unknown)
        result = await handler(message, context)
        
        # Add metadata
        result['timestamp'] = datetime.now().isoformat()
        if 'correlation_id' in message:
            result['correlation_id'] = message['correlation_id']
            
        return result
        
    except Exception as e:
        logger.error(f"Error handling message: {e}", exc_info=True)
        return {
            'type': 'error',
            'error': str(e),
            'original_message_type': message.get('type', 'unknown')
        }

async def handle_chat(message: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Handle chat messages"""
    chatbot = get_chatbot()
    
    if not chatbot:
        return {
            'type': 'chat_response',
            'response': "I'm sorry, but the chat service is not available right now.",
            'success': False
        }
    
    try:
        user_message = message.get('message', '')
        
        # Get response from Claude
        response = await chatbot.generate_response(user_message)
        
        # Check if this was a monitoring command and add metadata
        monitoring_info = None
        if hasattr(chatbot, '_monitoring_active'):
            monitoring_info = {
                'monitoring_active': chatbot._monitoring_active,
                'capture_method': getattr(chatbot, '_capture_method', 'unknown')
            }
        
        result = {
            'type': 'chat_response',
            'response': response,
            'success': True
        }
        
        if monitoring_info:
            result['monitoring'] = monitoring_info
            
        return result
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return {
            'type': 'chat_response',
            'response': "I apologize, but I encountered an error processing your message.",
            'success': False,
            'error': str(e)
        }

async def handle_get_status(message: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Get system status"""
    try:
        # Gather system info
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get process info
        process = psutil.Process()
        process_info = {
            'cpu_percent': process.cpu_percent(),
            'memory_mb': process.memory_info().rss / 1024 / 1024,
            'threads': process.num_threads()
        }
        
        return {
            'type': 'system_status',
            'status': {
                'system': {
                    'platform': platform.system(),
                    'python_version': platform.python_version(),
                    'cpu_percent': cpu_percent,
                    'memory': {
                        'total_gb': memory.total / 1024**3,
                        'available_gb': memory.available / 1024**3,
                        'percent': memory.percent
                    },
                    'disk': {
                        'total_gb': disk.total / 1024**3,
                        'free_gb': disk.free / 1024**3,
                        'percent': disk.percent
                    }
                },
                'process': process_info,
                'services': {
                    'chatbot': get_chatbot() is not None,
                    'websocket': True
                }
            },
            'success': True
        }
        
    except Exception as e:
        logger.error(f"Status error: {e}")
        return {
            'type': 'system_status',
            'success': False,
            'error': str(e)
        }

async def handle_echo(message: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Echo back the message"""
    return {
        'type': 'echo_response',
        'original_message': message,
        'context': context,
        'success': True
    }

async def handle_health_check(message: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Perform health check"""
    checks = {}
    
    # Check chatbot
    try:
        chatbot = get_chatbot()
        checks['chatbot'] = chatbot is not None
    except:
        checks['chatbot'] = False
    
    # Check memory
    try:
        memory = psutil.virtual_memory()
        checks['memory'] = memory.percent < 90
    except:
        checks['memory'] = False
    
    # Check CPU
    try:
        cpu = psutil.cpu_percent(interval=0.1)
        checks['cpu'] = cpu < 90
    except:
        checks['cpu'] = False
    
    # Overall health
    all_healthy = all(checks.values())
    
    return {
        'type': 'health_check_response',
        'healthy': all_healthy,
        'checks': checks,
        'success': True
    }

async def handle_ping(message: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Handle ping"""
    return {
        'type': 'pong',
        'success': True
    }

async def handle_unknown(message: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Handle unknown message types"""
    logger.warning(f"Unknown message type: {message.get('type')}")
    return {
        'type': 'error',
        'error': f"Unknown message type: {message.get('type')}",
        'supported_types': ['chat', 'get_status', 'echo', 'health_check', 'ping'],
        'success': False
    }

# Export for WebSocket integration
__all__ = ['handle_websocket_message']