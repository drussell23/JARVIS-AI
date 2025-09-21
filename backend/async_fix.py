"""
Async fix for DynamicChatbot initialization
"""

import asyncio
from functools import wraps

def defer_async_tasks(func):
    """Decorator to defer async task creation until event loop is running"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        instance = func(*args, **kwargs)
        
        # Check if we have pending async tasks
        if hasattr(instance, '_deferred_tasks'):
            # Try to schedule them if event loop is running
            try:
                loop = asyncio.get_running_loop()
                for task_func in instance._deferred_tasks:
                    loop.create_task(task_func())
                instance._deferred_tasks = []
            except RuntimeError:
                # No event loop yet, tasks will be started later
                pass
                
        return instance
    return wrapper

def start_monitoring_when_ready(chatbot_instance):
    """Start monitoring when the event loop is ready"""
    async def _start():
        if hasattr(chatbot_instance, 'start_monitoring'):
            await chatbot_instance.start_monitoring()
    
    try:
        asyncio.create_task(_start())
    except RuntimeError:
        # Store for later
        if not hasattr(chatbot_instance, '_deferred_tasks'):
            chatbot_instance._deferred_tasks = []
        chatbot_instance._deferred_tasks.append(_start)