"""
JARVIS Core - Advanced Architecture for Scale & Memory Efficiency
"""

from .model_manager import ModelManager, ModelTier, ModelInfo
from .memory_controller import MemoryController, MemoryPressure, MemorySnapshot
from .task_router import TaskRouter, TaskType, TaskAnalysis
from .jarvis_core import JARVISCore, JARVISAssistant

__all__ = [
    "ModelManager",
    "ModelTier", 
    "ModelInfo",
    "MemoryController",
    "MemoryPressure",
    "MemorySnapshot",
    "TaskRouter",
    "TaskType",
    "TaskAnalysis",
    "JARVISCore",
    "JARVISAssistant"
]

__version__ = "2.0.0"