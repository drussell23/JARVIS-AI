"""
JARVIS Neural Mesh - Production Agents

This module contains all production-ready agents for the Neural Mesh system.
Each agent extends BaseNeuralMeshAgent and provides specific capabilities.

Agent Categories:
- Core: Memory, Coordination, Health monitoring
- Intelligence: Analysis, Reasoning, Pattern recognition
- Vision: Screen capture, Error detection, OCR
- Voice: TTS, STT, Speaker verification
- Context: Environment awareness, State tracking
- Action: Task execution, Tool orchestration

Quick Start:
    from neural_mesh import start_neural_mesh
    from neural_mesh.agents import initialize_production_agents

    coordinator = await start_neural_mesh()
    agents = await initialize_production_agents(coordinator)

    # All 6 production agents are now running!
"""

from .memory_agent import MemoryAgent
from .coordinator_agent import CoordinatorAgent
from .health_monitor_agent import HealthMonitorAgent
from .context_tracker_agent import ContextTrackerAgent
from .error_analyzer_agent import ErrorAnalyzerAgent
from .pattern_recognition_agent import PatternRecognitionAgent

from .agent_initializer import (
    AgentInitializer,
    PRODUCTION_AGENTS,
    get_agent_initializer,
    initialize_production_agents,
    shutdown_production_agents,
)

__all__ = [
    # Agent Classes
    "MemoryAgent",
    "CoordinatorAgent",
    "HealthMonitorAgent",
    "ContextTrackerAgent",
    "ErrorAnalyzerAgent",
    "PatternRecognitionAgent",
    # Initializer
    "AgentInitializer",
    "PRODUCTION_AGENTS",
    "get_agent_initializer",
    "initialize_production_agents",
    "shutdown_production_agents",
]
