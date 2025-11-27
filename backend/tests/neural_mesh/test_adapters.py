"""
Integration Tests for Neural Mesh Adapters

Tests the IntelligenceEngineAdapter, AutonomyEngineAdapter, VoiceSystemAdapter,
and JARVISNeuralMeshBridge to ensure proper integration with existing JARVIS systems.
"""

import asyncio
import pytest
import sys
import os
from typing import Any, Dict, Set
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from neural_mesh.data_models import (
    AgentMessage,
    KnowledgeType,
    MessageType,
    MessagePriority,
)
from neural_mesh.neural_mesh_coordinator import NeuralMeshCoordinator
from neural_mesh.base.base_neural_mesh_agent import BaseNeuralMeshAgent
from neural_mesh.adapters.legacy_agent_adapter import LegacyAgentAdapter, adapt_agent
from neural_mesh.adapters.intelligence_adapter import (
    IntelligenceEngineAdapter,
    IntelligenceEngineType,
    IntelligenceCapabilities,
)
from neural_mesh.adapters.autonomy_adapter import (
    AutonomyEngineAdapter,
    AutonomyComponentType,
    AutonomyCapabilities,
)
from neural_mesh.adapters.voice_adapter import (
    VoiceSystemAdapter,
    VoiceComponentType,
    VoiceCapabilities,
)
from neural_mesh.jarvis_bridge import (
    JARVISNeuralMeshBridge,
    AgentDiscoveryConfig,
    SystemCategory,
)


# =============================================================================
# Mock Components for Testing
# =============================================================================

class MockIntelligenceEngine:
    """Mock UAE/SAI engine for testing."""

    def __init__(self) -> None:
        self.initialized = False
        self.context = {}
        self.state = {"awareness": "active"}

    async def initialize(self) -> None:
        self.initialized = True

    async def analyze_workspace(self, space_id: int = None, include_windows: bool = True) -> Dict:
        return {
            "space_id": space_id,
            "windows": 5 if include_windows else 0,
            "summary": "Test workspace analysis",
        }

    async def process_query(self, query: str, context: Dict = None) -> str:
        return f"Response to: {query}"

    async def get_awareness_state(self) -> Dict:
        return self.state

    async def update_context(self, context: Dict) -> None:
        self.context.update(context)


class MockAutonomousAgent:
    """Mock AutonomousAgent for testing."""

    def __init__(self) -> None:
        self._initialized = False
        self._metrics = MagicMock(
            total_sessions=5,
            successful_sessions=4,
            total_actions=50,
        )
        self.reasoning_engine = MagicMock()
        self.tool_orchestrator = MagicMock()
        self.memory_manager = MagicMock()

    async def initialize(self) -> bool:
        self._initialized = True
        return True

    async def run(self, goal: str, context: Dict = None) -> Dict:
        return {"success": True, "goal": goal, "result": "Task completed"}

    async def chat(self, message: str, context: Dict = None) -> str:
        return f"Chat response to: {message}"

    def get_status(self) -> Dict:
        return {"status": "active", "sessions": 5}


class MockVoiceMemoryAgent:
    """Mock VoiceMemoryAgent for testing."""

    def __init__(self) -> None:
        self.voice_memory = {
            "Derek": {"samples": 10, "freshness": 0.95},
            "Guest": {"samples": 3, "freshness": 0.60},
        }
        self.initialized = False

    async def initialize(self) -> None:
        self.initialized = True

    async def check_voice_freshness(self, speaker: str = None) -> Dict:
        if speaker:
            mem = self.voice_memory.get(speaker, {})
            return {"speaker": speaker, "freshness": mem.get("freshness", 0)}
        return {"overall_freshness": 0.85}

    async def startup_freshness_check(self) -> Dict:
        return {"status": "ok", "speakers_checked": len(self.voice_memory)}

    def get_all_profiles(self) -> list:
        return [
            {"speaker_name": k, **v}
            for k, v in self.voice_memory.items()
        ]


class MockSpeakerVerificationService:
    """Mock SpeakerVerificationService for testing."""

    def __init__(self) -> None:
        self.enrolled_speakers = {"Derek": True}

    async def verify_speaker(
        self,
        audio_data: bytes,
        expected_speaker: str = None,
        threshold: float = 0.85,
    ) -> Dict:
        # Simulate verification
        verified = expected_speaker in self.enrolled_speakers
        return {
            "verified": verified,
            "speaker": expected_speaker,
            "confidence": 0.92 if verified else 0.3,
        }

    async def enroll_speaker(
        self,
        speaker_name: str,
        audio_samples: list,
    ) -> Dict:
        self.enrolled_speakers[speaker_name] = True
        return {"success": True, "speaker": speaker_name}


# =============================================================================
# Intelligence Adapter Tests
# =============================================================================

@pytest.mark.asyncio
async def test_intelligence_adapter_initialization():
    """Test IntelligenceEngineAdapter initialization."""
    engine = MockIntelligenceEngine()
    adapter = IntelligenceEngineAdapter(
        engine=engine,
        engine_type=IntelligenceEngineType.UAE,
        agent_name="test_uae",
    )

    assert adapter.agent_name == "test_uae"
    assert adapter.agent_type == "intelligence"
    assert adapter.engine_type == IntelligenceEngineType.UAE
    assert "spatial_awareness" in adapter.capabilities
    assert "contextual_understanding" in adapter.capabilities


@pytest.mark.asyncio
async def test_intelligence_adapter_execute_task():
    """Test task execution through intelligence adapter."""
    engine = MockIntelligenceEngine()
    adapter = IntelligenceEngineAdapter(
        engine=engine,
        engine_type=IntelligenceEngineType.UAE,
    )

    # Execute workspace analysis
    result = await adapter.execute_task({
        "action": "analyze_workspace",
        "input": {"space_id": 3, "include_windows": True},
    })

    assert result["space_id"] == 3
    assert result["windows"] == 5
    assert "summary" in result


@pytest.mark.asyncio
async def test_intelligence_adapter_process_query():
    """Test query processing through intelligence adapter."""
    engine = MockIntelligenceEngine()
    adapter = IntelligenceEngineAdapter(
        engine=engine,
        engine_type=IntelligenceEngineType.UAE,
    )

    result = await adapter.execute_task({
        "action": "process_query",
        "input": {"query": "What windows are open?"},
    })

    assert "response" in result
    assert "What windows are open?" in result["response"]


@pytest.mark.asyncio
async def test_intelligence_capabilities_matrix():
    """Test capability matrix for different engine types."""
    # UAE should have most capabilities
    uae_caps = IntelligenceCapabilities(
        spatial_awareness=True,
        contextual_understanding=True,
        proactive_suggestions=True,
    )
    assert "spatial_awareness" in uae_caps.to_set()
    assert "proactive_suggestions" in uae_caps.to_set()

    # COT should have reasoning capabilities
    cot_caps = IntelligenceCapabilities(
        chain_of_thought=True,
        reasoning_graphs=True,
    )
    assert "chain_of_thought" in cot_caps.to_set()
    assert "spatial_awareness" not in cot_caps.to_set()


# =============================================================================
# Autonomy Adapter Tests
# =============================================================================

@pytest.mark.asyncio
async def test_autonomy_adapter_initialization():
    """Test AutonomyEngineAdapter initialization."""
    agent = MockAutonomousAgent()
    adapter = AutonomyEngineAdapter(
        autonomy_component=agent,
        component_type=AutonomyComponentType.AGENT,
        agent_name="test_autonomy",
    )

    assert adapter.agent_name == "test_autonomy"
    assert adapter.agent_type == "autonomy"
    assert "autonomous_execution" in adapter.capabilities
    assert "reasoning" in adapter.capabilities


@pytest.mark.asyncio
async def test_autonomy_adapter_run_task():
    """Test autonomous task execution."""
    agent = MockAutonomousAgent()
    adapter = AutonomyEngineAdapter(
        autonomy_component=agent,
        component_type=AutonomyComponentType.AGENT,
    )

    result = await adapter.execute_task({
        "action": "run",
        "input": {"goal": "Organize workspace"},
    })

    assert "result" in result
    assert result["result"]["success"] is True


@pytest.mark.asyncio
async def test_autonomy_adapter_chat():
    """Test chat functionality."""
    agent = MockAutonomousAgent()
    adapter = AutonomyEngineAdapter(
        autonomy_component=agent,
        component_type=AutonomyComponentType.AGENT,
    )

    result = await adapter.execute_task({
        "action": "chat",
        "input": {"message": "Hello JARVIS"},
    })

    assert "response" in result
    assert "Hello JARVIS" in result["response"]


@pytest.mark.asyncio
async def test_autonomy_adapter_session_management():
    """Test session management."""
    agent = MockAutonomousAgent()
    adapter = AutonomyEngineAdapter(
        autonomy_component=agent,
        component_type=AutonomyComponentType.AGENT,
    )

    # Start session
    session = await adapter.execute_task({
        "action": "start_session",
        "input": {"goal": "Test goal", "mode": "supervised"},
    })

    assert "session_id" in session
    assert session["status"] == "active"

    # End session
    ended = await adapter.execute_task({
        "action": "end_session",
        "input": {"session_id": session["session_id"]},
    })

    assert ended["status"] == "completed"


# =============================================================================
# Voice Adapter Tests
# =============================================================================

@pytest.mark.asyncio
async def test_voice_adapter_initialization():
    """Test VoiceSystemAdapter initialization."""
    agent = MockVoiceMemoryAgent()
    adapter = VoiceSystemAdapter(
        voice_component=agent,
        component_type=VoiceComponentType.MEMORY,
        agent_name="test_voice",
    )

    assert adapter.agent_name == "test_voice"
    assert adapter.agent_type == "voice"
    assert "voice_memory" in adapter.capabilities
    assert "profile_management" in adapter.capabilities


@pytest.mark.asyncio
async def test_voice_adapter_get_profiles():
    """Test getting voice profiles."""
    agent = MockVoiceMemoryAgent()
    adapter = VoiceSystemAdapter(
        voice_component=agent,
        component_type=VoiceComponentType.MEMORY,
    )

    result = await adapter.execute_task({
        "action": "get_profiles",
        "input": {},
    })

    assert "profiles" in result
    assert result["count"] == 2


@pytest.mark.asyncio
async def test_voice_adapter_check_freshness():
    """Test freshness checking."""
    agent = MockVoiceMemoryAgent()
    adapter = VoiceSystemAdapter(
        voice_component=agent,
        component_type=VoiceComponentType.MEMORY,
    )

    result = await adapter.execute_task({
        "action": "check_freshness",
        "input": {"speaker": "Derek"},
    })

    assert "freshness" in result
    assert result["freshness"] == 0.95


@pytest.mark.asyncio
async def test_voice_verification_adapter():
    """Test speaker verification adapter."""
    service = MockSpeakerVerificationService()
    adapter = VoiceSystemAdapter(
        voice_component=service,
        component_type=VoiceComponentType.VERIFICATION,
    )

    result = await adapter.execute_task({
        "action": "verify_speaker",
        "input": {
            "audio_data": b"test_audio",
            "expected_speaker": "Derek",
        },
    })

    assert result["verified"] is True
    assert result["confidence"] > 0.85


@pytest.mark.asyncio
async def test_voice_enrollment():
    """Test speaker enrollment."""
    service = MockSpeakerVerificationService()
    adapter = VoiceSystemAdapter(
        voice_component=service,
        component_type=VoiceComponentType.VERIFICATION,
    )

    result = await adapter.execute_task({
        "action": "enroll_speaker",
        "input": {
            "speaker_name": "NewUser",
            "audio_samples": [b"sample1", b"sample2"],
        },
    })

    assert result["success"] is True


# =============================================================================
# Legacy Adapter Tests
# =============================================================================

@pytest.mark.asyncio
async def test_legacy_adapter_wrapping():
    """Test wrapping a legacy agent."""
    class LegacyAgent:
        def analyze(self, data: Dict) -> Dict:
            return {"analyzed": True, "data": data}

    legacy = LegacyAgent()
    adapter = LegacyAgentAdapter(
        wrapped_agent=legacy,
        agent_name="legacy_agent",
        agent_type="custom",
        capabilities={"analyze"},
        task_handler=legacy.analyze,
    )

    assert adapter.agent_name == "legacy_agent"
    assert "analyze" in adapter.capabilities


@pytest.mark.asyncio
async def test_adapt_agent_auto_detection():
    """Test auto-detection of handlers in adapt_agent."""
    class AutoAgent:
        def capture(self, **kwargs) -> Dict:
            return {"captured": True}

        def analyze(self, **kwargs) -> Dict:
            return {"analyzed": True}

    agent = AutoAgent()
    adapter = adapt_agent(
        agent,
        name="auto_agent",
        agent_type="vision",
        capabilities={"capture", "analyze"},
    )

    # Handlers should be auto-detected
    result = await adapter.execute_task({
        "action": "capture",
        "input": {},
    })
    assert result["captured"] is True


# =============================================================================
# JARVIS Bridge Tests
# =============================================================================

@pytest.mark.asyncio
async def test_bridge_initialization():
    """Test JARVISNeuralMeshBridge initialization."""
    config = AgentDiscoveryConfig(
        enabled_categories={SystemCategory.INTELLIGENCE},
        skip_agents={"intelligence_uae"},  # Skip for faster test
    )

    bridge = JARVISNeuralMeshBridge(config=config)

    assert bridge._config == config
    assert not bridge._initialized
    assert not bridge._running


@pytest.mark.asyncio
async def test_bridge_custom_agent_registration():
    """Test registering custom agents with bridge."""
    bridge = JARVISNeuralMeshBridge(
        config=AgentDiscoveryConfig(
            enabled_categories=set(),  # Don't auto-discover
        )
    )

    # Add custom agent
    mock_agent = MockIntelligenceEngine()
    bridge.add_custom_agent("custom_intel", mock_agent)

    assert "custom_intel" in bridge._custom_agents


@pytest.mark.asyncio
async def test_bridge_get_agents_by_capability():
    """Test getting agents by capability."""
    bridge = JARVISNeuralMeshBridge(
        config=AgentDiscoveryConfig(enabled_categories=set())
    )

    # Create and register mock adapters
    engine = MockIntelligenceEngine()
    adapter = IntelligenceEngineAdapter(
        engine=engine,
        engine_type=IntelligenceEngineType.UAE,
        agent_name="test_uae",
    )

    bridge._adapters["test_uae"] = adapter

    # Get by capability
    agents = bridge.get_agents_by_capability("spatial_awareness")
    assert len(agents) == 1
    assert agents[0].agent_name == "test_uae"


@pytest.mark.asyncio
async def test_bridge_event_callbacks():
    """Test bridge event callbacks."""
    bridge = JARVISNeuralMeshBridge(
        config=AgentDiscoveryConfig(enabled_categories=set())
    )

    callback_data = {}

    def on_agent_registered(data):
        callback_data["registered"] = data

    bridge.on("agent_registered", on_agent_registered)

    # Simulate callback trigger
    await bridge._trigger_callback("agent_registered", {"name": "test_agent"})

    assert "registered" in callback_data
    assert callback_data["registered"]["name"] == "test_agent"


# =============================================================================
# Full Integration Test
# =============================================================================

@pytest.mark.asyncio
async def test_full_coordinator_integration():
    """Test full integration with NeuralMeshCoordinator."""
    # Create coordinator
    coordinator = NeuralMeshCoordinator()
    await coordinator.initialize()
    await coordinator.start()

    try:
        # Create adapters with mock components
        intel_engine = MockIntelligenceEngine()
        intel_adapter = IntelligenceEngineAdapter(
            engine=intel_engine,
            engine_type=IntelligenceEngineType.UAE,
            agent_name="intel_test",
        )

        voice_agent = MockVoiceMemoryAgent()
        voice_adapter = VoiceSystemAdapter(
            voice_component=voice_agent,
            component_type=VoiceComponentType.MEMORY,
            agent_name="voice_test",
        )

        # Register with coordinator
        await coordinator.register_agent(intel_adapter)
        await coordinator.register_agent(voice_adapter)

        # Verify registration
        registry = coordinator.registry
        intel_info = await registry.get("intel_test")
        voice_info = await registry.get("voice_test")

        assert intel_info is not None
        assert voice_info is not None
        assert intel_info.agent_type == "intelligence"
        assert voice_info.agent_type == "voice"

        # Test cross-agent communication
        bus = coordinator.bus
        received_messages = []

        async def handler(msg: AgentMessage):
            received_messages.append(msg)

        await bus.subscribe("voice_test", MessageType.CUSTOM, handler)

        # Intel agent broadcasts
        await intel_adapter.broadcast(
            message_type=MessageType.CUSTOM,
            payload={"event": "workspace_analyzed", "space_id": 3},
        )

        await asyncio.sleep(0.1)  # Allow message propagation

        assert len(received_messages) >= 1

        # Test health check
        health = await coordinator.health_check()
        assert health["healthy"] is True

    finally:
        await coordinator.stop()


@pytest.mark.asyncio
async def test_multi_adapter_workflow():
    """Test workflow across multiple adapters."""
    coordinator = NeuralMeshCoordinator()
    await coordinator.initialize()
    await coordinator.start()

    try:
        # Create multiple adapters
        adapters = []

        intel_engine = MockIntelligenceEngine()
        intel = IntelligenceEngineAdapter(
            engine=intel_engine,
            engine_type=IntelligenceEngineType.UAE,
            agent_name="workflow_intel",
        )
        adapters.append(intel)

        autonomy_agent = MockAutonomousAgent()
        autonomy = AutonomyEngineAdapter(
            autonomy_component=autonomy_agent,
            component_type=AutonomyComponentType.AGENT,
            agent_name="workflow_autonomy",
        )
        adapters.append(autonomy)

        # Register all
        for adapter in adapters:
            await coordinator.register_agent(adapter)

        # Create workflow
        from neural_mesh.data_models import WorkflowTask, ExecutionStrategy

        tasks = [
            WorkflowTask(
                task_id="analyze",
                required_capability="spatial_awareness",
                payload={
                    "action": "analyze_workspace",
                    "input": {"space_id": 1},
                },
            ),
            WorkflowTask(
                task_id="execute",
                required_capability="autonomous_execution",
                payload={
                    "action": "run",
                    "input": {"goal": "Process analysis"},
                },
                dependencies=["analyze"],
            ),
        ]

        # Execute workflow
        result = await coordinator.orchestrator.execute_workflow(
            name="test_workflow",
            tasks=tasks,
            strategy=ExecutionStrategy.HYBRID,
        )

        assert result.success is True
        assert len(result.task_results) == 2

    finally:
        await coordinator.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
