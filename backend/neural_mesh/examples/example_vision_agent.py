"""
Example Vision Agent for Neural Mesh

This demonstrates how to create a Neural Mesh agent that:
1. Inherits from BaseNeuralMeshAgent
2. Implements required methods
3. Uses knowledge graph for learning
4. Communicates with other agents
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, Dict, List, Optional, Set

from ..base.base_neural_mesh_agent import BaseNeuralMeshAgent
from ..data_models import (
    KnowledgeType,
    MessageType,
    AgentMessage,
)

logger = logging.getLogger(__name__)


class ExampleVisionAgent(BaseNeuralMeshAgent):
    """
    Example vision agent that demonstrates Neural Mesh integration.

    This agent can:
    - Capture screens (simulated)
    - Detect errors in captured content
    - Learn from past errors via knowledge graph
    - Collaborate with other agents

    Usage:
        agent = ExampleVisionAgent()

        # Initialize with Neural Mesh components
        await coordinator.register_agent(agent)

        # Or manually:
        await agent.initialize(bus, registry, knowledge_graph)
        await agent.start()

        # Agent is now active and processing tasks
    """

    def __init__(self) -> None:
        """Initialize the vision agent."""
        super().__init__(
            agent_name="example_vision_agent",
            agent_type="vision",
            capabilities={
                "screen_capture",
                "error_detection",
                "ocr",
                "visual_analysis",
            },
            version="1.0.0",
        )

        # Agent-specific state
        self._capture_count = 0
        self._error_cache: Dict[str, List[str]] = {}

    async def on_initialize(self) -> None:
        """Initialize agent-specific resources."""
        logger.info("Initializing ExampleVisionAgent")

        # Subscribe to vision-related custom messages
        await self.subscribe(
            MessageType.CUSTOM,
            self._handle_custom_message,
        )

        # Load any cached knowledge about errors
        if self.knowledge_graph:
            past_errors = await self.query_knowledge(
                query="screen errors visual bugs",
                knowledge_types=[KnowledgeType.ERROR],
                limit=10,
            )
            for error in past_errors:
                error_type = error.data.get("error_type", "unknown")
                if error_type not in self._error_cache:
                    self._error_cache[error_type] = []
                self._error_cache[error_type].append(error.data.get("solution", ""))

            logger.info(
                "Loaded %d past error patterns from knowledge graph",
                len(past_errors),
            )

    async def on_start(self) -> None:
        """Called when agent starts."""
        logger.info("ExampleVisionAgent started - ready for visual tasks")

    async def on_stop(self) -> None:
        """Cleanup when agent stops."""
        logger.info(
            "ExampleVisionAgent stopping - processed %d captures",
            self._capture_count,
        )

    async def execute_task(self, payload: Dict[str, Any]) -> Any:
        """
        Execute a vision task.

        This is the main entry point for task execution from the orchestrator.
        """
        action = payload.get("action", "")
        input_data = payload.get("input", {})

        logger.info("Executing vision task: %s", action)

        if action == "screen_capture":
            return await self.capture_screen(
                space_id=input_data.get("space_id"),
            )
        elif action == "error_detection":
            return await self.detect_errors(
                screenshot=input_data.get("screenshot"),
            )
        elif action == "ocr":
            return await self.extract_text(
                screenshot=input_data.get("screenshot"),
            )
        elif action == "visual_analysis":
            return await self.analyze_visual(
                screenshot=input_data.get("screenshot"),
            )
        else:
            raise ValueError(f"Unknown action: {action}")

    async def capture_screen(
        self,
        space_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Capture a screen (simulated).

        In a real implementation, this would use actual screen capture.
        """
        self._capture_count += 1

        # Simulate screen capture
        await asyncio.sleep(0.1)  # Simulate capture time

        screenshot = {
            "capture_id": f"cap_{self._capture_count}",
            "space_id": space_id or 1,
            "width": 1920,
            "height": 1080,
            "timestamp": asyncio.get_event_loop().time(),
            "data": f"[simulated screenshot data for space {space_id}]",
        }

        logger.debug("Captured screen %s", screenshot["capture_id"])

        # Store in knowledge graph for future reference
        if self.knowledge_graph:
            await self.add_knowledge(
                knowledge_type=KnowledgeType.OBSERVATION,
                data={
                    "type": "screen_capture",
                    "space_id": space_id,
                    "capture_id": screenshot["capture_id"],
                },
                tags={"vision", "capture", f"space_{space_id}"},
                ttl_seconds=300,  # 5 minute cache
            )

        # Notify other agents
        await self.broadcast(
            message_type=MessageType.CUSTOM,
            payload={
                "event": "screen_captured",
                "capture_id": screenshot["capture_id"],
                "space_id": space_id,
            },
        )

        return screenshot

    async def detect_errors(
        self,
        screenshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Detect errors in a screenshot (simulated).

        In a real implementation, this would use vision AI for error detection.
        """
        # Simulate error detection
        await asyncio.sleep(0.2)  # Simulate analysis time

        # Simulate finding some errors (randomly for demo)
        errors = []
        if random.random() > 0.5:
            errors.append({
                "type": "syntax_error",
                "description": "Potential syntax error detected",
                "location": {"x": 100, "y": 200},
                "confidence": 0.85,
            })

        if random.random() > 0.7:
            errors.append({
                "type": "ui_glitch",
                "description": "UI element misalignment detected",
                "location": {"x": 500, "y": 300},
                "confidence": 0.75,
            })

        # Check knowledge graph for known solutions
        solutions = []
        for error in errors:
            if self.knowledge_graph:
                # Query for similar past errors
                similar = await self.query_knowledge(
                    query=f"{error['type']} {error['description']}",
                    knowledge_types=[KnowledgeType.ERROR, KnowledgeType.SOLUTION],
                    limit=3,
                )
                for entry in similar:
                    if "solution" in entry.data:
                        solutions.append({
                            "error_type": error["type"],
                            "suggested_solution": entry.data["solution"],
                            "confidence": entry.confidence,
                        })

        result = {
            "errors_found": len(errors),
            "errors": errors,
            "solutions": solutions,
            "analysis_confidence": 0.9,
        }

        # Store detected errors in knowledge graph
        if errors and self.knowledge_graph:
            for error in errors:
                await self.add_knowledge(
                    knowledge_type=KnowledgeType.ERROR,
                    data={
                        "error_type": error["type"],
                        "description": error["description"],
                        "source": "visual_detection",
                    },
                    tags={"error", error["type"], "visual"},
                    confidence=error["confidence"],
                )

        return result

    async def extract_text(
        self,
        screenshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Extract text from a screenshot (simulated OCR).
        """
        await asyncio.sleep(0.15)  # Simulate OCR time

        return {
            "text": "[Simulated extracted text from screenshot]",
            "confidence": 0.92,
            "regions": [
                {"text": "def main():", "location": {"x": 50, "y": 100}},
                {"text": "# TODO: fix bug", "location": {"x": 50, "y": 120}},
            ],
        }

    async def analyze_visual(
        self,
        screenshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive visual analysis.
        """
        # Combine multiple analysis types
        errors = await self.detect_errors(screenshot)
        text = await self.extract_text(screenshot)

        return {
            "analysis_type": "comprehensive",
            "errors": errors,
            "text_content": text,
            "visual_elements": {
                "windows": 3,
                "ui_elements": 45,
                "code_regions": 2,
            },
        }

    async def _handle_custom_message(self, message: AgentMessage) -> None:
        """Handle custom messages from other agents."""
        event = message.payload.get("event", "")

        if event == "request_screen":
            # Another agent is requesting a screen capture
            space_id = message.payload.get("space_id")
            screenshot = await self.capture_screen(space_id)

            # Send response
            if self.message_bus:
                await self.message_bus.respond(
                    message,
                    payload={"screenshot": screenshot},
                    from_agent=self.agent_name,
                )


# Example usage
async def demo():
    """Demonstrate the vision agent."""
    from ..neural_mesh_coordinator import NeuralMeshCoordinator

    # Create and start Neural Mesh
    coordinator = NeuralMeshCoordinator()
    await coordinator.initialize()
    await coordinator.start()

    try:
        # Create and register vision agent
        agent = ExampleVisionAgent()
        await coordinator.register_agent(agent)

        print("Vision agent registered and running!")
        print(f"Agent: {agent}")
        print(f"Capabilities: {agent.capabilities}")

        # Simulate some tasks
        result = await agent.capture_screen(space_id=3)
        print(f"Capture result: {result}")

        errors = await agent.detect_errors(result)
        print(f"Error detection: {errors}")

        # Show metrics
        metrics = agent.get_metrics()
        print(f"Agent metrics: tasks_completed={metrics.tasks_completed}")

        # Show system health
        health = await coordinator.health_check()
        print(f"System health: {health}")

    finally:
        await coordinator.stop()


if __name__ == "__main__":
    asyncio.run(demo())
