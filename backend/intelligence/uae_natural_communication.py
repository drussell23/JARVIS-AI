#!/usr/bin/env python3
"""
UAE Natural Communication Layer
================================

Provides natural language communication during UAE intelligence fusion process.

JARVIS speaks to you naturally about what it's thinking, learning, and doing
as it combines Context Intelligence and Situational Awareness.

Example Flow:
    User: "Living Room TV"

    JARVIS: "Looking for Control Center... I remember it's usually in the top-right."
    JARVIS: "Found it! Position has shifted slightly from last time - updating my memory."
    JARVIS: "Connecting to Living Room TV..."
    JARVIS: "Connected! I'll remember this works better now."

Author: Derek J. Russell
Date: October 2025
Version: 1.0.0
"""

import asyncio
import logging
import random
from typing import Optional, Dict, Any, Callable
from enum import Enum

from backend.intelligence.unified_awareness_engine import (
    UnifiedDecision,
    ExecutionResult,
    DecisionSource,
    ContextualData,
    SituationalData
)
from backend.intelligence.uae_communication_config import (
    ResponseStyle,
    get_response_templates,
    format_confidence,
    DEFAULT_RESPONSE_STYLE
)

logger = logging.getLogger(__name__)


class CommunicationMode(Enum):
    """Communication verbosity modes"""
    SILENT = "silent"           # No communication
    MINIMAL = "minimal"         # Only important updates
    NORMAL = "normal"           # Balanced feedback
    VERBOSE = "verbose"         # Detailed explanations
    DEBUG = "debug"            # Full technical details


class UAENaturalCommunicator:
    """
    Natural language communicator for UAE intelligence process

    Translates technical UAE operations into natural, conversational feedback
    """

    def __init__(
        self,
        voice_callback: Optional[Callable] = None,
        text_callback: Optional[Callable] = None,
        mode: CommunicationMode = CommunicationMode.NORMAL,
        response_style: ResponseStyle = DEFAULT_RESPONSE_STYLE
    ):
        """
        Initialize communicator

        Args:
            voice_callback: Async function to speak text
            text_callback: Async function to send text
            mode: Communication verbosity mode
            response_style: Response personality style
        """
        self.voice_callback = voice_callback
        self.text_callback = text_callback
        self.mode = mode
        self.response_style = response_style

        # State tracking for contextual responses
        self.last_element = None
        self.last_decision_source = None
        self.learning_count = 0
        self.correction_count = 0

    async def on_decision_start(self, element_id: str):
        """Communicate when starting to make a decision"""
        if self.mode == CommunicationMode.SILENT:
            return

        self.last_element = element_id

        # Get template and format
        templates = get_response_templates('decision_start', style=self.response_style)
        template = random.choice(templates)
        message = template.format(element=element_id.replace('_', ' ').title())

        if self.mode == CommunicationMode.VERBOSE:
            message += " I'm combining what I remember with what I currently see."

        await self._communicate(message, priority="low")

    async def on_context_data(self, context_data: Optional[ContextualData]):
        """Communicate about context intelligence findings"""
        if self.mode in [CommunicationMode.SILENT, CommunicationMode.MINIMAL]:
            return

        if not context_data:
            if self.mode in [CommunicationMode.VERBOSE, CommunicationMode.DEBUG]:
                await self._communicate(
                    "This is new to me - I haven't encountered this before.",
                    priority="low"
                )
            return

        # Build natural message about context
        if context_data.usage_count > 10:
            confidence_desc = "very familiar"
        elif context_data.usage_count > 5:
            confidence_desc = "familiar"
        else:
            confidence_desc = "somewhat familiar"

        if self.mode == CommunicationMode.VERBOSE:
            message = (
                f"I'm {confidence_desc} with this - I've used it {context_data.usage_count} times. "
                f"I remember it's usually around position {context_data.expected_position}."
            )
            await self._communicate(message, priority="low")
        elif self.mode == CommunicationMode.DEBUG:
            message = (
                f"[Context] Usage: {context_data.usage_count}, "
                f"Position: {context_data.expected_position}, "
                f"Pattern strength: {context_data.pattern_strength:.2f}, "
                f"Confidence: {context_data.confidence:.2f}"
            )
            await self._communicate(message, priority="low")

    async def on_situation_data(self, situation_data: Optional[SituationalData]):
        """Communicate about situational awareness findings"""
        if self.mode in [CommunicationMode.SILENT, CommunicationMode.MINIMAL]:
            return

        if not situation_data:
            if self.mode in [CommunicationMode.VERBOSE, CommunicationMode.DEBUG]:
                await self._communicate(
                    "I can't see it right now - I'll use what I remember.",
                    priority="low"
                )
            return

        if self.mode == CommunicationMode.VERBOSE:
            confidence_desc = "clearly" if situation_data.confidence > 0.8 else "probably"
            message = f"I can {confidence_desc} see it at position {situation_data.detected_position}."
            await self._communicate(message, priority="low")
        elif self.mode == CommunicationMode.DEBUG:
            message = (
                f"[Situation] Position: {situation_data.detected_position}, "
                f"Confidence: {situation_data.confidence:.2f}, "
                f"Method: {situation_data.detection_method}"
            )
            await self._communicate(message, priority="low")

    async def on_decision_made(self, decision: UnifiedDecision):
        """Communicate about the decision fusion process"""
        self.last_decision_source = decision.decision_source

        if self.mode == CommunicationMode.SILENT:
            return

        confidence_pct = format_confidence(decision.confidence)

        # Map decision source to template source
        source_map = {
            DecisionSource.FUSION: 'fused',
            DecisionSource.CONTEXT: 'context',
            DecisionSource.SITUATION: 'situation'
        }

        # Get appropriate templates
        if decision.decision_source == DecisionSource.FALLBACK:
            message = "Hmm, I'm having trouble locating this. Let me try a different approach..."
        else:
            # Check if position changed (for FUSION only)
            if decision.decision_source == DecisionSource.FUSION and not decision.metadata.get('agreement'):
                # Position changed - use special template
                templates = get_response_templates('decision_made', source='position_changed', style=self.response_style)
                self.correction_count += 1
            else:
                # Normal decision - use source-specific template
                source = source_map.get(decision.decision_source, 'fused')
                templates = get_response_templates('decision_made', source=source, style=self.response_style)

            template = random.choice(templates)
            message = template.format(
                element=self.last_element.replace('_', ' ').title() if self.last_element else 'target',
                confidence=confidence_pct
            )

            # Add extra detail for verbose/debug modes
            if self.mode == CommunicationMode.DEBUG:
                message += f" [Source: {decision.decision_source.value}, Pos: {decision.chosen_position}, " \
                          f"Ctx: {decision.context_weight:.2f}, Sit: {decision.situation_weight:.2f}]"

        await self._communicate(message, priority="normal")

    async def on_execution_start(self, decision: UnifiedDecision, action: str):
        """Communicate before executing action"""
        if self.mode in [CommunicationMode.SILENT, CommunicationMode.MINIMAL]:
            return

        # Natural action descriptions
        action_messages = {
            'click': f"Clicking at {decision.chosen_position}...",
            'connect': "Connecting...",
            'navigate': "Navigating..."
        }

        message = action_messages.get(action, f"Executing {action}...")

        if self.mode == CommunicationMode.VERBOSE:
            message += f" Using {decision.decision_source.value} intelligence."

        await self._communicate(message, priority="low")

    async def on_execution_complete(self, result: ExecutionResult):
        """Communicate after execution completes"""
        if result.success:
            # Get success templates
            templates = get_response_templates('execution_complete', source='success', style=self.response_style)
            template = random.choice(templates)

            message = template.format(
                action=getattr(result.decision, 'action', 'operation'),
                duration=f"{result.execution_time:.1f}",
                confidence=format_confidence(result.decision.confidence),
                verified='yes' if result.verification_passed else 'no'
            )

            # Add learning context for normal mode
            if self.mode == CommunicationMode.NORMAL and self.correction_count > 0 and self.learning_count % 3 == 0:
                message += f" I'm learning - {self.correction_count} improvements made so far."

            await self._communicate(message, priority="normal")
        else:
            # Failure - get failure templates
            templates = get_response_templates('execution_complete', source='failed', style=self.response_style)
            template = random.choice(templates)

            error_desc = result.error or "unknown issue"
            message = template.format(
                action=getattr(result.decision, 'action', 'operation'),
                error=error_desc
            )

            await self._communicate(message, priority="high")

    async def on_learning_event(self, result: ExecutionResult):
        """Communicate about learning from execution"""
        self.learning_count += 1

        if self.mode in [CommunicationMode.SILENT, CommunicationMode.MINIMAL]:
            return

        if not result.success:
            return  # Don't announce learning from failures

        # Announce significant learning milestones
        if self.mode == CommunicationMode.NORMAL and self.learning_count % 10 == 0:
            templates = get_response_templates('learning_event', style=self.response_style)
            template = random.choice(templates)
            message = template.format(count=self.learning_count)
            await self._communicate(message, priority="low")
        elif self.mode in [CommunicationMode.VERBOSE, CommunicationMode.DEBUG]:
            # More frequent updates in verbose mode
            if result.decision.decision_source == DecisionSource.SITUATION:
                message = "Updated my memory with this new position. Next time will be even faster!"
                await self._communicate(message, priority="low")
            elif result.decision.decision_source == DecisionSource.FUSION:
                if not result.decision.metadata.get('agreement'):
                    message = "Interesting - I've adjusted my understanding based on what I just saw."
                    await self._communicate(message, priority="low")

        if self.mode == CommunicationMode.DEBUG:
            message = (
                f"[Learning] Event #{self.learning_count}, "
                f"Source: {result.decision.decision_source.value}, "
                f"Success: {result.success}, "
                f"Corrections: {self.correction_count}"
            )
            await self._communicate(message, priority="low")

    async def on_device_connection(self, device_name: str, step: str, step_result: Dict[str, Any]):
        """Communicate during multi-step device connection"""
        if self.mode == CommunicationMode.SILENT:
            return

        # Get appropriate template based on step
        templates = get_response_templates('device_connection', source=step, style=self.response_style)

        if not templates:
            # Fallback for unknown steps
            message = f"Step: {step}"
        else:
            template = random.choice(templates)
            duration = step_result.get('duration', 0)
            error = step_result.get('error', 'unknown error')

            message = template.format(
                device=device_name,
                duration=f"{duration:.1f}" if duration else "0.0",
                error=error,
                step=step
            )

        # Determine priority
        phase = step_result.get('phase', 'start')
        priority = "normal" if step_result.get('success', True) else "high"

        # Only communicate on certain phases to avoid spam
        if self.mode == CommunicationMode.MINIMAL:
            # Only communicate start and final result
            if step not in ['start', 'complete', 'error']:
                return

        await self._communicate(message, priority=priority)

    async def _communicate(self, message: str, priority: str = "normal"):
        """
        Send message via voice and/or text

        Args:
            message: Message to communicate
            priority: Priority level (low, normal, high)
        """
        try:
            # Send via text callback
            if self.text_callback:
                if asyncio.iscoroutinefunction(self.text_callback):
                    await self.text_callback(message, priority=priority)
                else:
                    self.text_callback(message, priority=priority)

            # Send via voice callback (for important messages)
            if self.voice_callback and priority in ['normal', 'high']:
                if asyncio.iscoroutinefunction(self.voice_callback):
                    await self.voice_callback(message)
                else:
                    self.voice_callback(message)

            # Always log
            logger.info(f"[UAE-COMM] {message}")

        except Exception as e:
            logger.error(f"[UAE-COMM] Error communicating: {e}")

    def set_mode(self, mode: CommunicationMode):
        """Change communication mode"""
        self.mode = mode
        logger.info(f"[UAE-COMM] Mode changed to: {mode.value}")

    def get_stats(self) -> Dict[str, Any]:
        """Get communication statistics"""
        return {
            'learning_count': self.learning_count,
            'correction_count': self.correction_count,
            'last_element': self.last_element,
            'last_decision_source': self.last_decision_source.value if self.last_decision_source else None,
            'mode': self.mode.value
        }


# ============================================================================
# Global Communicator Instance
# ============================================================================

_global_communicator: Optional[UAENaturalCommunicator] = None


def initialize_communicator(
    voice_callback: Optional[Callable] = None,
    text_callback: Optional[Callable] = None,
    mode: CommunicationMode = CommunicationMode.NORMAL,
    response_style: ResponseStyle = DEFAULT_RESPONSE_STYLE
) -> UAENaturalCommunicator:
    """
    Initialize global UAE communicator

    Args:
        voice_callback: Voice output function
        text_callback: Text output function
        mode: Communication mode
        response_style: Response personality style

    Returns:
        UAENaturalCommunicator instance
    """
    global _global_communicator

    _global_communicator = UAENaturalCommunicator(
        voice_callback=voice_callback,
        text_callback=text_callback,
        mode=mode,
        response_style=response_style
    )

    logger.info(f"[UAE-COMM] Initialized with mode: {mode.value}, style: {response_style.value}")
    return _global_communicator


def get_communicator() -> Optional[UAENaturalCommunicator]:
    """Get global communicator instance"""
    return _global_communicator


# ============================================================================
# Example Usage
# ============================================================================

async def main():
    """Demo natural communication"""
    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 80)
    print("UAE Natural Communication - Demo")
    print("=" * 80)

    # Voice callback
    async def speak(text):
        print(f"\n🗣️  JARVIS: \"{text}\"")

    # Text callback
    async def send_text(text, priority="normal"):
        emoji = {"low": "💬", "normal": "💭", "high": "⚠️"}
        print(f"{emoji.get(priority, '💬')} {text}")

    # Initialize communicator
    comm = initialize_communicator(
        voice_callback=speak,
        text_callback=send_text,
        mode=CommunicationMode.NORMAL
    )

    print("\n📱 User: \"Living Room TV\"\n")

    # Simulate UAE flow
    await comm.on_decision_start("control_center")
    await asyncio.sleep(0.5)

    # Context data
    from intelligence.unified_awareness_engine import ContextualData
    context = ContextualData(
        element_id="control_center",
        expected_position=(1236, 12),
        confidence=0.85,
        usage_count=47,
        last_success=1234567890,
        pattern_strength=0.9
    )
    await comm.on_context_data(context)
    await asyncio.sleep(0.5)

    # Situation data
    from intelligence.unified_awareness_engine import SituationalData
    situation = SituationalData(
        element_id="control_center",
        detected_position=(1287, 12),
        confidence=0.92,
        detection_method="vision_claude",
        detection_time=1234567890
    )
    await comm.on_situation_data(situation)
    await asyncio.sleep(0.5)

    # Decision
    from intelligence.unified_awareness_engine import UnifiedDecision, DecisionSource
    decision = UnifiedDecision(
        element_id="control_center",
        chosen_position=(1287, 12),
        confidence=0.92,
        decision_source=DecisionSource.FUSION,
        context_weight=0.4,
        situation_weight=0.6,
        reasoning="Situation is recent, choose current detection",
        timestamp=1234567890,
        metadata={'agreement': False}
    )
    await comm.on_decision_made(decision)
    await asyncio.sleep(0.5)

    # Execution
    await comm.on_execution_start(decision, "click")
    await asyncio.sleep(1.0)

    # Result
    from intelligence.unified_awareness_engine import ExecutionResult
    result = ExecutionResult(
        decision=decision,
        success=True,
        execution_time=1.2,
        verification_passed=True
    )
    await comm.on_execution_complete(result)
    await asyncio.sleep(0.5)

    await comm.on_learning_event(result)

    # Stats
    print("\n\n📊 Communication Stats:")
    stats = comm.get_stats()
    print(f"   Learning events: {stats['learning_count']}")
    print(f"   Corrections made: {stats['correction_count']}")
    print(f"   Mode: {stats['mode']}")

    print("\n" + "=" * 80)
    print("✅ Demo complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
