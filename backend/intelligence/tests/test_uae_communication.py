#!/usr/bin/env python3
"""
UAE Natural Communication Tests
================================

Tests for UAE natural communication layer integration.

Author: Derek J. Russell
Date: October 2025
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import List

# Add repo root to path
repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

from backend.intelligence.uae_natural_communication import (
    UAENaturalCommunicator,
    CommunicationMode,
    initialize_communicator
)
from backend.intelligence.uae_communication_config import ResponseStyle
from backend.intelligence.unified_awareness_engine import (
    UnifiedDecision,
    ExecutionResult,
    DecisionSource
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class MessageCollector:
    """Collects messages for testing"""

    def __init__(self):
        self.messages: List[str] = []
        self.voice_messages: List[str] = []

    async def text_callback(self, message: str, priority: str = "normal"):
        """Collect text messages"""
        self.messages.append(f"[{priority.upper()}] {message}")
        logger.info(f"📝 TEXT: {message}")

    async def voice_callback(self, message: str):
        """Collect voice messages"""
        self.voice_messages.append(message)
        logger.info(f"🔊 VOICE: {message}")

    def clear(self):
        """Clear collected messages"""
        self.messages.clear()
        self.voice_messages.clear()


async def test_decision_flow():
    """Test complete decision-making flow"""
    print("\n" + "=" * 80)
    print("Test 1: Complete Decision Flow (Normal Mode)")
    print("=" * 80)

    collector = MessageCollector()
    communicator = UAENaturalCommunicator(
        voice_callback=collector.voice_callback,
        text_callback=collector.text_callback,
        mode=CommunicationMode.NORMAL,
        response_style=ResponseStyle.CASUAL
    )

    # Simulate decision flow
    print("\n1️⃣  Starting decision...")
    await communicator.on_decision_start("control_center")

    # Simulate decision made (using fusion)
    print("\n2️⃣  Decision made...")
    decision = UnifiedDecision(
        element_id="control_center",
        chosen_position=(1400, 25),
        confidence=0.92,
        decision_source=DecisionSource.FUSION,
        context_weight=0.5,
        situation_weight=0.5,
        reasoning="Both layers agree on position",
        timestamp=time.time(),
        metadata={'agreement': True}
    )
    await communicator.on_decision_made(decision)

    # Simulate execution
    print("\n3️⃣  Executing...")
    await communicator.on_execution_start(decision, 'click')

    # Simulate completion
    print("\n4️⃣  Execution complete...")
    result = ExecutionResult(
        decision=decision,
        success=True,
        execution_time=0.45,
        verification_passed=True,
        metadata={'coordinates': (1400, 25), 'method_used': 'uae_fusion'}
    )
    await communicator.on_execution_complete(result)

    # Simulate learning
    print("\n5️⃣  Learning event...")
    await communicator.on_learning_event(result)

    print(f"\n📊 Collected {len(collector.messages)} text messages")
    print(f"🔊 Collected {len(collector.voice_messages)} voice messages")

    return True


async def test_device_connection_flow():
    """Test multi-step device connection flow"""
    print("\n" + "=" * 80)
    print("Test 2: Device Connection Flow (Normal Mode)")
    print("=" * 80)

    collector = MessageCollector()
    communicator = UAENaturalCommunicator(
        voice_callback=collector.voice_callback,
        text_callback=collector.text_callback,
        mode=CommunicationMode.NORMAL,
        response_style=ResponseStyle.CASUAL
    )

    device_name = "Living Room TV"

    # Start
    print("\n1️⃣  Starting connection...")
    await communicator.on_device_connection(
        device_name,
        step='start',
        step_result={'phase': 'start', 'success': True}
    )

    # Step 1: Control Center
    print("\n2️⃣  Opening Control Center...")
    await communicator.on_device_connection(
        device_name,
        step='open_control_center',
        step_result={'phase': 'start', 'success': True}
    )
    await asyncio.sleep(0.2)
    await communicator.on_device_connection(
        device_name,
        step='open_control_center',
        step_result={'phase': 'success', 'success': True}
    )

    # Step 2: Screen Mirroring
    print("\n3️⃣  Opening Screen Mirroring...")
    await communicator.on_device_connection(
        device_name,
        step='open_screen_mirroring',
        step_result={'phase': 'start', 'success': True}
    )
    await asyncio.sleep(0.2)
    await communicator.on_device_connection(
        device_name,
        step='open_screen_mirroring',
        step_result={'phase': 'success', 'success': True}
    )

    # Step 3: Select Device
    print("\n4️⃣  Selecting device...")
    await communicator.on_device_connection(
        device_name,
        step='select_device',
        step_result={'phase': 'start', 'success': True}
    )
    await asyncio.sleep(0.2)
    await communicator.on_device_connection(
        device_name,
        step='select_device',
        step_result={'phase': 'success', 'success': True}
    )

    # Complete
    print("\n5️⃣  Connection complete...")
    await communicator.on_device_connection(
        device_name,
        step='complete',
        step_result={'phase': 'success', 'success': True, 'duration': 2.1}
    )

    print(f"\n📊 Collected {len(collector.messages)} text messages")
    print(f"🔊 Collected {len(collector.voice_messages)} voice messages")

    return True


async def test_position_change_detection():
    """Test position change communication"""
    print("\n" + "=" * 80)
    print("Test 3: Position Change Detection (Verbose Mode)")
    print("=" * 80)

    collector = MessageCollector()
    communicator = UAENaturalCommunicator(
        voice_callback=collector.voice_callback,
        text_callback=collector.text_callback,
        mode=CommunicationMode.VERBOSE,
        response_style=ResponseStyle.CASUAL
    )

    # Simulate position disagreement (position has changed)
    print("\n1️⃣  Starting decision...")
    await communicator.on_decision_start("control_center")

    print("\n2️⃣  Position has changed...")
    decision = UnifiedDecision(
        element_id="control_center",
        chosen_position=(1420, 25),  # New position
        confidence=0.88,
        decision_source=DecisionSource.FUSION,
        context_weight=0.3,
        situation_weight=0.7,  # Situation has higher weight
        reasoning="Position shifted - preferring visual detection",
        timestamp=time.time(),
        metadata={'agreement': False, 'position_shifted': True}
    )
    await communicator.on_decision_made(decision)

    print(f"\n📊 Correction count: {communicator.correction_count}")
    print(f"📝 Messages: {len(collector.messages)}")

    return True


async def test_different_styles():
    """Test different communication styles"""
    print("\n" + "=" * 80)
    print("Test 4: Different Communication Styles")
    print("=" * 80)

    styles = [
        (ResponseStyle.MINIMAL, "Minimal"),
        (ResponseStyle.CASUAL, "Casual"),
        (ResponseStyle.PROFESSIONAL, "Professional"),
        (ResponseStyle.TECHNICAL, "Technical")
    ]

    for style, name in styles:
        print(f"\n--- {name} Style ---")
        collector = MessageCollector()
        communicator = UAENaturalCommunicator(
            voice_callback=collector.voice_callback,
            text_callback=collector.text_callback,
            mode=CommunicationMode.NORMAL,
            response_style=style
        )

        await communicator.on_decision_start("control_center")

        decision = UnifiedDecision(
            element_id="control_center",
            chosen_position=(1400, 25),
            confidence=0.95,
            decision_source=DecisionSource.CONTEXT,
            context_weight=1.0,
            situation_weight=0.0,
            reasoning="Using historical data",
            timestamp=time.time(),
            metadata={'agreement': True}
        )
        await communicator.on_decision_made(decision)

        await asyncio.sleep(0.1)

    return True


async def test_factory_initialization():
    """Test communicator factory initialization"""
    print("\n" + "=" * 80)
    print("Test 5: Factory Initialization")
    print("=" * 80)

    collector = MessageCollector()

    # Test factory
    print("\n1️⃣  Initializing via factory...")
    communicator = initialize_communicator(
        voice_callback=collector.voice_callback,
        text_callback=collector.text_callback,
        mode=CommunicationMode.NORMAL,
        response_style=ResponseStyle.CASUAL
    )

    print(f"   Type: {type(communicator).__name__}")
    print(f"   Mode: {communicator.mode}")
    print(f"   Style: {communicator.response_style}")

    # Test it works
    print("\n2️⃣  Testing functionality...")
    await communicator.on_decision_start("wifi")

    print(f"\n📝 Messages collected: {len(collector.messages)}")

    return True


async def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("UAE Natural Communication Test Suite")
    print("=" * 80)

    tests = [
        ("Decision Flow", test_decision_flow),
        ("Device Connection Flow", test_device_connection_flow),
        ("Position Change Detection", test_position_change_detection),
        ("Different Styles", test_different_styles),
        ("Factory Initialization", test_factory_initialization)
    ]

    results = []

    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, "✅ PASS" if result else "❌ FAIL"))
        except Exception as e:
            logger.error(f"Test failed: {e}", exc_info=True)
            results.append((name, f"❌ ERROR: {e}"))

    # Summary
    print("\n" + "=" * 80)
    print("Test Results Summary")
    print("=" * 80)
    for name, result in results:
        print(f"{name:.<50} {result}")

    passed = sum(1 for _, r in results if "✅" in r)
    total = len(results)

    print(f"\n{passed}/{total} tests passed")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
