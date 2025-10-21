#!/usr/bin/env python3
"""
UAE-Enhanced Adaptive Control Center Clicker
=============================================

Integrates Unified Awareness Engine (UAE) with Control Center automation
for ultimate intelligence and self-healing capabilities.

Features:
- Fusion of context and situational awareness
- Bidirectional learning from every execution
- Confidence-weighted decision making
- Priority-based monitoring
- Automatic self-correction
- Continuous adaptation

Architecture:
    UAE Engine → Context + Situation → Fused Decision → Execute → Learn

Author: Derek J. Russell
Date: October 2025
Version: 3.0.0
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# Import base clicker
from backend.display.adaptive_control_center_clicker import (
    AdaptiveControlCenterClicker,
    ClickResult,
    DetectionResult
)

# Import UAE
from backend.intelligence.unified_awareness_engine import (
    get_uae_engine,
    UnifiedAwarenessEngine,
    UnifiedDecision,
    ExecutionResult,
    DecisionSource
)

# Import SAI
from backend.vision.situational_awareness import get_sai_engine

# Import Natural Communication
from backend.intelligence.uae_natural_communication import (
    get_communicator,
    initialize_communicator,
    CommunicationMode
)

logger = logging.getLogger(__name__)


class UAEEnhancedControlCenterClicker(AdaptiveControlCenterClicker):
    """
    Ultimate Control Center clicker with full unified awareness

    Combines:
    - Context Intelligence (historical patterns)
    - Situational Awareness (real-time perception)
    - Adaptive Integration (confidence-weighted fusion)
    - Bidirectional Learning (continuous improvement)

    Result: Never fails, always adapts, continuously improves
    """

    def __init__(
        self,
        vision_analyzer=None,
        cache_ttl: int = 86400,
        enable_verification: bool = True,
        enable_uae: bool = True,
        uae_monitoring_interval: float = 10.0,
        enable_communication: bool = True,
        communication_mode: CommunicationMode = CommunicationMode.NORMAL,
        voice_callback: Optional[Callable] = None,
        text_callback: Optional[Callable] = None
    ):
        """
        Initialize UAE-enhanced clicker

        Args:
            vision_analyzer: Claude Vision analyzer
            cache_ttl: Cache TTL in seconds
            enable_verification: Enable click verification
            enable_uae: Enable UAE system
            uae_monitoring_interval: UAE monitoring interval
            enable_communication: Enable natural communication
            communication_mode: Communication verbosity mode
            voice_callback: Voice output callback
            text_callback: Text output callback
        """
        # Initialize parent
        super().__init__(
            vision_analyzer=vision_analyzer,
            cache_ttl=cache_ttl,
            enable_verification=enable_verification
        )

        # UAE integration
        self.enable_uae = enable_uae
        self.uae_engine: Optional[UnifiedAwarenessEngine] = None
        self.uae_monitoring_interval = uae_monitoring_interval

        # Communication
        self.enable_communication = enable_communication
        self.communicator = None
        if enable_communication:
            self.communicator = initialize_communicator(
                voice_callback=voice_callback,
                text_callback=text_callback,
                mode=communication_mode
            )

        # UAE metrics
        self.uae_metrics = {
            'uae_decisions': 0,
            'context_based_clicks': 0,
            'situation_based_clicks': 0,
            'fusion_clicks': 0,
            'learning_events': 0,
            'self_corrections': 0
        }

        if enable_uae:
            self._initialize_uae()

        logger.info(
            f"[UAE-CLICKER] UAE-Enhanced Control Center Clicker initialized "
            f"(UAE={'enabled' if enable_uae else 'disabled'}, "
            f"Communication={'enabled' if enable_communication else 'disabled'})"
        )

    def _initialize_uae(self):
        """Initialize UAE engine"""
        try:
            # Create SAI engine
            sai_engine = get_sai_engine(
                vision_analyzer=self.vision_analyzer if hasattr(self, 'vision_analyzer') else None,
                monitoring_interval=self.uae_monitoring_interval
            )

            # Create UAE engine
            self.uae_engine = get_uae_engine(
                sai_engine=sai_engine,
                vision_analyzer=self.vision_analyzer if hasattr(self, 'vision_analyzer') else None
            )

            # Register callbacks
            self.uae_engine.register_decision_callback(self._on_uae_decision)
            self.uae_engine.register_learning_callback(self._on_uae_learning)

            logger.info("[UAE-CLICKER] ✅ UAE engine initialized")

        except Exception as e:
            logger.error(f"[UAE-CLICKER] Failed to initialize UAE: {e}")
            self.enable_uae = False

    async def start_uae(self):
        """Start UAE system"""
        if not self.enable_uae or not self.uae_engine:
            logger.warning("[UAE-CLICKER] UAE not enabled")
            return

        await self.uae_engine.start()
        logger.info("[UAE-CLICKER] ✅ UAE system started")

    async def stop_uae(self):
        """Stop UAE system"""
        if self.uae_engine and self.uae_engine.is_active:
            await self.uae_engine.stop()
            logger.info("[UAE-CLICKER] UAE system stopped")

    async def _on_uae_decision(self, decision: UnifiedDecision):
        """
        Callback when UAE makes a decision

        Args:
            decision: UAE decision
        """
        self.uae_metrics['uae_decisions'] += 1

        # Track decision source
        if decision.decision_source == DecisionSource.CONTEXT:
            self.uae_metrics['context_based_clicks'] += 1
        elif decision.decision_source == DecisionSource.SITUATION:
            self.uae_metrics['situation_based_clicks'] += 1
        elif decision.decision_source == DecisionSource.FUSION:
            self.uae_metrics['fusion_clicks'] += 1

        logger.debug(
            f"[UAE-CLICKER] Decision: {decision.decision_source.value} → "
            f"{decision.chosen_position} (conf={decision.confidence:.2f})"
        )

    async def _on_uae_learning(self, result: ExecutionResult):
        """
        Callback when UAE learns from execution

        Args:
            result: Execution result
        """
        self.uae_metrics['learning_events'] += 1

        if result.success and result.decision.decision_source == DecisionSource.SITUATION:
            # Situation-based decision succeeded - this is a self-correction
            self.uae_metrics['self_corrections'] += 1
            logger.info(
                f"[UAE-CLICKER] 🔧 Self-correction: Used situational data "
                f"(context might have been stale)"
            )

    async def click(
        self,
        target: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ClickResult:
        """
        Enhanced click using UAE unified awareness

        Args:
            target: Target to click
            context: Optional context

        Returns:
            ClickResult
        """
        if not self.enable_uae or not self.uae_engine:
            # Fall back to parent implementation
            return await super().click(target, context)

        logger.info(f"[UAE-CLICKER] 🎯 UAE-aware click: {target}")

        # Communicate start
        if self.communicator:
            await self.communicator.on_decision_start(target)

        # Step 1: Get unified decision from UAE
        decision = await self.uae_engine.get_element_position(target)

        # Communicate decision
        if self.communicator:
            await self.communicator.on_decision_made(decision)

        # Step 2: Check decision confidence
        if decision.confidence < 0.3:
            logger.warning(
                f"[UAE-CLICKER] Low confidence ({decision.confidence:.2f}), "
                "falling back to traditional detection"
            )
            return await super().click(target, context)

        # Step 3: Execute using UAE
        result = await self._execute_with_uae(target, decision, context)

        return result

    async def _execute_with_uae(
        self,
        target: str,
        decision: UnifiedDecision,
        context: Optional[Dict[str, Any]]
    ) -> ClickResult:
        """
        Execute click with UAE awareness and learning

        Args:
            target: Target element
            decision: UAE decision
            context: Optional context

        Returns:
            ClickResult
        """
        start_time = asyncio.get_event_loop().time()

        try:
            # Communicate execution start
            if self.communicator:
                await self.communicator.on_execution_start(decision, 'click')

            # Perform click
            logger.info(
                f"[UAE-CLICKER] Clicking {target} at {decision.chosen_position} "
                f"(source={decision.decision_source.value}, conf={decision.confidence:.2f})"
            )

            # Use parent's click mechanism but with UAE coordinates
            result = await self._perform_click_at_coordinates(
                decision.chosen_position,
                target
            )

            # Verify if enabled
            verification_passed = False
            if self.enable_verification:
                verification_passed = await self._verify_click_result(target, result)

            # Create ClickResult
            click_result = ClickResult(
                success=result.get('success', False),
                target=target,
                coordinates=decision.chosen_position,
                method_used=f"uae_{decision.decision_source.value}",
                verification_passed=verification_passed,
                duration=asyncio.get_event_loop().time() - start_time,
                fallback_attempts=0,
                metadata={
                    'decision_source': decision.decision_source.value,
                    'context_weight': decision.context_weight,
                    'situation_weight': decision.situation_weight,
                    'reasoning': decision.reasoning,
                    **result
                }
            )

            # Learn from execution via UAE
            exec_result = ExecutionResult(
                decision=decision,
                success=click_result.success,
                execution_time=click_result.duration,
                verification_passed=verification_passed,
                metadata={'target': target}
            )

            await self.uae_engine._learn_from_execution(exec_result)

            # Communicate completion
            if self.communicator:
                await self.communicator.on_execution_complete(exec_result)
                if exec_result.success:
                    await self.communicator.on_learning_event(exec_result)

            return click_result

        except Exception as e:
            logger.error(f"[UAE-CLICKER] Execution failed: {e}", exc_info=True)

            # Communicate failure
            if self.communicator:
                exec_result = ExecutionResult(
                    decision=decision,
                    success=False,
                    execution_time=asyncio.get_event_loop().time() - start_time,
                    verification_passed=False,
                    error=str(e)
                )
                await self.communicator.on_execution_complete(exec_result)

            return ClickResult(
                success=False,
                target=target,
                coordinates=decision.chosen_position,
                method_used=f"uae_{decision.decision_source.value}_failed",
                verification_passed=False,
                duration=asyncio.get_event_loop().time() - start_time,
                fallback_attempts=0,
                metadata={},
                error=str(e)
            )

    async def _perform_click_at_coordinates(
        self,
        coordinates: Tuple[int, int],
        target: str
    ) -> Dict[str, Any]:
        """
        Perform actual click at coordinates

        Args:
            coordinates: (x, y) coordinates
            target: Target identifier

        Returns:
            Click result dict
        """
        try:
            import pyautogui

            x, y = coordinates

            # Move and click
            pyautogui.moveTo(x, y, duration=0.2)
            await asyncio.sleep(0.1)
            pyautogui.click(x, y)

            return {
                'success': True,
                'coordinates': coordinates,
                'target': target
            }

        except Exception as e:
            logger.error(f"Click execution failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def _verify_click_result(
        self,
        target: str,
        result: Dict[str, Any]
    ) -> bool:
        """
        Verify click result

        Args:
            target: Target element
            result: Click result

        Returns:
            Whether verification passed
        """
        # TODO: Implement actual verification logic
        # For now, assume success if click succeeded
        return result.get('success', False)

    async def connect_to_device(self, device_name: str) -> Dict[str, Any]:
        """
        Connect to AirPlay device using UAE intelligence

        Args:
            device_name: Device name

        Returns:
            Connection result
        """
        logger.info(f"[UAE-CLICKER] 🔗 Connecting to device: {device_name}")

        # Communicate connection start
        if self.communicator:
            await self.communicator.on_device_connection(
                device_name,
                step='start',
                step_result={'phase': 'start', 'success': True}
            )

        start_time = time.time()
        steps = {}

        try:
            # Step 1: Click Control Center
            logger.info("[UAE-CLICKER] Step 1: Opening Control Center...")

            if self.communicator:
                await self.communicator.on_device_connection(
                    device_name,
                    step='open_control_center',
                    step_result={'phase': 'start', 'success': True}
                )

            cc_result = await self.click("control_center")
            steps['open_control_center'] = {
                'success': cc_result.success,
                'method': cc_result.method_used,
                'confidence': cc_result.confidence
            }

            if not cc_result.success:
                if self.communicator:
                    await self.communicator.on_device_connection(
                        device_name,
                        step='open_control_center',
                        step_result={'phase': 'fail', 'success': False}
                    )
                return {
                    'success': False,
                    'message': 'Failed to open Control Center',
                    'step_failed': 'open_control_center',
                    'steps': steps,
                    'duration': time.time() - start_time
                }

            if self.communicator:
                await self.communicator.on_device_connection(
                    device_name,
                    step='open_control_center',
                    step_result={'phase': 'success', 'success': True}
                )

            # Wait for Control Center to open
            await asyncio.sleep(0.5)

            # Step 2: Click Screen Mirroring
            logger.info("[UAE-CLICKER] Step 2: Opening Screen Mirroring...")

            if self.communicator:
                await self.communicator.on_device_connection(
                    device_name,
                    step='open_screen_mirroring',
                    step_result={'phase': 'start', 'success': True}
                )

            sm_result = await self.click("screen_mirroring")
            steps['open_screen_mirroring'] = {
                'success': sm_result.success,
                'method': sm_result.method_used,
                'confidence': sm_result.confidence
            }

            if not sm_result.success:
                if self.communicator:
                    await self.communicator.on_device_connection(
                        device_name,
                        step='open_screen_mirroring',
                        step_result={'phase': 'fail', 'success': False}
                    )
                return {
                    'success': False,
                    'message': 'Failed to open Screen Mirroring',
                    'step_failed': 'open_screen_mirroring',
                    'steps': steps,
                    'duration': time.time() - start_time
                }

            if self.communicator:
                await self.communicator.on_device_connection(
                    device_name,
                    step='open_screen_mirroring',
                    step_result={'phase': 'success', 'success': True}
                )

            # Wait for menu
            await asyncio.sleep(0.5)

            # Step 3: Select device
            logger.info(f"[UAE-CLICKER] Step 3: Selecting device '{device_name}'...")

            if self.communicator:
                await self.communicator.on_device_connection(
                    device_name,
                    step='select_device',
                    step_result={'phase': 'start', 'success': True}
                )

            device_result = await self.click(device_name)
            steps['select_device'] = {
                'success': device_result.success,
                'method': device_result.method_used,
                'confidence': device_result.confidence
            }

            if not device_result.success:
                if self.communicator:
                    await self.communicator.on_device_connection(
                        device_name,
                        step='select_device',
                        step_result={'phase': 'fail', 'success': False}
                    )
                return {
                    'success': False,
                    'message': f'Failed to select device {device_name}',
                    'step_failed': 'select_device',
                    'steps': steps,
                    'duration': time.time() - start_time
                }

            if self.communicator:
                await self.communicator.on_device_connection(
                    device_name,
                    step='select_device',
                    step_result={'phase': 'success', 'success': True}
                )

            # Success - communicate completion
            if self.communicator:
                await self.communicator.on_device_connection(
                    device_name,
                    step='complete',
                    step_result={'phase': 'success', 'success': True, 'duration': time.time() - start_time}
                )

            return {
                'success': True,
                'message': f'Connected to {device_name}',
                'device': device_name,
                'steps': steps,
                'duration': time.time() - start_time
            }

        except Exception as e:
            logger.error(f"[UAE-CLICKER] Connection failed: {e}", exc_info=True)

            if self.communicator:
                await self.communicator.on_device_connection(
                    device_name,
                    step='error',
                    step_result={'phase': 'fail', 'success': False, 'error': str(e)}
                )

            return {
                'success': False,
                'message': f'Error: {str(e)}',
                'steps': steps,
                'duration': time.time() - start_time
            }

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics including UAE"""
        metrics = super().get_metrics()

        # Add UAE metrics
        metrics['uae'] = {
            'enabled': self.enable_uae,
            'active': self.uae_engine.is_active if self.uae_engine else False,
            **self.uae_metrics
        }

        # Add UAE engine metrics if available
        if self.uae_engine:
            metrics['uae']['engine_metrics'] = self.uae_engine.get_comprehensive_metrics()

        return metrics

    async def __aenter__(self):
        """Async context manager entry"""
        if self.enable_uae:
            await self.start_uae()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.enable_uae:
            await self.stop_uae()


# ============================================================================
# Singleton Instance
# ============================================================================

_uae_clicker: Optional[UAEEnhancedControlCenterClicker] = None


def get_uae_clicker(
    vision_analyzer=None,
    cache_ttl: int = 86400,
    enable_verification: bool = True,
    enable_uae: bool = True,
    uae_monitoring_interval: float = 10.0
) -> UAEEnhancedControlCenterClicker:
    """
    Get singleton UAE-enhanced clicker

    Args:
        vision_analyzer: Claude Vision analyzer
        cache_ttl: Cache TTL in seconds
        enable_verification: Enable verification
        enable_uae: Enable UAE
        uae_monitoring_interval: UAE monitoring interval

    Returns:
        UAEEnhancedControlCenterClicker instance
    """
    global _uae_clicker

    if _uae_clicker is None:
        _uae_clicker = UAEEnhancedControlCenterClicker(
            vision_analyzer=vision_analyzer,
            cache_ttl=cache_ttl,
            enable_verification=enable_verification,
            enable_uae=enable_uae,
            uae_monitoring_interval=uae_monitoring_interval
        )
    elif vision_analyzer is not None:
        _uae_clicker.set_vision_analyzer(vision_analyzer)

    return _uae_clicker


# ============================================================================
# Example Usage
# ============================================================================

async def main():
    """Demo UAE-enhanced clicker"""
    import time
    from dataclasses import asdict

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("\n" + "=" * 80)
    print("UAE-Enhanced Adaptive Control Center Clicker - Demo")
    print("=" * 80)

    # Create clicker with UAE
    async with get_uae_clicker(enable_uae=True, uae_monitoring_interval=5.0) as clicker:
        print("\n✅ UAE monitoring active")
        print("   Context Intelligence: Learning patterns")
        print("   Situational Awareness: Monitoring environment")
        print("   Integration Layer: Fusing decisions\n")

        # Attempt click
        print("🎯 Attempting to click Control Center...\n")
        result = await clicker.click("control_center")

        print(f"📊 Result:")
        print(f"   Success: {'✅' if result.success else '❌'}")
        print(f"   Method: {result.method_used}")
        print(f"   Confidence: {result.confidence:.2%}")
        print(f"   Coordinates: {result.coordinates}")
        print(f"   Duration: {result.duration:.2f}s")
        print(f"   Verification: {'✅' if result.verification_passed else '❌'}")

        # Show UAE intelligence breakdown
        if 'decision_source' in result.metadata:
            print(f"\n🧠 UAE Intelligence:")
            print(f"   Decision source: {result.metadata['decision_source']}")
            print(f"   Context weight: {result.metadata.get('context_weight', 0):.2%}")
            print(f"   Situation weight: {result.metadata.get('situation_weight', 0):.2%}")
            print(f"   Reasoning: {result.metadata.get('reasoning', 'N/A')}")

        # Monitor for a bit
        print("\n⏳ Monitoring for 20 seconds...")
        print("   (UAE continuously learns and adapts)\n")
        await asyncio.sleep(20)

        # Show comprehensive metrics
        print("📈 Comprehensive Metrics:")
        metrics = clicker.get_metrics()

        print(f"\nClicks:")
        print(f"   Total: {metrics['total_attempts']}")
        print(f"   Successful: {metrics['successful_clicks']}")
        print(f"   Cache hit rate: {metrics['cache_hit_rate']:.1%}")

        if 'uae' in metrics:
            uae = metrics['uae']
            print(f"\nUAE:")
            print(f"   Decisions: {uae['uae_decisions']}")
            print(f"   Context-based: {uae['context_based_clicks']}")
            print(f"   Situation-based: {uae['situation_based_clicks']}")
            print(f"   Fusion: {uae['fusion_clicks']}")
            print(f"   Learning events: {uae['learning_events']}")
            print(f"   Self-corrections: {uae['self_corrections']}")

    print("\n" + "=" * 80)
    print("✅ Demo complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
