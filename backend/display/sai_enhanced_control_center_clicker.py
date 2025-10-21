#!/usr/bin/env python3
"""
SAI-Enhanced Adaptive Control Center Clicker
=============================================

Integrates Situational Awareness Intelligence (SAI) with the Adaptive Control Center Clicker
for ultimate reliability and self-healing capabilities.

Features:
- Real-time UI layout awareness
- Automatic coordinate revalidation
- Environment change detection
- Multi-display adaptive clicking
- Proactive cache invalidation
- Zero manual intervention

Architecture:
    SAI Engine monitors environment → Detects changes → Invalidates cache → Re-detects coordinates

Author: Derek J. Russell
Date: October 2025
Version: 2.0.0
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# Import original adaptive clicker
from backend.display.adaptive_control_center_clicker import (
    AdaptiveControlCenterClicker,
    ClickResult,
    DetectionResult
)

# Import SAI
from backend.vision.situational_awareness import (
    get_sai_engine,
    SituationalAwarenessEngine,
    ChangeEvent,
    ChangeType,
    UIElementDescriptor,
    ElementType
)

logger = logging.getLogger(__name__)


class SAIEnhancedControlCenterClicker(AdaptiveControlCenterClicker):
    """
    Enhanced Control Center clicker with full situational awareness

    Inherits all capabilities from AdaptiveControlCenterClicker and adds:
    - Real-time environment monitoring
    - Automatic cache invalidation on UI changes
    - Proactive coordinate revalidation
    - Multi-monitor awareness
    - Continuous adaptation
    """

    def __init__(
        self,
        vision_analyzer=None,
        cache_ttl: int = 86400,
        enable_verification: bool = True,
        enable_sai: bool = True,
        sai_monitoring_interval: float = 10.0
    ):
        """
        Initialize SAI-enhanced clicker

        Args:
            vision_analyzer: Claude Vision analyzer
            cache_ttl: Cache TTL in seconds
            enable_verification: Enable click verification
            enable_sai: Enable SAI monitoring
            sai_monitoring_interval: SAI scan interval
        """
        # Initialize parent
        super().__init__(
            vision_analyzer=vision_analyzer,
            cache_ttl=cache_ttl,
            enable_verification=enable_verification
        )

        # SAI integration
        self.enable_sai = enable_sai
        self.sai_engine: Optional[SituationalAwarenessEngine] = None
        self.sai_monitoring_interval = sai_monitoring_interval

        # SAI metrics
        self.sai_metrics = {
            'environment_changes_detected': 0,
            'automatic_revalidations': 0,
            'proactive_cache_invalidations': 0,
            'sai_assisted_detections': 0
        }

        if enable_sai:
            self._initialize_sai()

        logger.info(
            f"[SAI-CLICKER] SAI-Enhanced Control Center Clicker initialized "
            f"(SAI={'enabled' if enable_sai else 'disabled'})"
        )

    def _initialize_sai(self):
        """Initialize SAI engine"""
        try:
            self.sai_engine = get_sai_engine(
                vision_analyzer=self.vision_analyzer if hasattr(self, 'vision_analyzer') else None,
                monitoring_interval=self.sai_monitoring_interval,
                enable_auto_revalidation=True
            )

            # Register change callback
            self.sai_engine.register_change_callback(self._on_environment_change)

            # Register tracked elements in SAI
            self._register_elements_with_sai()

            logger.info("[SAI-CLICKER] ✅ SAI engine initialized")

        except Exception as e:
            logger.error(f"[SAI-CLICKER] Failed to initialize SAI: {e}")
            self.enable_sai = False

    def _register_elements_with_sai(self):
        """Register elements we track with SAI"""
        # Control Center
        self.sai_engine.tracker.add_custom_element(UIElementDescriptor(
            element_id="control_center_click_target",
            element_type=ElementType.MENU_BAR_ICON,
            display_characteristics={
                'icon_description': 'Control Center icon - two toggle switches stacked vertically',
                'location': 'top-right menu bar',
                'typical_position': 'near battery and WiFi icons'
            }
        ))

        # Screen Mirroring (submenu)
        self.sai_engine.tracker.add_custom_element(UIElementDescriptor(
            element_id="screen_mirroring_menu_item",
            element_type=ElementType.MENU_ITEM,
            display_characteristics={
                'text_label': 'Screen Mirroring',
                'alternate_labels': ['Display'],
                'location': 'Control Center dropdown menu'
            }
        ))

        logger.info("[SAI-CLICKER] Registered elements with SAI tracker")

    async def start_sai_monitoring(self):
        """Start SAI environmental monitoring"""
        if not self.enable_sai or not self.sai_engine:
            logger.warning("[SAI-CLICKER] SAI not enabled")
            return

        await self.sai_engine.start_monitoring()
        logger.info("[SAI-CLICKER] ✅ SAI monitoring started")

    async def stop_sai_monitoring(self):
        """Stop SAI monitoring"""
        if self.sai_engine and self.sai_engine.is_monitoring:
            await self.sai_engine.stop_monitoring()
            logger.info("[SAI-CLICKER] SAI monitoring stopped")

    async def _on_environment_change(self, change: ChangeEvent):
        """
        Callback when SAI detects environmental change

        Args:
            change: Change event from SAI
        """
        self.sai_metrics['environment_changes_detected'] += 1

        logger.info(
            f"[SAI-CLICKER] 🔔 Environment change: {change.change_type.value} "
            f"(element={change.element_id})"
        )

        # Handle different change types
        if change.change_type == ChangeType.POSITION_CHANGED:
            # Specific element moved
            if change.element_id in ['control_center', 'screen_mirroring']:
                # Invalidate our cache
                self.cache.invalidate(change.element_id)
                self.sai_metrics['proactive_cache_invalidations'] += 1

                logger.info(
                    f"[SAI-CLICKER] 🔄 Invalidated cache for {change.element_id} "
                    f"(moved from {change.old_value} to {change.new_value})"
                )

        elif change.change_type in [
            ChangeType.DISPLAY_CHANGED,
            ChangeType.RESOLUTION_CHANGED,
            ChangeType.SYSTEM_UPDATE
        ]:
            # Major environment change - clear all caches
            self.cache.clear()
            self.sai_metrics['proactive_cache_invalidations'] += 1

            logger.warning(
                f"[SAI-CLICKER] ⚠️  Major environment change detected "
                f"({change.change_type.value}), cleared all caches"
            )

            # Trigger automatic revalidation
            asyncio.create_task(self._revalidate_critical_elements())

    async def _revalidate_critical_elements(self):
        """Revalidate critical UI elements after major change"""
        logger.info("[SAI-CLICKER] 🔄 Starting automatic revalidation of critical elements...")

        self.sai_metrics['automatic_revalidations'] += 1

        critical_elements = ['control_center', 'screen_mirroring']

        for element_id in critical_elements:
            try:
                # Use SAI to detect new position
                position = await self.sai_engine.get_element_position(
                    element_id,
                    use_cache=False,
                    force_detect=True
                )

                if position:
                    logger.info(
                        f"[SAI-CLICKER] ✅ Revalidated {element_id}: {position.coordinates}"
                    )
                else:
                    logger.warning(f"[SAI-CLICKER] ⚠️  Could not revalidate {element_id}")

            except Exception as e:
                logger.error(f"[SAI-CLICKER] Error revalidating {element_id}: {e}")

        logger.info("[SAI-CLICKER] ✅ Revalidation complete")

    async def click(
        self,
        target: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ClickResult:
        """
        Enhanced click with SAI integration

        Args:
            target: Target to click
            context: Optional context

        Returns:
            ClickResult
        """
        # Check if SAI has a validated position
        if self.enable_sai and self.sai_engine:
            try:
                # Query SAI for element position
                sai_position = await self.sai_engine.get_element_position(
                    target,
                    use_cache=True,
                    force_detect=False
                )

                if sai_position:
                    logger.info(
                        f"[SAI-CLICKER] 🎯 Using SAI-validated position for {target}: "
                        f"{sai_position.coordinates}"
                    )

                    # Update our cache with SAI position
                    self.cache.set(
                        target,
                        sai_position.coordinates,
                        sai_position.confidence,
                        f"sai_{sai_position.detection_method}"
                    )

                    self.sai_metrics['sai_assisted_detections'] += 1

            except Exception as e:
                logger.error(f"[SAI-CLICKER] Error querying SAI: {e}")

        # Use parent click logic
        return await super().click(target, context)

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics including SAI"""
        metrics = super().get_metrics()

        # Add SAI metrics
        metrics['sai'] = {
            'enabled': self.enable_sai,
            'monitoring_active': self.sai_engine.is_monitoring if self.sai_engine else False,
            **self.sai_metrics
        }

        # Add SAI engine metrics if available
        if self.sai_engine:
            metrics['sai']['engine_metrics'] = self.sai_engine.get_metrics()

        return metrics

    async def __aenter__(self):
        """Async context manager entry"""
        if self.enable_sai:
            await self.start_sai_monitoring()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.enable_sai:
            await self.stop_sai_monitoring()


# ============================================================================
# Singleton Instance
# ============================================================================

_sai_clicker: Optional[SAIEnhancedControlCenterClicker] = None


def get_sai_clicker(
    vision_analyzer=None,
    cache_ttl: int = 86400,
    enable_verification: bool = True,
    enable_sai: bool = True,
    sai_monitoring_interval: float = 10.0
) -> SAIEnhancedControlCenterClicker:
    """
    Get singleton SAI-enhanced clicker

    Args:
        vision_analyzer: Claude Vision analyzer
        cache_ttl: Cache TTL in seconds
        enable_verification: Enable verification
        enable_sai: Enable SAI
        sai_monitoring_interval: SAI monitoring interval

    Returns:
        SAIEnhancedControlCenterClicker instance
    """
    global _sai_clicker

    if _sai_clicker is None:
        _sai_clicker = SAIEnhancedControlCenterClicker(
            vision_analyzer=vision_analyzer,
            cache_ttl=cache_ttl,
            enable_verification=enable_verification,
            enable_sai=enable_sai,
            sai_monitoring_interval=sai_monitoring_interval
        )
    elif vision_analyzer is not None:
        _sai_clicker.set_vision_analyzer(vision_analyzer)

    return _sai_clicker


# ============================================================================
# Example Usage
# ============================================================================

async def main():
    """Demo SAI-enhanced clicker"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("\n" + "=" * 80)
    print("SAI-Enhanced Adaptive Control Center Clicker - Demo")
    print("=" * 80)

    # Create clicker with SAI
    async with get_sai_clicker(enable_sai=True, sai_monitoring_interval=5.0) as clicker:
        print("\n✅ SAI monitoring active - watching for environment changes...")
        print("🎯 Attempting to click Control Center...\n")

        # Click Control Center
        result = await clicker.click("control_center")

        print(f"\n📊 Result: {result.success}")
        print(f"   Method: {result.method_used}")
        print(f"   Duration: {result.duration:.2f}s")
        print(f"   Verification: {'✅' if result.verification_passed else '❌'}")

        # Wait a bit for SAI to monitor
        print("\n⏳ Monitoring environment for 30 seconds...")
        print("   (Try moving menu bar icons or changing displays)\n")
        await asyncio.sleep(30)

        # Show metrics
        print("\n📊 Final Metrics:")
        metrics = clicker.get_metrics()

        print(f"\nClicks: {metrics['total_attempts']} total, {metrics['successful_clicks']} successful")
        print(f"Cache: {metrics['cache_hit_rate']:.1%} hit rate")

        if 'sai' in metrics:
            sai = metrics['sai']
            print(f"\nSAI:")
            print(f"  Environment changes detected: {sai['environment_changes_detected']}")
            print(f"  Automatic revalidations: {sai['automatic_revalidations']}")
            print(f"  Proactive cache invalidations: {sai['proactive_cache_invalidations']}")
            print(f"  SAI-assisted detections: {sai['sai_assisted_detections']}")

    print("\n" + "=" * 80)
    print("✅ Demo complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
