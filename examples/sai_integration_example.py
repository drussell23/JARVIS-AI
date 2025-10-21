#!/usr/bin/env python3
"""
SAI Integration Example for JARVIS
===================================

Complete example showing how to integrate Situational Awareness Intelligence
with JARVIS automation workflows.

This demonstrates:
1. Initializing SAI with JARVIS vision analyzer
2. Using SAI-enhanced Control Center clicker
3. Monitoring environment changes
4. Handling UI drift automatically
5. Collecting metrics and monitoring health

Author: Derek J. Russell
Date: October 2025
"""

import asyncio
import logging
from pathlib import Path
import sys

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from vision.situational_awareness import (
    get_sai_engine,
    ChangeEvent,
    ChangeType,
    UIElementDescriptor,
    ElementType
)
from display.sai_enhanced_control_center_clicker import get_sai_clicker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def example_basic_monitoring():
    """Example 1: Basic SAI monitoring"""
    print("\n" + "=" * 80)
    print("Example 1: Basic SAI Monitoring")
    print("=" * 80)

    # Initialize SAI (without vision analyzer for this example)
    engine = get_sai_engine(monitoring_interval=5.0)

    # Register change callback
    def on_change(change: ChangeEvent):
        logger.info(f"🔔 Environment change: {change.change_type.value}")
        if change.element_id:
            logger.info(f"   Element: {change.element_id}")
        logger.info(f"   Old: {change.old_value} → New: {change.new_value}")

    engine.register_change_callback(on_change)

    # Start monitoring
    logger.info("Starting SAI monitoring...")
    await engine.start_monitoring()

    # Monitor for 30 seconds
    logger.info("👀 Monitoring environment for 30 seconds...")
    logger.info("   (Try moving menu bar icons or changing displays)")
    await asyncio.sleep(30)

    # Stop and show metrics
    await engine.stop_monitoring()

    metrics = engine.get_metrics()
    logger.info(f"\n📊 Monitoring Results:")
    logger.info(f"   Changes detected: {metrics['changes']['total_detected']}")
    logger.info(f"   Cache hit rate: {metrics['cache']['hit_rate']:.1%}")
    logger.info(f"   Tracked elements: {len(metrics['tracked_elements'])}")


async def example_sai_enhanced_clicking():
    """Example 2: SAI-Enhanced Control Center clicking"""
    print("\n" + "=" * 80)
    print("Example 2: SAI-Enhanced Control Center Clicking")
    print("=" * 80)

    # Use SAI-enhanced clicker with context manager
    async with get_sai_clicker(
        enable_sai=True,
        sai_monitoring_interval=5.0
    ) as clicker:

        logger.info("✅ SAI-enhanced clicker initialized")
        logger.info("🔍 SAI monitoring active in background")

        # Attempt to click Control Center
        logger.info("\n🎯 Clicking Control Center...")
        result = await clicker.click("control_center")

        if result.success:
            logger.info(f"✅ Success!")
            logger.info(f"   Method: {result.method_used}")
            logger.info(f"   Coordinates: {result.coordinates}")
            logger.info(f"   Verification: {'✅' if result.verification_passed else '❌'}")
            logger.info(f"   Duration: {result.duration:.2f}s")
        else:
            logger.error(f"❌ Failed: {result.error}")

        # Show SAI metrics
        metrics = clicker.get_metrics()
        sai_metrics = metrics.get('sai', {})

        logger.info(f"\n📊 SAI Metrics:")
        logger.info(f"   Environment changes: {sai_metrics.get('environment_changes_detected', 0)}")
        logger.info(f"   Automatic revalidations: {sai_metrics.get('automatic_revalidations', 0)}")
        logger.info(f"   Cache invalidations: {sai_metrics.get('proactive_cache_invalidations', 0)}")
        logger.info(f"   SAI-assisted detections: {sai_metrics.get('sai_assisted_detections', 0)}")

        # Wait a bit for monitoring
        logger.info("\n⏳ Monitoring for additional 20 seconds...")
        logger.info("   (SAI watches for UI changes in background)")
        await asyncio.sleep(20)

    logger.info("✅ SAI monitoring automatically stopped")


async def example_custom_element_tracking():
    """Example 3: Track custom UI elements"""
    print("\n" + "=" * 80)
    print("Example 3: Custom Element Tracking")
    print("=" * 80)

    engine = get_sai_engine()

    # Register custom element (example: Slack icon in menu bar)
    custom_element = UIElementDescriptor(
        element_id="slack_menubar",
        element_type=ElementType.MENU_BAR_ICON,
        display_characteristics={
            'icon_description': 'Slack icon - colorful hash symbol (#)',
            'location': 'top menu bar, usually right side',
            'app_name': 'Slack',
            'color': 'multi-colored'
        },
        relative_position_rules={
            'anchor': 'top_right_corner',
            'search_region': 'menu_bar'
        }
    )

    engine.tracker.add_custom_element(custom_element)
    logger.info("✅ Registered custom element: slack_menubar")

    # Note: Actual detection would require vision analyzer
    logger.info("   (Vision detection requires Claude Vision analyzer)")
    logger.info("   Position detection: await engine.get_element_position('slack_menubar')")


async def example_automated_workflow():
    """Example 4: Automated workflow with SAI"""
    print("\n" + "=" * 80)
    print("Example 4: Automated Workflow with SAI Auto-Recovery")
    print("=" * 80)

    async with get_sai_clicker(enable_sai=True) as clicker:
        logger.info("🤖 Starting automated workflow...")

        # Workflow: Connect to AirPlay device
        logger.info("\n📺 Step 1: Connecting to 'Living Room TV'...")

        result = await clicker.connect_to_device("Living Room TV")

        if result['success']:
            logger.info(f"✅ Connected successfully in {result['duration']:.2f}s")

            # Show step details
            steps = result.get('steps', {})
            for step_name, step_result in steps.items():
                logger.info(f"   {step_name}: {step_result.get('success', False)}")

        else:
            logger.error(f"❌ Failed: {result['message']}")
            logger.error(f"   Failed at step: {result.get('step_failed', 'unknown')}")

        # SAI automatically handled any UI changes during workflow
        logger.info("\n✨ SAI ensured coordinates stayed valid throughout workflow")


async def example_health_monitoring():
    """Example 5: Monitor SAI health and performance"""
    print("\n" + "=" * 80)
    print("Example 5: SAI Health Monitoring")
    print("=" * 80)

    engine = get_sai_engine(monitoring_interval=5.0)
    await engine.start_monitoring()

    logger.info("🏥 Monitoring SAI health for 30 seconds...")

    # Monitor health every 10 seconds
    for i in range(3):
        await asyncio.sleep(10)

        metrics = engine.get_metrics()

        logger.info(f"\n📊 Health Check #{i+1}:")
        logger.info(f"   Status: {'🟢 Active' if metrics['monitoring']['active'] else '🔴 Inactive'}")
        logger.info(f"   Current env hash: {metrics['monitoring']['current_hash']}")

        cache = metrics['cache']
        logger.info(f"   Cache size: {cache['cache_size']}")
        logger.info(f"   Cache hit rate: {cache['hit_rate']:.1%}")
        logger.info(f"   Cache hits: {cache['hits']}")
        logger.info(f"   Cache misses: {cache['misses']}")

        changes = metrics['changes']
        logger.info(f"   Total changes detected: {changes['total_detected']}")

        # Performance check
        if cache['hit_rate'] < 0.7:
            logger.warning("⚠️  Cache hit rate below 70% - investigate environment stability")

        if changes['total_detected'] > 10:
            logger.warning("⚠️  Many environment changes - UI may be unstable")

    await engine.stop_monitoring()
    logger.info("\n✅ Health monitoring complete")


async def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("SAI Integration Examples for JARVIS")
    print("=" * 80)
    print("\nThis will run 5 examples demonstrating SAI capabilities:")
    print("1. Basic monitoring")
    print("2. SAI-enhanced clicking")
    print("3. Custom element tracking")
    print("4. Automated workflow")
    print("5. Health monitoring")
    print("\nPress Ctrl+C to skip to next example\n")

    examples = [
        example_basic_monitoring,
        example_sai_enhanced_clicking,
        example_custom_element_tracking,
        example_automated_workflow,
        example_health_monitoring
    ]

    for i, example in enumerate(examples, 1):
        try:
            await example()
        except KeyboardInterrupt:
            logger.info(f"\n⏭️  Skipping to next example...")
            continue
        except Exception as e:
            logger.error(f"❌ Example {i} failed: {e}", exc_info=True)
            continue

        if i < len(examples):
            logger.info("\n⏸️  Press Enter to continue to next example (or Ctrl+C to skip)...")
            try:
                await asyncio.sleep(3)
            except KeyboardInterrupt:
                logger.info("⏭️  Skipping...")

    print("\n" + "=" * 80)
    print("✅ All examples complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Read documentation: backend/vision/situational_awareness/README.md")
    print("2. Run tests: pytest backend/vision/situational_awareness/tests/ -v")
    print("3. Integrate SAI into your JARVIS workflows")
    print("=" * 80)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n\n⏹️  Examples stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
