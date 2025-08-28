"""
Test script for M1 Memory Manager
Validates memory management functionality
"""

import asyncio
import psutil
import numpy as np
from memory.memory_manager import M1MemoryManager, ComponentPriority
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class MockComponent:
    """Mock component that uses specified amount of memory"""

    def __init__(self, name: str, size_mb: int):
        self.name = name
        self.size_mb = size_mb
        # Allocate memory
        self.data = np.zeros((size_mb * 1024 * 1024 // 8,), dtype=np.float64)
        logger.info(f"Created {name} with {size_mb}MB")

    async def cleanup(self):
        """Cleanup method called when unloading"""
        logger.info(f"Cleaning up {self.name}")
        self.data = None

async def test_basic_functionality():
    """Test basic memory manager functionality"""
    logger.info("=== Testing Basic Functionality ===")

    manager = M1MemoryManager()

    # Register components
    manager.register_component("test_small", ComponentPriority.HIGH, 100)
    manager.register_component("test_medium", ComponentPriority.MEDIUM, 500)
    manager.register_component("test_large", ComponentPriority.LOW, 1000)

    # Start monitoring
    await manager.start_monitoring()

    # Get initial status
    snapshot = await manager.get_memory_snapshot()
    logger.info(
        f"Initial memory state: {snapshot.state.value} ({snapshot.percent * 100:.1f}% used)"
    )

    # Test loading components
    small_component = MockComponent("test_small", 50)
    can_load, reason = await manager.can_load_component("test_small")
    logger.info(f"Can load small component: {can_load} - {reason}")

    if can_load:
        success = await manager.load_component("test_small", small_component)
        logger.info(f"Small component loaded: {success}")

    # Get status after loading
    await asyncio.sleep(2)
    snapshot = await manager.get_memory_snapshot()
    logger.info(
        f"After loading: {snapshot.state.value} ({snapshot.percent * 100:.1f}% used)"
    )

    # Get report
    report = await manager.get_memory_report()
    logger.info(f"Memory Report:\n{json.dumps(report, indent=2, default=str)}")

    # Cleanup
    await manager.cleanup()

async def test_memory_pressure():
    """Test behavior under memory pressure"""
    logger.info("\n=== Testing Memory Pressure ===")

    manager = M1MemoryManager()

    # Register components
    components = [
        ("critical_component", ComponentPriority.CRITICAL, 100),
        ("high_component", ComponentPriority.HIGH, 500),
        ("medium_component", ComponentPriority.MEDIUM, 1000),
        ("low_component", ComponentPriority.LOW, 500),
    ]

    for name, priority, size in components:
        manager.register_component(name, priority, size)

    # Start monitoring
    await manager.start_monitoring()

    # Load components progressively
    loaded_components = {}
    for name, priority, estimated_size in components:
        can_load, reason = await manager.can_load_component(name)
        logger.info(f"\nTrying to load {name}: {can_load} - {reason}")

        if can_load:
            # Create smaller mock to avoid actually running out of memory
            actual_size = min(estimated_size // 10, 50)  # Use 1/10th size for testing
            component = MockComponent(name, actual_size)
            success = await manager.load_component(name, component)
            if success:
                loaded_components[name] = component

        # Check memory state
        snapshot = await manager.get_memory_snapshot()
        logger.info(
            f"Memory state after {name}: {snapshot.state.value} ({snapshot.percent * 100:.1f}% used)"
        )

        # Wait a bit
        await asyncio.sleep(1)

    # Simulate memory pressure by forcing optimization
    logger.info("\n--- Simulating memory pressure ---")
    await manager._optimize_memory()

    # Check what got unloaded
    report = await manager.get_memory_report()
    logger.info(
        f"After optimization:\n{json.dumps(report['components'], indent=2, default=str)}"
    )

    # Cleanup
    await manager.cleanup()

async def test_emergency_scenarios():
    """Test emergency memory management"""
    logger.info("\n=== Testing Emergency Scenarios ===")

    manager = M1MemoryManager()

    # Register components
    manager.register_component("critical_service", ComponentPriority.CRITICAL, 100)
    manager.register_component("important_service", ComponentPriority.HIGH, 200)
    manager.register_component("optional_service", ComponentPriority.MEDIUM, 300)

    # Start monitoring
    await manager.start_monitoring()

    # Load all components
    components = {
        "critical_service": MockComponent("critical_service", 50),
        "important_service": MockComponent("important_service", 100),
        "optional_service": MockComponent("optional_service", 150),
    }

    for name, component in components.items():
        await manager.load_component(name, component)

    # Get initial state
    snapshot = await manager.get_memory_snapshot()
    logger.info(
        f"All components loaded: {snapshot.state.value} ({snapshot.percent * 100:.1f}% used)"
    )

    # Trigger emergency cleanup
    logger.info("\n--- Triggering emergency cleanup ---")
    await manager._emergency_shutdown()

    # Check results
    report = await manager.get_memory_report()
    remaining = [c["name"] for c in report["components"] if c["is_loaded"]]
    logger.info(f"Components remaining after emergency: {remaining}")

    # Verify critical component survived
    critical_survived = "critical_service" in remaining
    logger.info(f"Critical service survived: {critical_survived}")

    # Cleanup
    await manager.cleanup()

async def test_memory_prediction():
    """Test memory prediction accuracy"""
    logger.info("\n=== Testing Memory Prediction ===")

    manager = M1MemoryManager()

    # Register a component
    manager.register_component("test_predictor", ComponentPriority.MEDIUM, 200)

    # Load and unload multiple times to build prediction history
    sizes = [50, 75, 100, 125, 150]

    for i, size in enumerate(sizes):
        logger.info(f"\nIteration {i+1}: Loading with {size}MB")

        component = MockComponent("test_predictor", size)
        success = await manager.load_component("test_predictor", component)

        if success:
            # Let it measure
            await asyncio.sleep(1)

            # Get prediction for next load
            predicted = manager.predictor.predict_memory_need("test_predictor")
            logger.info(f"Predicted memory need for next load: {predicted:.1f}MB")

            # Unload
            await manager.unload_component("test_predictor")

    # Final prediction should be close to average + buffer
    final_prediction = manager.predictor.predict_memory_need("test_predictor")
    actual_average = np.mean(sizes)
    logger.info(f"\nFinal prediction: {final_prediction:.1f}MB")
    logger.info(f"Actual average: {actual_average:.1f}MB")
    logger.info(
        f"Prediction includes safety buffer: {final_prediction > actual_average}"
    )

    await manager.cleanup()

async def run_all_tests():
    """Run all memory manager tests"""
    print("Starting Memory Manager Tests...\n")

    # Show current system memory
    mem = psutil.virtual_memory()
    print(f"System Memory Status:")
    print(f"  Total: {mem.total / (1024**3):.1f} GB")
    print(f"  Available: {mem.available / (1024**3):.1f} GB")
    print(f"  Used: {mem.percent:.1f}%")
    print()

    try:
        await test_basic_functionality()
        await asyncio.sleep(2)

        await test_memory_pressure()
        await asyncio.sleep(2)

        await test_emergency_scenarios()
        await asyncio.sleep(2)

        await test_memory_prediction()

        print("\n✅ All tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_all_tests())

