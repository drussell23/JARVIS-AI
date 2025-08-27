#!/usr/bin/env python3
"""
Fix for the continuous learning asyncio issues
"""

import asyncio
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


def create_async_task(coro):
    """Safely create an async task whether we're in an event loop or not"""
    try:
        # Try to get the current event loop
        loop = asyncio.get_running_loop()
        # We're in an event loop, create a task
        return asyncio.create_task(coro)
    except RuntimeError:
        # No event loop running, use asyncio.run()
        return asyncio.run(coro)


async def safe_async_call(coro):
    """Safely call an async function from any context"""
    try:
        return await coro
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            # We're already in an event loop, just await it
            return await coro
        raise


def patch_advanced_continuous_learning():
    """Patch the advanced continuous learning to fix asyncio issues"""
    try:
        import vision.advanced_continuous_learning as acl
        
        # Store original method
        original_start = acl.AdvancedContinuousLearning._start_continuous_learning
        
        def fixed_start_continuous_learning(self):
            """Fixed version that properly handles async in threads"""
            def learning_loop():
                # Create a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                while self.running:
                    try:
                        # Run the learning cycle in the thread's event loop
                        loop.run_until_complete(self._learning_cycle())
                        time.sleep(60)  # Use time.sleep, not asyncio.sleep
                    except Exception as e:
                        logger.error(f"Learning cycle error: {e}")
                        time.sleep(300)  # Wait 5 minutes on error
                
                loop.close()
            
            import threading
            self.learning_thread = threading.Thread(target=learning_loop, daemon=True)
            self.learning_thread.start()
        
        # Apply the patch
        acl.AdvancedContinuousLearning._start_continuous_learning = fixed_start_continuous_learning
        
        logger.info("Applied continuous learning asyncio fix")
        return True
        
    except Exception as e:
        logger.error(f"Failed to apply continuous learning fix: {e}")
        return False


# Also fix the other asyncio.run() calls
def patch_statistics_calls():
    """Fix the statistics calls that use asyncio.run()"""
    try:
        import vision.advanced_continuous_learning as acl
        
        # Fix get_statistics method
        original_get_stats = acl.AdvancedContinuousLearning.get_statistics
        
        def fixed_get_statistics(self) -> dict:
            """Fixed version that handles async properly"""
            stats = {
                'learning_enabled': self.enabled,
                'running': self.running,
                'learning_tasks': len(self.learning_queue),
                'meta_strategies': len(self.meta_learning.strategies),
                'federated_updates': len(self.federated_updates),
                'last_training': self.last_training_time.isoformat() if self.last_training_time else None,
                'training_stats': dict(self.training_stats)
            }
            
            # Handle async call properly
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, create a task
                task = loop.create_task(self.experience_replay.get_statistics())
                replay_stats = loop.run_until_complete(task)
            except RuntimeError:
                # Not in an async context, create new event loop
                replay_stats = asyncio.run(self.experience_replay.get_statistics())
            
            stats['experience_replay'] = replay_stats
            return stats
        
        acl.AdvancedContinuousLearning.get_statistics = fixed_get_statistics
        
        logger.info("Applied statistics asyncio fix")
        return True
        
    except Exception as e:
        logger.error(f"Failed to apply statistics fix: {e}")
        return False


def apply_all_fixes():
    """Apply all asyncio fixes"""
    fixes_applied = []
    
    if patch_advanced_continuous_learning():
        fixes_applied.append("continuous_learning")
    
    if patch_statistics_calls():
        fixes_applied.append("statistics")
    
    return fixes_applied


if __name__ == "__main__":
    # Test the fixes
    fixes = apply_all_fixes()
    print(f"Applied fixes: {fixes}")