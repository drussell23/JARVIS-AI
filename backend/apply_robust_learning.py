#!/usr/bin/env python3
"""
Apply robust continuous learning patch to the system
Run this before starting the backend to enable robust learning
"""

import os
import sys
import logging

# Add backend to path
backend_path = os.path.dirname(os.path.abspath(__file__))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def apply_robust_learning_patches():
    """Apply all patches for robust continuous learning"""
    
    # 1. Import and apply the integration patch
    try:
        from vision.integrate_robust_learning import apply_robust_learning_patch
        
        if apply_robust_learning_patch():
            logger.info("✓ Applied robust continuous learning patch")
        else:
            logger.error("✗ Failed to apply robust learning patch")
            return False
            
    except Exception as e:
        logger.error(f"✗ Error applying robust learning patch: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 2. Patch vision_system_v2 imports
    try:
        import vision.vision_system_v2 as vision_v2
        
        # Replace the import
        from vision.integrate_robust_learning import get_advanced_continuous_learning
        vision_v2.get_advanced_continuous_learning = get_advanced_continuous_learning
        
        logger.info("✓ Patched vision_system_v2 imports")
        
    except Exception as e:
        logger.error(f"✗ Error patching vision_system_v2: {e}")
        return False
    
    # 3. Set environment variable to enable robust learning
    os.environ.pop('DISABLE_CONTINUOUS_LEARNING', None)
    logger.info("✓ Enabled continuous learning")
    
    return True


def verify_robust_learning():
    """Verify that robust learning is properly configured"""
    try:
        # Test import
        from vision.integrate_robust_learning import ROBUST_AVAILABLE
        
        if ROBUST_AVAILABLE:
            logger.info("✓ Robust continuous learning is available")
            
            # Test configuration
            from vision.robust_continuous_learning import LearningConfig
            config = LearningConfig()
            
            logger.info(f"  - Max CPU: {config.max_cpu_percent}%")
            logger.info(f"  - Max Memory: {config.max_memory_percent}%")
            logger.info(f"  - Adaptive scheduling: {config.enable_adaptive_scheduling}")
            logger.info(f"  - Load threshold: {config.load_factor_threshold}")
            
            return True
        else:
            logger.error("✗ Robust continuous learning not available")
            return False
            
    except Exception as e:
        logger.error(f"✗ Error verifying robust learning: {e}")
        return False


def main():
    """Main function to apply patches"""
    logger.info("Starting robust continuous learning setup...")
    
    # Apply patches
    if not apply_robust_learning_patches():
        logger.error("Failed to apply patches")
        return 1
    
    # Verify
    if not verify_robust_learning():
        logger.error("Failed to verify robust learning")
        return 1
    
    logger.info("✓ Robust continuous learning setup complete!")
    logger.info("You can now start the backend with continuous learning enabled")
    logger.info("The system will automatically manage resources and throttle as needed")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())