#!/usr/bin/env python3
"""Test ML loader directly to debug issues"""

import asyncio
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_ml_loader():
    """Test the ML loader directly"""
    try:
        logger.info("Testing ML model loader...")
        
        # Import and test
        from ml_model_loader import initialize_models
        
        # Define progress callback
        async def progress_callback(loaded: int, total: int):
            percentage = (loaded / total * 100) if total > 0 else 0
            logger.info(f"Progress: {loaded}/{total} ({percentage:.1f}%)")
        
        # Run with timeout
        logger.info("Starting model initialization...")
        models = await asyncio.wait_for(
            initialize_models(progress_callback=progress_callback),
            timeout=60.0  # 60 second timeout
        )
        
        logger.info(f"Successfully loaded {len(models)} models!")
        
        # Print loaded models
        for name, model in models.items():
            logger.info(f"  - {name}")
            
    except asyncio.TimeoutError:
        logger.error("Model loading timed out after 60 seconds")
    except Exception as e:
        logger.error(f"Error during model loading: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(test_ml_loader())