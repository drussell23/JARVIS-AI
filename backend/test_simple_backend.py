#!/usr/bin/env python3
"""Simple test of backend startup without ML models"""

import asyncio
import logging
import time
import signal
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_backend():
    """Test backend startup"""
    logger.info("Starting backend test...")
    
    # Set environment to skip ML loading
    os.environ['SKIP_ML_LOADING'] = 'true'
    
    # Import FastAPI app
    try:
        from main import app
        logger.info("Successfully imported FastAPI app")
        
        # Check if app has startup event
        logger.info("FastAPI app imported successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to import main app: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = asyncio.run(test_backend())
    if success:
        logger.info("✅ Backend import successful!")
    else:
        logger.error("❌ Backend import failed!")