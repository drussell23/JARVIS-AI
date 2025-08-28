#!/usr/bin/env python3
"""Test imports one by one to find what's hanging"""

import time
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_import(module_name):
    """Test importing a module"""
    logger.info(f"Testing import: {module_name}")
    start = time.time()
    try:
        exec(f"import {module_name}")
        elapsed = time.time() - start
        logger.info(f"✅ {module_name} imported in {elapsed:.2f}s")
        return True
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"❌ {module_name} failed after {elapsed:.2f}s: {e}")
        return False

# Test imports from main.py
imports_to_test = [
    "os",
    "sys", 
    "tensorflow",
    "fastapi",
    "pydantic",
    "chatbots.claude_chatbot",
    "asyncio",
    "json",
    "logging",
    "dotenv",
    "memory.memory_manager",
    "memory.memory_api",
    "ml_model_loader",
    "api.model_status_api",
    "api.voice_api",
    "api.jarvis_voice_api",
]

logger.info("Starting import tests...")

for module in imports_to_test:
    test_import(module)
    
logger.info("Import tests complete!")