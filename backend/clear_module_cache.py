#!/usr/bin/env python3
"""
Module cache cleaner for JARVIS
Ensures fresh imports of all modules on startup
"""

import sys
import os
import shutil
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def clear_all_caches():
    """Clear all Python caches to ensure fresh imports"""
    
    # 1. Clear __pycache__ directories
    backend_dir = Path(__file__).parent
    pycache_dirs = list(backend_dir.rglob("__pycache__"))
    
    for pycache in pycache_dirs:
        try:
            shutil.rmtree(pycache)
            logger.info(f"Removed cache dir: {pycache}")
        except Exception as e:
            logger.debug(f"Could not remove {pycache}: {e}")
    
    # 2. Clear .pyc files
    pyc_files = list(backend_dir.rglob("*.pyc"))
    for pyc in pyc_files:
        try:
            os.remove(pyc)
            logger.info(f"Removed cache file: {pyc}")
        except Exception as e:
            logger.debug(f"Could not remove {pyc}: {e}")
    
    # 3. Clear sys.modules for our code
    modules_to_remove = []
    critical_modules = [
        'api.unified_command_processor',
        'api.vision_command_handler', 
        'vision.multi_space_capture_engine',
        'backend.api.unified_command_processor',
        'backend.api.vision_command_handler',
        'backend.vision.multi_space_capture_engine'
    ]
    
    # First remove critical modules
    for module_name in critical_modules:
        if module_name in sys.modules:
            del sys.modules[module_name]
            logger.info(f"Cleared critical module: {module_name}")
            modules_to_remove.append(module_name)
    
    # Then remove all our modules
    for module_name in list(sys.modules.keys()):
        if any(pattern in module_name for pattern in [
            'unified', 'vision', 'command', 'multi_space',
            'api.', 'vision.', 'system_control.',
            'backend.api', 'backend.vision'
        ]):
            if module_name not in modules_to_remove:
                modules_to_remove.append(module_name)
    
    for module_name in modules_to_remove:
        if module_name in sys.modules:
            del sys.modules[module_name]
            logger.info(f"Cleared module: {module_name}")
    
    # 4. Set environment to prevent bytecode generation
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    
    # 5. Clear import cache
    if hasattr(sys, 'path_importer_cache'):
        sys.path_importer_cache.clear()
    
    # 6. Invalidate finder caches
    import importlib
    importlib.invalidate_caches()
    
    logger.info(f"✅ Cleared {len(modules_to_remove)} cached modules")
    logger.info("✅ All Python caches cleared - imports will be fresh")
    
    return len(modules_to_remove)

def verify_fresh_imports():
    """Verify that critical modules will be imported fresh"""
    
    # Test import of unified command processor
    if 'api.unified_command_processor' in sys.modules:
        logger.warning("unified_command_processor still cached!")
        return False
        
    if 'api.vision_command_handler' in sys.modules:
        logger.warning("vision_command_handler still cached!")
        return False
    
    logger.info("✅ Verified: All critical modules will be imported fresh")
    return True

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cleared = clear_all_caches()
    verify_fresh_imports()
    print(f"\n✅ Module cache cleared: {cleared} modules removed")
    print("   JARVIS will now use fresh code with all fixes!")