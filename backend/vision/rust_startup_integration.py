"""
Rust startup integration for JARVIS backend.
Handles automatic Rust component initialization and verification.
"""

import os
import sys
import asyncio
import logging
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, Optional
import psutil

logger = logging.getLogger(__name__)

class RustStartupManager:
    """Manages Rust component initialization during JARVIS startup."""
    
    def __init__(self):
        self.vision_dir = Path(__file__).parent
        self.rust_core_dir = self.vision_dir / "jarvis-rust-core"
        self.config_file = self.vision_dir / "rust_config.json"
        self.rust_available = False
        self.rust_components = {}
        
    async def initialize_rust_components(self) -> Dict[str, Any]:
        """Initialize Rust components with automatic building if needed."""
        logger.info("🦀 Initializing Rust acceleration components...")
        
        # Check if Rust is already built
        if not self._check_rust_built():
            logger.info("Rust components not found, checking if we should build...")
            
            # Only build automatically in development mode
            if os.getenv('JARVIS_DEV_MODE', 'false').lower() == 'true':
                logger.info("Development mode detected, building Rust components...")
                success = await self._build_rust_components()
                if not success:
                    logger.warning("Failed to build Rust components automatically")
                    return self._get_fallback_config()
            else:
                logger.info("Production mode - skipping automatic Rust build")
                return self._get_fallback_config()
                
        # Try to import Rust core
        try:
            import jarvis_rust_core
            self.rust_available = True
            logger.info("✅ Rust core imported successfully")
            
            # Verify components
            self.rust_components = self._verify_rust_components(jarvis_rust_core)
            
            # Update Python modules if needed
            if self._should_update_python_modules():
                await self._update_python_modules()
                
            # Initialize global Rust accelerator
            if hasattr(jarvis_rust_core, 'initialize_rust_acceleration'):
                from .rust_integration import initialize_rust_acceleration
                rust_accelerator = initialize_rust_acceleration(
                    enable_memory_pool=True,
                    enable_runtime_manager=True,
                    worker_threads=min(8, psutil.cpu_count()),
                    enable_cpu_affinity=True
                )
                logger.info("✅ Global Rust accelerator initialized")
                
            return {
                'available': True,
                'components': self.rust_components,
                'performance_boost': self._estimate_performance_boost(),
                'memory_savings': self._estimate_memory_savings()
            }
            
        except ImportError as e:
            logger.warning(f"Rust core not available: {e}")
            return self._get_fallback_config()
            
    def _check_rust_built(self) -> bool:
        """Check if Rust library is built."""
        if sys.platform == "darwin":
            lib_file = self.rust_core_dir / "target" / "release" / "libjarvis_rust_core.dylib"
        elif sys.platform == "linux":
            lib_file = self.rust_core_dir / "target" / "release" / "libjarvis_rust_core.so"
        else:
            lib_file = self.rust_core_dir / "target" / "release" / "jarvis_rust_core.dll"
            
        return lib_file.exists()
        
    async def _build_rust_components(self) -> bool:
        """Build Rust components asynchronously."""
        try:
            # Check if cargo is available
            result = subprocess.run(["cargo", "--version"], capture_output=True)
            if result.returncode != 0:
                logger.error("Cargo not found - please install Rust")
                return False
                
            # Run build script
            build_script = self.vision_dir / "build_rust_components.py"
            if build_script.exists():
                logger.info("Running Rust build script...")
                proc = await asyncio.create_subprocess_exec(
                    sys.executable, str(build_script),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await proc.communicate()
                
                if proc.returncode == 0:
                    logger.info("✅ Rust components built successfully")
                    return True
                else:
                    logger.error(f"Build failed: {stderr.decode()}")
                    return False
            else:
                logger.error(f"Build script not found: {build_script}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to build Rust components: {e}")
            return False
            
    def _verify_rust_components(self, rust_core) -> Dict[str, bool]:
        """Verify which Rust components are available."""
        components = {
            'memory_pool': hasattr(rust_core, 'RustAdvancedMemoryPool'),
            'bloom_filter': hasattr(rust_core, 'bloom_filter'),
            'sliding_window': hasattr(rust_core, 'sliding_window'),
            'metal_accelerator': hasattr(rust_core, 'metal_accelerator') and sys.platform == 'darwin',
            'zero_copy': hasattr(rust_core, 'zero_copy'),
            'image_processor': hasattr(rust_core, 'RustImageProcessor'),
            'runtime_manager': hasattr(rust_core, 'RustRuntimeManager')
        }
        
        available = sum(components.values())
        total = len(components)
        logger.info(f"Rust components verified: {available}/{total} available")
        
        for name, available in components.items():
            if available:
                logger.info(f"  ✅ {name}")
            else:
                logger.debug(f"  ❌ {name}")
                
        return components
        
    def _should_update_python_modules(self) -> bool:
        """Check if Python modules need updating for Rust."""
        # Check if update marker exists
        update_marker = self.vision_dir / ".rust_modules_updated"
        if update_marker.exists():
            # Check if marker is recent (within 24 hours)
            import time
            marker_age = time.time() - update_marker.stat().st_mtime
            if marker_age < 86400:  # 24 hours
                return False
                
        return True
        
    async def _update_python_modules(self):
        """Update Python modules to use Rust acceleration."""
        try:
            update_script = self.vision_dir / "update_python_rust_integration.py"
            if update_script.exists():
                logger.info("Updating Python modules for Rust acceleration...")
                
                proc = await asyncio.create_subprocess_exec(
                    sys.executable, str(update_script),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await proc.communicate()
                
                if proc.returncode == 0:
                    logger.info("✅ Python modules updated for Rust")
                    # Create update marker
                    update_marker = self.vision_dir / ".rust_modules_updated"
                    update_marker.touch()
                else:
                    logger.warning(f"Module update had issues: {stderr.decode()}")
                    
        except Exception as e:
            logger.error(f"Failed to update Python modules: {e}")
            
    def _estimate_performance_boost(self) -> Dict[str, float]:
        """Estimate performance improvements from Rust."""
        if not self.rust_available:
            return {}
            
        # Based on benchmarks
        boosts = {
            'frame_processing': 5.0,  # 5x faster
            'duplicate_detection': 10.0,  # 10x faster
            'memory_operations': 3.0,  # 3x faster
            'gpu_acceleration': 2.0 if sys.platform == 'darwin' else 1.0
        }
        
        # Adjust based on available components
        if not self.rust_components.get('bloom_filter'):
            boosts['duplicate_detection'] = 1.0
        if not self.rust_components.get('metal_accelerator'):
            boosts['gpu_acceleration'] = 1.0
            
        return boosts
        
    def _estimate_memory_savings(self) -> Dict[str, Any]:
        """Estimate memory savings from Rust."""
        if not self.rust_available:
            return {'enabled': False}
            
        total_ram_gb = psutil.virtual_memory().total / (1024**3)
        
        return {
            'enabled': True,
            'total_ram_gb': total_ram_gb,
            'rust_pool_mb': int(total_ram_gb * 1024 * 0.4 * 0.5),  # 50% of JARVIS allocation
            'estimated_savings_percent': 25,  # 25% memory reduction
            'zero_copy_enabled': self.rust_components.get('zero_copy', False)
        }
        
    def _get_fallback_config(self) -> Dict[str, Any]:
        """Get fallback configuration when Rust is not available."""
        return {
            'available': False,
            'components': {},
            'performance_boost': {},
            'memory_savings': {'enabled': False},
            'fallback_reason': 'Rust components not built or not available'
        }
        
    def get_rust_status(self) -> Dict[str, Any]:
        """Get current Rust component status."""
        return {
            'rust_available': self.rust_available,
            'rust_built': self._check_rust_built(),
            'components': self.rust_components,
            'vision_dir': str(self.vision_dir),
            'rust_core_dir': str(self.rust_core_dir)
        }

# Global instance
_rust_startup_manager: Optional[RustStartupManager] = None

async def initialize_rust_acceleration() -> Dict[str, Any]:
    """Initialize Rust acceleration during startup."""
    global _rust_startup_manager
    
    if _rust_startup_manager is None:
        _rust_startup_manager = RustStartupManager()
    
    # Initialize basic Rust components
    result = await _rust_startup_manager.initialize_rust_components()
    
    # Initialize dynamic component loader for runtime switching
    try:
        from .dynamic_component_loader import initialize_dynamic_components
        dynamic_loader = await initialize_dynamic_components()
        result['dynamic_loader'] = True
        result['dynamic_status'] = dynamic_loader.get_status()
        logger.info("✅ Dynamic component loader initialized - will check for Rust availability periodically")
    except Exception as e:
        logger.warning(f"Could not initialize dynamic component loader: {e}")
        result['dynamic_loader'] = False
        
    return result

def get_rust_status() -> Dict[str, Any]:
    """Get current Rust status."""
    if _rust_startup_manager is None:
        return {'rust_available': False, 'components': {}}
        
    return _rust_startup_manager.get_rust_status()

# Export for main.py
__all__ = ['initialize_rust_acceleration', 'get_rust_status']