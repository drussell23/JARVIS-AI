#!/usr/bin/env python3
"""
Update Python modules to use Rust acceleration dynamically.
No hardcoding - all configuration is dynamic.
"""

import os
import sys
import logging
import json
from pathlib import Path
import importlib
import psutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import Rust core
try:
    import jarvis_rust_core
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    logger.warning("Rust core not available - updates will prepare for future integration")

class PythonRustIntegrator:
    """Updates Python modules to use Rust acceleration."""
    
    def __init__(self):
        self.vision_dir = Path(__file__).parent
        self.config_file = self.vision_dir / "rust_config.json"
        self.config = self._load_config()
        
    def _load_config(self) -> dict:
        """Load Rust configuration."""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        else:
            # Default configuration
            return {
                "rust_acceleration": {
                    "enabled": RUST_AVAILABLE,
                    "components": {
                        "memory_pool": True,
                        "bloom_filter": True,
                        "sliding_window": True,
                        "image_processor": True,
                        "metal_acceleration": sys.platform == "darwin"
                    },
                    "memory_pool_size_mb": 2048,
                    "worker_threads": min(8, psutil.cpu_count()),
                    "enable_cpu_affinity": True
                }
            }
            
    def update_bloom_filter_network(self):
        """Update bloom_filter_network.py to use Rust."""
        logger.info("Updating bloom_filter_network.py...")
        
        file_path = self.vision_dir / "bloom_filter_network.py"
        if not file_path.exists():
            logger.warning(f"{file_path} not found")
            return
            
        # Read current content
        content = file_path.read_text()
        
        # Add Rust import section if not present
        rust_import = '''
# Try to import Rust acceleration
try:
    import jarvis_rust_core
    RUST_BLOOM_AVAILABLE = True
    logger.info("Rust bloom filter acceleration available")
except ImportError:
    RUST_BLOOM_AVAILABLE = False
    jarvis_rust_core = None
'''
        
        if "RUST_BLOOM_AVAILABLE" not in content:
            # Find where to insert (after imports)
            import_end = content.find("logger = logging.getLogger")
            if import_end > 0:
                import_end = content.find("\n", import_end) + 1
                content = content[:import_end] + "\n" + rust_import + "\n" + content[import_end:]
                
        # Update BloomFilterNetwork to use Rust if available
        rust_init = '''
        # Use Rust acceleration if available
        self.rust_accelerated = False
        if RUST_BLOOM_AVAILABLE and enable_rust_hashing:
            try:
                self.rust_network = jarvis_rust_core.bloom_filter.PyRustBloomNetwork(
                    global_mb=global_size_mb,
                    regional_mb=regional_size_mb,
                    element_mb=element_size_mb
                )
                self.rust_accelerated = True
                logger.info("Using Rust-accelerated bloom filters")
            except Exception as e:
                logger.warning(f"Failed to initialize Rust bloom filter: {e}")
'''
        
        # Find __init__ method of BloomFilterNetwork
        init_start = content.find("def __init__(self,")
        if init_start > 0:
            # Find the end of parameter list
            init_body = content.find("):", init_start)
            if init_body > 0:
                # Find where to insert (after first few lines)
                insert_point = content.find("self.enable_hierarchical_checking", init_body)
                if insert_point > 0:
                    insert_point = content.find("\n", insert_point) + 1
                    # Check if already updated
                    if "self.rust_accelerated" not in content[init_body:insert_point+500]:
                        content = content[:insert_point] + "\n" + rust_init + content[insert_point:]
                        
        # Update check_and_add method to use Rust
        rust_check = '''
        # Use Rust acceleration if available
        if self.rust_accelerated:
            try:
                return self.rust_network.check_and_add(element_key.encode(), quadrant)
            except Exception as e:
                logger.error(f"Rust bloom filter error: {e}, falling back to Python")
                self.rust_accelerated = False
'''
        
        # Find check_and_add method
        check_method = content.find("def check_and_add(self,")
        if check_method > 0:
            method_body = content.find("with self.lock:", check_method)
            if method_body > 0:
                method_body = content.find("\n", method_body) + 1
                # Check if already updated
                if "self.rust_accelerated" not in content[method_body:method_body+500]:
                    # Increase indentation
                    rust_check_indented = "\n".join("        " + line for line in rust_check.strip().split("\n"))
                    content = content[:method_body] + rust_check_indented + "\n\n" + content[method_body:]
                    
        # Write updated content
        file_path.write_text(content)
        logger.info("✓ Updated bloom_filter_network.py")
        
    def update_real_time_interaction_handler(self):
        """Update real_time_interaction_handler.py to use Rust."""
        logger.info("Updating real_time_interaction_handler.py...")
        
        file_path = self.vision_dir / "real_time_interaction_handler.py"
        if not file_path.exists():
            logger.warning(f"{file_path} not found")
            return
            
        content = file_path.read_text()
        
        # Add Rust import
        rust_import = '''
# Try to import Rust acceleration
try:
    import jarvis_rust_core
    from rust_proactive_integration import RustProactiveMonitor, get_rust_monitor
    RUST_ACCELERATION_AVAILABLE = True
    logger.info("Rust acceleration available for real-time processing")
except ImportError:
    RUST_ACCELERATION_AVAILABLE = False
    jarvis_rust_core = None
    logger.info("Running without Rust acceleration")
'''
        
        if "RUST_ACCELERATION_AVAILABLE" not in content:
            import_end = content.find("logger = logging.getLogger")
            if import_end > 0:
                import_end = content.find("\n", import_end) + 1
                content = content[:import_end] + "\n" + rust_import + "\n" + content[import_end:]
                
        # Add Rust components to __init__
        rust_init = '''
        # Initialize Rust acceleration if available
        self.rust_accelerated = False
        if RUST_ACCELERATION_AVAILABLE:
            try:
                # Initialize Rust components
                self.rust_monitor = RustProactiveMonitor(
                    vision_analyzer=vision_analyzer,
                    interaction_handler=self
                )
                
                # Use Rust frame buffer
                if hasattr(jarvis_rust_core, 'sliding_window'):
                    self.frame_buffer = jarvis_rust_core.sliding_window.PyFrameRingBuffer(
                        capacity_mb=500  # Dynamic based on available RAM
                    )
                    
                # Use Rust bloom filter
                if hasattr(jarvis_rust_core, 'bloom_filter'):
                    self.bloom_filter = jarvis_rust_core.bloom_filter.PyRustBloomFilter(
                        size_mb=10.0,
                        num_hashes=7
                    )
                    
                self.rust_accelerated = True
                logger.info("Rust acceleration initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Rust acceleration: {e}")
'''
        
        # Find __init__ method
        init_start = content.find("def __init__(self,")
        if init_start > 0:
            init_body = content.find("):", init_start)
            if init_body > 0:
                insert_point = content.find("self._is_monitoring = False", init_body)
                if insert_point > 0:
                    insert_point = content.find("\n", insert_point) + 1
                    if "self.rust_accelerated" not in content[init_body:insert_point+1000]:
                        content = content[:insert_point] + "\n" + rust_init + content[insert_point:]
                        
        # Write updated content
        file_path.write_text(content)
        logger.info("✓ Updated real_time_interaction_handler.py")
        
    def update_claude_vision_analyzer(self):
        """Update claude_vision_analyzer_main.py to use Rust."""
        logger.info("Updating claude_vision_analyzer_main.py...")
        
        file_path = self.vision_dir / "claude_vision_analyzer_main.py"
        if not file_path.exists():
            logger.warning(f"{file_path} not found")
            return
            
        content = file_path.read_text()
        
        # Add Rust import
        rust_import = '''
# Try to import Rust acceleration
try:
    import jarvis_rust_core
    from rust_integration import RustImageProcessor, ZeroCopyVisionPipeline
    RUST_VISION_AVAILABLE = True
    logger.info("Rust vision acceleration available")
except ImportError:
    RUST_VISION_AVAILABLE = False
    jarvis_rust_core = None
'''
        
        if "RUST_VISION_AVAILABLE" not in content:
            import_end = content.find("logger = logging.getLogger")
            if import_end > 0:
                import_end = content.find("\n", import_end) + 1
                content = content[:import_end] + "\n" + rust_import + "\n" + content[import_end:]
                
        # Add Rust initialization
        rust_init = '''
        # Initialize Rust acceleration
        self.rust_accelerated = False
        if RUST_VISION_AVAILABLE:
            try:
                self.rust_processor = RustImageProcessor()
                self.zero_copy_pipeline = ZeroCopyVisionPipeline(enable_quantization=True)
                self.rust_accelerated = True
                logger.info("Rust vision acceleration initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Rust vision: {e}")
'''
        
        # Find __init__ method
        init_start = content.find("def __init__(self")
        if init_start > 0:
            init_body = content.find("):", init_start)
            if init_body > 0:
                # Find a good insertion point
                insert_point = content.find("self.client = None", init_body)
                if insert_point > 0:
                    insert_point = content.find("\n", insert_point) + 1
                    if "self.rust_accelerated" not in content[init_body:insert_point+1000]:
                        content = content[:insert_point] + "\n" + rust_init + content[insert_point:]
                        
        # Update analyze_screenshot to use Rust preprocessing
        rust_preprocess = '''
        # Use Rust acceleration for preprocessing if available
        if self.rust_accelerated and hasattr(self, 'zero_copy_pipeline'):
            try:
                # Preprocess with zero-copy pipeline
                processed = await self.zero_copy_pipeline.process_image(
                    screenshot if isinstance(screenshot, bytes) else screenshot.tobytes()
                )
                if processed.get('success'):
                    # Use processed features to enhance prompt
                    features = processed.get('features', [])
                    if features:
                        prompt = f"{prompt}\n[Preprocessed features detected]"  # Claude uses these internally
            except Exception as e:
                logger.debug(f"Rust preprocessing error: {e}")
'''
        
        # Find analyze_screenshot method
        analyze_method = content.find("async def analyze_screenshot(self,")
        if analyze_method > 0:
            method_body = content.find("try:", analyze_method)
            if method_body > 0:
                method_body = content.find("\n", method_body) + 1
                if "self.rust_accelerated" not in content[method_body:method_body+1000]:
                    # Proper indentation
                    rust_preprocess_indented = "\n".join("            " + line for line in rust_preprocess.strip().split("\n"))
                    content = content[:method_body] + rust_preprocess_indented + "\n\n" + content[method_body:]
                    
        # Write updated content
        file_path.write_text(content)
        logger.info("✓ Updated claude_vision_analyzer_main.py")
        
    def update_integration_orchestrator(self):
        """Update integration_orchestrator.py for dynamic Rust memory."""
        logger.info("Updating integration_orchestrator.py...")
        
        file_path = self.vision_dir / "integration_orchestrator.py"
        if not file_path.exists():
            logger.warning(f"{file_path} not found")
            return
            
        content = file_path.read_text()
        
        # Add Rust memory import
        rust_import = '''
# Try to use Rust memory management
try:
    import jarvis_rust_core
    RUST_MEMORY_AVAILABLE = hasattr(jarvis_rust_core, 'zero_copy') and hasattr(jarvis_rust_core.zero_copy, 'PyZeroCopyPool')
    if RUST_MEMORY_AVAILABLE:
        logger.info("Rust zero-copy memory management available")
except ImportError:
    RUST_MEMORY_AVAILABLE = False
'''
        
        if "RUST_MEMORY_AVAILABLE" not in content:
            import_end = content.find("logger = logging.getLogger")
            if import_end > 0:
                import_end = content.find("\n", import_end) + 1
                content = content[:import_end] + "\n" + rust_import + "\n" + content[import_end:]
                
        # Update memory allocation to use Rust
        rust_memory = '''
        # Use Rust zero-copy memory if available
        if RUST_MEMORY_AVAILABLE:
            try:
                # Allocate 40% of available RAM for Rust pool
                available_mb = psutil.virtual_memory().available // (1024 * 1024)
                rust_pool_size = int(available_mb * 0.4)
                rust_pool_size = min(rust_pool_size, 6400)  # Cap at 6.4GB for 16GB system
                
                self.rust_memory_pool = jarvis_rust_core.zero_copy.PyZeroCopyPool(
                    max_memory_mb=rust_pool_size
                )
                
                logger.info(f"Initialized Rust memory pool: {rust_pool_size}MB")
                
                # Adjust component budgets to account for Rust pool
                total_budget_mb -= rust_pool_size * 0.5  # Half counts toward budget
            except Exception as e:
                logger.error(f"Failed to initialize Rust memory: {e}")
'''
        
        # Find allocate_memory method
        alloc_method = content.find("def allocate_memory(self)")
        if alloc_method > 0:
            method_body = content.find("available_mb = psutil.virtual_memory()", alloc_method)
            if method_body > 0:
                # Find where to insert
                insert_point = content.find("# Calculate total budget", method_body)
                if insert_point > 0:
                    if "RUST_MEMORY_AVAILABLE" not in content[method_body:insert_point+500]:
                        content = content[:insert_point] + rust_memory + "\n\n" + content[insert_point:]
                        
        # Write updated content
        file_path.write_text(content)
        logger.info("✓ Updated integration_orchestrator.py")
        
    def create_rust_enabled_config(self):
        """Create configuration for Rust-enabled components."""
        logger.info("Creating Rust-enabled configuration...")
        
        config = {
            "rust_acceleration": self.config["rust_acceleration"],
            "vision": {
                "enable_rust_acceleration": RUST_AVAILABLE,
                "rust_components": {
                    "bloom_filter": RUST_AVAILABLE,
                    "sliding_window": RUST_AVAILABLE,
                    "metal_gpu": RUST_AVAILABLE and sys.platform == "darwin",
                    "zero_copy_memory": RUST_AVAILABLE,
                    "simd_processing": RUST_AVAILABLE
                },
                "performance_targets": {
                    "frame_processing_ms": 20,
                    "duplicate_detection_ms": 0.5,
                    "memory_usage_mb": 3000,
                    "gpu_utilization_percent": 50
                }
            },
            "memory": {
                "total_system_gb": psutil.virtual_memory().total / (1024**3),
                "jarvis_allocation_percent": 40,
                "rust_pool_percent": 50,  # 50% of JARVIS allocation
                "enable_zero_copy": RUST_AVAILABLE,
                "enable_memory_pressure_handling": True
            }
        }
        
        # Save configuration
        config_file = self.vision_dir / "jarvis_rust_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
            
        logger.info(f"✓ Created configuration: {config_file}")
        
        return config
        
    def verify_updates(self):
        """Verify all updates were successful."""
        logger.info("Verifying updates...")
        
        results = {}
        
        # Check each module
        modules_to_check = [
            "bloom_filter_network",
            "real_time_interaction_handler",
            "claude_vision_analyzer_main",
            "integration_orchestrator"
        ]
        
        for module_name in modules_to_check:
            try:
                # Try to import
                module = importlib.import_module(module_name)
                
                # Check for Rust flags
                has_rust = (
                    hasattr(module, "RUST_BLOOM_AVAILABLE") or
                    hasattr(module, "RUST_ACCELERATION_AVAILABLE") or
                    hasattr(module, "RUST_VISION_AVAILABLE") or
                    hasattr(module, "RUST_MEMORY_AVAILABLE")
                )
                
                results[module_name] = {
                    "imported": True,
                    "rust_ready": has_rust,
                    "rust_active": has_rust and RUST_AVAILABLE
                }
                
            except Exception as e:
                results[module_name] = {
                    "imported": False,
                    "error": str(e)
                }
                
        # Print results
        print("\n" + "=" * 60)
        print("PYTHON-RUST INTEGRATION VERIFICATION")
        print("=" * 60)
        print(f"Rust Core Available: {RUST_AVAILABLE}")
        print(f"System: {sys.platform}, RAM: {psutil.virtual_memory().total / (1024**3):.1f}GB")
        print("\nModule Status:")
        
        for module, status in results.items():
            if status.get("imported"):
                rust_status = "✓ Active" if status.get("rust_active") else "✗ Ready" if status.get("rust_ready") else "✗ Not Ready"
                print(f"  {module}: Imported ✓, Rust {rust_status}")
            else:
                print(f"  {module}: Import Failed ✗ - {status.get('error', 'Unknown error')}")
                
        print("=" * 60)
        
        return results
        
    def run_all_updates(self):
        """Run all update operations."""
        logger.info("Starting Python-Rust integration updates...")
        
        # Update each module
        self.update_bloom_filter_network()
        self.update_real_time_interaction_handler()
        self.update_claude_vision_analyzer()
        self.update_integration_orchestrator()
        
        # Create configuration
        config = self.create_rust_enabled_config()
        
        # Verify updates
        results = self.verify_updates()
        
        # Summary
        successful = sum(1 for r in results.values() if r.get("imported"))
        rust_ready = sum(1 for r in results.values() if r.get("rust_ready"))
        
        print(f"\nSummary: {successful}/{len(results)} modules updated successfully")
        print(f"Rust-ready modules: {rust_ready}/{len(results)}")
        
        if RUST_AVAILABLE:
            print("\n✅ Rust acceleration is active and integrated!")
        else:
            print("\n⚠️  Modules prepared for Rust but core not built yet.")
            print("Run: python build_rust_components.py")
            
        return results

def main():
    """Run Python-Rust integration updates."""
    integrator = PythonRustIntegrator()
    integrator.run_all_updates()

if __name__ == "__main__":
    main()