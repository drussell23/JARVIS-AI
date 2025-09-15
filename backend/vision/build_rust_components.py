#!/usr/bin/env python3
"""
Build and verify Rust components for JARVIS vision system.
This script handles the complete Rust integration setup.
"""

import os
import sys
import subprocess
import logging
import json
from pathlib import Path
import platform
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RustBuilder:
    def __init__(self):
        self.vision_dir = Path(__file__).parent
        self.rust_core_dir = self.vision_dir / "jarvis-rust-core"
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.is_macos = platform.system() == "Darwin"
        self.is_m1 = self.is_macos and platform.machine() == "arm64"
        
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are installed."""
        logger.info("Checking prerequisites...")
        
        # Check Rust
        try:
            result = subprocess.run(["rustc", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"✓ Rust installed: {result.stdout.strip()}")
            else:
                logger.error("✗ Rust not found. Install from https://rustup.rs/")
                return False
        except FileNotFoundError:
            logger.error("✗ Rust not found. Install from https://rustup.rs/")
            return False
            
        # Check Cargo
        try:
            result = subprocess.run(["cargo", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"✓ Cargo installed: {result.stdout.strip()}")
            else:
                logger.error("✗ Cargo not found")
                return False
        except FileNotFoundError:
            logger.error("✗ Cargo not found")
            return False
            
        # Check maturin
        try:
            result = subprocess.run(["maturin", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"✓ Maturin installed: {result.stdout.strip()}")
            else:
                logger.info("Maturin not found, installing...")
                subprocess.run([sys.executable, "-m", "pip", "install", "maturin"], check=True)
                logger.info("✓ Maturin installed")
        except FileNotFoundError:
            logger.info("Maturin not found, installing...")
            subprocess.run([sys.executable, "-m", "pip", "install", "maturin"], check=True)
            logger.info("✓ Maturin installed")
            
        return True
        
    def setup_rust_toolchain(self):
        """Setup Rust toolchain for the target platform."""
        logger.info("Setting up Rust toolchain...")
        
        if self.is_m1:
            # Set up for M1 Mac
            logger.info("Configuring for Apple Silicon (M1)...")
            subprocess.run(["rustup", "target", "add", "aarch64-apple-darwin"], check=True)
            
            # Create .cargo/config.toml for optimizations
            cargo_config_dir = self.rust_core_dir / ".cargo"
            cargo_config_dir.mkdir(exist_ok=True)
            
            config_content = """
[target.aarch64-apple-darwin]
rustflags = ["-C", "target-cpu=native"]

[build]
target = "aarch64-apple-darwin"

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
strip = true
"""
            
            config_file = cargo_config_dir / "config.toml"
            config_file.write_text(config_content)
            logger.info("✓ Created optimized cargo config for M1")
            
    def build_rust_library(self) -> bool:
        """Build the Rust library with all optimizations."""
        logger.info("Building Rust library...")
        
        if not self.rust_core_dir.exists():
            logger.error(f"Rust core directory not found: {self.rust_core_dir}")
            return False
            
        os.chdir(self.rust_core_dir)
        
        # Clean previous builds
        if (self.rust_core_dir / "target").exists():
            logger.info("Cleaning previous build...")
            shutil.rmtree(self.rust_core_dir / "target")
            
        # Build with cargo first
        logger.info("Running cargo build...")
        env = os.environ.copy()
        if self.is_m1:
            env["CARGO_BUILD_TARGET"] = "aarch64-apple-darwin"
            
        result = subprocess.run(
            ["cargo", "build", "--release", "--features", "python-bindings,simd"],
            env=env,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Cargo build failed: {result.stderr}")
            return False
            
        logger.info("✓ Cargo build successful")
        
        # Build Python bindings with maturin
        logger.info("Building Python bindings with maturin...")
        result = subprocess.run(
            ["maturin", "develop", "--release", "--features", "python-bindings,simd"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Maturin build failed: {result.stderr}")
            return False
            
        logger.info("✓ Maturin build successful")
        return True
        
    def verify_installation(self) -> bool:
        """Verify the Rust library is properly installed."""
        logger.info("Verifying Rust library installation...")
        
        # Change back to vision directory
        os.chdir(self.vision_dir)
        
        # Try to import the module
        try:
            import jarvis_rust_core
            logger.info("✓ Successfully imported jarvis_rust_core")
            
            # Check available functions
            available_components = []
            if hasattr(jarvis_rust_core, 'RustImageProcessor'):
                available_components.append('RustImageProcessor')
            if hasattr(jarvis_rust_core, 'RustAdvancedMemoryPool'):
                available_components.append('RustAdvancedMemoryPool')
            if hasattr(jarvis_rust_core, 'RustRuntimeManager'):
                available_components.append('RustRuntimeManager')
            if hasattr(jarvis_rust_core, 'RustQuantizedModel'):
                available_components.append('RustQuantizedModel')
                
            logger.info(f"Available components: {', '.join(available_components)}")
            
            # Test basic functionality
            logger.info("Testing basic functionality...")
            
            # Test memory pool
            try:
                pool = jarvis_rust_core.RustAdvancedMemoryPool()
                stats = pool.stats()
                logger.info(f"✓ Memory pool working: {stats}")
            except Exception as e:
                logger.error(f"✗ Memory pool test failed: {e}")
                return False
                
            # Test runtime manager
            try:
                runtime = jarvis_rust_core.RustRuntimeManager(
                    worker_threads=4,
                    enable_cpu_affinity=True
                )
                stats = runtime.stats()
                logger.info(f"✓ Runtime manager working: {stats}")
            except Exception as e:
                logger.error(f"✗ Runtime manager test failed: {e}")
                return False
                
            return True
            
        except ImportError as e:
            logger.error(f"✗ Failed to import jarvis_rust_core: {e}")
            return False
            
    def update_python_modules(self):
        """Update Python modules to use Rust acceleration."""
        logger.info("Updating Python modules for Rust integration...")
        
        # Update rust_bridge.py
        rust_bridge_file = self.vision_dir / "rust_bridge.py"
        if rust_bridge_file.exists():
            content = rust_bridge_file.read_text()
            
            # Update RUST_AVAILABLE check
            if "RUST_AVAILABLE = False" in content:
                logger.info("Updating rust_bridge.py...")
                content = content.replace(
                    "RUST_AVAILABLE = False",
                    "RUST_AVAILABLE = True  # Updated by build script"
                )
                rust_bridge_file.write_text(content)
                logger.info("✓ Updated rust_bridge.py")
                
        # Create configuration file
        config = {
            "rust_acceleration": {
                "enabled": True,
                "components": {
                    "memory_pool": True,
                    "runtime_manager": True,
                    "image_processor": True,
                    "bloom_filter": True,
                    "metal_acceleration": self.is_macos
                },
                "memory_pool_size_mb": 2048,
                "worker_threads": 8,
                "enable_cpu_affinity": True
            }
        }
        
        config_file = self.vision_dir / "rust_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"✓ Created rust_config.json")
        
    def run_performance_test(self):
        """Run a quick performance test."""
        logger.info("Running performance test...")
        
        test_script = self.vision_dir / "rust_integration.py"
        if test_script.exists():
            result = subprocess.run(
                [sys.executable, str(test_script)],
                capture_output=True,
                text=True
            )
            
            if "Benchmarking Rust acceleration" in result.stdout:
                logger.info("Performance test output:")
                print(result.stdout)
            else:
                logger.warning("Performance test did not run as expected")
                
    def create_test_script(self):
        """Create a test script for developers."""
        test_content = '''
#!/usr/bin/env python3
"""Test Rust components integration."""

import sys
import time
import numpy as np

try:
    import jarvis_rust_core
    print("✓ Rust core imported successfully")
except ImportError as e:
    print(f"✗ Failed to import Rust core: {e}")
    sys.exit(1)

# Test memory pool
print("\nTesting memory pool...")
pool = jarvis_rust_core.RustAdvancedMemoryPool()
print(f"Pool stats: {pool.stats()}")

# Test runtime manager
print("\nTesting runtime manager...")
runtime = jarvis_rust_core.RustRuntimeManager(worker_threads=4)
print(f"Runtime stats: {runtime.stats()}")

# Test image processor
print("\nTesting image processor...")
processor = jarvis_rust_core.RustImageProcessor()
test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
start = time.time()
result = processor.process_numpy_image(test_image)
print(f"Image processing took: {(time.time() - start) * 1000:.2f}ms")

print("\n✓ All tests passed!")
'''
        
        test_file = self.vision_dir / "test_rust_components.py"
        test_file.write_text(test_content)
        test_file.chmod(0o755)
        logger.info(f"✓ Created test script: {test_file}")
        
    def main(self):
        """Main build process."""
        logger.info("Starting Rust components build process...")
        logger.info(f"Platform: {platform.system()} {platform.machine()}")
        logger.info(f"Python: {sys.version}")
        
        # Step 1: Check prerequisites
        if not self.check_prerequisites():
            logger.error("Prerequisites check failed")
            return False
            
        # Step 2: Setup toolchain
        self.setup_rust_toolchain()
        
        # Step 3: Build library
        if not self.build_rust_library():
            logger.error("Build failed")
            return False
            
        # Step 4: Verify installation
        if not self.verify_installation():
            logger.error("Verification failed")
            return False
            
        # Step 5: Update Python modules
        self.update_python_modules()
        
        # Step 6: Create test script
        self.create_test_script()
        
        # Step 7: Run performance test
        self.run_performance_test()
        
        logger.info("\n" + "=" * 50)
        logger.info("✓ Rust components build completed successfully!")
        logger.info("=" * 50)
        logger.info("\nNext steps:")
        logger.info("1. Run: python test_rust_components.py")
        logger.info("2. Update your JARVIS configuration to enable Rust acceleration")
        logger.info("3. Monitor performance improvements in real-time monitoring")
        
        return True

if __name__ == "__main__":
    builder = RustBuilder()
    success = builder.main()
    sys.exit(0 if success else 1)
