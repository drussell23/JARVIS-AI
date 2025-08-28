"""
Rust-Python Bridge for Vision System
Provides seamless integration between Rust acceleration and Python vision processing
"""

import os
import sys
import logging
import ctypes
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Try to import Rust components
RUST_AVAILABLE = False
rust_processor = None

# Check if Rust library is built
rust_lib_path = Path(__file__).parent / "jarvis-rust-core" / "target" / "release"
if rust_lib_path.exists():
    # Try loading the dynamic library
    try:
        if sys.platform == "darwin":
            lib_file = rust_lib_path / "libjarvis_rust_core.dylib"
        elif sys.platform == "linux":
            lib_file = rust_lib_path / "libjarvis_rust_core.so"
        else:
            lib_file = rust_lib_path / "jarvis_rust_core.dll"
        
        if lib_file.exists():
            rust_lib = ctypes.CDLL(str(lib_file))
            RUST_AVAILABLE = True
            logger.info(f"Loaded Rust library from {lib_file}")
    except Exception as e:
        logger.warning(f"Failed to load Rust library: {e}")

# Try PyO3 bindings as fallback
if not RUST_AVAILABLE:
    try:
        import jarvis_rust_core
        RUST_AVAILABLE = True
        logger.info("Using PyO3 Rust bindings")
    except ImportError:
        logger.warning("Rust acceleration not available")

class RustImageProcessor:
    """Wrapper for Rust image processing functions"""
    
    def __init__(self):
        self.use_ctypes = hasattr(self, 'rust_lib')
        self._init_rust_functions()
    
    def _init_rust_functions(self):
        """Initialize Rust function signatures"""
        if RUST_AVAILABLE and self.use_ctypes:
            # Define ctypes signatures
            self.rust_lib.process_image.argtypes = [
                ctypes.POINTER(ctypes.c_uint8),  # image data
                ctypes.c_uint32,  # width
                ctypes.c_uint32,  # height
                ctypes.c_uint8,   # channels
                ctypes.POINTER(ctypes.c_uint8),  # output buffer
            ]
            self.rust_lib.process_image.restype = ctypes.c_int
    
    def process_numpy_image(self, image: np.ndarray) -> np.ndarray:
        """Process image using Rust acceleration"""
        if not RUST_AVAILABLE:
            # Fallback to Python processing
            return image
        
        if self.use_ctypes:
            # Use ctypes interface
            height, width = image.shape[:2]
            channels = image.shape[2] if len(image.shape) > 2 else 1
            
            # Prepare output buffer
            output_size = width * height * channels
            output_buffer = (ctypes.c_uint8 * output_size)()
            
            # Call Rust function
            result = self.rust_lib.process_image(
                image.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                width,
                height,
                channels,
                output_buffer
            )
            
            if result == 0:
                # Success - convert back to numpy
                output_array = np.frombuffer(output_buffer, dtype=np.uint8)
                return output_array.reshape((height, width, channels))
            else:
                # Error - return original
                logger.error(f"Rust processing failed with code {result}")
                return image
        else:
            # Use PyO3 interface
            try:
                processor = jarvis_rust_core.RustImageProcessor()
                return processor.process_numpy_image(image)
            except Exception as e:
                logger.error(f"PyO3 processing error: {e}")
                return image

class RustAdvancedMemoryPool:
    """Wrapper for Rust memory pool"""
    
    def __init__(self):
        if RUST_AVAILABLE and hasattr(jarvis_rust_core, 'RustAdvancedMemoryPool'):
            self.pool = jarvis_rust_core.RustAdvancedMemoryPool()
        else:
            self.pool = None
    
    def allocate(self, size: int):
        """Allocate memory buffer"""
        if self.pool:
            return self.pool.allocate(size)
        else:
            # Fallback to numpy
            return np.zeros(size, dtype=np.uint8)
    
    def stats(self) -> Dict[str, Any]:
        """Get memory pool statistics"""
        if self.pool:
            return self.pool.stats()
        else:
            return {'available': False}

class RustRuntimeManager:
    """Wrapper for Rust runtime manager"""
    
    def __init__(self, worker_threads: Optional[int] = None, enable_cpu_affinity: bool = True):
        self.worker_threads = worker_threads
        self.enable_cpu_affinity = enable_cpu_affinity
        
        if RUST_AVAILABLE and hasattr(jarvis_rust_core, 'RustRuntimeManager'):
            self.runtime = jarvis_rust_core.RustRuntimeManager(
                worker_threads=worker_threads,
                enable_cpu_affinity=enable_cpu_affinity
            )
        else:
            self.runtime = None
    
    def run_cpu_task(self, func):
        """Run CPU-intensive task"""
        if self.runtime:
            return self.runtime.run_cpu_task(func)
        else:
            # Fallback to direct execution
            return func()
    
    def stats(self) -> Dict[str, Any]:
        """Get runtime statistics"""
        if self.runtime:
            return self.runtime.stats()
        else:
            return {'available': False}

def process_image_batch(images: List[np.ndarray]) -> List[np.ndarray]:
    """Process batch of images using Rust acceleration"""
    if RUST_AVAILABLE and hasattr(jarvis_rust_core, 'process_image_batch'):
        return jarvis_rust_core.process_image_batch(images)
    else:
        # Fallback to sequential processing
        processor = RustImageProcessor()
        return [processor.process_numpy_image(img) for img in images]

# Visual feature extraction functions
def extract_dominant_colors_rust(image: np.ndarray, num_colors: int = 5) -> List[Tuple[int, int, int]]:
    """Extract dominant colors using Rust"""
    if RUST_AVAILABLE and hasattr(jarvis_rust_core, 'extract_dominant_colors'):
        return jarvis_rust_core.extract_dominant_colors(image, num_colors)
    else:
        # Simple Python fallback
        from sklearn.cluster import KMeans
        pixels = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=num_colors, random_state=42)
        kmeans.fit(pixels)
        colors = kmeans.cluster_centers_.astype(int)
        return [tuple(color) for color in colors]

def calculate_edge_density_rust(image: np.ndarray) -> float:
    """Calculate edge density using Rust SIMD operations"""
    if RUST_AVAILABLE and hasattr(jarvis_rust_core, 'calculate_edge_density'):
        return jarvis_rust_core.calculate_edge_density(image)
    else:
        # Python fallback using Sobel
        import cv2
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) > 2 else image
        edges = cv2.Canny(gray, 50, 150)
        return np.sum(edges > 0) / edges.size

def analyze_texture_rust(image: np.ndarray) -> Dict[str, float]:
    """Analyze texture patterns using Rust"""
    if RUST_AVAILABLE and hasattr(jarvis_rust_core, 'analyze_texture'):
        return jarvis_rust_core.analyze_texture(image)
    else:
        # Simple Python fallback
        return {
            'contrast': float(np.std(image)),
            'homogeneity': 1.0 / (1.0 + np.var(image)),
            'energy': float(np.sum(image ** 2)) / image.size
        }

def analyze_spatial_layout_rust(image: np.ndarray) -> Dict[str, Any]:
    """Analyze spatial layout using Rust"""
    if RUST_AVAILABLE and hasattr(jarvis_rust_core, 'analyze_spatial_layout'):
        return jarvis_rust_core.analyze_spatial_layout(image)
    else:
        # Python fallback
        height, width = image.shape[:2]
        return {
            'quadrants': {
                'top_left': image[:height//2, :width//2].mean(),
                'top_right': image[:height//2, width//2:].mean(),
                'bottom_left': image[height//2:, :width//2].mean(),
                'bottom_right': image[height//2:, width//2:].mean()
            },
            'center_weight': float(image[height//4:3*height//4, width//4:3*width//4].mean()),
            'edge_weight': float(np.concatenate([
                image[0, :], image[-1, :], image[:, 0], image[:, -1]
            ]).mean())
        }

# Build Rust library if needed
def build_rust_library():
    """Build the Rust library if not already built"""
    rust_project_path = Path(__file__).parent / "jarvis-rust-core"
    
    if not rust_project_path.exists():
        logger.warning(f"Rust project not found at {rust_project_path}")
        return False
    
    # Check if already built
    lib_path = rust_project_path / "target" / "release"
    if lib_path.exists() and any(lib_path.glob("*jarvis_rust_core*")):
        return True
    
    logger.info("Building Rust library...")
    try:
        import subprocess
        result = subprocess.run(
            ["cargo", "build", "--release", "--features", "python-bindings"],
            cwd=rust_project_path,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info("Rust library built successfully")
            return True
        else:
            logger.error(f"Rust build failed: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Failed to build Rust library: {e}")
        return False

# Attempt to build on import if needed
if not RUST_AVAILABLE:
    if build_rust_library():
        # Try importing again
        try:
            import jarvis_rust_core
            RUST_AVAILABLE = True
            logger.info("Successfully loaded Rust library after building")
        except ImportError:
            pass