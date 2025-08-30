"""
Core ML and Metal Acceleration for macOS
Leverages Apple Silicon and GPU for ML inference
"""

import os
import platform
import logging
import numpy as np
from typing import Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass
import time

# Core ML imports (only on macOS)
COREML_AVAILABLE = False
METAL_AVAILABLE = False

if platform.system() == "Darwin":
    try:
        import coremltools as ct
        from coremltools.models import MLModel
        COREML_AVAILABLE = True
    except ImportError:
        pass
    
    try:
        import Metal
        import MetalPerformanceShaders as mps
        METAL_AVAILABLE = True
    except ImportError:
        pass

# PyTorch with Metal backend
try:
    import torch
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        TORCH_METAL_AVAILABLE = True
    else:
        TORCH_METAL_AVAILABLE = False
except ImportError:
    TORCH_METAL_AVAILABLE = False

from .optimization_config import MacOSAcceleration, OPTIMIZATION_CONFIG

logger = logging.getLogger(__name__)

@dataclass
class AcceleratorInfo:
    """Information about available accelerators"""
    has_coreml: bool = False
    has_metal: bool = False
    has_neural_engine: bool = False
    has_gpu: bool = False
    gpu_name: Optional[str] = None
    max_compute_units: int = 1
    recommended_batch_size: int = 1

class CoreMLAccelerator:
    """
    Accelerates ML models using Core ML on macOS
    """
    
    def __init__(self, config: MacOSAcceleration = None):
        self.config = config or OPTIMIZATION_CONFIG.macos
        self.models: Dict[str, MLModel] = {}
        self.info = self._detect_capabilities()
        
        logger.info(f"Core ML Accelerator initialized: {self.info}")
    
    def _detect_capabilities(self) -> AcceleratorInfo:
        """Detect available acceleration capabilities"""
        info = AcceleratorInfo()
        
        if not platform.system() == "Darwin":
            return info
        
        info.has_coreml = COREML_AVAILABLE
        info.has_metal = METAL_AVAILABLE or TORCH_METAL_AVAILABLE
        
        # Check for Apple Silicon
        if platform.processor() == "arm":
            info.has_neural_engine = True
            info.max_compute_units = 8  # M1 has 8, M2 has 10+
            info.recommended_batch_size = 4
        
        # Check for GPU
        if TORCH_METAL_AVAILABLE:
            info.has_gpu = True
            info.gpu_name = "Apple Metal GPU"
        
        return info
    
    def convert_model_to_coreml(self, 
                               model: Any,
                               model_name: str,
                               input_shape: Tuple[int, ...],
                               output_dir: str = None) -> Optional[str]:
        """
        Convert a PyTorch or TensorFlow model to Core ML
        """
        if not COREML_AVAILABLE:
            logger.warning("Core ML not available")
            return None
        
        output_dir = output_dir or os.path.expanduser("~/.jarvis/coreml_models")
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f"{model_name}.mlmodel")
        
        try:
            # Check if already converted
            if os.path.exists(output_path):
                logger.info(f"Using existing Core ML model: {output_path}")
                return output_path
            
            logger.info(f"Converting {model_name} to Core ML...")
            
            # PyTorch conversion
            if hasattr(model, 'eval'):
                import torch
                model.eval()
                
                # Create example input
                example_input = torch.randn(1, *input_shape)
                
                # Trace the model
                traced_model = torch.jit.trace(model, example_input)
                
                # Convert to Core ML
                coreml_model = ct.convert(
                    traced_model,
                    inputs=[ct.TensorType(shape=example_input.shape)],
                    compute_units=ct.ComputeUnit[self.config.coreml_compute_units],
                    minimum_deployment_target=ct.target.macOS13
                )
                
                # Save model
                coreml_model.save(output_path)
                logger.info(f"Saved Core ML model to {output_path}")
                
                return output_path
            
            else:
                logger.error(f"Unsupported model type for {model_name}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to convert model to Core ML: {e}")
            return None
    
    def load_coreml_model(self, model_path: str, model_name: str) -> bool:
        """Load a Core ML model"""
        if not COREML_AVAILABLE:
            return False
        
        try:
            model = MLModel(model_path)
            self.models[model_name] = model
            logger.info(f"Loaded Core ML model: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to load Core ML model: {e}")
            return False
    
    def predict(self, model_name: str, input_data: np.ndarray) -> Optional[np.ndarray]:
        """Run inference using Core ML"""
        if model_name not in self.models:
            logger.error(f"Model {model_name} not loaded")
            return None
        
        try:
            model = self.models[model_name]
            
            # Prepare input
            input_dict = {
                model.input_names[0]: input_data
            }
            
            # Run prediction
            start_time = time.time()
            output = model.predict(input_dict)
            inference_time = time.time() - start_time
            
            # Get output
            output_data = output[model.output_names[0]]
            
            logger.debug(f"Core ML inference took {inference_time*1000:.2f}ms")
            
            return output_data
            
        except Exception as e:
            logger.error(f"Core ML prediction failed: {e}")
            return None

class MetalAccelerator:
    """
    Accelerates operations using Metal Performance Shaders
    """
    
    def __init__(self, config: MacOSAcceleration = None):
        self.config = config or OPTIMIZATION_CONFIG.macos
        self.device = None
        
        if TORCH_METAL_AVAILABLE and self.config.use_metal:
            self.device = torch.device("mps")
            logger.info("Metal acceleration enabled via PyTorch")
    
    def to_device(self, tensor: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Move tensor to Metal device"""
        if not TORCH_METAL_AVAILABLE:
            return tensor
        
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        
        if self.device:
            return tensor.to(self.device)
        
        return tensor
    
    def accelerated_fft(self, audio_data: np.ndarray) -> np.ndarray:
        """Accelerated FFT using Metal"""
        if not self.device:
            # Fallback to numpy
            return np.fft.fft(audio_data)
        
        # Convert to PyTorch and use Metal
        tensor = self.to_device(torch.from_numpy(audio_data))
        fft_result = torch.fft.fft(tensor)
        
        return fft_result.cpu().numpy()
    
    def accelerated_conv1d(self, signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Accelerated 1D convolution using Metal"""
        if not self.device:
            return np.convolve(signal, kernel, mode='same')
        
        # Use PyTorch on Metal
        signal_tensor = self.to_device(torch.from_numpy(signal).float().unsqueeze(0).unsqueeze(0))
        kernel_tensor = self.to_device(torch.from_numpy(kernel).float().unsqueeze(0).unsqueeze(0))
        
        # Pad signal
        padding = len(kernel) // 2
        result = torch.nn.functional.conv1d(signal_tensor, kernel_tensor, padding=padding)
        
        return result.squeeze().cpu().numpy()

class UnifiedAccelerator:
    """
    Unified interface for all acceleration methods
    """
    
    def __init__(self, config: MacOSAcceleration = None):
        self.config = config or OPTIMIZATION_CONFIG.macos
        
        # Initialize accelerators
        self.coreml = None
        self.metal = None
        
        if platform.system() == "Darwin":
            if self.config.use_coreml and COREML_AVAILABLE:
                self.coreml = CoreMLAccelerator(config)
            
            if self.config.use_metal and (METAL_AVAILABLE or TORCH_METAL_AVAILABLE):
                self.metal = MetalAccelerator(config)
        
        self._log_capabilities()
    
    def _log_capabilities(self):
        """Log available acceleration capabilities"""
        caps = []
        
        if self.coreml:
            caps.append("Core ML")
            if self.coreml.info.has_neural_engine:
                caps.append("Neural Engine")
        
        if self.metal:
            caps.append("Metal GPU")
        
        if caps:
            logger.info(f"Acceleration available: {', '.join(caps)}")
        else:
            logger.info("No hardware acceleration available")
    
    def accelerate_model(self, model: Any, model_name: str, 
                        input_shape: Tuple[int, ...]) -> Any:
        """
        Accelerate a model using best available method
        """
        # Try Core ML first (best for inference)
        if self.coreml and self.config.use_coreml:
            coreml_path = self.coreml.convert_model_to_coreml(
                model, model_name, input_shape
            )
            
            if coreml_path and self.coreml.load_coreml_model(coreml_path, model_name):
                logger.info(f"Model {model_name} accelerated with Core ML")
                return lambda x: self.coreml.predict(model_name, x)
        
        # Fall back to Metal/MPS
        if self.metal and hasattr(model, 'to'):
            model = model.to(self.metal.device)
            logger.info(f"Model {model_name} accelerated with Metal")
            return model
        
        # No acceleration available
        return model
    
    def accelerate_audio_processing(self, audio_data: np.ndarray, 
                                  operation: str = "fft") -> np.ndarray:
        """
        Accelerate audio processing operations
        """
        if self.metal:
            if operation == "fft":
                return self.metal.accelerated_fft(audio_data)
            elif operation == "stft":
                # Implement accelerated STFT
                window_size = 2048
                hop_size = 512
                # Use Metal-accelerated FFT in loop
                # ... implementation ...
        
        # Fallback to CPU
        if operation == "fft":
            return np.fft.fft(audio_data)
        else:
            return audio_data

# Optimized audio feature extraction using acceleration
class AcceleratedFeatureExtractor:
    """
    Hardware-accelerated audio feature extraction
    """
    
    def __init__(self, accelerator: UnifiedAccelerator = None):
        self.accelerator = accelerator or UnifiedAccelerator()
    
    def extract_mfcc(self, audio: np.ndarray, sr: int = 16000, 
                    n_mfcc: int = 13) -> np.ndarray:
        """Extract MFCC features with acceleration"""
        # Use accelerated FFT
        stft = self.accelerator.accelerate_audio_processing(audio, "fft")
        
        # Mel filterbank (could also be accelerated)
        # ... mel filterbank computation ...
        
        # DCT (could use Accelerate framework)
        # ... DCT computation ...
        
        return np.zeros((n_mfcc, 100))  # Placeholder
    
    def extract_spectral_features(self, audio: np.ndarray, 
                                 sr: int = 16000) -> Dict[str, float]:
        """Extract spectral features with acceleration"""
        # Accelerated FFT
        fft = self.accelerator.accelerate_audio_processing(audio, "fft")
        magnitude = np.abs(fft)
        
        features = {
            "spectral_centroid": np.sum(magnitude * np.arange(len(magnitude))) / np.sum(magnitude),
            "spectral_bandwidth": np.sqrt(np.sum(magnitude * (np.arange(len(magnitude)) ** 2)) / np.sum(magnitude)),
            "spectral_rolloff": np.argmax(np.cumsum(magnitude) > 0.85 * np.sum(magnitude))
        }
        
        return features

# Example usage
def benchmark_acceleration():
    """Benchmark acceleration performance"""
    accelerator = UnifiedAccelerator()
    
    # Test data
    audio = np.random.randn(16000)  # 1 second of audio
    
    # Benchmark FFT
    import time
    
    # CPU baseline
    start = time.time()
    for _ in range(100):
        np.fft.fft(audio)
    cpu_time = time.time() - start
    
    # Accelerated
    start = time.time()
    for _ in range(100):
        accelerator.accelerate_audio_processing(audio, "fft")
    accel_time = time.time() - start
    
    print(f"FFT Benchmark (100 iterations):")
    print(f"  CPU: {cpu_time:.3f}s")
    print(f"  Accelerated: {accel_time:.3f}s")
    print(f"  Speedup: {cpu_time/accel_time:.2f}x")

if __name__ == "__main__":
    if platform.system() == "Darwin":
        benchmark_acceleration()
    else:
        print("Acceleration only available on macOS")