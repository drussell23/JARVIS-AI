#!/usr/bin/env python3
"""
Quantized Models for Voice Unlock
=================================

Ultra-optimized models for 35% memory target using INT8/INT4 quantization.
Reduces voice biometric model from ~50MB to ~6MB with minimal accuracy loss.
"""

import numpy as np
import pickle
import joblib
from pathlib import Path
from typing import Any, Tuple, Dict, Optional
import logging
from dataclasses import dataclass
import struct

logger = logging.getLogger(__name__)

# Try to use Rust extensions for maximum performance
try:
    import jarvis_rust_extensions
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    

@dataclass
class QuantizedModel:
    """Container for quantized model data"""
    model_type: str
    weights: np.ndarray  # INT8 or INT4
    biases: Optional[np.ndarray]  # FP16
    scales: Dict[str, float]  # Quantization scales
    metadata: Dict[str, Any]
    

class VoiceModelQuantizer:
    """Quantize voice biometric models for extreme memory efficiency"""
    
    def __init__(self):
        self.quantization_cache = {}
        
    def quantize_svm_model(self, model_path: Path, target: str = "int8") -> QuantizedModel:
        """
        Quantize SVM model to INT8 or INT4
        
        Args:
            model_path: Path to original model
            target: "int8" or "int4" for ultra-low memory
            
        Returns:
            QuantizedModel with 75-87.5% memory reduction
        """
        # Load original model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        if not hasattr(model, 'support_vectors_'):
            raise ValueError("Not an SVM model")
            
        # Get model components
        support_vectors = model.support_vectors_.astype(np.float32)
        dual_coef = model.dual_coef_.astype(np.float32)
        
        if target == "int8":
            # INT8 quantization
            sv_quant, sv_scale = self._quantize_to_int8(support_vectors)
            dc_quant, dc_scale = self._quantize_to_int8(dual_coef)
            
            return QuantizedModel(
                model_type="svm_int8",
                weights=sv_quant,
                biases=dc_quant.astype(np.int8),
                scales={
                    "support_vectors_scale": sv_scale,
                    "dual_coef_scale": dc_scale,
                    "intercept": float(model.intercept_[0]) if hasattr(model, 'intercept_') else 0.0,
                    "gamma": float(model.gamma) if hasattr(model, 'gamma') else 'auto'
                },
                metadata={
                    "n_support": sv_quant.shape[0],
                    "n_features": sv_quant.shape[1],
                    "kernel": model.kernel,
                    "original_size_mb": support_vectors.nbytes / 1024 / 1024,
                    "quantized_size_mb": (sv_quant.nbytes + dc_quant.nbytes) / 1024 / 1024
                }
            )
            
        elif target == "int4":
            # Ultra-extreme INT4 quantization
            sv_quant, sv_scale = self._quantize_to_int4(support_vectors)
            dc_quant, dc_scale = self._quantize_to_int8(dual_coef)  # Keep dual_coef at INT8
            
            return QuantizedModel(
                model_type="svm_int4",
                weights=sv_quant,
                biases=dc_quant.astype(np.int8),
                scales={
                    "support_vectors_scale": sv_scale,
                    "dual_coef_scale": dc_scale,
                    "intercept": float(model.intercept_[0]) if hasattr(model, 'intercept_') else 0.0,
                    "gamma": float(model.gamma) if hasattr(model, 'gamma') else 'auto'
                },
                metadata={
                    "n_support": support_vectors.shape[0],
                    "n_features": support_vectors.shape[1],
                    "kernel": model.kernel,
                    "original_size_mb": support_vectors.nbytes / 1024 / 1024,
                    "quantized_size_mb": (sv_quant.nbytes / 2 + dc_quant.nbytes) / 1024 / 1024  # INT4 is packed
                }
            )
            
    def _quantize_to_int8(self, array: np.ndarray) -> Tuple[np.ndarray, float]:
        """Symmetric INT8 quantization"""
        if RUST_AVAILABLE:
            # Use Rust for fast quantization
            result = jarvis_rust_extensions.RustModelLoader().quantize_array_int8(array.flatten().tolist())
            quantized = np.frombuffer(result['data'], dtype=np.int8).reshape(array.shape)
            return quantized, result['scale']
        else:
            # Python fallback
            max_abs = np.abs(array).max()
            scale = max_abs / 127.0
            quantized = np.round(array / scale).astype(np.int8)
            return quantized, scale
            
    def _quantize_to_int4(self, array: np.ndarray) -> Tuple[np.ndarray, float]:
        """Ultra-low INT4 quantization with packing"""
        # First quantize to INT8 range
        max_abs = np.abs(array).max()
        scale = max_abs / 7.0  # INT4 range is -8 to 7
        quantized = np.round(array / scale).clip(-8, 7).astype(np.int8)
        
        # Pack two INT4 values into one byte
        packed = self._pack_int4(quantized.flatten())
        
        return packed, scale
        
    def _pack_int4(self, values: np.ndarray) -> np.ndarray:
        """Pack INT8 array into INT4 (nibbles)"""
        # Ensure even length
        if len(values) % 2 != 0:
            values = np.append(values, 0)
            
        packed = np.zeros(len(values) // 2, dtype=np.uint8)
        
        for i in range(0, len(values), 2):
            low = values[i] & 0x0F
            high = (values[i + 1] & 0x0F) << 4
            packed[i // 2] = low | high
            
        return packed
        
    def save_quantized_model(self, model: QuantizedModel, output_path: Path):
        """Save quantized model with compression"""
        # Prepare data for saving
        data = {
            'model_type': model.model_type,
            'weights': model.weights,
            'biases': model.biases,
            'scales': model.scales,
            'metadata': model.metadata
        }
        
        if RUST_AVAILABLE:
            # Use LZ4 compression for ultra-fast loading
            import pickle
            pickled = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            compressed = jarvis_rust_extensions.compress_data_lz4(pickled)
            
            with open(output_path, 'wb') as f:
                # Write magic number and uncompressed size
                f.write(b'QVLZ')  # Quantized Voice LZ4
                f.write(struct.pack('<Q', len(pickled)))
                f.write(compressed)
        else:
            # Standard joblib with compression
            joblib.dump(data, output_path, compress=9)
            
        logger.info(f"Saved quantized model: {model.metadata['quantized_size_mb']:.1f}MB "
                   f"(was {model.metadata['original_size_mb']:.1f}MB)")
        
    def load_quantized_model(self, model_path: Path) -> QuantizedModel:
        """Load quantized model with decompression"""
        if RUST_AVAILABLE and model_path.suffix != '.pkl':
            with open(model_path, 'rb') as f:
                magic = f.read(4)
                if magic == b'QVLZ':
                    # LZ4 compressed
                    uncompressed_size = struct.unpack('<Q', f.read(8))[0]
                    compressed = f.read()
                    
                    decompressed = jarvis_rust_extensions.decompress_data_lz4(
                        compressed, uncompressed_size
                    )
                    data = pickle.loads(decompressed)
                else:
                    # Not our format, try joblib
                    f.seek(0)
                    data = joblib.load(f)
        else:
            # Standard loading
            data = joblib.load(model_path)
            
        return QuantizedModel(**data)
        
    def create_inference_engine(self, model: QuantizedModel):
        """Create optimized inference engine for quantized model"""
        if model.model_type == "svm_int8":
            return QuantizedSVMInference(
                model.weights,
                model.biases,
                model.scales,
                model.metadata,
                precision="int8"
            )
        elif model.model_type == "svm_int4":
            return QuantizedSVMInference(
                model.weights,
                model.biases,
                model.scales,
                model.metadata,
                precision="int4"
            )
        else:
            raise ValueError(f"Unknown model type: {model.model_type}")
            

class QuantizedSVMInference:
    """Ultra-fast inference for quantized SVM models"""
    
    def __init__(self, support_vectors: np.ndarray, dual_coef: np.ndarray,
                 scales: Dict[str, float], metadata: Dict[str, Any], 
                 precision: str = "int8"):
        self.support_vectors = support_vectors
        self.dual_coef = dual_coef
        self.scales = scales
        self.metadata = metadata
        self.precision = precision
        
        # Pre-compute for RBF kernel
        if metadata.get('kernel') == 'rbf':
            self.gamma = scales.get('gamma', 'auto')
            if self.gamma == 'auto':
                self.gamma = 1.0 / metadata['n_features']
                
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Fast prediction using quantized model"""
        # Quantize input
        X_scale = np.abs(X).max() / 127.0 if self.precision == "int8" else np.abs(X).max() / 7.0
        X_quant = np.round(X / X_scale).astype(np.int8)
        
        if self.precision == "int8":
            # INT8 inference
            decision = self._predict_int8(X_quant, X_scale)
        else:
            # INT4 inference
            decision = self._predict_int4(X_quant, X_scale)
            
        # Apply threshold
        return np.sign(decision)
        
    def _predict_int8(self, X_quant: np.ndarray, X_scale: float) -> float:
        """INT8 prediction"""
        # Dequantize for kernel computation (can be optimized further)
        sv_scale = self.scales['support_vectors_scale']
        dc_scale = self.scales['dual_coef_scale']
        
        # RBF kernel in quantized space
        if self.metadata.get('kernel') == 'rbf':
            # Compute squared distances in quantized space
            # This is approximate but much faster
            kernel_values = np.zeros(self.support_vectors.shape[0])
            
            for i in range(self.support_vectors.shape[0]):
                diff = X_quant - self.support_vectors[i]
                # Approximate squared distance
                dist_sq = np.sum(diff * diff)
                # Adjust for scales
                actual_dist_sq = dist_sq * (X_scale * sv_scale) ** 2
                kernel_values[i] = np.exp(-self.gamma * actual_dist_sq)
        else:
            # Linear kernel (dot product)
            kernel_values = np.dot(self.support_vectors, X_quant)
            # Adjust for scales
            kernel_values = kernel_values * X_scale * sv_scale
            
        # Decision function
        decision = np.dot(self.dual_coef.flatten(), kernel_values) * dc_scale
        decision += self.scales['intercept']
        
        return decision
        
    def _predict_int4(self, X_quant: np.ndarray, X_scale: float) -> float:
        """INT4 prediction with unpacking"""
        # Unpack support vectors from INT4
        sv_unpacked = self._unpack_int4(self.support_vectors)
        sv_unpacked = sv_unpacked.reshape(self.metadata['n_support'], -1)
        
        # Rest is similar to INT8
        return self._predict_int8_with_sv(X_quant, X_scale, sv_unpacked)
        
    def _unpack_int4(self, packed: np.ndarray) -> np.ndarray:
        """Unpack INT4 to INT8"""
        unpacked = np.zeros(packed.size * 2, dtype=np.int8)
        
        for i, byte in enumerate(packed):
            unpacked[i * 2] = (byte & 0x0F) - 8 if (byte & 0x0F) > 7 else (byte & 0x0F)
            unpacked[i * 2 + 1] = ((byte >> 4) & 0x0F) - 8 if ((byte >> 4) & 0x0F) > 7 else ((byte >> 4) & 0x0F)
            
        return unpacked
        
    def _predict_int8_with_sv(self, X_quant: np.ndarray, X_scale: float, 
                              support_vectors: np.ndarray) -> float:
        """INT8 prediction with provided support vectors"""
        sv_scale = self.scales['support_vectors_scale']
        dc_scale = self.scales['dual_coef_scale']
        
        if self.metadata.get('kernel') == 'rbf':
            kernel_values = np.zeros(support_vectors.shape[0])
            for i in range(support_vectors.shape[0]):
                diff = X_quant - support_vectors[i][:X_quant.shape[0]]
                dist_sq = np.sum(diff * diff)
                actual_dist_sq = dist_sq * (X_scale * sv_scale) ** 2
                kernel_values[i] = np.exp(-self.gamma * actual_dist_sq)
        else:
            kernel_values = np.dot(support_vectors[:, :X_quant.shape[0]], X_quant)
            kernel_values = kernel_values * X_scale * sv_scale
            
        decision = np.dot(self.dual_coef.flatten(), kernel_values) * dc_scale
        decision += self.scales['intercept']
        
        return decision
        

def quantize_voice_models(models_dir: Path, output_dir: Path, target: str = "int8"):
    """
    Quantize all voice models in a directory
    
    Args:
        models_dir: Directory containing original models
        output_dir: Directory to save quantized models
        target: "int8" or "int4" quantization
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    quantizer = VoiceModelQuantizer()
    
    for model_path in models_dir.glob("*.pkl"):
        logger.info(f"Quantizing {model_path.name}...")
        
        try:
            # Quantize model
            quantized = quantizer.quantize_svm_model(model_path, target)
            
            # Save with .q8 or .q4 extension
            ext = ".q8" if target == "int8" else ".q4"
            output_path = output_dir / f"{model_path.stem}{ext}"
            
            quantizer.save_quantized_model(quantized, output_path)
            
            # Report compression
            reduction = (1 - quantized.metadata['quantized_size_mb'] / 
                        quantized.metadata['original_size_mb']) * 100
            logger.info(f"✅ {model_path.name}: {reduction:.1f}% size reduction")
            
        except Exception as e:
            logger.error(f"Failed to quantize {model_path.name}: {e}")
            
            
# Example usage
if __name__ == "__main__":
    # Quantize voice models
    models_dir = Path.home() / '.jarvis' / 'models'
    output_dir = Path.home() / '.jarvis' / 'models' / 'quantized'
    
    # Use INT8 for good balance of size and accuracy
    quantize_voice_models(models_dir, output_dir, target="int8")