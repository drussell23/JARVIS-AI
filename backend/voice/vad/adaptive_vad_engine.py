#!/usr/bin/env python3
"""
Adaptive VAD Engine - Dynamic Multi-Platform Voice Activity Detection
Intelligently selects optimal acceleration based on platform, UI mode, and capabilities

Architecture:
- macOS HUD: MPS (Metal Performance Shaders) - Apple Neural Engine
- macOS Web: CoreML or MPS (user preference)
- Linux/Windows: ONNX or CPU fallback
- Async/await throughout for non-blocking operation
- Zero hardcoding - all configuration dynamic
"""

import asyncio
import logging
import platform
import os
from typing import Optional, Dict, Any, Literal
from dataclasses import dataclass
from enum import Enum
import torch
import numpy as np

logger = logging.getLogger(__name__)


class AccelerationType(Enum):
    """Available acceleration types"""
    MPS = "mps"  # Metal Performance Shaders (Apple Neural Engine)
    COREML = "coreml"  # CoreML (Apple Neural Engine via different API)
    ONNX = "onnx"  # ONNX Runtime (cross-platform)
    CUDA = "cuda"  # NVIDIA CUDA
    CPU = "cpu"  # CPU fallback


class UIMode(Enum):
    """UI modes for JARVIS"""
    HUD = "macos"  # macOS native HUD
    WEBAPP = "webapp"  # Web application
    HEADLESS = "headless"  # No UI


@dataclass
class PlatformCapabilities:
    """Detected platform capabilities"""
    os_type: str  # darwin, linux, windows
    has_mps: bool = False
    has_coreml: bool = False
    has_cuda: bool = False
    has_onnx: bool = False
    cpu_count: int = 1
    total_memory_gb: float = 0.0
    is_apple_silicon: bool = False


@dataclass
class VADConfig:
    """VAD configuration - no hardcoding"""
    acceleration: AccelerationType
    model_path: Optional[str] = None
    sample_rate: int = 16000
    chunk_size: int = 512
    threshold: float = 0.5
    warmup_iterations: int = 10

    @classmethod
    def from_env(cls, ui_mode: UIMode, capabilities: PlatformCapabilities) -> "VADConfig":
        """Create config from environment and capabilities"""
        # Select best acceleration method
        acceleration = cls._select_acceleration(ui_mode, capabilities)

        # Dynamic threshold based on environment variable or default
        threshold = float(os.getenv("JARVIS_VAD_THRESHOLD", "0.5"))

        # Dynamic sample rate
        sample_rate = int(os.getenv("JARVIS_SAMPLE_RATE", "16000"))

        return cls(
            acceleration=acceleration,
            sample_rate=sample_rate,
            threshold=threshold
        )

    @staticmethod
    def _select_acceleration(ui_mode: UIMode, caps: PlatformCapabilities) -> AccelerationType:
        """Intelligently select best acceleration method"""
        # HUD mode on Apple Silicon: Prefer MPS for lowest latency
        if ui_mode == UIMode.HUD and caps.has_mps:
            return AccelerationType.MPS

        # Web-app mode on macOS: Prefer CoreML (can run in browser worker)
        # But fall back to MPS if CoreML unavailable
        if ui_mode == UIMode.WEBAPP and caps.os_type == "darwin":
            if caps.has_coreml:
                return AccelerationType.COREML
            elif caps.has_mps:
                return AccelerationType.MPS

        # CUDA on Linux/Windows if available
        if caps.has_cuda:
            return AccelerationType.CUDA

        # ONNX as cross-platform fallback
        if caps.has_onnx:
            return AccelerationType.ONNX

        # CPU fallback
        return AccelerationType.CPU


class AdaptiveVADEngine:
    """
    Dynamic VAD engine that adapts to platform and UI mode

    Features:
    - Automatic acceleration selection
    - Async operation for non-blocking inference
    - Lazy model loading
    - Automatic warmup
    - Performance metrics
    """

    def __init__(self, ui_mode: UIMode = UIMode.HUD):
        self.ui_mode = ui_mode
        self.capabilities = self._detect_capabilities()
        self.config = VADConfig.from_env(ui_mode, self.capabilities)
        self.model = None
        self.is_initialized = False

        logger.info("=" * 80)
        logger.info("🎯 Adaptive VAD Engine Initializing")
        logger.info("=" * 80)
        logger.info(f"UI Mode: {ui_mode.value}")
        logger.info(f"Platform: {self.capabilities.os_type}")
        logger.info(f"Selected Acceleration: {self.config.acceleration.value}")
        logger.info(f"Apple Silicon: {self.capabilities.is_apple_silicon}")
        logger.info(f"MPS Available: {self.capabilities.has_mps}")
        logger.info(f"CoreML Available: {self.capabilities.has_coreml}")
        logger.info(f"CUDA Available: {self.capabilities.has_cuda}")
        logger.info("=" * 80)

    @staticmethod
    def _detect_capabilities() -> PlatformCapabilities:
        """Detect platform capabilities dynamically"""
        import psutil

        caps = PlatformCapabilities(
            os_type=platform.system().lower(),
            cpu_count=psutil.cpu_count(),
            total_memory_gb=psutil.virtual_memory().total / (1024**3)
        )

        # Detect Apple Silicon
        if caps.os_type == "darwin":
            caps.is_apple_silicon = platform.processor() == "arm"

        # Detect MPS (Metal Performance Shaders)
        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                caps.has_mps = True
        except ImportError:
            pass

        # Detect CoreML
        if caps.os_type == "darwin":
            try:
                import coremltools
                caps.has_coreml = True
            except ImportError:
                pass

        # Detect CUDA
        try:
            import torch
            if torch.cuda.is_available():
                caps.has_cuda = True
        except ImportError:
            pass

        # Detect ONNX Runtime
        try:
            import onnxruntime
            caps.has_onnx = True
        except ImportError:
            pass

        return caps

    async def initialize(self) -> None:
        """Async initialization of VAD model"""
        if self.is_initialized:
            return

        logger.info("🔄 Loading VAD model asynchronously...")

        # Load model based on selected acceleration
        if self.config.acceleration == AccelerationType.MPS:
            await self._load_mps_model()
        elif self.config.acceleration == AccelerationType.COREML:
            await self._load_coreml_model()
        elif self.config.acceleration == AccelerationType.ONNX:
            await self._load_onnx_model()
        elif self.config.acceleration == AccelerationType.CUDA:
            await self._load_cuda_model()
        else:
            await self._load_cpu_model()

        # Warmup
        await self._warmup()

        self.is_initialized = True
        logger.info("✅ VAD engine initialized and ready")

    async def _load_mps_model(self) -> None:
        """Load Silero VAD with MPS acceleration"""
        logger.info("⚡ Loading Silero VAD with MPS (Apple Neural Engine)...")

        def _load():
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            model = model.to('mps')
            model.eval()
            return model, utils

        # Run in thread pool to avoid blocking
        self.model, self.utils = await asyncio.to_thread(_load)
        logger.info("✅ MPS model loaded")

    async def _load_coreml_model(self) -> None:
        """Load CoreML VAD model"""
        logger.info("🍎 Loading CoreML VAD model...")

        # Check if CoreML model exists, if not convert it
        coreml_path = os.path.expanduser("~/.jarvis/models/vad_model.mlpackage")

        if not os.path.exists(coreml_path):
            logger.info("CoreML model not found, converting from PyTorch...")
            await self._convert_to_coreml(coreml_path)

        def _load():
            import coremltools as ct
            return ct.models.MLModel(coreml_path)

        self.model = await asyncio.to_thread(_load)
        logger.info("✅ CoreML model loaded")

    async def _convert_to_coreml(self, output_path: str) -> None:
        """Convert Silero VAD to CoreML asynchronously"""
        logger.info("🔄 Converting Silero VAD to CoreML...")

        def _convert():
            import torch
            import coremltools as ct

            # Load PyTorch model
            model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            model.eval()

            # Trace model
            example_input = torch.randn(1, 512)
            traced_model = torch.jit.trace(model, example_input)

            # Convert to CoreML
            coreml_model = ct.convert(
                traced_model,
                inputs=[ct.TensorType(name="audio", shape=(1, 512))],
                compute_units=ct.ComputeUnit.ALL,
                minimum_deployment_target=ct.target.macOS12
            )

            # Save
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            coreml_model.save(output_path)

            return coreml_model

        self.model = await asyncio.to_thread(_convert)
        logger.info(f"✅ CoreML model converted and saved: {output_path}")

    async def _load_onnx_model(self) -> None:
        """Load ONNX VAD model"""
        logger.info("🔧 Loading ONNX VAD model...")

        def _load():
            import onnxruntime
            # Load Silero VAD ONNX model
            model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=True
            )
            return model

        self.model = await asyncio.to_thread(_load)
        logger.info("✅ ONNX model loaded")

    async def _load_cuda_model(self) -> None:
        """Load model with CUDA acceleration"""
        logger.info("🚀 Loading Silero VAD with CUDA...")

        def _load():
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            model = model.to('cuda')
            model.eval()
            return model, utils

        self.model, self.utils = await asyncio.to_thread(_load)
        logger.info("✅ CUDA model loaded")

    async def _load_cpu_model(self) -> None:
        """Load model with CPU fallback"""
        logger.info("💻 Loading Silero VAD with CPU...")

        def _load():
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            model.eval()
            return model, utils

        self.model, self.utils = await asyncio.to_thread(_load)
        logger.info("✅ CPU model loaded")

    async def _warmup(self) -> None:
        """Warmup model for consistent latency"""
        logger.info(f"🔥 Warming up VAD model ({self.config.warmup_iterations} iterations)...")

        dummy_audio = np.random.randn(self.config.chunk_size).astype(np.float32)

        for _ in range(self.config.warmup_iterations):
            await self.detect_voice(dummy_audio)

        logger.info("✅ Warmup complete")

    async def detect_voice(self, audio: np.ndarray) -> tuple[bool, float]:
        """
        Async voice detection

        Args:
            audio: Audio samples (float32, normalized)

        Returns:
            (is_voice, confidence) tuple
        """
        if not self.is_initialized:
            await self.initialize()

        def _infer():
            if self.config.acceleration == AccelerationType.MPS:
                return self._infer_mps(audio)
            elif self.config.acceleration == AccelerationType.COREML:
                return self._infer_coreml(audio)
            elif self.config.acceleration == AccelerationType.CUDA:
                return self._infer_cuda(audio)
            else:
                return self._infer_cpu(audio)

        return await asyncio.to_thread(_infer)

    def _infer_mps(self, audio: np.ndarray) -> tuple[bool, float]:
        """MPS inference"""
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to('mps')
        with torch.no_grad():
            confidence = self.model(audio_tensor, self.config.sample_rate).item()
        return confidence >= self.config.threshold, confidence

    def _infer_coreml(self, audio: np.ndarray) -> tuple[bool, float]:
        """CoreML inference"""
        # CoreML inference implementation
        # This would use coremltools API
        import coremltools as ct
        audio_input = {"audio": audio.reshape(1, -1)}
        prediction = self.model.predict(audio_input)
        confidence = float(prediction["output"])
        return confidence >= self.config.threshold, confidence

    def _infer_cuda(self, audio: np.ndarray) -> tuple[bool, float]:
        """CUDA inference"""
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to('cuda')
        with torch.no_grad():
            confidence = self.model(audio_tensor, self.config.sample_rate).item()
        return confidence >= self.config.threshold, confidence

    def _infer_cpu(self, audio: np.ndarray) -> tuple[bool, float]:
        """CPU inference"""
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
        with torch.no_grad():
            confidence = self.model(audio_tensor, self.config.sample_rate).item()
        return confidence >= self.config.threshold, confidence

    async def shutdown(self) -> None:
        """Graceful shutdown"""
        logger.info("🛑 Shutting down Adaptive VAD Engine...")
        self.model = None
        self.is_initialized = False
        logger.info("✅ Shutdown complete")


# Factory function for easy instantiation
async def create_vad_engine(ui_mode: str = "macos") -> AdaptiveVADEngine:
    """
    Create and initialize VAD engine

    Args:
        ui_mode: "macos" for HUD, "webapp" for web-app, "headless" for no UI

    Returns:
        Initialized VAD engine
    """
    mode_map = {
        "macos": UIMode.HUD,
        "webapp": UIMode.WEBAPP,
        "headless": UIMode.HEADLESS
    }

    mode = mode_map.get(ui_mode, UIMode.HUD)
    engine = AdaptiveVADEngine(ui_mode=mode)
    await engine.initialize()
    return engine
