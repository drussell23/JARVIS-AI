#!/usr/bin/env python3
"""
Intelligent VAD Auto-Initialization System
Automatically prepares optimal VAD models based on UI mode and platform

This module is called during JARVIS startup and intelligently:
- Detects UI mode (HUD vs web-app)
- Determines platform capabilities
- Downloads/converts models as needed
- Caches everything for fast subsequent startups
- Zero manual intervention required
"""

import asyncio
import logging
import os
import platform
from pathlib import Path
from typing import Optional, Dict, Any
import sys

logger = logging.getLogger(__name__)


class VADAutoInitializer:
    """
    Intelligent VAD initialization that happens automatically during startup

    Workflow:
    1. Detect UI mode from startup args
    2. Detect platform capabilities (MPS, CoreML, CUDA, etc.)
    3. Check what models are already available (cache)
    4. Download/convert only what's needed
    5. Report readiness to main system
    """

    def __init__(self, ui_mode: str = "macos"):
        self.ui_mode = ui_mode  # "macos", "webapp", "headless"
        self.platform = platform.system().lower()
        self.is_apple_silicon = platform.processor() == "arm"
        self.models_dir = Path.home() / ".jarvis" / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Capability detection
        self.has_mps = False
        self.has_coreml_tools = False
        self.has_cuda = False
        self.has_torch = False

        logger.info("=" * 80)
        logger.info("🎯 VAD Auto-Initialization System")
        logger.info("=" * 80)
        logger.info(f"UI Mode: {ui_mode}")
        logger.info(f"Platform: {self.platform}")
        logger.info(f"Apple Silicon: {self.is_apple_silicon}")
        logger.info(f"Models Directory: {self.models_dir}")

    async def detect_capabilities(self) -> Dict[str, bool]:
        """Detect what acceleration methods are available"""
        logger.info("\n🔍 Detecting platform capabilities...")

        # Check PyTorch + MPS
        try:
            import torch
            self.has_torch = True
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.has_mps = True
                logger.info("✅ PyTorch + MPS (Apple Neural Engine): Available")
            else:
                logger.info("✅ PyTorch: Available (CPU mode)")
        except ImportError:
            logger.warning("⚠️  PyTorch not installed")

        # Check CoreML Tools
        try:
            import coremltools
            self.has_coreml_tools = True
            logger.info("✅ CoreML Tools: Available")
        except ImportError:
            logger.info("ℹ️  CoreML Tools not installed (not needed for HUD mode)")

        # Check CUDA
        try:
            import torch
            if torch.cuda.is_available():
                self.has_cuda = True
                logger.info("✅ CUDA: Available")
        except:
            pass

        return {
            "mps": self.has_mps,
            "coreml_tools": self.has_coreml_tools,
            "cuda": self.has_cuda,
            "torch": self.has_torch
        }

    def determine_required_models(self) -> Dict[str, bool]:
        """Determine which models are needed based on UI mode"""
        logger.info(f"\n🎯 Determining required models for '{self.ui_mode}' mode...")

        needs = {
            "pytorch_mps": False,
            "coreml": False,
            "onnx": False
        }

        if self.ui_mode == "macos":  # HUD mode
            # HUD uses MPS (PyTorch + Metal)
            if self.has_mps:
                needs["pytorch_mps"] = True
                logger.info("   → PyTorch + MPS model (for HUD)")
            else:
                needs["pytorch_mps"] = True  # CPU fallback
                logger.info("   → PyTorch CPU model (MPS not available)")

        elif self.ui_mode == "webapp":  # Web-app mode
            # Web-app prefers CoreML on macOS, ONNX elsewhere
            if self.platform == "darwin" and self.has_coreml_tools:
                needs["coreml"] = True
                logger.info("   → CoreML model (for web-app on macOS)")
            else:
                needs["onnx"] = True
                logger.info("   → ONNX model (for web-app)")

        else:  # Headless or other
            # Use best available
            if self.has_mps:
                needs["pytorch_mps"] = True
            elif self.has_cuda:
                needs["pytorch_mps"] = True  # Same PyTorch model, different device
            else:
                needs["onnx"] = True

        return needs

    async def check_model_cache(self, model_type: str) -> bool:
        """Check if a model is already cached"""
        if model_type == "pytorch_mps":
            # PyTorch models are cached by torch.hub
            cache_dir = Path.home() / ".cache" / "torch" / "hub" / "snakers4_silero-vad_master"
            exists = cache_dir.exists()
            if exists:
                logger.info(f"   ✅ PyTorch model: Cached at {cache_dir}")
            return exists

        elif model_type == "coreml":
            model_path = self.models_dir / "vad_model.mlpackage"
            exists = model_path.exists()
            if exists:
                logger.info(f"   ✅ CoreML model: Cached at {model_path}")
            return exists

        elif model_type == "onnx":
            # ONNX model check
            cache_dir = Path.home() / ".cache" / "torch" / "hub" / "snakers4_silero-vad_master"
            exists = cache_dir.exists()
            if exists:
                logger.info(f"   ✅ ONNX model: Cached at {cache_dir}")
            return exists

        return False

    async def download_pytorch_model(self) -> bool:
        """Download Silero VAD PyTorch model via torch.hub"""
        logger.info("\n📥 Downloading Silero VAD PyTorch model...")
        logger.info("   Repository: snakers4/silero-vad")
        logger.info("   This happens once and is cached permanently")

        def _download():
            import torch
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            return model

        try:
            await asyncio.to_thread(_download)
            logger.info("✅ PyTorch model downloaded and cached")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to download PyTorch model: {e}")
            return False

    async def convert_to_coreml(self) -> bool:
        """Convert PyTorch model to CoreML for web-app"""
        logger.info("\n🔄 Converting Silero VAD to CoreML...")
        logger.info("   This happens once and is cached permanently")
        logger.info("   Typical conversion time: 30-60 seconds")

        # Import the converter
        sys.path.insert(0, str(Path(__file__).parent.parent / "coreml"))

        try:
            from convert_jit_to_coreml import CoreMLConverter

            converter = CoreMLConverter(
                output_dir=str(self.models_dir),
                force=False
            )

            success = await converter.convert()

            if success:
                logger.info("✅ CoreML model converted and cached")
            else:
                logger.error("❌ CoreML conversion failed")

            return success

        except Exception as e:
            logger.error(f"❌ CoreML conversion error: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def download_onnx_model(self) -> bool:
        """Download ONNX version of Silero VAD"""
        logger.info("\n📥 Downloading Silero VAD ONNX model...")

        def _download():
            import torch
            model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=True
            )
            return model

        try:
            await asyncio.to_thread(_download)
            logger.info("✅ ONNX model downloaded and cached")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to download ONNX model: {e}")
            return False

    async def initialize(self) -> bool:
        """
        Main initialization workflow - called automatically during startup

        Returns:
            True if initialization succeeded, False otherwise
        """
        try:
            # Step 1: Detect capabilities
            capabilities = await self.detect_capabilities()

            if not capabilities.get("torch"):
                logger.error("❌ PyTorch not available - VAD cannot function")
                logger.error("   Install: pip install torch")
                return False

            # Step 2: Determine what models we need
            needs = self.determine_required_models()

            # Step 3: Check cache and download/convert as needed
            logger.info("\n🔍 Checking model cache...")

            all_ready = True

            if needs.get("pytorch_mps"):
                if not await self.check_model_cache("pytorch_mps"):
                    logger.info("   ⬇️  PyTorch model not cached, downloading...")
                    if not await self.download_pytorch_model():
                        all_ready = False

            if needs.get("coreml"):
                if not await self.check_model_cache("coreml"):
                    logger.info("   🔄 CoreML model not cached, converting...")
                    if not await self.convert_to_coreml():
                        logger.warning("⚠️  CoreML conversion failed, will fall back to MPS")
                        # Not critical - can fall back to MPS

            if needs.get("onnx"):
                if not await self.check_model_cache("onnx"):
                    logger.info("   ⬇️  ONNX model not cached, downloading...")
                    if not await self.download_onnx_model():
                        all_ready = False

            # Step 4: Report status
            logger.info("\n" + "=" * 80)
            if all_ready:
                logger.info("✅ VAD SYSTEM READY")
                logger.info("=" * 80)
                logger.info(f"Mode: {self.ui_mode}")
                logger.info(f"Acceleration: {self._get_acceleration_description()}")
                logger.info(f"All required models cached and ready")
                logger.info("=" * 80)
            else:
                logger.warning("⚠️  VAD SYSTEM PARTIALLY READY")
                logger.warning("Some models failed to initialize - check logs above")

            return all_ready

        except Exception as e:
            logger.error(f"❌ VAD initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _get_acceleration_description(self) -> str:
        """Get human-readable description of acceleration being used"""
        if self.ui_mode == "macos":
            if self.has_mps:
                return "MPS (Apple Neural Engine)"
            else:
                return "CPU (MPS not available)"
        elif self.ui_mode == "webapp":
            if self.platform == "darwin" and self.has_coreml_tools:
                return "CoreML (Apple Neural Engine)"
            else:
                return "ONNX Runtime"
        else:
            if self.has_mps:
                return "MPS (Apple Neural Engine)"
            elif self.has_cuda:
                return "CUDA (NVIDIA GPU)"
            else:
                return "CPU"


# Convenience function for startup integration
async def auto_initialize_vad(ui_mode: str = "macos") -> bool:
    """
    Auto-initialize VAD system - called by start_system.py

    Args:
        ui_mode: "macos" for HUD, "webapp" for web-app, "headless" for no UI

    Returns:
        True if initialization succeeded

    Usage in start_system.py:
        from backend.voice.vad.vad_auto_init import auto_initialize_vad

        # During startup, before launching backend:
        vad_ready = await auto_initialize_vad(ui_mode=args.ui)
        if not vad_ready:
            logger.warning("VAD initialization incomplete - some features may be limited")
    """
    initializer = VADAutoInitializer(ui_mode=ui_mode)
    return await initializer.initialize()


# For testing/manual runs
if __name__ == "__main__":
    import sys

    # Parse UI mode from args
    ui_mode = sys.argv[1] if len(sys.argv) > 1 else "macos"

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )

    # Run initialization
    success = asyncio.run(auto_initialize_vad(ui_mode=ui_mode))

    sys.exit(0 if success else 1)
