"""
Optimized Claude Vision Analyzer - Faster response times through image optimization
"""

import base64
import io
from typing import Dict, Any, Optional
from PIL import Image
import numpy as np
from anthropic import Anthropic
import json
import time
import logging

logger = logging.getLogger(__name__)


class OptimizedClaudeVisionAnalyzer:
    """Optimized Claude vision analyzer with faster response times"""

    def __init__(self, api_key: str, use_intelligent_selection: bool = True):
        """Initialize Claude vision analyzer"""
        self.client = Anthropic(api_key=api_key)
        self.model = "claude-3-5-sonnet-20241022"
        self.use_intelligent_selection = use_intelligent_selection

        # Optimization settings
        self.max_image_size = (1280, 720)  # Reduced from full resolution
        self.jpeg_quality = 60  # Lower quality for faster upload
        self.max_file_size = 500 * 1024  # 500KB max

    async def _analyze_screenshot_fast_with_intelligent_selection(
        self, image_base64: str, image_data: bytes, optimized_prompt: str, start_time: float
    ) -> Dict[str, Any]:
        """Analyze screenshot using intelligent model selection"""
        try:
            from backend.core.hybrid_orchestrator import HybridOrchestrator

            orchestrator = HybridOrchestrator()
            if not orchestrator.is_running:
                await orchestrator.start()

            # Build rich context
            context = {
                "task_type": "vision_analysis",
                "image_size_kb": len(image_data) / 1024,
                "optimization_level": "fast",
                "compression_quality": self.jpeg_quality,
                "max_image_size": self.max_image_size,
                "performance_target": "low_latency",
            }

            # Execute with intelligent selection
            # Note: Vision requires special handling - we pass image data separately
            api_start = time.time()

            result = await orchestrator.execute_with_intelligent_model_selection(
                query=optimized_prompt,
                intent="vision_analysis",
                required_capabilities={"vision", "vision_analyze_heavy", "multimodal"},
                context=context,
                max_tokens=512,
                temperature=0.3,
                image_data=image_base64,  # Pass image data
            )

            if not result.get("success"):
                raise Exception(result.get("error", "Unknown error"))

            response_text = result.get("text", "").strip()
            model_used = result.get("model_used", "intelligent_selection")

            logger.info(f"✨ Vision analysis using {model_used}")
            logger.info(f"Claude API call took {time.time() - api_start:.2f}s")

            return {
                "description": response_text,
                "response_time": time.time() - start_time,
                "image_size_kb": len(image_data) / 1024,
                "model_used": model_used,
            }

        except ImportError:
            logger.warning("Hybrid orchestrator not available, falling back to direct API")
            raise
        except Exception as e:
            logger.error(f"Error in intelligent selection: {e}")
            raise

    async def analyze_screenshot_fast(self, image: Any, prompt: str) -> Dict[str, Any]:
        """Fast screenshot analysis with optimizations

        Args:
            image: Screenshot as PIL Image or numpy array
            prompt: What to analyze in the image

        Returns:
            Analysis results from Claude
        """
        start_time = time.time()

        # Convert to PIL Image
        if isinstance(image, np.ndarray):
            if image.dtype == object:
                raise ValueError("Invalid numpy array dtype. Expected uint8 array.")
            pil_image = Image.fromarray(image.astype(np.uint8))
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Optimize image
        optimized_image = self._optimize_image(pil_image)

        # Convert to base64
        buffer = io.BytesIO()
        optimized_image.save(
            buffer, format="JPEG", quality=self.jpeg_quality, optimize=True
        )
        image_data = buffer.getvalue()

        # Further compress if needed
        while len(image_data) > self.max_file_size and self.jpeg_quality > 30:
            self.jpeg_quality -= 10
            buffer = io.BytesIO()
            optimized_image.save(
                buffer, format="JPEG", quality=self.jpeg_quality, optimize=True
            )
            image_data = buffer.getvalue()

        image_base64 = base64.b64encode(image_data).decode()

        logger.info(
            f"Image optimization took {time.time() - start_time:.2f}s, size: {len(image_data)/1024:.1f}KB"
        )

        # Use shorter, more focused prompt for faster response
        optimized_prompt = self._optimize_prompt(prompt)

        # Try intelligent selection first
        if self.use_intelligent_selection:
            try:
                return await self._analyze_screenshot_fast_with_intelligent_selection(
                    image_base64, image_data, optimized_prompt, start_time
                )
            except Exception as e:
                logger.warning(f"Intelligent selection failed, falling back to direct API: {e}")

        # Fallback to direct API
        # Send to Claude with optimized settings
        api_start = time.time()
        message = self.client.messages.create(
            model=self.model,
            max_tokens=512,  # Reduced from 1024
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_base64,
                            },
                        },
                        {"type": "text", "text": optimized_prompt},
                    ],
                }
            ],
            temperature=0.3,  # Lower temperature for more consistent responses
        )

        logger.info(f"Claude API call took {time.time() - api_start:.2f}s")

        result = {
            "description": message.content[0].text,
            "response_time": time.time() - start_time,
            "image_size_kb": len(image_data) / 1024,
        }

        return result

    def _optimize_image(self, image: Image.Image) -> Image.Image:
        """Optimize image for faster upload and processing"""
        # Convert RGBA to RGB if needed
        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        elif image.mode != "RGB":
            image = image.convert("RGB")

        # Resize if larger than max size
        if (
            image.size[0] > self.max_image_size[0]
            or image.size[1] > self.max_image_size[1]
        ):
            image.thumbnail(self.max_image_size, Image.Resampling.LANCZOS)

        return image

    def _optimize_prompt(self, prompt: str) -> str:
        """Optimize prompt for faster response"""
        # For common queries, use shorter prompts
        if "can you see my screen" in prompt.lower():
            return "Briefly describe the main application and activity visible on screen. Focus on: 1) Primary app open, 2) What the user is doing. Keep response under 100 words."
        elif "what's on my screen" in prompt.lower():
            return "List the main elements visible: 1) Open application, 2) Current activity. Be concise."
        else:
            # For other prompts, add conciseness instruction
            return f"{prompt}\n\nBe concise and direct, limiting response to essential information only."

    async def analyze_screenshot(self, image: Any, prompt: str) -> Dict[str, Any]:
        """Backward compatible method that uses optimized version"""
        return await self.analyze_screenshot_fast(image, prompt)
