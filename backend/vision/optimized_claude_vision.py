"""
Optimized Claude Vision Analyzer - Faster response times through image optimization
Includes YOLO hybrid integration for fast UI detection
"""

import base64
import io
import logging
import time
from typing import Any, Dict, Optional

import numpy as np
from anthropic import Anthropic
from PIL import Image

logger = logging.getLogger(__name__)


class OptimizedClaudeVisionAnalyzer:
    """Optimized Claude vision analyzer with faster response times and YOLO integration"""

    def __init__(
        self,
        api_key: str,
        use_intelligent_selection: bool = True,
        use_yolo_hybrid: bool = True,
    ):
        """Initialize Claude vision analyzer with optional YOLO integration"""
        self.client = Anthropic(api_key=api_key)
        self.model = "claude-3-5-sonnet-20241022"
        self.use_intelligent_selection = use_intelligent_selection
        self.use_yolo_hybrid = use_yolo_hybrid

        # Optimization settings
        self.max_image_size = (1280, 720)  # Reduced from full resolution
        self.jpeg_quality = 60  # Lower quality for faster upload
        self.max_file_size = 500 * 1024  # 500KB max

        # Lazy-loaded YOLO hybrid vision (only if requested)
        self._yolo_hybrid = None

    def _get_yolo_hybrid(self):
        """Lazy load YOLO hybrid vision system"""
        if self._yolo_hybrid is None and self.use_yolo_hybrid:
            try:
                from backend.vision.yolo_claude_hybrid import YOLOClaudeHybridVision

                self._yolo_hybrid = YOLOClaudeHybridVision()
                logger.info("✅ YOLO-Claude hybrid vision loaded")
            except Exception as e:
                logger.warning(f"YOLO hybrid vision not available: {e}")
        return self._yolo_hybrid

    def _determine_vision_task_type(self, prompt: str):
        """Determine the best vision task type based on prompt"""
        prompt_lower = prompt.lower()

        # Import task types
        try:
            from backend.vision.yolo_claude_hybrid import VisionTaskType

            # UI detection keywords
            if any(
                kw in prompt_lower
                for kw in ["ui", "button", "icon", "menu", "window", "control", "interface"]
            ):
                return VisionTaskType.UI_DETECTION

            # Control Center keywords
            if "control center" in prompt_lower:
                return VisionTaskType.CONTROL_CENTER

            # TV connection keywords
            if any(kw in prompt_lower for kw in ["tv", "television", "connection", "remote"]):
                return VisionTaskType.TV_CONNECTION

            # Multi-monitor keywords
            if any(kw in prompt_lower for kw in ["monitor", "display", "screen layout"]):
                return VisionTaskType.MULTI_MONITOR

            # Text extraction keywords
            if any(kw in prompt_lower for kw in ["text", "read", "ocr", "content"]):
                return VisionTaskType.TEXT_EXTRACTION

            # Default to comprehensive
            return VisionTaskType.COMPREHENSIVE

        except ImportError:
            return None

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
            Analysis results from Claude or YOLO
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

        # Try YOLO hybrid first for eligible tasks
        if self.use_yolo_hybrid:
            task_type = self._determine_vision_task_type(prompt)
            if task_type:
                try:
                    yolo_hybrid = self._get_yolo_hybrid()
                    if yolo_hybrid:
                        logger.info(
                            f"🎯 Attempting YOLO hybrid analysis for task type: {task_type.value}"
                        )
                        result = await yolo_hybrid.analyze_screen(
                            image=pil_image,
                            task_type=task_type,
                            prompt=prompt,
                            claude_analyzer=self,
                        )

                        # Add timing info
                        result["response_time"] = time.time() - start_time
                        result["method"] = "yolo_hybrid"

                        logger.info(
                            f"✅ YOLO hybrid analysis completed in {result['response_time']:.2f}s"
                        )
                        return result
                except Exception as e:
                    logger.warning(f"YOLO hybrid failed, falling back to Claude: {e}")

        # Optimize image
        optimized_image = self._optimize_image(pil_image)

        # Convert to base64
        buffer = io.BytesIO()
        optimized_image.save(buffer, format="JPEG", quality=self.jpeg_quality, optimize=True)
        image_data = buffer.getvalue()

        # Further compress if needed
        while len(image_data) > self.max_file_size and self.jpeg_quality > 30:
            self.jpeg_quality -= 10
            buffer = io.BytesIO()
            optimized_image.save(buffer, format="JPEG", quality=self.jpeg_quality, optimize=True)
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
        if image.size[0] > self.max_image_size[0] or image.size[1] > self.max_image_size[1]:
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

    async def detect_control_center(self, screenshot: Any) -> Optional[Dict[str, Any]]:
        """Detect macOS Control Center using YOLO"""
        yolo_hybrid = self._get_yolo_hybrid()
        if not yolo_hybrid:
            logger.warning("YOLO hybrid not available for Control Center detection")
            return None

        try:
            result = await yolo_hybrid.detect_control_center(screenshot)
            return result
        except Exception as e:
            logger.error(f"Control Center detection failed: {e}")
            return None

    async def detect_monitors(self, screenshot: Any) -> Optional[Dict[str, Any]]:
        """Detect multiple monitors using YOLO"""
        yolo_hybrid = self._get_yolo_hybrid()
        if not yolo_hybrid:
            logger.warning("YOLO hybrid not available for monitor detection")
            return None

        try:
            result = await yolo_hybrid.detect_monitors(screenshot)
            return result
        except Exception as e:
            logger.error(f"Monitor detection failed: {e}")
            return None

    async def detect_tv_connection_ui(self, screenshot: Any) -> Optional[Dict[str, Any]]:
        """Detect Living Room TV connection UI using YOLO"""
        yolo_hybrid = self._get_yolo_hybrid()
        if not yolo_hybrid:
            logger.warning("YOLO hybrid not available for TV connection UI detection")
            return None

        try:
            result = await yolo_hybrid.detect_tv_connection_ui(screenshot)
            return result
        except Exception as e:
            logger.error(f"TV connection UI detection failed: {e}")
            return None
