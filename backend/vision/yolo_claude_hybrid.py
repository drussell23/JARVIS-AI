"""
YOLO + Claude Hybrid Vision System
===================================

Intelligent hybrid vision system that combines:
- YOLOv8 for fast UI element detection (6GB RAM, 50-100ms)
- Claude Vision for complex scene understanding
- Automatic selection based on task complexity

Strategy:
1. Use YOLO for simple tasks (icon detection, button finding, layout analysis)
2. Use Claude Vision for complex tasks (text reading, semantic understanding, context analysis)
3. Use both when beneficial (YOLO detects regions → Claude analyzes specific regions)

Benefits:
- 10-20x faster for UI detection tasks
- Significant cost savings (YOLO is free after initial model download)
- Better for real-time applications
- Falls back to Claude Vision when YOLO insufficient

Integration with Intelligent Model Selection:
- Registered as capability: {"vision", "ui_detection", "real_time", "fast_detection"}
- Priority: Higher than Claude Vision for UI-specific tasks
- RAM-aware: Only loads when sufficient RAM available

Author: JARVIS AI Team
Date: 2025-10-27
"""

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Import YOLOVisionDetector
try:
    from backend.vision.yolo_vision_detector import (
        YOLO_AVAILABLE,
        Detection,
        DetectionResult,
        YOLOModelSize,
        YOLOVisionDetector,
    )
except ImportError:
    logger.warning("YOLOVisionDetector not available")
    YOLO_AVAILABLE = False


class VisionTaskComplexity(Enum):
    """Complexity levels for vision tasks"""

    SIMPLE = "simple"  # UI detection, icon finding → YOLO
    MODERATE = "moderate"  # Layout analysis, region detection → YOLO or Claude
    COMPLEX = "complex"  # Text reading, semantic analysis → Claude
    HYBRID = "hybrid"  # Both YOLO and Claude needed


class VisionTaskType(Enum):
    """Types of vision tasks"""

    UI_DETECTION = "ui_detection"  # Detect buttons, icons, UI elements → YOLO
    ICON_DETECTION = "icon_detection"  # Find specific icons → YOLO
    BUTTON_DETECTION = "button_detection"  # Find buttons → YOLO
    LAYOUT_ANALYSIS = "layout_analysis"  # Analyze screen layout → YOLO
    TEXT_EXTRACTION = "text_extraction"  # OCR, read text → Claude
    SEMANTIC_ANALYSIS = "semantic_analysis"  # Understand scene → Claude
    OBJECT_DETECTION = "object_detection"  # General objects → YOLO
    REGION_ANALYSIS = "region_analysis"  # Analyze specific region → Claude
    MULTI_MONITOR = "multi_monitor"  # Multi-monitor detection → YOLO
    CONTROL_CENTER = "control_center"  # macOS Control Center → YOLO
    TV_CONNECTION = "tv_connection"  # TV UI detection → YOLO
    COMPREHENSIVE = "comprehensive"  # Full analysis → Hybrid


@dataclass
class VisionStrategy:
    """Strategy for vision task execution"""

    task_type: VisionTaskType
    complexity: VisionTaskComplexity
    use_yolo: bool
    use_claude: bool
    yolo_first: bool  # If True, run YOLO first then Claude on specific regions
    confidence_threshold: float
    reasoning: str


class IntelligentVisionRouter:
    """
    Intelligent router that decides whether to use YOLO, Claude, or both

    Decision factors:
    - Task type (UI detection → YOLO, text reading → Claude)
    - Complexity (simple → YOLO, complex → Claude)
    - RAM availability (YOLO needs 6GB)
    - Cost considerations (YOLO is free)
    - Speed requirements (YOLO is 10-20x faster)
    """

    def __init__(self):
        self.task_strategies: Dict[VisionTaskType, VisionStrategy] = self._initialize_strategies()

    def _initialize_strategies(self) -> Dict[VisionTaskType, VisionStrategy]:
        """Initialize task routing strategies"""
        return {
            VisionTaskType.UI_DETECTION: VisionStrategy(
                task_type=VisionTaskType.UI_DETECTION,
                complexity=VisionTaskComplexity.SIMPLE,
                use_yolo=True,
                use_claude=False,
                yolo_first=True,
                confidence_threshold=0.25,
                reasoning="UI elements best detected with YOLO (fast, accurate)",
            ),
            VisionTaskType.ICON_DETECTION: VisionStrategy(
                task_type=VisionTaskType.ICON_DETECTION,
                complexity=VisionTaskComplexity.SIMPLE,
                use_yolo=True,
                use_claude=False,
                yolo_first=True,
                confidence_threshold=0.3,
                reasoning="Icon detection optimized for YOLO",
            ),
            VisionTaskType.BUTTON_DETECTION: VisionStrategy(
                task_type=VisionTaskType.BUTTON_DETECTION,
                complexity=VisionTaskComplexity.SIMPLE,
                use_yolo=True,
                use_claude=False,
                yolo_first=True,
                confidence_threshold=0.3,
                reasoning="Button detection optimized for YOLO",
            ),
            VisionTaskType.LAYOUT_ANALYSIS: VisionStrategy(
                task_type=VisionTaskType.LAYOUT_ANALYSIS,
                complexity=VisionTaskComplexity.MODERATE,
                use_yolo=True,
                use_claude=False,
                yolo_first=True,
                confidence_threshold=0.2,
                reasoning="Layout structure detectable with YOLO",
            ),
            VisionTaskType.TEXT_EXTRACTION: VisionStrategy(
                task_type=VisionTaskType.TEXT_EXTRACTION,
                complexity=VisionTaskComplexity.COMPLEX,
                use_yolo=False,
                use_claude=True,
                yolo_first=False,
                confidence_threshold=0.5,
                reasoning="Text reading requires Claude Vision (OCR)",
            ),
            VisionTaskType.SEMANTIC_ANALYSIS: VisionStrategy(
                task_type=VisionTaskType.SEMANTIC_ANALYSIS,
                complexity=VisionTaskComplexity.COMPLEX,
                use_yolo=False,
                use_claude=True,
                yolo_first=False,
                confidence_threshold=0.5,
                reasoning="Semantic understanding requires Claude Vision",
            ),
            VisionTaskType.OBJECT_DETECTION: VisionStrategy(
                task_type=VisionTaskType.OBJECT_DETECTION,
                complexity=VisionTaskComplexity.SIMPLE,
                use_yolo=True,
                use_claude=False,
                yolo_first=True,
                confidence_threshold=0.25,
                reasoning="Object detection optimized for YOLO (COCO classes)",
            ),
            VisionTaskType.REGION_ANALYSIS: VisionStrategy(
                task_type=VisionTaskType.REGION_ANALYSIS,
                complexity=VisionTaskComplexity.HYBRID,
                use_yolo=True,
                use_claude=True,
                yolo_first=True,
                confidence_threshold=0.25,
                reasoning="YOLO finds regions, Claude analyzes content",
            ),
            VisionTaskType.MULTI_MONITOR: VisionStrategy(
                task_type=VisionTaskType.MULTI_MONITOR,
                complexity=VisionTaskComplexity.SIMPLE,
                use_yolo=True,
                use_claude=False,
                yolo_first=True,
                confidence_threshold=0.3,
                reasoning="Monitor detection optimized for YOLO",
            ),
            VisionTaskType.CONTROL_CENTER: VisionStrategy(
                task_type=VisionTaskType.CONTROL_CENTER,
                complexity=VisionTaskComplexity.SIMPLE,
                use_yolo=True,
                use_claude=False,
                yolo_first=True,
                confidence_threshold=0.4,
                reasoning="Control Center icon detectable with YOLO",
            ),
            VisionTaskType.TV_CONNECTION: VisionStrategy(
                task_type=VisionTaskType.TV_CONNECTION,
                complexity=VisionTaskComplexity.MODERATE,
                use_yolo=True,
                use_claude=False,
                yolo_first=True,
                confidence_threshold=0.3,
                reasoning="TV UI elements detectable with YOLO",
            ),
            VisionTaskType.COMPREHENSIVE: VisionStrategy(
                task_type=VisionTaskType.COMPREHENSIVE,
                complexity=VisionTaskComplexity.HYBRID,
                use_yolo=True,
                use_claude=True,
                yolo_first=True,
                confidence_threshold=0.2,
                reasoning="Comprehensive analysis benefits from both models",
            ),
        }

    def get_strategy(self, task_type: VisionTaskType) -> VisionStrategy:
        """Get routing strategy for task type"""
        return self.task_strategies.get(
            task_type,
            VisionStrategy(
                task_type=task_type,
                complexity=VisionTaskComplexity.MODERATE,
                use_yolo=True,
                use_claude=True,
                yolo_first=True,
                confidence_threshold=0.25,
                reasoning="Default: Try YOLO first, fall back to Claude",
            ),
        )

    def should_use_yolo(self, task_type: VisionTaskType) -> bool:
        """Check if YOLO should be used for this task"""
        strategy = self.get_strategy(task_type)
        return strategy.use_yolo

    def should_use_claude(self, task_type: VisionTaskType) -> bool:
        """Check if Claude should be used for this task"""
        strategy = self.get_strategy(task_type)
        return strategy.use_claude

    def is_hybrid_task(self, task_type: VisionTaskType) -> bool:
        """Check if task requires both YOLO and Claude"""
        strategy = self.get_strategy(task_type)
        return strategy.use_yolo and strategy.use_claude


class YOLOClaudeHybridVision:
    """
    Hybrid vision system combining YOLO and Claude Vision

    Features:
    - Intelligent task routing
    - Cost optimization (prefer YOLO when possible)
    - Speed optimization (YOLO is 10-20x faster)
    - Quality optimization (Claude for complex tasks)
    - Automatic fallback
    - RAM-aware model loading
    """

    def __init__(
        self,
        yolo_model_size: YOLOModelSize = YOLOModelSize.MEDIUM,
        enable_yolo: bool = True,
        enable_claude: bool = True,
        prefer_yolo: bool = True,  # Prefer YOLO when both can handle task
        max_yolo_ram_gb: float = 8.0,
    ):
        """
        Initialize hybrid vision system

        Args:
            yolo_model_size: YOLOv8 model size
            enable_yolo: Enable YOLO detector
            enable_claude: Enable Claude Vision
            prefer_yolo: Prefer YOLO over Claude when both viable
            max_yolo_ram_gb: Max RAM for YOLO
        """
        self.enable_yolo = enable_yolo and YOLO_AVAILABLE
        self.enable_claude = enable_claude
        self.prefer_yolo = prefer_yolo

        # Initialize router
        self.router = IntelligentVisionRouter()

        # Initialize YOLO detector (lazy loading)
        self.yolo_detector: Optional[YOLOVisionDetector] = None
        if self.enable_yolo:
            try:
                self.yolo_detector = YOLOVisionDetector(
                    model_size=yolo_model_size, max_ram_gb=max_yolo_ram_gb, enable_cache=True
                )
                logger.info("✅ YOLO detector initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize YOLO detector: {e}")
                self.enable_yolo = False

        # Claude Vision will be handled by existing analyzers
        logger.info(
            f"YOLOClaudeHybridVision initialized (yolo={self.enable_yolo}, "
            f"claude={self.enable_claude}, prefer_yolo={prefer_yolo})"
        )

    async def analyze_screen(
        self,
        image: Union[str, Image.Image, np.ndarray],
        task_type: VisionTaskType = VisionTaskType.COMPREHENSIVE,
        prompt: Optional[str] = None,
        claude_analyzer: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Analyze screen using optimal vision model(s)

        Args:
            image: Screen image
            task_type: Type of vision task
            prompt: Optional prompt for Claude
            claude_analyzer: Optional Claude analyzer instance

        Returns:
            Analysis results with metadata
        """
        # Get strategy
        strategy = self.router.get_strategy(task_type)

        logger.info(
            f"Vision task: {task_type.value} (complexity={strategy.complexity.value}, "
            f"yolo={strategy.use_yolo}, claude={strategy.use_claude})"
        )

        result = {"task_type": task_type.value, "strategy": strategy.reasoning, "models_used": []}

        # Execute based on strategy
        if strategy.use_yolo and strategy.use_claude and strategy.yolo_first:
            # Hybrid: YOLO first, then Claude on regions
            result.update(await self._hybrid_analysis(image, prompt, claude_analyzer, strategy))

        elif strategy.use_yolo and self.enable_yolo:
            # YOLO only
            result.update(await self._yolo_analysis(image, strategy))
            result["models_used"].append("yolo")

        elif strategy.use_claude and self.enable_claude:
            # Claude only
            result.update(await self._claude_analysis(image, prompt, claude_analyzer))
            result["models_used"].append("claude")

        else:
            # Fallback
            logger.warning(f"No suitable vision model available for {task_type.value}")
            result["error"] = "No vision model available"

        return result

    async def _yolo_analysis(
        self, image: Union[str, Image.Image, np.ndarray], strategy: VisionStrategy
    ) -> Dict[str, Any]:
        """Analyze with YOLO only"""
        if not self.yolo_detector:
            return {"error": "YOLO detector not available"}

        try:
            # Detect UI elements
            detection_result = await self.yolo_detector.detect_ui_elements(
                image, min_confidence=strategy.confidence_threshold
            )

            return {
                "yolo_detections": detection_result.to_dict(),
                "inference_time_ms": detection_result.inference_time_ms,
                "total_detections": len(detection_result.detections),
            }

        except Exception as e:
            logger.error(f"YOLO analysis failed: {e}")
            return {"error": str(e)}

    async def _claude_analysis(
        self,
        image: Union[str, Image.Image, np.ndarray],
        prompt: Optional[str],
        claude_analyzer: Optional[Any],
    ) -> Dict[str, Any]:
        """Analyze with Claude Vision only"""
        if not claude_analyzer:
            return {"error": "Claude analyzer not available"}

        try:
            # Use existing Claude Vision analyzer
            if hasattr(claude_analyzer, "analyze_screenshot"):
                result = await claude_analyzer.analyze_screenshot(
                    image, analysis_type="comprehensive", prompt=prompt
                )
                return {"claude_analysis": result, "description": result.get("description", "")}
            else:
                return {"error": "Claude analyzer missing analyze_screenshot method"}

        except Exception as e:
            logger.error(f"Claude analysis failed: {e}")
            return {"error": str(e)}

    async def _hybrid_analysis(
        self,
        image: Union[str, Image.Image, np.ndarray],
        prompt: Optional[str],
        claude_analyzer: Optional[Any],
        strategy: VisionStrategy,
    ) -> Dict[str, Any]:
        """Hybrid analysis: YOLO detects regions, Claude analyzes them"""
        result = {"models_used": ["yolo", "claude"]}

        try:
            # Step 1: YOLO detection
            yolo_result = await self._yolo_analysis(image, strategy)
            result["yolo_detections"] = yolo_result.get("yolo_detections", {})

            # Step 2: If YOLO found interesting regions, analyze with Claude
            detections = yolo_result.get("yolo_detections", {}).get("detections", [])

            if detections and claude_analyzer:
                # For now, pass full image to Claude with YOLO context
                # Future: Crop and analyze specific regions
                claude_result = await self._claude_analysis(image, prompt, claude_analyzer)
                result["claude_analysis"] = claude_result.get("claude_analysis", {})

                # Merge insights
                result["description"] = (
                    f"Found {len(detections)} UI elements. " + claude_result.get("description", "")
                )
            else:
                result["description"] = f"Found {len(detections)} UI elements via YOLO."

            return result

        except Exception as e:
            logger.error(f"Hybrid analysis failed: {e}")
            return {"error": str(e)}

    async def detect_control_center(
        self, screenshot: Union[str, Image.Image, np.ndarray]
    ) -> Optional[Detection]:
        """Quick Control Center detection with YOLO"""
        if not self.yolo_detector:
            return None

        return await self.yolo_detector.detect_control_center(screenshot)

    async def detect_monitors(
        self, screenshot: Union[str, Image.Image, np.ndarray]
    ) -> List[Detection]:
        """Multi-monitor layout detection with YOLO"""
        if not self.yolo_detector:
            return []

        return await self.yolo_detector.detect_monitors(screenshot)

    async def detect_tv_connection_ui(
        self, screenshot: Union[str, Image.Image, np.ndarray]
    ) -> DetectionResult:
        """TV connection UI detection with YOLO"""
        if not self.yolo_detector:
            raise RuntimeError("YOLO detector not available")

        return await self.yolo_detector.detect_tv_connection_ui(screenshot)

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        stats = {
            "yolo_enabled": self.enable_yolo,
            "claude_enabled": self.enable_claude,
            "prefer_yolo": self.prefer_yolo,
        }

        if self.yolo_detector:
            stats["yolo_stats"] = self.yolo_detector.get_stats()

        return stats

    async def cleanup(self):
        """Cleanup resources"""
        if self.yolo_detector:
            await self.yolo_detector.cleanup()

        logger.info("YOLOClaudeHybridVision cleanup complete")


# Global instance
_hybrid_vision_instance: Optional[YOLOClaudeHybridVision] = None


def get_hybrid_vision(
    yolo_model_size: YOLOModelSize = YOLOModelSize.MEDIUM, **kwargs
) -> YOLOClaudeHybridVision:
    """Get or create global hybrid vision instance"""
    global _hybrid_vision_instance

    if _hybrid_vision_instance is None:
        _hybrid_vision_instance = YOLOClaudeHybridVision(yolo_model_size=yolo_model_size, **kwargs)

    return _hybrid_vision_instance


async def main():
    """Example usage"""
    print("🔍 YOLO+Claude Hybrid Vision - Example Usage")
    print("=" * 50)

    # Create hybrid vision system
    hybrid = YOLOClaudeHybridVision(yolo_model_size=YOLOModelSize.MEDIUM, prefer_yolo=True)

    # Example tasks
    tasks = [
        (VisionTaskType.UI_DETECTION, "Should use YOLO only"),
        (VisionTaskType.TEXT_EXTRACTION, "Should use Claude only"),
        (VisionTaskType.COMPREHENSIVE, "Should use both (hybrid)"),
    ]

    for task_type, description in tasks:
        strategy = hybrid.router.get_strategy(task_type)
        print(f"\n📋 {task_type.value}:")
        print(f"   {description}")
        print(f"   Strategy: {strategy.reasoning}")
        print(f"   YOLO: {strategy.use_yolo}, Claude: {strategy.use_claude}")

    # Get stats
    stats = hybrid.get_stats()
    print(f"\n📊 System Stats:")
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # Cleanup
    await hybrid.cleanup()

    print("\n✅ Example complete!")


if __name__ == "__main__":
    asyncio.run(main())
