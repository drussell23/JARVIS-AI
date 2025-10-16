"""
Enhanced Vision Pipeline v1.0
==============================

Multi-stage vision processing pipeline for UI automation.

Stages:
    1. Screen Region Segmentation
    2. Icon Pattern Recognition
    3. Coordinate Calculation
    4. Multi-Model Validation
    5. Mouse Automation Execution

Author: Derek J. Russell
Date: October 2025
"""

from .pipeline_manager import (
    VisionPipelineManager,
    get_vision_pipeline,
    PipelineStage,
    PipelineResult,
    PipelineMetrics
)

from .screen_region_analyzer import (
    ScreenRegionAnalyzer,
    ScreenRegion,
    QuadTreeNode
)

from .icon_detection_engine import (
    IconDetectionEngine,
    DetectionResult
)

from .coordinate_calculator import (
    CoordinateCalculator,
    CoordinateResult
)

from .multi_model_validator import (
    MultiModelValidator,
    ValidationResult
)

from .mouse_automation_controller import (
    MouseAutomationController,
    MouseActionResult
)

__all__ = [
    # Pipeline Manager
    'VisionPipelineManager',
    'get_vision_pipeline',
    'PipelineStage',
    'PipelineResult',
    'PipelineMetrics',
    
    # Stage 1
    'ScreenRegionAnalyzer',
    'ScreenRegion',
    'QuadTreeNode',
    
    # Stage 2
    'IconDetectionEngine',
    'DetectionResult',
    
    # Stage 3
    'CoordinateCalculator',
    'CoordinateResult',
    
    # Stage 4
    'MultiModelValidator',
    'ValidationResult',
    
    # Stage 5
    'MouseAutomationController',
    'MouseActionResult',
]

__version__ = '1.0.0'
__author__ = 'Derek J. Russell'
