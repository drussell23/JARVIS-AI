#!/usr/bin/env python3
"""
Icon Detection Engine - Enhanced Vision Pipeline v1.0
Stage 2: Icon Pattern Recognition
==================================================

Multi-method icon detection using:
- Template matching (OpenCV)
- Edge detection + contour analysis
- Shape recognition
- Feature matching (SIFT/ORB)

Author: Derek J. Russell
Date: October 2025
"""

import asyncio
import logging
import cv2
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Icon detection result"""
    found: bool
    bounding_box: Optional[Tuple[int, int, int, int]]  # (x, y, w, h)
    confidence: float
    method: str
    center_point: Optional[Tuple[int, int]]
    metadata: Dict[str, Any]


class IconDetectionEngine:
    """
    Icon Detection Engine
    
    Uses multiple detection methods with confidence scoring:
    1. Template Matching - Fast, pixel-perfect matching
    2. Edge Detection - Robust to scaling/rotation
    3. Shape Recognition - Geometric analysis
    4. Feature Matching - SIFT/ORB keypoints
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize icon detection engine"""
        self.config = config
        self.detection_config = config.get('detection', {})
        
        # Detection methods to use
        self.methods = self.detection_config.get('methods', [
            'template_matching',
            'edge_detection',
            'shape_recognition'
        ])
        
        # Confidence thresholds
        self.min_confidence = self.detection_config.get('min_confidence', 0.85)
        
        # Template cache
        self.templates: Dict[str, np.ndarray] = {}
        self.template_dir = Path(__file__).parent.parent.parent / 'assets' / 'templates'
        self.template_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("[ICON DETECTOR] Icon Detection Engine initialized")
    
    async def initialize(self):
        """Initialize detection engine"""
        # Load templates
        await self._load_templates()
        logger.info(f"[ICON DETECTOR] Loaded {len(self.templates)} templates")
    
    async def detect_icon(
        self,
        region,  # ScreenRegion from Stage 1
        target: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Detect icon in screen region
        
        Args:
            region: ScreenRegion from Stage 1
            target: Target icon to find
            context: Optional context
            
        Returns:
            Dict with detection results
        """
        logger.info(f"[ICON DETECTOR] Detecting icon: {target}")
        
        try:
            # Convert PIL Image to OpenCV format
            img_cv = self._pil_to_cv(region.image)
            
            # Run all detection methods in parallel
            tasks = []
            
            if 'template_matching' in self.methods:
                tasks.append(self._template_matching(img_cv, target))
            
            if 'edge_detection' in self.methods:
                tasks.append(self._edge_detection(img_cv, target))
            
            if 'shape_recognition' in self.methods:
                tasks.append(self._shape_recognition(img_cv, target))
            
            # Wait for all methods to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions
            valid_results = [r for r in results if isinstance(r, DetectionResult)]
            
            if not valid_results:
                return {'success': False, 'error': 'No detection methods succeeded'}
            
            # Find best result (highest confidence)
            best_result = max(valid_results, key=lambda r: r.confidence)
            
            logger.info(f"[ICON DETECTOR] ✅ Icon found via {best_result.method}")
            logger.info(f"[ICON DETECTOR] Confidence: {best_result.confidence:.2%}")
            logger.info(f"[ICON DETECTOR] Bounding box: {best_result.bounding_box}")
            
            return {
                'success': True,
                'detection_results': {
                    'best': best_result,
                    'all_results': valid_results,
                    'methods_tried': len(tasks),
                    'methods_succeeded': len(valid_results)
                }
            }
            
        except Exception as e:
            logger.error(f"[ICON DETECTOR] Detection failed: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}
    
    async def _template_matching(self, img: np.ndarray, target: str) -> DetectionResult:
        """Template matching detection"""
        try:
            # Get template for target
            template = self.templates.get(target)
            
            if template is None:
                # No template available, create generic one based on target description
                template = await self._generate_template(target)
            
            if template is None:
                raise ValueError(f"No template available for {target}")
            
            # Convert to grayscale
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            
            # Multi-scale template matching
            best_match = None
            best_confidence = 0.0
            
            for scale in np.linspace(0.7, 1.3, 10):
                # Resize template
                scaled_template = cv2.resize(
                    template_gray,
                    (int(template_gray.shape[1] * scale), int(template_gray.shape[0] * scale))
                )
                
                if scaled_template.shape[0] > img_gray.shape[0] or scaled_template.shape[1] > img_gray.shape[1]:
                    continue
                
                # Match template
                result = cv2.matchTemplate(img_gray, scaled_template, cv2.TM_CCOEFF_NORMED)
                
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                
                if max_val > best_confidence:
                    best_confidence = max_val
                    best_match = (max_loc, scaled_template.shape)
            
            if best_match and best_confidence > self.min_confidence:
                (x, y), (h, w) = best_match
                
                return DetectionResult(
                    found=True,
                    bounding_box=(x, y, w, h),
                    confidence=float(best_confidence),
                    method='template_matching',
                    center_point=(x + w // 2, y + h // 2),
                    metadata={'scale_tested': 10, 'best_scale': scale}
                )
            
            raise ValueError(f"Template confidence too low: {best_confidence:.2%}")
            
        except Exception as e:
            logger.debug(f"[ICON DETECTOR] Template matching failed: {e}")
            raise
    
    async def _edge_detection(self, img: np.ndarray, target: str) -> DetectionResult:
        """Edge detection + contour analysis"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Canny edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by target-specific criteria
            target_criteria = self._get_target_criteria(target)
            
            best_contour = None
            best_score = 0.0
            
            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate score based on size, aspect ratio, position
                score = self._score_contour(contour, target_criteria)
                
                if score > best_score:
                    best_score = score
                    best_contour = (x, y, w, h)
            
            if best_contour and best_score > self.min_confidence:
                x, y, w, h = best_contour
                
                return DetectionResult(
                    found=True,
                    bounding_box=(x, y, w, h),
                    confidence=float(best_score),
                    method='edge_detection',
                    center_point=(x + w // 2, y + h // 2),
                    metadata={'contours_analyzed': len(contours)}
                )
            
            raise ValueError(f"No suitable contour found (score: {best_score:.2%})")
            
        except Exception as e:
            logger.debug(f"[ICON DETECTOR] Edge detection failed: {e}")
            raise
    
    async def _shape_recognition(self, img: np.ndarray, target: str) -> DetectionResult:
        """Shape-based recognition"""
        try:
            # For Control Center: look for two overlapping rounded rectangles
            if target == 'control_center':
                return await self._detect_control_center_shape(img)
            
            # Generic shape detection
            raise NotImplementedError(f"Shape recognition not implemented for {target}")
            
        except Exception as e:
            logger.debug(f"[ICON DETECTOR] Shape recognition failed: {e}")
            raise
    
    async def _detect_control_center_shape(self, img: np.ndarray) -> DetectionResult:
        """Detect Control Center icon by shape (two overlapping rectangles)"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Threshold
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Look for rounded rectangles with specific aspect ratio
            for contour in contours:
                # Approximate polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Control Center icon has ~8 vertices (rounded rectangle)
                if 6 <= len(approx) <= 12:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Check size (icon should be ~20-30px)
                    if 15 <= w <= 35 and 15 <= h <= 35:
                        # Check aspect ratio (should be square-ish)
                        aspect_ratio = w / h
                        if 0.8 <= aspect_ratio <= 1.2:
                            return DetectionResult(
                                found=True,
                                bounding_box=(x, y, w, h),
                                confidence=0.90,
                                method='shape_recognition',
                                center_point=(x + w // 2, y + h // 2),
                                metadata={'vertices': len(approx), 'aspect_ratio': aspect_ratio}
                            )
            
            raise ValueError("Control Center shape not found")
            
        except Exception as e:
            raise
    
    def _pil_to_cv(self, pil_image: Image.Image) -> np.ndarray:
        """Convert PIL Image to OpenCV format"""
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    def _get_target_criteria(self, target: str) -> Dict[str, Any]:
        """Get detection criteria for target"""
        criteria = {
            'control_center': {
                'min_width': 18,
                'max_width': 32,
                'min_height': 18,
                'max_height': 32,
                'aspect_ratio': (0.8, 1.2),
                'position': 'top_right'
            },
            'screen_mirroring': {
                'min_width': 40,
                'max_width': 80,
                'min_height': 40,
                'max_height': 80,
                'aspect_ratio': (0.9, 1.1)
            }
        }
        
        return criteria.get(target, {
            'min_width': 20,
            'max_width': 100,
            'min_height': 20,
            'max_height': 100,
            'aspect_ratio': (0.5, 2.0)
        })
    
    def _score_contour(self, contour: np.ndarray, criteria: Dict[str, Any]) -> float:
        """Score contour based on criteria"""
        x, y, w, h = cv2.boundingRect(contour)
        
        score = 1.0
        
        # Size check
        if w < criteria.get('min_width', 0) or w > criteria.get('max_width', 10000):
            score *= 0.5
        if h < criteria.get('min_height', 0) or h > criteria.get('max_height', 10000):
            score *= 0.5
        
        # Aspect ratio check
        aspect_ratio = w / h if h > 0 else 0
        min_ar, max_ar = criteria.get('aspect_ratio', (0, 100))
        if not (min_ar <= aspect_ratio <= max_ar):
            score *= 0.7
        
        return score
    
    async def _load_templates(self):
        """Load icon templates"""
        # Initialize ML template generator
        try:
            from vision.enhanced_vision_pipeline.ml_template_generator import get_ml_template_generator

            self.ml_generator = get_ml_template_generator({
                'max_memory_mb': 500,
                'cache_dir': self.template_dir.parent / 'ml_cache'
            })
            logger.info("[ICON DETECTOR] ✅ ML template generator initialized")
        except Exception as e:
            logger.warning(f"[ICON DETECTOR] ML generator not available: {e}")
            self.ml_generator = None

        # Load pre-made templates from assets folder if they exist
        for template_file in self.template_dir.glob('*.png'):
            try:
                template_name = template_file.stem
                template_img = cv2.imread(str(template_file))
                if template_img is not None:
                    self.templates[template_name] = template_img
                    logger.debug(f"[ICON DETECTOR] Loaded template: {template_name}")
            except Exception as e:
                logger.warning(f"[ICON DETECTOR] Failed to load {template_file}: {e}")

    async def _generate_template(self, target: str) -> Optional[np.ndarray]:
        """
        Generate template for target using ML

        Uses hybrid approach:
        1. Traditional ML: HOG + LBP feature extraction
        2. Deep Learning: MobileNetV3 feature extraction
        3. Template synthesis and augmentation

        Args:
            target: Target icon name

        Returns:
            Generated template as numpy array or None
        """
        if self.ml_generator is None:
            logger.warning("[ICON DETECTOR] ML generator not available, using fallback")
            return await self._generate_fallback_template(target)

        try:
            # Generate template using ML
            template = await self.ml_generator.generate_template(
                target=target,
                context={
                    'screen_region': None,  # Could pass screen region if available
                    'detection_config': self.detection_config
                }
            )

            if template is not None:
                # Cache for future use
                self.templates[target] = template
                logger.info(f"[ICON DETECTOR] ✅ Generated ML template for {target}")

            return template

        except Exception as e:
            logger.error(f"[ICON DETECTOR] ML template generation failed: {e}")
            return await self._generate_fallback_template(target)

    async def _generate_fallback_template(self, target: str) -> Optional[np.ndarray]:
        """
        Generate simple fallback template without ML

        Creates basic geometric shapes based on target name
        """
        size = 48
        template = np.zeros((size, size, 3), dtype=np.uint8)
        template[:] = (255, 255, 255)  # White background

        if target == 'control_center':
            # Two overlapping rectangles (toggle switch)
            cv2.rectangle(template, (12, 16), (22, 32), (100, 100, 100), -1)
            cv2.rectangle(template, (22, 16), (36, 32), (100, 100, 100), -1)
        elif target == 'screen_mirroring':
            # Monitor with wireless waves
            cv2.rectangle(template, (12, 20), (36, 32), (50, 50, 50), 2)
            cv2.ellipse(template, (24, 26), (8, 8), 0, -45, 45, (50, 150, 255), 2)
        else:
            # Generic rounded square
            cv2.rectangle(template, (12, 12), (36, 36), (100, 100, 100), -1)

        logger.debug(f"[ICON DETECTOR] Generated fallback template for {target}")
        return template
