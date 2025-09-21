"""
Element Detector - Extracts visual elements from screenshots for Scene Graph
Uses computer vision techniques to identify UI elements, text, and content
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import pytesseract
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class DetectedElement:
    """Represents a detected visual element"""
    type: str
    bounds: Dict[str, int]  # x, y, width, height
    confidence: float
    properties: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'type': self.type,
            'bounds': self.bounds,
            'confidence': self.confidence,
            **self.properties
        }


class ElementDetector:
    """Detects visual elements in screenshots"""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.min_element_size = 10  # Minimum size for valid elements
        self.text_confidence_threshold = 60
        
    async def detect_elements(self, screenshot: np.ndarray) -> List[Dict[str, Any]]:
        """Detect all visual elements in screenshot"""
        elements = []
        
        # Run detection methods in parallel
        tasks = [
            self._detect_windows(screenshot),
            self._detect_ui_elements(screenshot),
            self._detect_text_elements(screenshot),
            self._detect_content_regions(screenshot)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                elements.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Element detection error: {result}")
        
        # Merge overlapping elements
        elements = self._merge_overlapping_elements(elements)
        
        # Sort by z-order (approximate by area and position)
        elements.sort(key=lambda e: (
            -e['bounds']['width'] * e['bounds']['height'],  # Larger first
            e['bounds']['y'],  # Top to bottom
            e['bounds']['x']   # Left to right
        ))
        
        return elements
    
    async def _detect_windows(self, screenshot: np.ndarray) -> List[Dict[str, Any]]:
        """Detect application windows and containers"""
        elements = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY) if len(screenshot.shape) == 3 else screenshot
        
        # Find large rectangular regions (windows)
        # Use edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process large contours as potential windows
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 10000:  # Skip small regions
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if it's a reasonable window size
            if w > screenshot.shape[1] * 0.2 and h > screenshot.shape[0] * 0.2:
                elements.append({
                    'type': 'window',
                    'bounds': {'x': x, 'y': y, 'width': w, 'height': h},
                    'confidence': 0.8,
                    'properties': {
                        'area': area,
                        'aspect_ratio': w / h if h > 0 else 1,
                        'is_fullscreen': w >= screenshot.shape[1] * 0.95 and h >= screenshot.shape[0] * 0.95
                    }
                })
        
        return elements
    
    async def _detect_ui_elements(self, screenshot: np.ndarray) -> List[Dict[str, Any]]:
        """Detect UI elements like buttons, inputs, etc."""
        elements = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY) if len(screenshot.shape) == 3 else screenshot
        
        # Detect button-like elements (rectangular with consistent color)
        # Use threshold to find uniform regions
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100 or area > 50000:  # Skip too small or too large
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check aspect ratio for button-like shapes
            aspect_ratio = w / h if h > 0 else 1
            
            # Classify based on shape and size
            element_type = self._classify_ui_element(w, h, aspect_ratio)
            
            if element_type:
                # Extract region properties
                roi = gray[y:y+h, x:x+w]
                mean_intensity = np.mean(roi)
                std_intensity = np.std(roi)
                
                elements.append({
                    'type': element_type,
                    'bounds': {'x': x, 'y': y, 'width': w, 'height': h},
                    'confidence': 0.7,
                    'properties': {
                        'is_interactive': True,
                        'is_enabled': mean_intensity > 100,  # Brighter = enabled
                        'has_uniform_background': std_intensity < 30,
                        'aspect_ratio': aspect_ratio
                    }
                })
        
        return elements
    
    def _classify_ui_element(self, width: int, height: int, aspect_ratio: float) -> Optional[str]:
        """Classify UI element based on dimensions"""
        # Button-like
        if 2 <= aspect_ratio <= 8 and 20 <= height <= 60:
            return 'button'
        
        # Input field-like
        elif aspect_ratio > 5 and 20 <= height <= 40:
            return 'input'
        
        # Checkbox/radio button
        elif 0.8 <= aspect_ratio <= 1.2 and 10 <= width <= 30:
            return 'checkbox'
        
        # Dropdown
        elif 3 <= aspect_ratio <= 10 and 25 <= height <= 40:
            return 'dropdown'
        
        # Generic interactive element
        elif 20 <= width <= 200 and 20 <= height <= 100:
            return 'ui_element'
        
        return None
    
    async def _detect_text_elements(self, screenshot: np.ndarray) -> List[Dict[str, Any]]:
        """Detect text regions using OCR"""
        elements = []
        
        try:
            # Run OCR with location data
            ocr_data = pytesseract.image_to_data(screenshot, output_type=pytesseract.Output.DICT)
            
            # Group text by lines
            current_line = []
            current_line_num = -1
            
            for i in range(len(ocr_data['text'])):
                if ocr_data['conf'][i] < self.text_confidence_threshold:
                    continue
                
                text = ocr_data['text'][i].strip()
                if not text:
                    continue
                
                line_num = ocr_data['line_num'][i]
                
                if line_num != current_line_num and current_line:
                    # Process previous line
                    element = self._create_text_element(current_line)
                    if element:
                        elements.append(element)
                    current_line = []
                
                current_line_num = line_num
                current_line.append({
                    'text': text,
                    'x': ocr_data['left'][i],
                    'y': ocr_data['top'][i],
                    'width': ocr_data['width'][i],
                    'height': ocr_data['height'][i],
                    'conf': ocr_data['conf'][i]
                })
            
            # Process last line
            if current_line:
                element = self._create_text_element(current_line)
                if element:
                    elements.append(element)
                    
        except Exception as e:
            logger.error(f"OCR error: {e}")
        
        return elements
    
    def _create_text_element(self, text_parts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Create text element from OCR parts"""
        if not text_parts:
            return None
        
        # Calculate bounding box
        min_x = min(p['x'] for p in text_parts)
        min_y = min(p['y'] for p in text_parts)
        max_x = max(p['x'] + p['width'] for p in text_parts)
        max_y = max(p['y'] + p['height'] for p in text_parts)
        
        # Combine text
        text = ' '.join(p['text'] for p in text_parts)
        avg_conf = sum(p['conf'] for p in text_parts) / len(text_parts)
        
        # Determine text type
        text_type = 'text'
        font_size = max_y - min_y
        
        if font_size > 24:
            text_type = 'heading'
        elif len(text) < 20 and ':' in text:
            text_type = 'label'
        
        return {
            'type': text_type,
            'bounds': {
                'x': min_x,
                'y': min_y,
                'width': max_x - min_x,
                'height': max_y - min_y
            },
            'confidence': avg_conf / 100,
            'properties': {
                'text': text,
                'format': 'plain',
                'is_editable': False,
                'estimated_font_size': font_size,
                'word_count': len(text.split())
            }
        }
    
    async def _detect_content_regions(self, screenshot: np.ndarray) -> List[Dict[str, Any]]:
        """Detect content regions like images, videos, etc."""
        elements = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY) if len(screenshot.shape) == 3 else screenshot
        
        # Use variance to find content regions
        # High variance = likely image content
        # Low variance = likely solid color or text
        
        # Sliding window approach
        window_size = 100
        stride = 50
        
        for y in range(0, screenshot.shape[0] - window_size, stride):
            for x in range(0, screenshot.shape[1] - window_size, stride):
                roi = gray[y:y+window_size, x:x+window_size]
                variance = np.var(roi)
                
                # High variance suggests image content
                if variance > 1000:
                    # Expand region to find full bounds
                    bounds = self._expand_content_region(gray, x, y, window_size)
                    
                    if bounds['width'] > 50 and bounds['height'] > 50:
                        elements.append({
                            'type': 'content',
                            'bounds': bounds,
                            'confidence': 0.6,
                            'properties': {
                                'content_type': 'image',
                                'variance': float(variance),
                                'is_modified': False
                            }
                        })
        
        # Merge overlapping content regions
        elements = self._merge_overlapping_elements(elements)
        
        return elements
    
    def _expand_content_region(self, gray: np.ndarray, x: int, y: int, 
                              initial_size: int) -> Dict[str, int]:
        """Expand content region to find full bounds"""
        h, w = gray.shape
        
        # Start with initial bounds
        left, top = x, y
        right, bottom = x + initial_size, y + initial_size
        
        # Expand in all directions while variance remains high
        threshold_variance = 500
        expand_step = 10
        
        # Expand right
        while right < w - expand_step:
            roi = gray[top:bottom, right:right+expand_step]
            if roi.size > 0 and np.var(roi) > threshold_variance:
                right += expand_step
            else:
                break
        
        # Expand bottom
        while bottom < h - expand_step:
            roi = gray[bottom:bottom+expand_step, left:right]
            if roi.size > 0 and np.var(roi) > threshold_variance:
                bottom += expand_step
            else:
                break
        
        # Expand left
        while left > expand_step:
            roi = gray[top:bottom, left-expand_step:left]
            if roi.size > 0 and np.var(roi) > threshold_variance:
                left -= expand_step
            else:
                break
        
        # Expand top
        while top > expand_step:
            roi = gray[top-expand_step:top, left:right]
            if roi.size > 0 and np.var(roi) > threshold_variance:
                top -= expand_step
            else:
                break
        
        return {
            'x': left,
            'y': top,
            'width': right - left,
            'height': bottom - top
        }
    
    def _merge_overlapping_elements(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge overlapping elements of the same type"""
        if not elements:
            return elements
        
        merged = []
        used = set()
        
        for i, elem1 in enumerate(elements):
            if i in used:
                continue
            
            bounds1 = elem1['bounds']
            merge_group = [elem1]
            
            for j, elem2 in enumerate(elements[i+1:], i+1):
                if j in used:
                    continue
                
                if elem1['type'] != elem2['type']:
                    continue
                
                bounds2 = elem2['bounds']
                
                # Check overlap
                if self._bounds_overlap(bounds1, bounds2, threshold=0.5):
                    merge_group.append(elem2)
                    used.add(j)
            
            # Merge the group
            if len(merge_group) > 1:
                merged_elem = self._merge_elements(merge_group)
                merged.append(merged_elem)
            else:
                merged.append(elem1)
        
        return merged
    
    def _bounds_overlap(self, b1: Dict[str, int], b2: Dict[str, int], 
                       threshold: float = 0.5) -> bool:
        """Check if two bounds overlap by threshold percentage"""
        # Calculate intersection
        x_overlap = max(0, min(b1['x'] + b1['width'], b2['x'] + b2['width']) - max(b1['x'], b2['x']))
        y_overlap = max(0, min(b1['y'] + b1['height'], b2['y'] + b2['height']) - max(b1['y'], b2['y']))
        
        intersection_area = x_overlap * y_overlap
        
        # Calculate union
        area1 = b1['width'] * b1['height']
        area2 = b2['width'] * b2['height']
        union_area = area1 + area2 - intersection_area
        
        # IoU (Intersection over Union)
        if union_area > 0:
            iou = intersection_area / union_area
            return iou > threshold
        
        return False
    
    def _merge_elements(self, elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple elements into one"""
        # Calculate combined bounds
        min_x = min(e['bounds']['x'] for e in elements)
        min_y = min(e['bounds']['y'] for e in elements)
        max_x = max(e['bounds']['x'] + e['bounds']['width'] for e in elements)
        max_y = max(e['bounds']['y'] + e['bounds']['height'] for e in elements)
        
        # Average confidence
        avg_confidence = sum(e['confidence'] for e in elements) / len(elements)
        
        # Merge properties
        merged_props = {}
        for elem in elements:
            if 'properties' in elem:
                merged_props.update(elem['properties'])
        
        return {
            'type': elements[0]['type'],
            'bounds': {
                'x': min_x,
                'y': min_y,
                'width': max_x - min_x,
                'height': max_y - min_y
            },
            'confidence': avg_confidence,
            'properties': merged_props
        }