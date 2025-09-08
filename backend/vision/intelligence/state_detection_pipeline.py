"""
State Detection Pipeline - Core detection logic for VSMS
Implements the complete detection pipeline with visual signatures and confidence scoring
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from datetime import datetime
import hashlib
from collections import defaultdict
from sklearn.cluster import MiniBatchKMeans
import logging

logger = logging.getLogger(__name__)


@dataclass
class VisualSignature:
    """Visual signature for state identification"""
    signature_type: str  # layout, color, text, icon, etc.
    features: Dict[str, Any]
    confidence_weight: float = 1.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def match(self, other_features: Dict[str, Any]) -> float:
        """Calculate match score against other features"""
        if self.signature_type == "layout":
            return self._match_layout(other_features)
        elif self.signature_type == "color":
            return self._match_color(other_features)
        elif self.signature_type == "text":
            return self._match_text(other_features)
        elif self.signature_type == "element":
            return self._match_elements(other_features)
        else:
            return 0.0
    
    def _match_layout(self, features: Dict[str, Any]) -> float:
        """Match layout signatures"""
        if 'layout_hash' not in features or 'layout_hash' not in self.features:
            return 0.0
        
        # Simplified hash comparison
        if features['layout_hash'] == self.features['layout_hash']:
            return 1.0
        
        # Fuzzy matching based on element positions
        if 'element_positions' in features and 'element_positions' in self.features:
            return self._calculate_layout_similarity(
                features['element_positions'], 
                self.features['element_positions']
            )
        
        return 0.0
    
    def _match_color(self, features: Dict[str, Any]) -> float:
        """Match color signatures"""
        if 'dominant_colors' not in features or 'dominant_colors' not in self.features:
            return 0.0
        
        # Color histogram comparison
        return self._compare_color_histograms(
            features['dominant_colors'],
            self.features['dominant_colors']
        )
    
    def _match_text(self, features: Dict[str, Any]) -> float:
        """Match text content signatures"""
        if 'text_elements' not in features or 'text_elements' not in self.features:
            return 0.0
        
        # Text similarity
        return self._calculate_text_similarity(
            features['text_elements'],
            self.features['text_elements']
        )
    
    def _match_elements(self, features: Dict[str, Any]) -> float:
        """Match UI element signatures"""
        if 'ui_elements' not in features or 'ui_elements' not in self.features:
            return 0.0
        
        # Element type and count matching
        return self._compare_ui_elements(
            features['ui_elements'],
            self.features['ui_elements']
        )
    
    def _calculate_layout_similarity(self, pos1: List, pos2: List) -> float:
        """Calculate similarity between layout positions"""
        # Simplified - in production use proper spatial similarity
        common = len(set(map(tuple, pos1)) & set(map(tuple, pos2)))
        total = max(len(pos1), len(pos2))
        return common / total if total > 0 else 0.0
    
    def _compare_color_histograms(self, colors1: List, colors2: List) -> float:
        """Compare color distributions"""
        # Simplified color comparison
        if not colors1 or not colors2:
            return 0.0
        
        # Convert to numpy arrays for easier comparison
        c1 = np.array(colors1[:5])  # Top 5 colors
        c2 = np.array(colors2[:5])
        
        # Calculate color distance
        distances = []
        for color1 in c1:
            min_dist = min(np.linalg.norm(color1 - color2) for color2 in c2)
            distances.append(min_dist)
        
        # Normalize to 0-1 (max RGB distance is ~441)
        avg_distance = np.mean(distances)
        similarity = 1.0 - (avg_distance / 441.0)
        
        return max(0.0, similarity)
    
    def _calculate_text_similarity(self, text1: List[str], text2: List[str]) -> float:
        """Calculate text content similarity"""
        if not text1 or not text2:
            return 0.0
        
        # Simple Jaccard similarity
        set1 = set(text1)
        set2 = set(text2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _compare_ui_elements(self, elements1: Dict, elements2: Dict) -> float:
        """Compare UI element compositions"""
        if not elements1 or not elements2:
            return 0.0
        
        # Compare element types and counts
        all_types = set(elements1.keys()) | set(elements2.keys())
        
        similarity_sum = 0.0
        for elem_type in all_types:
            count1 = elements1.get(elem_type, 0)
            count2 = elements2.get(elem_type, 0)
            
            if count1 == count2:
                similarity_sum += 1.0
            else:
                # Partial credit for similar counts
                similarity_sum += 1.0 - abs(count1 - count2) / max(count1, count2, 1)
        
        return similarity_sum / len(all_types) if all_types else 0.0


class StateDetectionPipeline:
    """Advanced state detection pipeline with multiple detection strategies"""
    
    def __init__(self):
        self.color_clusterer = MiniBatchKMeans(n_clusters=5, random_state=42)
        self.detection_strategies = [
            self.detect_by_layout,
            self.detect_by_color_pattern,
            self.detect_by_text_content,
            self.detect_by_ui_elements,
            self.detect_by_modal_overlay
        ]
        self.confidence_weights = {
            'layout': 0.3,
            'color': 0.2,
            'text': 0.3,
            'elements': 0.15,
            'modal': 0.05
        }
    
    async def extract_state_features(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """Extract comprehensive features for state detection"""
        features = {}
        
        # Layout features
        layout_features = self._extract_layout_features(screenshot)
        features.update(layout_features)
        
        # Color features
        color_features = self._extract_color_features(screenshot)
        features.update(color_features)
        
        # Text detection features (mock for now)
        text_features = self._extract_text_features(screenshot)
        features.update(text_features)
        
        # UI element features
        element_features = self._extract_ui_element_features(screenshot)
        features.update(element_features)
        
        # Modal/overlay detection
        modal_features = self._detect_modal_features(screenshot)
        features.update(modal_features)
        
        # Additional metadata
        features['timestamp'] = datetime.now()
        features['resolution'] = screenshot.shape[:2]
        
        return features
    
    def _extract_layout_features(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """Extract layout-based features"""
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours (representing UI elements)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Extract bounding boxes
        bboxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                bboxes.append((x, y, w, h))
        
        # Create layout hash
        layout_hash = self._create_layout_hash(bboxes)
        
        # Grid-based layout analysis
        grid_features = self._analyze_grid_layout(screenshot.shape[:2], bboxes)
        
        return {
            'layout_hash': layout_hash,
            'element_positions': bboxes[:50],  # Limit to 50 elements
            'element_count': len(bboxes),
            'grid_layout': grid_features,
            'layout_complexity': self._calculate_layout_complexity(bboxes)
        }
    
    def _extract_color_features(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """Extract color-based features"""
        # Resize for faster processing
        small = cv2.resize(screenshot, (100, 100))
        
        # Reshape to list of pixels
        pixels = small.reshape(-1, 3)
        
        # Cluster colors
        self.color_clusterer.partial_fit(pixels)
        dominant_colors = self.color_clusterer.cluster_centers_.astype(int)
        
        # Color histogram
        hist_b = cv2.calcHist([screenshot], [0], None, [32], [0, 256])
        hist_g = cv2.calcHist([screenshot], [1], None, [32], [0, 256])
        hist_r = cv2.calcHist([screenshot], [2], None, [32], [0, 256])
        
        # Normalize histograms
        hist_b = hist_b.flatten() / hist_b.sum()
        hist_g = hist_g.flatten() / hist_g.sum()
        hist_r = hist_r.flatten() / hist_r.sum()
        
        return {
            'dominant_colors': dominant_colors.tolist(),
            'color_histogram': {
                'blue': hist_b.tolist(),
                'green': hist_g.tolist(),
                'red': hist_r.tolist()
            },
            'color_variance': np.std(pixels, axis=0).tolist(),
            'brightness': np.mean(cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY))
        }
    
    def _extract_text_features(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """Extract text-based features (mock implementation)"""
        # In production, use OCR (Tesseract, EasyOCR, or Vision API)
        # For now, return mock data
        return {
            'text_elements': ['File', 'Edit', 'View', 'Window', 'Help'],  # Mock menu items
            'text_density': 0.15,  # Percentage of image with text
            'has_error_text': False,
            'has_loading_text': False
        }
    
    def _extract_ui_element_features(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """Extract UI element features"""
        # Detect common UI elements using template matching or ML
        ui_elements = defaultdict(int)
        
        # Simple heuristic-based detection
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        
        # Detect buttons (rectangular regions with uniform color)
        ui_elements['buttons'] = self._detect_buttons(gray)
        
        # Detect text fields (white/light rectangles)
        ui_elements['text_fields'] = self._detect_text_fields(screenshot)
        
        # Detect checkboxes/radio buttons (small squares/circles)
        ui_elements['checkboxes'] = self._detect_checkboxes(gray)
        
        # Detect scrollbars
        ui_elements['scrollbars'] = self._detect_scrollbars(screenshot)
        
        return {
            'ui_elements': dict(ui_elements),
            'interaction_density': sum(ui_elements.values()) / (screenshot.shape[0] * screenshot.shape[1]) * 10000
        }
    
    def _detect_modal_features(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """Detect modal/overlay features"""
        # Check for darkened background (common modal pattern)
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        
        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # Check for bimodal distribution (dark background + bright modal)
        hist_normalized = hist.flatten() / hist.sum()
        
        # Find peaks
        peaks = []
        for i in range(1, 255):
            if hist_normalized[i] > hist_normalized[i-1] and hist_normalized[i] > hist_normalized[i+1]:
                if hist_normalized[i] > 0.01:  # Threshold
                    peaks.append(i)
        
        has_modal = len(peaks) >= 2 and max(peaks) - min(peaks) > 100
        
        # Detect centered rectangle (common modal pattern)
        if has_modal:
            modal_rect = self._detect_centered_rectangle(screenshot)
        else:
            modal_rect = None
        
        return {
            'has_modal': has_modal,
            'modal_rect': modal_rect,
            'overlay_darkness': min(peaks) if peaks else 0,
            'histogram_peaks': len(peaks)
        }
    
    def _create_layout_hash(self, bboxes: List[Tuple[int, int, int, int]]) -> str:
        """Create a hash representing the layout"""
        if not bboxes:
            return "empty"
        
        # Sort by position
        sorted_boxes = sorted(bboxes, key=lambda b: (b[1], b[0]))
        
        # Create simplified representation
        layout_str = ""
        for x, y, w, h in sorted_boxes[:20]:  # Use top 20 elements
            # Quantize positions to reduce noise
            qx = x // 10 * 10
            qy = y // 10 * 10
            qw = w // 10 * 10
            qh = h // 10 * 10
            layout_str += f"{qx},{qy},{qw},{qh};"
        
        # Create hash
        return hashlib.md5(layout_str.encode()).hexdigest()[:16]
    
    def _analyze_grid_layout(self, shape: Tuple[int, int], bboxes: List) -> Dict[str, Any]:
        """Analyze grid-based layout patterns"""
        if not bboxes:
            return {'type': 'empty', 'rows': 0, 'cols': 0}
        
        height, width = shape
        
        # Divide screen into grid
        grid_size = 3
        grid = np.zeros((grid_size, grid_size))
        
        for x, y, w, h in bboxes:
            # Find which grid cell this element belongs to
            grid_x = min(int(x / width * grid_size), grid_size - 1)
            grid_y = min(int(y / height * grid_size), grid_size - 1)
            grid[grid_y, grid_x] += 1
        
        # Determine layout type
        if grid[0, :].sum() > grid[1:, :].sum():
            layout_type = "top_heavy"
        elif grid[:, 0].sum() > grid[:, 1:].sum():
            layout_type = "left_sidebar"
        elif grid[1, 1] > grid.sum() / 2:
            layout_type = "centered"
        else:
            layout_type = "distributed"
        
        return {
            'type': layout_type,
            'grid_distribution': grid.tolist(),
            'primary_region': np.unravel_index(grid.argmax(), grid.shape)
        }
    
    def _calculate_layout_complexity(self, bboxes: List) -> float:
        """Calculate layout complexity score"""
        if not bboxes:
            return 0.0
        
        # Factors: number of elements, size variance, overlap
        complexity = len(bboxes) / 100.0  # Normalize by typical max
        
        # Size variance
        if len(bboxes) > 1:
            sizes = [(w * h) for _, _, w, h in bboxes]
            size_variance = np.std(sizes) / np.mean(sizes)
            complexity += size_variance * 0.5
        
        return min(1.0, complexity)
    
    def _detect_buttons(self, gray_image: np.ndarray) -> int:
        """Detect button-like elements"""
        # Simple detection using morphological operations
        _, binary = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        button_count = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Button-like aspect ratio
            if 0.2 < h/w < 0.5 and 20 < w < 200 and 10 < h < 50:
                button_count += 1
        
        return button_count
    
    def _detect_text_fields(self, image: np.ndarray) -> int:
        """Detect text field-like elements"""
        # Look for white/light rectangles
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # White/light color range
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        
        mask = cv2.inRange(hsv, lower_white, upper_white)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_field_count = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Text field-like aspect ratio
            if 0.05 < h/w < 0.3 and w > 100 and 15 < h < 40:
                text_field_count += 1
        
        return text_field_count
    
    def _detect_checkboxes(self, gray_image: np.ndarray) -> int:
        """Detect checkbox-like elements"""
        # Look for small square regions
        edges = cv2.Canny(gray_image, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        checkbox_count = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Square-like with small size
            if 0.8 < h/w < 1.2 and 10 < w < 30 and 10 < h < 30:
                checkbox_count += 1
        
        return checkbox_count
    
    def _detect_scrollbars(self, image: np.ndarray) -> int:
        """Detect scrollbar-like elements"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        scrollbar_count = 0
        
        # Vertical scrollbar (right edge)
        right_edge = gray[:, -20:]
        if np.std(right_edge) < 20:  # Low variance indicates scrollbar
            scrollbar_count += 1
        
        # Horizontal scrollbar (bottom edge)
        bottom_edge = gray[-20:, :]
        if np.std(bottom_edge) < 20:
            scrollbar_count += 1
        
        return scrollbar_count
    
    def _detect_centered_rectangle(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect a centered rectangular region (modal)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Look in center region
        center_y, center_x = height // 2, width // 2
        roi = gray[center_y-100:center_y+100, center_x-100:center_x+100]
        
        # Find edges
        edges = cv2.Canny(roi, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            # Adjust for ROI offset
            return (x + center_x - 100, y + center_y - 100, w, h)
        
        return None
    
    # Detection strategy methods
    async def detect_by_layout(self, features: Dict[str, Any], known_signatures: List[VisualSignature]) -> Tuple[Optional[str], float]:
        """Detect state by layout matching"""
        layout_signatures = [s for s in known_signatures if s.signature_type == "layout"]
        
        if not layout_signatures:
            return None, 0.0
        
        best_match = None
        best_score = 0.0
        
        for sig in layout_signatures:
            score = sig.match(features)
            if score > best_score:
                best_score = score
                best_match = sig.features.get('state_id')
        
        return best_match, best_score * self.confidence_weights['layout']
    
    async def detect_by_color_pattern(self, features: Dict[str, Any], known_signatures: List[VisualSignature]) -> Tuple[Optional[str], float]:
        """Detect state by color patterns"""
        color_signatures = [s for s in known_signatures if s.signature_type == "color"]
        
        if not color_signatures:
            return None, 0.0
        
        best_match = None
        best_score = 0.0
        
        for sig in color_signatures:
            score = sig.match(features)
            if score > best_score:
                best_score = score
                best_match = sig.features.get('state_id')
        
        return best_match, best_score * self.confidence_weights['color']
    
    async def detect_by_text_content(self, features: Dict[str, Any], known_signatures: List[VisualSignature]) -> Tuple[Optional[str], float]:
        """Detect state by text content"""
        text_signatures = [s for s in known_signatures if s.signature_type == "text"]
        
        if not text_signatures:
            return None, 0.0
        
        # Special handling for error/loading states
        if features.get('has_error_text'):
            return 'error_state', 0.9 * self.confidence_weights['text']
        
        if features.get('has_loading_text'):
            return 'loading_state', 0.9 * self.confidence_weights['text']
        
        best_match = None
        best_score = 0.0
        
        for sig in text_signatures:
            score = sig.match(features)
            if score > best_score:
                best_score = score
                best_match = sig.features.get('state_id')
        
        return best_match, best_score * self.confidence_weights['text']
    
    async def detect_by_ui_elements(self, features: Dict[str, Any], known_signatures: List[VisualSignature]) -> Tuple[Optional[str], float]:
        """Detect state by UI element composition"""
        element_signatures = [s for s in known_signatures if s.signature_type == "element"]
        
        if not element_signatures:
            return None, 0.0
        
        best_match = None
        best_score = 0.0
        
        for sig in element_signatures:
            score = sig.match(features)
            if score > best_score:
                best_score = score
                best_match = sig.features.get('state_id')
        
        return best_match, best_score * self.confidence_weights['elements']
    
    async def detect_by_modal_overlay(self, features: Dict[str, Any], known_signatures: List[VisualSignature]) -> Tuple[Optional[str], float]:
        """Detect modal/dialog states"""
        if features.get('has_modal'):
            # High confidence for modal detection
            return 'modal_state', 0.95 * self.confidence_weights['modal']
        
        return None, 0.0
    
    async def detect_state_ensemble(self, 
                                  screenshot: np.ndarray,
                                  known_signatures: List[VisualSignature]) -> Tuple[Optional[str], float, Dict[str, Any]]:
        """Ensemble state detection using multiple strategies"""
        # Extract features
        features = await self.extract_state_features(screenshot)
        
        # Run all detection strategies
        detection_results = []
        
        for strategy in self.detection_strategies:
            state_id, confidence = await strategy(features, known_signatures)
            if state_id:
                detection_results.append({
                    'state_id': state_id,
                    'confidence': confidence,
                    'strategy': strategy.__name__
                })
        
        if not detection_results:
            return None, 0.0, features
        
        # Aggregate results
        state_votes = defaultdict(float)
        strategy_votes = defaultdict(list)
        
        for result in detection_results:
            state_votes[result['state_id']] += result['confidence']
            strategy_votes[result['state_id']].append(result['strategy'])
        
        # Find best state
        best_state = max(state_votes.items(), key=lambda x: x[1])
        
        # Calculate final confidence
        # Bonus for multiple strategies agreeing
        agreement_bonus = len(strategy_votes[best_state[0]]) * 0.1
        final_confidence = min(1.0, best_state[1] + agreement_bonus)
        
        # Add detection metadata to features
        features['detection_metadata'] = {
            'strategies_used': [r['strategy'] for r in detection_results],
            'state_votes': dict(state_votes),
            'agreement_level': len(strategy_votes[best_state[0]]) / len(self.detection_strategies)
        }
        
        return best_state[0], final_confidence, features