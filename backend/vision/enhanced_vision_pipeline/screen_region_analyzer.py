#!/usr/bin/env python3
"""
Screen Region Analyzer - Enhanced Vision Pipeline v1.0
Stage 1: Screen Region Segmentation
======================================================

Captures screen and performs intelligent region segmentation using:
- Quadtree spatial partitioning
- Contrast enhancement
- Noise normalization
- DPI/Retina scaling detection

Author: Derek J. Russell
Date: October 2025
"""

import asyncio
import logging
import subprocess
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ScreenRegion:
    """Screen region with metadata"""
    x: int
    y: int
    width: int
    height: int
    image: Image.Image
    dpi_scale: float
    metadata: Dict[str, Any]


class QuadTreeNode:
    """Quadtree node for spatial partitioning"""
    
    def __init__(self, x: int, y: int, width: int, height: int, depth: int = 0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.depth = depth
        self.children: List['QuadTreeNode'] = []
        self.is_leaf = True
        self.variance = 0.0
    
    def subdivide(self):
        """Subdivide node into 4 quadrants"""
        if not self.is_leaf:
            return
        
        half_w = self.width // 2
        half_h = self.height // 2
        
        # Create 4 child nodes
        self.children = [
            QuadTreeNode(self.x, self.y, half_w, half_h, self.depth + 1),
            QuadTreeNode(self.x + half_w, self.y, half_w, half_h, self.depth + 1),
            QuadTreeNode(self.x, self.y + half_h, half_w, half_h, self.depth + 1),
            QuadTreeNode(self.x + half_w, self.y + half_h, half_w, half_h, self.depth + 1)
        ]
        
        self.is_leaf = False


class ScreenRegionAnalyzer:
    """
    Screen Region Analyzer
    
    Captures and segments screen regions using advanced techniques:
    - Quadtree hierarchical partitioning
    - Adaptive contrast enhancement
    - DPI-aware scaling
    - Multi-method capture with fallback
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize screen region analyzer"""
        self.config = config
        self.segment_config = config.get('segmentation', {})
        
        # Quadtree settings
        self.max_depth = self.segment_config.get('quadtree_max_depth', 4)
        self.variance_threshold = self.segment_config.get('variance_threshold', 50)
        
        # Image processing
        self.contrast_factor = self.segment_config.get('contrast_factor', 1.3)
        self.noise_reduction = self.segment_config.get('noise_reduction_enabled', True)
        
        # DPI detection
        self.dpi_scale = None
        
        # Cache directory
        self.cache_dir = Path.home() / '.jarvis' / 'screenshots' / 'pipeline_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("[SCREEN ANALYZER] Screen Region Analyzer initialized")
    
    async def initialize(self):
        """Initialize analyzer"""
        # Detect DPI scaling
        self.dpi_scale = await self._detect_dpi_scale()
        logger.info(f"[SCREEN ANALYZER] DPI scale detected: {self.dpi_scale}x")
    
    async def analyze_region(
        self,
        target: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze and segment screen region
        
        Args:
            target: Target element type ("control_center", "screen_mirroring", etc.)
            context: Optional context info
            
        Returns:
            Dict with segmented region and metadata
        """
        logger.info(f"[SCREEN ANALYZER] Analyzing region for target: {target}")
        
        try:
            # Determine region to capture based on target
            region_config = self._get_region_config(target)
            
            # Capture screen region
            screenshot = await self._capture_region(region_config)
            
            if screenshot is None:
                return {'success': False, 'error': 'Failed to capture screen'}
            
            # Enhance image
            enhanced = await self._enhance_image(screenshot)
            
            # Apply quadtree segmentation
            quadtree_result = await self._apply_quadtree_segmentation(enhanced, region_config)
            
            # Create screen region object
            region = ScreenRegion(
                x=region_config['x'],
                y=region_config['y'],
                width=region_config['width'],
                height=region_config['height'],
                image=enhanced,
                dpi_scale=self.dpi_scale,
                metadata={
                    'target': target,
                    'quadtree_nodes': quadtree_result['nodes'],
                    'interesting_regions': quadtree_result['interesting_regions'],
                    'capture_method': region_config.get('method', 'screencapture')
                }
            )
            
            logger.info(f"[SCREEN ANALYZER] ✅ Region analyzed: {region.width}x{region.height}")
            logger.info(f"[SCREEN ANALYZER] Found {len(quadtree_result['interesting_regions'])} interesting regions")
            
            return {
                'success': True,
                'segmented_region': region,
                'quadtree_result': quadtree_result
            }
            
        except Exception as e:
            logger.error(f"[SCREEN ANALYZER] Region analysis failed: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}
    
    def _get_region_config(self, target: str) -> Dict[str, Any]:
        """Get region configuration for target"""
        # Dynamic region configuration based on target
        region_configs = {
            'control_center': {
                'x': 0,
                'y': 0,
                'width': -1,  # Full width
                'height': 30,  # Menu bar height
                'name': 'menu_bar'
            },
            'screen_mirroring': {
                'x': -400,  # Right side
                'y': 0,
                'width': 400,
                'height': 600,
                'name': 'control_center_panel'
            },
            'display_list': {
                'x': -400,
                'y': 0,
                'width': 400,
                'height': 800,
                'name': 'screen_mirroring_menu'
            }
        }
        
        return region_configs.get(target, region_configs['control_center'])
    
    async def _capture_region(self, region_config: Dict[str, Any]) -> Optional[Image.Image]:
        """Capture screen region using best available method"""
        try:
            # Method 1: screencapture command (macOS native)
            temp_path = self.cache_dir / f'capture_{region_config["name"]}.png'
            
            # Build screencapture command
            if region_config['width'] == -1:
                # Full screen
                cmd = ['screencapture', '-x', str(temp_path)]
            else:
                # Specific region
                # Note: screencapture -R uses x,y,width,height
                x = region_config['x']
                y = region_config['y']
                w = region_config['width']
                h = region_config['height']
                
                cmd = ['screencapture', '-R', f'{x},{y},{w},{h}', '-x', str(temp_path)]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await process.communicate()
            
            if temp_path.exists():
                image = Image.open(temp_path)
                logger.debug(f"[SCREEN ANALYZER] Captured {image.size[0]}x{image.size[1]} region")
                return image
            
            return None
            
        except Exception as e:
            logger.error(f"[SCREEN ANALYZER] Capture failed: {e}")
            return None
    
    async def _enhance_image(self, image: Image.Image) -> Image.Image:
        """Enhance image for better detection"""
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Contrast enhancement
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(self.contrast_factor)
            
            # Noise reduction
            if self.noise_reduction:
                image = image.filter(ImageFilter.MedianFilter(size=3))
            
            # Sharpening
            image = image.filter(ImageFilter.SHARPEN)
            
            logger.debug("[SCREEN ANALYZER] Image enhanced")
            return image
            
        except Exception as e:
            logger.warning(f"[SCREEN ANALYZER] Enhancement failed: {e}")
            return image
    
    async def _apply_quadtree_segmentation(
        self,
        image: Image.Image,
        region_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply quadtree spatial partitioning"""
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Create root node
            root = QuadTreeNode(0, 0, image.width, image.height, 0)
            
            # Build quadtree
            nodes = await self._build_quadtree(root, img_array)
            
            # Find interesting regions (high variance = likely contains UI elements)
            interesting_regions = [
                {
                    'x': node.x,
                    'y': node.y,
                    'width': node.width,
                    'height': node.height,
                    'variance': node.variance
                }
                for node in nodes
                if node.is_leaf and node.variance > self.variance_threshold
            ]
            
            # Sort by variance (most interesting first)
            interesting_regions.sort(key=lambda r: r['variance'], reverse=True)
            
            logger.debug(f"[SCREEN ANALYZER] Quadtree: {len(nodes)} nodes, {len(interesting_regions)} interesting")
            
            return {
                'nodes': len(nodes),
                'interesting_regions': interesting_regions[:10]  # Top 10
            }
            
        except Exception as e:
            logger.warning(f"[SCREEN ANALYZER] Quadtree segmentation failed: {e}")
            return {'nodes': 0, 'interesting_regions': []}
    
    async def _build_quadtree(
        self,
        node: QuadTreeNode,
        img_array: np.ndarray
    ) -> List[QuadTreeNode]:
        """Recursively build quadtree"""
        nodes = [node]
        
        # Calculate variance of region
        region = img_array[node.y:node.y + node.height, node.x:node.x + node.width]
        node.variance = float(np.var(region))
        
        # Subdivide if high variance and not at max depth
        if node.variance > self.variance_threshold and node.depth < self.max_depth:
            if node.width >= 4 and node.height >= 4:  # Min size
                node.subdivide()
                
                # Recursively process children
                for child in node.children:
                    child_nodes = await self._build_quadtree(child, img_array)
                    nodes.extend(child_nodes)
        
        return nodes
    
    async def _detect_dpi_scale(self) -> float:
        """Detect DPI scaling factor (for Retina displays)"""
        try:
            import subprocess
            
            # Get screen info using system_profiler
            result = subprocess.run(
                ['system_profiler', 'SPDisplaysDataType'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            # Look for Retina indicators
            if 'Retina' in result.stdout:
                return 2.0
            
            # Default to 1.0
            return 1.0
            
        except Exception as e:
            logger.warning(f"[SCREEN ANALYZER] DPI detection failed: {e}")
            return 1.0
