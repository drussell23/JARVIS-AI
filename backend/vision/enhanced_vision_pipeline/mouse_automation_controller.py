#!/usr/bin/env python3
"""
Mouse Automation Controller - Enhanced Vision Pipeline v1.0
Stage 5: Mouse Automation Execution
===================================================

Physics-based mouse control with:
- Smooth acceleration/deceleration curves
- Bezier trajectory planning
- Human-like timing
- Click confirmation

Author: Derek J. Russell
Date: October 2025
"""

import asyncio
import logging
import time
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import pyautogui

logger = logging.getLogger(__name__)


@dataclass
class MouseActionResult:
    """Mouse action execution result"""
    success: bool
    action: str
    coordinates: Tuple[int, int]
    execution_time_ms: float
    trajectory_points: int
    metadata: Dict[str, Any]


class MouseAutomationController:
    """
    Mouse Automation Controller
    
    Executes mouse movements with human-like characteristics:
    - Bezier curves for smooth trajectories
    - Acceleration/deceleration (ease-in/ease-out)
    - Variable speed based on distance
    - Micro-adjustments for precision
    - Click confirmation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize mouse automation controller"""
        self.config = config
        self.mouse_config = config.get('mouse_automation', {})
        
        # Movement settings
        self.min_duration = self.mouse_config.get('min_duration_ms', 200) / 1000.0
        self.max_duration = self.mouse_config.get('max_duration_ms', 500) / 1000.0
        self.acceleration_factor = self.mouse_config.get('acceleration_factor', 0.3)
        
        # Click settings
        self.click_delay = self.mouse_config.get('click_delay_ms', 100) / 1000.0
        self.double_click = self.mouse_config.get('enable_double_click', False)
        
        # Bezier curve control
        self.curve_control_points = self.mouse_config.get('bezier_control_points', 3)
        
        # Safety
        pyautogui.PAUSE = 0.1
        pyautogui.FAILSAFE = True
        
        logger.info("[MOUSE CTRL] Mouse Automation Controller initialized")
    
    async def initialize(self):
        """Initialize mouse controller"""
        # Get current mouse position
        self.last_position = pyautogui.position()
        logger.info(f"[MOUSE CTRL] Current mouse position: {self.last_position}")
    
    async def execute_click(
        self,
        coordinates: Tuple[int, int],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute mouse click at coordinates
        
        Args:
            coordinates: (x, y) to click
            context: Optional context
            
        Returns:
            Dict with execution results
        """
        logger.info(f"[MOUSE CTRL] Executing click at {coordinates}")
        
        start_time = time.time()
        
        try:
            target_x, target_y = coordinates
            
            # Get current position
            current_pos = pyautogui.position()
            
            # Calculate distance
            distance = math.sqrt(
                (target_x - current_pos[0])**2 + (target_y - current_pos[1])**2
            )
            
            # Calculate movement duration based on distance
            duration = self._calculate_duration(distance)
            
            # Generate trajectory
            trajectory = self._generate_bezier_trajectory(
                current_pos,
                coordinates,
                num_points=max(10, int(distance / 10))
            )
            
            # Execute smooth movement
            await self._execute_trajectory(trajectory, duration)
            
            # Micro-adjustment for precision
            await self._micro_adjust(coordinates)
            
            # Execute click
            await self._execute_click_action(coordinates)
            
            # Confirm click (optional)
            clicked_successfully = await self._confirm_click()
            
            execution_time = (time.time() - start_time) * 1000
            
            result = MouseActionResult(
                success=clicked_successfully,
                action='click',
                coordinates=coordinates,
                execution_time_ms=execution_time,
                trajectory_points=len(trajectory),
                metadata={
                    'distance_px': distance,
                    'duration_ms': duration * 1000,
                    'current_position': pyautogui.position()
                }
            )
            
            logger.info(f"[MOUSE CTRL] ✅ Click executed in {execution_time:.1f}ms")
            logger.info(f"[MOUSE CTRL] Trajectory: {len(trajectory)} points over {distance:.1f}px")
            
            return {
                'success': True,
                'action_result': result
            }
            
        except Exception as e:
            logger.error(f"[MOUSE CTRL] Click execution failed: {e}", exc_info=True)
            
            execution_time = (time.time() - start_time) * 1000
            
            return {
                'success': False,
                'error': str(e),
                'execution_time_ms': execution_time
            }
    
    def _calculate_duration(self, distance: float) -> float:
        """Calculate movement duration based on distance"""
        # Longer distances get more time, but with diminishing returns
        # Uses logarithmic scaling for natural movement
        
        if distance < 10:
            return self.min_duration
        
        # Logarithmic scaling
        duration = self.min_duration + (math.log(distance) / math.log(1000)) * (self.max_duration - self.min_duration)
        
        return max(self.min_duration, min(duration, self.max_duration))
    
    def _generate_bezier_trajectory(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        num_points: int
    ) -> List[Tuple[int, int]]:
        """Generate Bezier curve trajectory"""
        # Generate control points for natural curve
        x0, y0 = start
        x1, y1 = end
        
        # Create control points slightly offset for natural curve
        dx = x1 - x0
        dy = y1 - y0
        
        # Control points create a slight arc
        cx1 = x0 + dx * 0.33 + dy * 0.1
        cy1 = y0 + dy * 0.33 - dx * 0.1
        
        cx2 = x0 + dx * 0.66 - dy * 0.1
        cy2 = y0 + dy * 0.66 + dx * 0.1
        
        # Generate points along Bezier curve
        trajectory = []
        for i in range(num_points):
            t = i / (num_points - 1)
            
            # Apply ease-in/ease-out
            t = self._ease_in_out(t)
            
            # Cubic Bezier formula
            x = (1-t)**3 * x0 + 3*(1-t)**2*t * cx1 + 3*(1-t)*t**2 * cx2 + t**3 * x1
            y = (1-t)**3 * y0 + 3*(1-t)**2*t * cy1 + 3*(1-t)*t**2 * cy2 + t**3 * y1
            
            trajectory.append((int(x), int(y)))
        
        return trajectory
    
    def _ease_in_out(self, t: float) -> float:
        """Ease-in/ease-out function for smooth acceleration"""
        # Smooth cubic easing
        if t < 0.5:
            return 4 * t**3
        else:
            return 1 - (-2 * t + 2)**3 / 2
    
    async def _execute_trajectory(
        self,
        trajectory: List[Tuple[int, int]],
        total_duration: float
    ):
        """Execute movement along trajectory"""
        if not trajectory:
            return
        
        delay_per_point = total_duration / len(trajectory)
        
        for x, y in trajectory:
            pyautogui.moveTo(x, y, duration=0)
            await asyncio.sleep(delay_per_point)
    
    async def _micro_adjust(self, target: Tuple[int, int]):
        """Make micro-adjustment to exact position"""
        # Final precise movement
        x, y = target
        pyautogui.moveTo(x, y, duration=0.05)
        await asyncio.sleep(0.05)
    
    async def _execute_click_action(self, coordinates: Tuple[int, int]):
        """Execute the actual click"""
        x, y = coordinates
        
        # Move to exact position
        pyautogui.moveTo(x, y)
        
        # Wait briefly (human-like)
        await asyncio.sleep(self.click_delay)
        
        # Click
        pyautogui.click(x, y)
        
        logger.debug(f"[MOUSE CTRL] Clicked at ({x}, {y})")
    
    async def _confirm_click(self) -> bool:
        """Confirm click was executed"""
        # In future: Check for UI changes, click feedback, etc.
        # For now: assume success
        return True
