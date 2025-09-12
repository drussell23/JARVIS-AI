#!/usr/bin/env python3
"""
Helper for robust Weather app navigation
Ensures Weather app stays focused and properly selects My Location
"""

import asyncio
import logging

logger = logging.getLogger(__name__)


class WeatherNavigationHelper:
    """Handles robust navigation in Weather app"""
    
    def __init__(self, controller):
        self.controller = controller
        
    async def select_my_location_robust(self):
        """
        Robustly select My Location in Weather app
        Uses direct Toronto selector
        """
        try:
            logger.info("Starting Toronto selection")
            
            # Import and use human-like clicker
            from .weather_human_clicker import WeatherHumanClicker
            clicker = WeatherHumanClicker(self.controller)
            
            # If vision handler is available, pass it
            if hasattr(self, 'vision_handler'):
                clicker.vision_handler = self.vision_handler
            
            # Use human-like clicking method
            success = await clicker.click_toronto_like_human()
            
            if success:
                logger.info("Successfully selected Toronto with human-like clicking")
                return True
                
            # If that fails, try the direct selector
            from .weather_toronto_selector import WeatherTorontoSelector
            selector = WeatherTorontoSelector(self.controller)
            
            if hasattr(self, 'vision_handler'):
                selector.vision_handler = self.vision_handler
                
            success = await selector.select_toronto()
            
            if success:
                logger.info("Successfully selected Toronto with direct selector")
                return True
            
            # If direct selection fails, try original approach as fallback
            logger.warning("Direct selection failed, trying original approach")
            
        except Exception as e:
            logger.error(f"Toronto selection error: {e}")
            
        # Original fallback approach
        try:
            logger.info("Using original keyboard navigation as final fallback")
            
            script = '''
            -- Force Weather to be absolutely frontmost
            tell application "Weather"
                activate
                reopen
                set frontmost to true
            end tell
            delay 1
            
            -- Now navigate in Weather
            tell application "System Events"
                tell process "Weather"
                    set frontmost to true
                    
                    -- Press up arrow 3 times to ensure we're at top
                    key code 126 -- Up arrow
                    delay 0.3
                    key code 126 -- Up arrow
                    delay 0.3
                    key code 126 -- Up arrow
                    delay 0.3
                    
                    -- Now press Enter to select first item (Toronto)
                    key code 36 -- Return
                    delay 3.0  -- Give it time to load
                end tell
            end tell
            '''
            
            success, result = self.controller.execute_applescript(script)
            if success:
                logger.info("Navigated with keyboard")
                await asyncio.sleep(2.5)
                await self.ensure_weather_focused()
                return True
                
        except Exception as e:
            logger.error(f"Original navigation also failed: {e}")
            
        return False
            
    async def ensure_weather_focused(self):
        """Ensure Weather app is focused before any operation"""
        script = '''
        tell application "Weather"
            if not frontmost then
                activate
                delay 0.2
            end if
        end tell
        '''
        self.controller.execute_applescript(script)
        await asyncio.sleep(0.2)