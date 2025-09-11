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
        Uses multiple strategies to ensure success
        """
        try:
            logger.info("Starting robust My Location selection")
            
            # Strategy 1: Use a more robust selection method
            # First, absolutely ensure Weather is the active app
            script = '''
            -- Force Weather to be absolutely frontmost
            tell application "Weather"
                activate
                reopen
                set frontmost to true
            end tell
            delay 1
            
            -- Make Weather the key window
            tell application "System Events"
                tell process "Weather"
                    set frontmost to true
                    if (count windows) > 0 then
                        set frontmost of window 1 to true
                    end if
                end tell
            end tell
            delay 0.5
            
            -- Now navigate in Weather
            tell application "System Events"
                tell process "Weather"
                    set frontmost to true
                    
                    -- Use arrow keys method which successfully selects Toronto
                    -- First ensure sidebar has focus
                    click at {125, 200}
                    delay 0.5
                    
                    -- Multiple up arrows to go to top
                    key code 126 -- Up arrow
                    delay 0.3
                    key code 126 -- Up arrow
                    delay 0.3
                    key code 126 -- Up arrow  
                    delay 0.3
                    key code 126 -- Up arrow
                    delay 0.3
                    
                    -- Return to select Toronto
                    key code 36 -- Return
                    delay 1.0
                    
                    -- Double-check by pressing Return again
                    key code 36 -- Return  
                    delay 2.5
                end tell
            end tell
            '''
            
            success, result = self.controller.execute_applescript(script)
            if success:
                logger.info("Successfully navigated to My Location using AppleScript")
                await asyncio.sleep(2.5)  # Wait for weather to load
                
                # Final ensure Weather is still active
                await self.ensure_weather_focused()
                return True
                
        except Exception as e:
            logger.error(f"Robust navigation failed: {e}")
            
        # Fallback: Simple click approach
        try:
            logger.info("Trying fallback click approach")
            
            # Ensure Weather is active
            self.controller.execute_applescript('''
                tell application "Weather" to activate
            ''')
            await asyncio.sleep(0.5)
            
            # Try extreme top coordinate for Toronto
            # Weather app sidebar starts around Y=45-50 based on testing
            # Toronto/My Location should be the very first item
            # First click
            await self.controller.click_at(125, 45)
            await asyncio.sleep(0.3)
            
            # Second click to ensure selection
            await self.controller.click_at(125, 45)
            await asyncio.sleep(2.5)
            
            # Keep Weather active
            self.controller.execute_applescript('''
                tell application "Weather" to activate
            ''')
            
            return True
            
        except Exception as e:
            logger.error(f"Fallback navigation failed: {e}")
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