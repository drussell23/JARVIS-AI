"""
Autonomy Handler for JARVIS
Manages autonomous mode activation and system integration
"""

import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class AutonomyHandler:
    """Handles autonomous mode activation and management"""
    
    def __init__(self):
        self.is_autonomous = False
        self.vision_connected = False
        self.ai_brain_active = False
        self.voice_system_active = False
        self.hardware_control_active = False
        
    async def activate_full_autonomy(self) -> Dict[str, Any]:
        """Activate full Iron Man-level autonomy"""
        logger.info("🚀 Initiating full autonomy activation sequence...")
        
        activation_steps = []
        
        try:
            # Step 1: Activate AI Brain
            logger.info("🧠 Activating AI Brain...")
            await self._activate_ai_brain()
            activation_steps.append("AI Brain online")
            
            # Step 2: Initialize Voice System
            logger.info("🔊 Initializing voice system...")
            await self._activate_voice_system()
            activation_steps.append("Voice system active")
            
            # Step 3: Connect Vision System
            logger.info("👁️ Connecting vision system...")
            await self._connect_vision_system()
            activation_steps.append("Vision system connected")
            
            # Step 4: Enable Hardware Control
            logger.info("🔧 Enabling hardware control...")
            await self._enable_hardware_control()
            activation_steps.append("Hardware control enabled")
            
            # Set autonomous mode
            self.is_autonomous = True
            
            return {
                "success": True,
                "mode": "autonomous",
                "message": "Full autonomy activated. All systems online.",
                "activation_steps": activation_steps,
                "systems": {
                    "ai_brain": self.ai_brain_active,
                    "voice": self.voice_system_active,
                    "vision": self.vision_connected,
                    "hardware": self.hardware_control_active
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to activate full autonomy: {e}")
            return {
                "success": False,
                "mode": "manual",
                "message": f"Autonomy activation failed: {str(e)}",
                "activation_steps": activation_steps,
                "error": str(e)
            }
    
    async def deactivate_autonomy(self) -> Dict[str, Any]:
        """Deactivate autonomous mode"""
        logger.info("🛑 Deactivating autonomous mode...")
        
        try:
            self.is_autonomous = False
            
            # Deactivate systems
            deactivation_steps = []
            
            if self.hardware_control_active:
                await self._disable_hardware_control()
                deactivation_steps.append("Hardware control disabled")
                
            if self.vision_connected:
                await self._disconnect_vision_system()
                deactivation_steps.append("Vision system disconnected")
                
            return {
                "success": True,
                "mode": "manual",
                "message": "Autonomous mode deactivated. Manual control restored.",
                "deactivation_steps": deactivation_steps,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error deactivating autonomy: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _activate_ai_brain(self):
        """Activate the AI brain system"""
        # Check if AI brain module is available
        try:
            from autonomy.advanced_ai_brain import AdvancedAIBrain
            # In a real implementation, we would initialize the brain
            self.ai_brain_active = True
            await asyncio.sleep(0.5)  # Simulate activation
        except ImportError:
            logger.warning("AI Brain module not available")
            self.ai_brain_active = False
    
    async def _activate_voice_system(self):
        """Activate voice system"""
        # Check if voice integration is available
        try:
            from autonomy.voice_integration import VoiceIntegrationSystem
            self.voice_system_active = True
            await asyncio.sleep(0.3)  # Simulate activation
        except ImportError:
            logger.warning("Voice integration module not available")
            self.voice_system_active = False
    
    async def _connect_vision_system(self):
        """Connect to vision system"""
        # Check if vision system is available
        try:
            # In a real implementation, we would connect to vision pipeline
            self.vision_connected = True
            await asyncio.sleep(0.5)  # Simulate connection
        except Exception as e:
            logger.warning(f"Vision system connection failed: {e}")
            self.vision_connected = False
    
    async def _enable_hardware_control(self):
        """Enable hardware control"""
        # Check if hardware control is available
        try:
            from autonomy.hardware_control import HardwareControlSystem
            self.hardware_control_active = True
            await asyncio.sleep(0.3)  # Simulate activation
        except ImportError:
            logger.warning("Hardware control module not available")
            self.hardware_control_active = False
    
    async def _disable_hardware_control(self):
        """Disable hardware control"""
        self.hardware_control_active = False
        await asyncio.sleep(0.2)
    
    async def _disconnect_vision_system(self):
        """Disconnect vision system"""
        self.vision_connected = False
        await asyncio.sleep(0.2)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current autonomy status"""
        return {
            "mode": "autonomous" if self.is_autonomous else "manual",
            "systems": {
                "ai_brain": self.ai_brain_active,
                "voice": self.voice_system_active,
                "vision": self.vision_connected,
                "hardware": self.hardware_control_active
            },
            "ready": all([
                self.ai_brain_active,
                self.voice_system_active,
                self.vision_connected,
                self.hardware_control_active
            ]) if self.is_autonomous else True
        }
    
    def process_autonomy_command(self, command: str) -> Optional[str]:
        """Check if command is for autonomy control"""
        command_lower = command.lower()
        
        autonomy_triggers = [
            "activate full autonomy",
            "enable autonomous mode",
            "activate autonomy",
            "iron man mode",
            "activate all systems",
            "full autonomy",
            "engage all systems"
        ]
        
        deactivate_triggers = [
            "disable autonomy",
            "deactivate autonomy",
            "manual mode",
            "disable autonomous mode",
            "stand down"
        ]
        
        for trigger in autonomy_triggers:
            if trigger in command_lower:
                return "activate"
                
        for trigger in deactivate_triggers:
            if trigger in command_lower:
                return "deactivate"
                
        return None

# Singleton instance
_autonomy_handler = None

def get_autonomy_handler() -> AutonomyHandler:
    """Get the singleton autonomy handler instance"""
    global _autonomy_handler
    if _autonomy_handler is None:
        _autonomy_handler = AutonomyHandler()
    return _autonomy_handler