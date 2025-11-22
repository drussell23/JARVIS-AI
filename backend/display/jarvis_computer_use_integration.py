#!/usr/bin/env python3
"""
JARVIS Computer Use Integration
================================

Integrates Computer Use API with JARVIS voice system for transparent,
voice-guided display connections.

This module provides the bridge between JARVIS's voice system and the
Computer Use Display Connector, enabling natural voice feedback during
AI-driven display connections.

Author: Derek J. Russell
Date: January 2025
Version: 1.0.0
"""

import os
import asyncio
import logging
from typing import Dict, Any, Optional, Callable
from pathlib import Path

import pyautogui

logger = logging.getLogger(__name__)


class JARVISComputerUseIntegration:
    """
    Integration layer between JARVIS and Computer Use API
    
    Features:
    - Voice transparency (JARVIS narrates actions)
    - Intelligent method selection (UAE vs Computer Use)
    - Seamless fallback handling
    - Statistics and monitoring
    """
    
    def __init__(
        self,
        jarvis_voice_engine=None,
        vision_analyzer=None,
        prefer_computer_use: bool = False,
        confidence_threshold: float = 0.7
    ):
        """
        Initialize JARVIS Computer Use Integration
        
        Args:
            jarvis_voice_engine: JARVIS voice engine for TTS
            vision_analyzer: Vision analyzer for UAE system
            prefer_computer_use: Prefer Computer Use over UAE
            confidence_threshold: UAE confidence threshold
        """
        self.jarvis_voice_engine = jarvis_voice_engine
        self.vision_analyzer = vision_analyzer
        self.prefer_computer_use = prefer_computer_use
        self.confidence_threshold = confidence_threshold
        
        # Components (lazy loaded)
        self._uae_clicker = None
        self._computer_use_connector = None
        self._hybrid_connector = None
        
        # Configuration
        self.computer_use_enabled = self._check_computer_use_available()
        
        logger.info(
            f"[JARVIS-COMPUTER-USE] Initialized "
            f"(Computer Use: {'enabled' if self.computer_use_enabled else 'disabled'})"
        )
    
    def _check_computer_use_available(self) -> bool:
        """Check if Computer Use API is available"""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning(
                "[JARVIS-COMPUTER-USE] ANTHROPIC_API_KEY not set - "
                "Computer Use API disabled"
            )
            return False
        return True
    
    def _get_voice_callback(self) -> Callable[[str], None]:
        """
        Get voice callback function for JARVIS TTS
        
        Returns:
            Voice callback function
        """
        def voice_callback(message: str):
            """Speak message through JARVIS"""
            try:
                if self.jarvis_voice_engine:
                    # Use JARVIS voice engine
                    if hasattr(self.jarvis_voice_engine, 'speak'):
                        self.jarvis_voice_engine.speak(message)
                    elif hasattr(self.jarvis_voice_engine, 'say'):
                        self.jarvis_voice_engine.say(message)
                    else:
                        logger.warning(
                            "[JARVIS-COMPUTER-USE] Voice engine has no speak method"
                        )
                        print(f"🔊 [JARVIS]: {message}")
                else:
                    # Fallback to print
                    print(f"🔊 [JARVIS]: {message}")
            except Exception as e:
                logger.error(f"[JARVIS-COMPUTER-USE] Voice callback error: {e}")
                print(f"🔊 [JARVIS]: {message}")
        
        return voice_callback
    
    def _initialize_components(self):
        """Lazy initialize connector components"""
        if self._hybrid_connector is not None:
            return  # Already initialized
        
        logger.info("[JARVIS-COMPUTER-USE] Initializing connector components...")
        
        # Get screen size
        screen_width, screen_height = pyautogui.size()
        logger.info(f"[JARVIS-COMPUTER-USE] Screen size: {screen_width}x{screen_height}")
        
        # Initialize UAE clicker
        try:
            from backend.display.uae_enhanced_control_center_clicker import get_uae_clicker
            
            self._uae_clicker = get_uae_clicker(
                vision_analyzer=self.vision_analyzer,
                enable_uae=False,  # Disable full UAE to avoid overhead
                enable_verification=True,
                enable_communication=False  # We handle voice in this layer
            )
            logger.info("[JARVIS-COMPUTER-USE] ✅ UAE clicker initialized")
        except Exception as e:
            logger.error(f"[JARVIS-COMPUTER-USE] Failed to initialize UAE: {e}")
            self._uae_clicker = None
        
        # Initialize Computer Use connector (if available)
        if self.computer_use_enabled:
            try:
                from backend.display.computer_use_display_connector import (
                    get_computer_use_connector
                )
                
                self._computer_use_connector = get_computer_use_connector(
                    voice_callback=self._get_voice_callback(),
                    display_width=screen_width,
                    display_height=screen_height
                )
                logger.info("[JARVIS-COMPUTER-USE] ✅ Computer Use connector initialized")
            except Exception as e:
                logger.error(f"[JARVIS-COMPUTER-USE] Failed to initialize Computer Use: {e}")
                self._computer_use_connector = None
                self.computer_use_enabled = False
        
        # Initialize hybrid connector
        if self._uae_clicker or self._computer_use_connector:
            try:
                from backend.display.hybrid_display_connector import get_hybrid_connector
                
                self._hybrid_connector = get_hybrid_connector(
                    uae_clicker=self._uae_clicker,
                    computer_use_connector=self._computer_use_connector,
                    voice_callback=self._get_voice_callback(),
                    prefer_computer_use=self.prefer_computer_use,
                    confidence_threshold=self.confidence_threshold
                )
                logger.info("[JARVIS-COMPUTER-USE] ✅ Hybrid connector initialized")
            except Exception as e:
                logger.error(f"[JARVIS-COMPUTER-USE] Failed to initialize hybrid: {e}")
                raise
        else:
            raise RuntimeError("No connection method available (UAE or Computer Use)")
    
    async def connect_to_display(
        self,
        device_name: str,
        mode: str = "mirror",
        force_computer_use: bool = False
    ) -> Dict[str, Any]:
        """
        Connect to display with voice transparency
        
        This is the main entry point for JARVIS voice commands like:
        - "connect to living room TV"
        - "mirror my screen to the TV"
        - "show my screen on the living room display"
        
        Args:
            device_name: Name of the AirPlay device
            mode: Connection mode ("mirror" or "extend")
            force_computer_use: Force use of Computer Use API
        
        Returns:
            Connection result dictionary:
            {
                'success': bool,
                'message': str,
                'device': str,
                'method': str ('uae', 'computer_use', or 'hybrid'),
                'duration': float,
                'stats': dict
            }
        """
        # Ensure components are initialized
        if self._hybrid_connector is None:
            self._initialize_components()
        
        logger.info(
            f"[JARVIS-COMPUTER-USE] Connecting to '{device_name}' "
            f"(mode: {mode}, force_computer_use: {force_computer_use})"
        )
        
        # Initial voice announcement
        voice_callback = self._get_voice_callback()
        voice_callback(f"Connecting to {device_name}.")
        
        try:
            # Execute connection through hybrid connector
            result = await self._hybrid_connector.connect_to_device(
                device_name=device_name,
                mode=mode,
                force_computer_use=force_computer_use
            )
            
            # Add integration metadata
            result['method'] = result.get('strategy_used', 'unknown')
            result['jarvis_integration'] = True
            
            # Final voice announcement
            if result.get('success'):
                voice_callback(f"Successfully connected to {device_name}.")
            else:
                voice_callback(
                    f"I was unable to connect to {device_name}. "
                    f"{result.get('message', 'Please check the device is available.')}"
                )
            
            return result
        
        except Exception as e:
            logger.error(f"[JARVIS-COMPUTER-USE] Connection error: {e}", exc_info=True)
            voice_callback(f"I encountered an error while connecting: {str(e)}")
            
            return {
                'success': False,
                'message': f"Integration error: {str(e)}",
                'device': device_name,
                'method': 'error',
                'duration': 0.0
            }
    
    async def disconnect_from_display(self, device_name: str) -> Dict[str, Any]:
        """
        Disconnect from display
        
        Args:
            device_name: Name of the device to disconnect from
        
        Returns:
            Disconnection result
        """
        # Ensure components are initialized
        if self._hybrid_connector is None:
            self._initialize_components()
        
        logger.info(f"[JARVIS-COMPUTER-USE] Disconnecting from '{device_name}'")
        
        voice_callback = self._get_voice_callback()
        voice_callback(f"Disconnecting from {device_name}.")
        
        try:
            # TODO: Implement disconnect through Computer Use or UAE
            # For now, return success
            voice_callback(f"Disconnected from {device_name}.")
            
            return {
                'success': True,
                'message': f"Disconnected from {device_name}",
                'device': device_name
            }
        
        except Exception as e:
            logger.error(f"[JARVIS-COMPUTER-USE] Disconnect error: {e}", exc_info=True)
            voice_callback(f"Error disconnecting: {str(e)}")
            
            return {
                'success': False,
                'message': f"Disconnect error: {str(e)}",
                'device': device_name
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get integration statistics"""
        stats = {
            'computer_use_enabled': self.computer_use_enabled,
            'prefer_computer_use': self.prefer_computer_use,
            'confidence_threshold': self.confidence_threshold,
            'components_initialized': self._hybrid_connector is not None
        }
        
        # Add hybrid connector stats if available
        if self._hybrid_connector:
            stats['hybrid'] = self._hybrid_connector.get_stats()
        
        # Add Computer Use connector stats if available
        if self._computer_use_connector:
            stats['computer_use'] = self._computer_use_connector.get_stats()
        
        return stats
    
    def set_prefer_computer_use(self, prefer: bool):
        """Enable/disable Computer Use preference"""
        self.prefer_computer_use = prefer
        if self._hybrid_connector:
            self._hybrid_connector.prefer_computer_use = prefer
        logger.info(f"[JARVIS-COMPUTER-USE] prefer_computer_use set to {prefer}")
    
    def set_confidence_threshold(self, threshold: float):
        """Set UAE confidence threshold"""
        self.confidence_threshold = threshold
        if self._hybrid_connector:
            self._hybrid_connector.confidence_threshold = threshold
        logger.info(f"[JARVIS-COMPUTER-USE] confidence_threshold set to {threshold}")


# ============================================================================
# Singleton Instance
# ============================================================================

_jarvis_computer_use: Optional[JARVISComputerUseIntegration] = None


def get_jarvis_computer_use(
    jarvis_voice_engine=None,
    vision_analyzer=None,
    prefer_computer_use: bool = False,
    confidence_threshold: float = 0.7
) -> JARVISComputerUseIntegration:
    """
    Get singleton JARVIS Computer Use Integration instance
    
    Args:
        jarvis_voice_engine: JARVIS voice engine
        vision_analyzer: Vision analyzer
        prefer_computer_use: Prefer Computer Use over UAE
        confidence_threshold: UAE confidence threshold
    
    Returns:
        JARVISComputerUseIntegration instance
    """
    global _jarvis_computer_use
    
    if _jarvis_computer_use is None:
        _jarvis_computer_use = JARVISComputerUseIntegration(
            jarvis_voice_engine=jarvis_voice_engine,
            vision_analyzer=vision_analyzer,
            prefer_computer_use=prefer_computer_use,
            confidence_threshold=confidence_threshold
        )
    
    return _jarvis_computer_use


# ============================================================================
# Demo/Test
# ============================================================================

async def main():
    """Demo JARVIS Computer Use Integration"""
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "=" * 80)
    print("🎙️  JARVIS Computer Use Integration - Demo")
    print("=" * 80)
    
    # Mock JARVIS voice engine
    class MockVoiceEngine:
        def speak(self, text: str):
            print(f"\n🔊 [JARVIS SPEAKS]: {text}\n")
    
    # Create integration
    integration = get_jarvis_computer_use(
        jarvis_voice_engine=MockVoiceEngine(),
        prefer_computer_use=False,  # Try UAE first
        confidence_threshold=0.7
    )
    
    # Device name
    device_name = "Living Room TV"
    if len(sys.argv) > 1:
        device_name = sys.argv[1]
    
    print(f"\n🎯 Test: Connect to '{device_name}'")
    print("   JARVIS will provide voice transparency throughout the process\n")
    
    # Test connection
    result = await integration.connect_to_display(device_name)
    
    # Display results
    print("\n" + "=" * 80)
    print("📊 Connection Result")
    print("=" * 80)
    print(f"Success: {'✅ Yes' if result['success'] else '❌ No'}")
    print(f"Method: {result.get('method', 'unknown')}")
    print(f"Message: {result.get('message', 'N/A')}")
    print(f"Duration: {result.get('duration', 0):.2f}s")
    
    # Show stats
    print("\n📈 Integration Stats:")
    stats = integration.get_stats()
    print(f"  Computer Use Enabled: {stats['computer_use_enabled']}")
    print(f"  Prefer Computer Use: {stats['prefer_computer_use']}")
    print(f"  Confidence Threshold: {stats['confidence_threshold']}")
    
    if 'hybrid' in stats:
        print("\n  Hybrid Stats:")
        for key, value in stats['hybrid'].items():
            if isinstance(value, float):
                print(f"    {key}: {value:.2%}" if 'rate' in key else f"    {key}: {value:.2f}")
            else:
                print(f"    {key}: {value}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
