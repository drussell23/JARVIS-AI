#!/usr/bin/env python3
"""
Hybrid Display Connector
=========================

Intelligently combines UAE-based coordinate detection with Computer Use API.

Strategy:
1. Fast Path: Try UAE-based detection (fast, free, local)
2. Robust Path: Fall back to Computer Use API (slower, costs $, but very robust)
3. Learning: Computer Use successes teach UAE system new patterns

Author: Derek J. Russell
Date: January 2025
Version: 1.0.0
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ConnectionStrategy:
    """Strategy selection for display connection"""
    use_computer_use: bool
    reason: str
    confidence: float


class HybridDisplayConnector:
    """
    Hybrid display connector combining UAE and Computer Use API
    
    Decision Logic:
    - If UAE confidence > 0.8 → Use UAE (fast path)
    - If UAE confidence < 0.5 → Use Computer Use (robust path)
    - If UAE fails → Fall back to Computer Use
    - Learn from Computer Use successes
    """
    
    def __init__(
        self,
        uae_clicker=None,
        computer_use_connector=None,
        voice_callback: Optional[Callable[[str], None]] = None,
        prefer_computer_use: bool = False,
        confidence_threshold: float = 0.7
    ):
        """
        Initialize Hybrid Display Connector
        
        Args:
            uae_clicker: UAE-enhanced clicker (existing system)
            computer_use_connector: Computer Use API connector (new system)
            voice_callback: JARVIS voice callback
            prefer_computer_use: Prefer Computer Use over UAE
            confidence_threshold: Confidence threshold for UAE (0.0-1.0)
        """
        self.uae_clicker = uae_clicker
        self.computer_use_connector = computer_use_connector
        self.voice_callback = voice_callback
        self.prefer_computer_use = prefer_computer_use
        self.confidence_threshold = confidence_threshold
        
        # Statistics
        self.stats = {
            'total_connections': 0,
            'uae_attempts': 0,
            'uae_successes': 0,
            'computer_use_attempts': 0,
            'computer_use_successes': 0,
            'fallback_triggers': 0,
            'learning_events': 0
        }
        
        logger.info(
            f"[HYBRID] Initialized Hybrid Display Connector "
            f"(prefer_computer_use={prefer_computer_use}, "
            f"threshold={confidence_threshold})"
        )
    
    def _speak(self, message: str):
        """Provide voice feedback through JARVIS"""
        if self.voice_callback:
            try:
                self.voice_callback(message)
            except Exception as e:
                logger.error(f"[HYBRID] Voice callback failed: {e}")
        logger.info(f"[JARVIS VOICE] {message}")
    
    async def connect_to_device(
        self,
        device_name: str,
        mode: str = "mirror",
        force_computer_use: bool = False
    ) -> Dict[str, Any]:
        """
        Connect to AirPlay device using hybrid approach
        
        Args:
            device_name: Name of the AirPlay device
            mode: Connection mode ("mirror" or "extend")
            force_computer_use: Force use of Computer Use API
        
        Returns:
            Connection result dictionary
        """
        start_time = asyncio.get_event_loop().time()
        self.stats['total_connections'] += 1
        
        logger.info(f"[HYBRID] 🔗 Connecting to '{device_name}' (mode: {mode})")
        
        # Decide strategy
        strategy = await self._select_strategy(device_name, force_computer_use)
        
        logger.info(
            f"[HYBRID] Strategy: {'Computer Use API' if strategy.use_computer_use else 'UAE'} "
            f"(reason: {strategy.reason}, confidence: {strategy.confidence:.2f})"
        )
        
        result = None
        
        try:
            if strategy.use_computer_use:
                # Use Computer Use API
                result = await self._connect_with_computer_use(device_name, mode)
            else:
                # Use UAE system
                result = await self._connect_with_uae(device_name, mode)
                
                # If UAE failed and Computer Use is available, fall back
                if not result['success'] and self.computer_use_connector:
                    logger.info("[HYBRID] UAE failed, falling back to Computer Use API")
                    self._speak("Trying a more robust connection method.")
                    self.stats['fallback_triggers'] += 1
                    
                    result = await self._connect_with_computer_use(device_name, mode)
                    result['fallback_used'] = True
            
            # Add hybrid metadata
            result['strategy_used'] = 'computer_use' if strategy.use_computer_use else 'uae'
            result['strategy_reason'] = strategy.reason
            result['strategy_confidence'] = strategy.confidence
            result['duration'] = asyncio.get_event_loop().time() - start_time
            
            # Learn from Computer Use successes
            if strategy.use_computer_use and result.get('success'):
                await self._learn_from_computer_use(device_name, result)
            
            return result
        
        except Exception as e:
            logger.error(f"[HYBRID] Connection error: {e}", exc_info=True)
            return {
                'success': False,
                'message': f"Hybrid connection error: {str(e)}",
                'device': device_name,
                'duration': asyncio.get_event_loop().time() - start_time
            }
    
    async def _select_strategy(
        self,
        device_name: str,
        force_computer_use: bool
    ) -> ConnectionStrategy:
        """
        Select connection strategy (UAE vs Computer Use)
        
        Args:
            device_name: Device name
            force_computer_use: Force Computer Use API
        
        Returns:
            ConnectionStrategy with decision
        """
        # If forced, use Computer Use
        if force_computer_use:
            return ConnectionStrategy(
                use_computer_use=True,
                reason="Forced by user",
                confidence=1.0
            )
        
        # If prefer_computer_use is set
        if self.prefer_computer_use and self.computer_use_connector:
            return ConnectionStrategy(
                use_computer_use=True,
                reason="Preferred mode",
                confidence=1.0
            )
        
        # If only one system is available
        if not self.uae_clicker:
            return ConnectionStrategy(
                use_computer_use=True,
                reason="UAE not available",
                confidence=1.0
            )
        
        if not self.computer_use_connector:
            return ConnectionStrategy(
                use_computer_use=False,
                reason="Computer Use not available",
                confidence=1.0
            )
        
        # Check UAE confidence
        uae_confidence = await self._estimate_uae_confidence(device_name)
        
        if uae_confidence >= self.confidence_threshold:
            # UAE is confident - use it (fast path)
            return ConnectionStrategy(
                use_computer_use=False,
                reason=f"UAE confidence high ({uae_confidence:.2f})",
                confidence=uae_confidence
            )
        else:
            # UAE confidence low - use Computer Use (robust path)
            return ConnectionStrategy(
                use_computer_use=True,
                reason=f"UAE confidence low ({uae_confidence:.2f})",
                confidence=uae_confidence
            )
    
    async def _estimate_uae_confidence(self, device_name: str) -> float:
        """
        Estimate UAE system confidence for this connection
        
        Args:
            device_name: Device name
        
        Returns:
            Confidence score (0.0 to 1.0)
        """
        if not self.uae_clicker:
            return 0.0
        
        try:
            # Check if UAE has cached coordinates for key elements
            has_control_center = await self._uae_has_cached("control_center")
            has_screen_mirroring = await self._uae_has_cached("screen_mirroring")
            has_device = await self._uae_has_cached(device_name)
            
            # Calculate confidence based on cache status
            confidence = 0.0
            if has_control_center:
                confidence += 0.3
            if has_screen_mirroring:
                confidence += 0.3
            if has_device:
                confidence += 0.4
            
            # Check UAE metrics if available
            if hasattr(self.uae_clicker, 'get_metrics'):
                metrics = self.uae_clicker.get_metrics()
                cache_hit_rate = metrics.get('cache_hit_rate', 0)
                confidence *= (0.5 + 0.5 * cache_hit_rate)  # Adjust by cache hit rate
            
            return min(confidence, 1.0)
        
        except Exception as e:
            logger.warning(f"[HYBRID] Error estimating UAE confidence: {e}")
            return 0.5  # Default to medium confidence
    
    async def _uae_has_cached(self, target: str) -> bool:
        """Check if UAE has cached coordinates for target"""
        try:
            if hasattr(self.uae_clicker, 'coordinate_cache'):
                return target in self.uae_clicker.coordinate_cache
            return False
        except:
            return False
    
    async def _connect_with_uae(
        self,
        device_name: str,
        mode: str
    ) -> Dict[str, Any]:
        """
        Connect using UAE-based system
        
        Args:
            device_name: Device name
            mode: Connection mode
        
        Returns:
            Connection result
        """
        self.stats['uae_attempts'] += 1
        
        logger.info(f"[HYBRID] 🚀 Using UAE system for '{device_name}'")
        self._speak(f"Connecting to {device_name} using fast method.")
        
        try:
            result = await self.uae_clicker.connect_to_device(device_name)
            
            if result.get('success'):
                self.stats['uae_successes'] += 1
                logger.info(f"[HYBRID] ✅ UAE succeeded")
            else:
                logger.warning(f"[HYBRID] ❌ UAE failed: {result.get('message')}")
            
            return result
        
        except Exception as e:
            logger.error(f"[HYBRID] UAE error: {e}", exc_info=True)
            return {
                'success': False,
                'message': f"UAE error: {str(e)}",
                'device': device_name
            }
    
    async def _connect_with_computer_use(
        self,
        device_name: str,
        mode: str
    ) -> Dict[str, Any]:
        """
        Connect using Computer Use API
        
        Args:
            device_name: Device name
            mode: Connection mode
        
        Returns:
            Connection result
        """
        self.stats['computer_use_attempts'] += 1
        
        logger.info(f"[HYBRID] 🤖 Using Computer Use API for '{device_name}'")
        self._speak(f"Using vision-based AI to connect to {device_name}.")
        
        try:
            result = await self.computer_use_connector.connect_to_device(device_name, mode)
            
            if result.get('success'):
                self.stats['computer_use_successes'] += 1
                logger.info(f"[HYBRID] ✅ Computer Use succeeded")
            else:
                logger.warning(f"[HYBRID] ❌ Computer Use failed: {result.get('message')}")
            
            return result
        
        except Exception as e:
            logger.error(f"[HYBRID] Computer Use error: {e}", exc_info=True)
            return {
                'success': False,
                'message': f"Computer Use error: {str(e)}",
                'device': device_name
            }
    
    async def _learn_from_computer_use(
        self,
        device_name: str,
        result: Dict[str, Any]
    ):
        """
        Learn from successful Computer Use execution
        
        When Computer Use succeeds, extract any useful information
        that could improve UAE system's cache or knowledge.
        
        Args:
            device_name: Device name
            result: Computer Use result
        """
        try:
            self.stats['learning_events'] += 1
            logger.info("[HYBRID] 📚 Learning from Computer Use success")
            
            # TODO: Extract coordinate information from Computer Use reasoning
            # TODO: Update UAE cache with discovered coordinates
            # TODO: Update Knowledge Graph with successful patterns
            
            # For now, just log
            if result.get('reasoning'):
                logger.info(f"[HYBRID] Computer Use reasoning had {len(result['reasoning'])} steps")
        
        except Exception as e:
            logger.warning(f"[HYBRID] Error learning from Computer Use: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get hybrid connector statistics"""
        return {
            **self.stats,
            'uae_success_rate': (
                self.stats['uae_successes'] / self.stats['uae_attempts']
                if self.stats['uae_attempts'] > 0 else 0.0
            ),
            'computer_use_success_rate': (
                self.stats['computer_use_successes'] / self.stats['computer_use_attempts']
                if self.stats['computer_use_attempts'] > 0 else 0.0
            ),
            'overall_success_rate': (
                (self.stats['uae_successes'] + self.stats['computer_use_successes']) /
                self.stats['total_connections']
                if self.stats['total_connections'] > 0 else 0.0
            )
        }


# ============================================================================
# Factory Function
# ============================================================================

def get_hybrid_connector(
    uae_clicker=None,
    computer_use_connector=None,
    voice_callback: Optional[Callable[[str], None]] = None,
    prefer_computer_use: bool = False,
    confidence_threshold: float = 0.7
) -> HybridDisplayConnector:
    """
    Get Hybrid Display Connector instance
    
    Args:
        uae_clicker: UAE-enhanced clicker
        computer_use_connector: Computer Use API connector
        voice_callback: JARVIS voice callback
        prefer_computer_use: Prefer Computer Use over UAE
        confidence_threshold: UAE confidence threshold
    
    Returns:
        HybridDisplayConnector instance
    """
    return HybridDisplayConnector(
        uae_clicker=uae_clicker,
        computer_use_connector=computer_use_connector,
        voice_callback=voice_callback,
        prefer_computer_use=prefer_computer_use,
        confidence_threshold=confidence_threshold
    )


# ============================================================================
# Demo
# ============================================================================

async def main():
    """Demo Hybrid Display Connector"""
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from backend.display.uae_enhanced_control_center_clicker import get_uae_clicker
    from backend.display.computer_use_display_connector import get_computer_use_connector
    import pyautogui
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "=" * 80)
    print("🔀 Hybrid Display Connector - Demo")
    print("=" * 80)
    
    # Mock voice callback
    def voice_callback(message: str):
        print(f"\n🔊 [JARVIS]: {message}\n")
    
    # Get screen size
    screen_width, screen_height = pyautogui.size()
    
    # Create UAE clicker
    print("\n📦 Initializing UAE clicker...")
    uae_clicker = get_uae_clicker(enable_uae=False)  # Disable full UAE for demo
    
    # Create Computer Use connector
    print("📦 Initializing Computer Use connector...")
    computer_use_connector = get_computer_use_connector(
        voice_callback=voice_callback,
        display_width=screen_width,
        display_height=screen_height
    )
    
    # Create hybrid connector
    print("📦 Initializing Hybrid connector...")
    hybrid = get_hybrid_connector(
        uae_clicker=uae_clicker,
        computer_use_connector=computer_use_connector,
        voice_callback=voice_callback,
        prefer_computer_use=False,  # Try UAE first
        confidence_threshold=0.7
    )
    
    # Device name
    device_name = "Living Room TV"
    if len(sys.argv) > 1:
        device_name = sys.argv[1]
    
    print(f"\n🎯 Connecting to: {device_name}")
    print("   (Hybrid will choose best method)\n")
    
    # Execute connection
    result = await hybrid.connect_to_device(device_name)
    
    # Display results
    print("\n" + "=" * 80)
    print("📊 Connection Result")
    print("=" * 80)
    print(f"Success: {'✅ Yes' if result['success'] else '❌ No'}")
    print(f"Strategy Used: {result.get('strategy_used', 'unknown')}")
    print(f"Strategy Reason: {result.get('strategy_reason', 'N/A')}")
    print(f"Message: {result.get('message', 'N/A')}")
    print(f"Duration: {result.get('duration', 0):.2f}s")
    
    if result.get('fallback_used'):
        print("⚠️  Fallback to Computer Use was triggered")
    
    # Show stats
    print("\n📈 Hybrid Stats:")
    stats = hybrid.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2%}" if 'rate' in key else f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    from pathlib import Path
    asyncio.run(main())
