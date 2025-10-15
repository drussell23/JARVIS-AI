"""
Proximity Display Bridge - Core Intelligence Layer
===================================================

Central intelligence layer that bridges Bluetooth proximity detection with
multi-monitor display management. Provides contextual display selection,
proximity scoring, and intelligent routing recommendations.

This is the "brain" of the proximity-aware display system - it aggregates
data from multiple sources and makes intelligent decisions about which
display to use based on user proximity, preferences, and context.

Key Responsibilities:
- Aggregate proximity and display data
- Calculate proximity scores for each display
- Select optimal display based on context
- Generate connection decisions with reasoning
- Manage display location configuration
- Provide context for command routing

Author: Derek Russell
Date: 2025-10-14
"""

import asyncio
import logging
import json
import os
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path

from .proximity_display_context import (
    ProximityData,
    DisplayLocation,
    ProximityDisplayContext,
    ConnectionDecision,
    ConnectionAction,
    ConnectionState,
    ProximityZone,
    ProximityThresholds
)
from .bluetooth_proximity_service import BluetoothProximityService, get_proximity_service

logger = logging.getLogger(__name__)


class ProximityDisplayBridge:
    """
    Central bridge between proximity detection and display management
    
    This class orchestrates the entire proximity-aware display system,
    integrating Bluetooth proximity data with multi-monitor detection
    to provide intelligent, context-aware display selection.
    """
    
    def __init__(
        self,
        proximity_service: Optional[BluetoothProximityService] = None,
        config_path: Optional[str] = None,
        thresholds: Optional[ProximityThresholds] = None
    ):
        self.logger = logging.getLogger(__name__)
        
        # Services
        self.proximity_service = proximity_service or get_proximity_service(thresholds)
        self.thresholds = thresholds or ProximityThresholds()
        
        # Configuration
        self.config_path = config_path or self._get_default_config_path()
        self.display_locations: Dict[int, DisplayLocation] = {}
        
        # State tracking
        self.last_context: Optional[ProximityDisplayContext] = None
        self.last_decision: Optional[ConnectionDecision] = None
        self.decision_history: List[ConnectionDecision] = []
        
        # Performance metrics
        self.context_generation_count = 0
        self.decision_count = 0
        
        # Load configuration (sync - will be called async later if needed)
        try:
            self._load_display_locations_sync()
        except Exception as e:
            self.logger.warning(f"[CONFIG] Could not load config synchronously: {e}")
        
        self.logger.info("[PROXIMITY BRIDGE] Initialized successfully")
    
    def _get_default_config_path(self) -> str:
        """Get default path for display location configuration"""
        backend_dir = Path(__file__).parent.parent
        config_dir = backend_dir / "config"
        config_dir.mkdir(exist_ok=True)
        return str(config_dir / "display_locations.json")
    
    def _load_display_locations_sync(self):
        """Load display locations synchronously (for initialization)"""
        try:
            if not os.path.exists(self.config_path):
                return
            
            with open(self.config_path, 'r') as f:
                data = json.load(f)
            
            self.display_locations = {
                int(k): DisplayLocation.from_dict(v) 
                for k, v in data.get("display_locations", {}).items()
            }
            
            self.logger.info(f"[CONFIG] Loaded {len(self.display_locations)} display locations")
            
        except Exception as e:
            self.logger.warning(f"[CONFIG] Could not load config: {e}")
    
    async def load_display_locations(self) -> bool:
        """
        Load display location configuration from JSON file
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(self.config_path):
                self.logger.info(f"[CONFIG] Creating default display locations at {self.config_path}")
                await self._create_default_config()
                return True
            
            with open(self.config_path, 'r') as f:
                data = json.load(f)
            
            self.display_locations = {
                int(k): DisplayLocation.from_dict(v) 
                for k, v in data.get("display_locations", {}).items()
            }
            
            self.logger.info(f"[CONFIG] Loaded {len(self.display_locations)} display locations")
            return True
            
        except Exception as e:
            self.logger.error(f"[CONFIG] Error loading display locations: {e}")
            return False
    
    async def save_display_locations(self) -> bool:
        """
        Save display location configuration to JSON file
        
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            data = {
                "display_locations": {
                    str(k): v.to_dict() 
                    for k, v in self.display_locations.items()
                },
                "last_updated": datetime.now().isoformat()
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.info(f"[CONFIG] Saved {len(self.display_locations)} display locations")
            return True
            
        except Exception as e:
            self.logger.error(f"[CONFIG] Error saving display locations: {e}")
            return False
    
    async def _create_default_config(self):
        """Create default display location configuration"""
        # Create example configuration for MacBook Pro + external displays
        default_locations = {
            1: DisplayLocation(
                display_id=1,
                location_name="MacBook Pro Built-in Display",
                zone="mobile",
                expected_proximity_range=(0.0, 2.0),
                auto_connect_enabled=False,  # Built-in always connected
                connection_priority=1.0,
                tags=["builtin", "primary", "mobile"]
            )
        }
        
        self.display_locations = default_locations
        await self.save_display_locations()
    
    async def register_display_location(
        self,
        display_id: int,
        location_name: str,
        zone: str,
        expected_proximity_range: Tuple[float, float],
        **kwargs
    ) -> bool:
        """
        Register or update a display location
        
        Args:
            display_id: Display ID from CoreGraphics
            location_name: Human-readable name (e.g., "Living Room TV")
            zone: Location zone (e.g., "living_room", "office")
            expected_proximity_range: (min_distance, max_distance) in meters
            **kwargs: Additional DisplayLocation parameters
            
        Returns:
            True if registered successfully
        """
        try:
            display_location = DisplayLocation(
                display_id=display_id,
                location_name=location_name,
                zone=zone,
                expected_proximity_range=expected_proximity_range,
                **kwargs
            )
            
            self.display_locations[display_id] = display_location
            await self.save_display_locations()
            
            self.logger.info(f"[CONFIG] Registered display: {location_name} (ID: {display_id})")
            return True
            
        except Exception as e:
            self.logger.error(f"[CONFIG] Error registering display: {e}")
            return False
    
    async def get_proximity_display_context(
        self,
        available_displays: Optional[List[Dict]] = None
    ) -> ProximityDisplayContext:
        """
        Generate comprehensive proximity-display context
        
        This is the main entry point for getting contextual information
        about user proximity and available displays.
        
        Args:
            available_displays: List of available displays (from MultiMonitorDetector)
                              If None, will attempt to detect displays
            
        Returns:
            ProximityDisplayContext with aggregated data
        """
        try:
            self.context_generation_count += 1
            
            # Get user proximity data
            user_proximity = await self.proximity_service.get_closest_device()
            
            # Get available displays (if not provided)
            if available_displays is None:
                available_displays = await self._detect_displays()
            
            # Calculate proximity scores for each display
            proximity_scores = await self._calculate_proximity_scores(
                user_proximity, available_displays
            )
            
            # Determine nearest display
            nearest_display = self._select_nearest_display(
                available_displays, proximity_scores
            )
            
            # Determine connection states
            connection_states = await self._determine_connection_states(
                available_displays
            )
            
            # Generate recommended action
            recommended_action = await self._recommend_action(
                user_proximity, nearest_display, proximity_scores
            )
            
            # Build context
            context = ProximityDisplayContext(
                user_proximity=user_proximity,
                available_displays=available_displays,
                display_locations=self.display_locations,
                proximity_scores=proximity_scores,
                nearest_display=nearest_display,
                connection_states=connection_states,
                recommended_action=recommended_action,
                context_metadata={
                    "generation_count": self.context_generation_count,
                    "bluetooth_available": self.proximity_service.bluetooth_available,
                    "tracked_devices": self.proximity_service.get_tracked_device_count()
                },
                timestamp=datetime.now()
            )
            
            self.last_context = context
            return context
            
        except Exception as e:
            self.logger.error(f"[BRIDGE] Error generating context: {e}")
            # Return empty context on error
            return ProximityDisplayContext(
                user_proximity=None,
                available_displays=available_displays or [],
                display_locations=self.display_locations,
                proximity_scores={},
                nearest_display=None,
                connection_states={},
                recommended_action=None,
                context_metadata={"error": str(e)},
                timestamp=datetime.now()
            )
    
    async def _detect_displays(self) -> List[Dict]:
        """
        Detect available displays using MultiMonitorDetector
        
        Returns:
            List of display info dicts
        """
        try:
            from vision.multi_monitor_detector import MultiMonitorDetector
            
            detector = MultiMonitorDetector()
            displays = await detector.detect_displays()
            
            # Convert to dict format
            return [
                {
                    "display_id": d.display_id,
                    "name": d.name,
                    "resolution": d.resolution,
                    "position": d.position,
                    "is_primary": d.is_primary
                }
                for d in displays
            ]
            
        except Exception as e:
            self.logger.error(f"[BRIDGE] Error detecting displays: {e}")
            return []
    
    async def _calculate_proximity_scores(
        self,
        user_proximity: Optional[ProximityData],
        displays: List[Dict]
    ) -> Dict[int, float]:
        """
        Calculate proximity scores for each display
        
        Score calculation factors:
        1. Distance from user (if proximity data available)
        2. Display location configuration (expected range)
        3. User preference history (if available)
        4. Connection priority from config
        
        Args:
            user_proximity: User proximity data
            displays: Available displays
            
        Returns:
            Dict mapping display_id -> proximity_score (0.0-1.0)
        """
        scores = {}
        
        for display in displays:
            display_id = display.get("display_id")
            
            # Base score from configuration priority
            location = self.display_locations.get(display_id)
            base_score = location.connection_priority if location else 0.5
            
            # If no proximity data, return base score
            if not user_proximity:
                scores[display_id] = base_score
                continue
            
            # Calculate distance-based score
            distance = user_proximity.estimated_distance
            
            # If display has location config, check if in expected range
            if location:
                if location.is_in_range(distance):
                    # Within expected range = high score
                    range_score = 1.0
                else:
                    # Outside range = penalize
                    min_dist, max_dist = location.expected_proximity_range
                    if distance < min_dist:
                        # Too close (unlikely but possible)
                        range_score = 0.7
                    else:
                        # Too far - exponential decay
                        excess_distance = distance - max_dist
                        range_score = max(0.0, 1.0 - (excess_distance / 10.0))
            else:
                # No location config - use generic distance scoring
                # Closer = higher score, with exponential decay
                range_score = max(0.0, 1.0 - (distance / 15.0))
            
            # Combine scores (weighted average)
            final_score = 0.4 * base_score + 0.6 * range_score
            
            # Apply proximity zone bonus
            if user_proximity.proximity_zone == ProximityZone.IMMEDIATE:
                final_score *= 1.2  # Boost for immediate proximity
            elif user_proximity.proximity_zone == ProximityZone.NEAR:
                final_score *= 1.1  # Small boost for near
            
            # Clamp to [0.0, 1.0]
            final_score = max(0.0, min(1.0, final_score))
            
            scores[display_id] = round(final_score, 3)
        
        return scores
    
    def _select_nearest_display(
        self,
        displays: List[Dict],
        proximity_scores: Dict[int, float]
    ) -> Optional[Dict]:
        """
        Select the nearest/most relevant display based on proximity scores
        
        Args:
            displays: Available displays
            proximity_scores: Calculated proximity scores
            
        Returns:
            Display with highest proximity score or None
        """
        if not displays or not proximity_scores:
            return None
        
        # Find display with highest score
        best_display_id = max(proximity_scores, key=proximity_scores.get)
        best_score = proximity_scores[best_display_id]
        
        # Only return if score is above threshold (0.3 = minimum relevance)
        if best_score < 0.3:
            return None
        
        # Find and return the display
        for display in displays:
            if display.get("display_id") == best_display_id:
                return display
        
        return None
    
    async def _determine_connection_states(
        self,
        displays: List[Dict]
    ) -> Dict[int, ConnectionState]:
        """
        Determine current connection state for each display
        
        Args:
            displays: Available displays
            
        Returns:
            Dict mapping display_id -> ConnectionState
        """
        states = {}
        
        for display in displays:
            display_id = display.get("display_id")
            
            # For now, assume all detected displays are connected
            # In future, could check actual connection status via CoreGraphics
            states[display_id] = ConnectionState.CONNECTED
        
        return states
    
    async def _recommend_action(
        self,
        user_proximity: Optional[ProximityData],
        nearest_display: Optional[Dict],
        proximity_scores: Dict[int, float]
    ) -> Optional[ConnectionAction]:
        """
        Recommend connection action based on context
        
        Args:
            user_proximity: User proximity data
            nearest_display: Nearest display
            proximity_scores: Proximity scores
            
        Returns:
            Recommended ConnectionAction or None
        """
        if not user_proximity or not nearest_display:
            return ConnectionAction.IGNORE
        
        display_id = nearest_display.get("display_id")
        score = proximity_scores.get(display_id, 0.0)
        distance = user_proximity.estimated_distance
        
        # Auto-connect criteria:
        # 1. Very close proximity (< 1.5m)
        # 2. High proximity score (> 0.8)
        # 3. High confidence (> 0.8)
        if (distance < self.thresholds.auto_connect_distance and
            score >= self.thresholds.auto_connect_confidence and
            user_proximity.confidence >= 0.8):
            return ConnectionAction.AUTO_CONNECT
        
        # Prompt user criteria:
        # 1. Moderate proximity (1.5-5m)
        # 2. Decent proximity score (> 0.5)
        elif (distance < 5.0 and
              score >= self.thresholds.prompt_user_confidence):
            return ConnectionAction.PROMPT_USER
        
        # Otherwise, ignore
        return ConnectionAction.IGNORE
    
    async def make_connection_decision(
        self,
        context: Optional[ProximityDisplayContext] = None
    ) -> Optional[ConnectionDecision]:
        """
        Make an intelligent connection decision based on context
        
        Args:
            context: ProximityDisplayContext (if None, will generate)
            
        Returns:
            ConnectionDecision with reasoning or None
        """
        try:
            self.decision_count += 1
            
            # Get or generate context
            if context is None:
                context = await self.get_proximity_display_context()
            
            # No nearest display = no decision
            if not context.nearest_display or not context.user_proximity:
                return None
            
            display_id = context.nearest_display.get("display_id")
            display_name = context.nearest_display.get("name", f"Display {display_id}")
            
            # Get display location for additional context
            location = self.display_locations.get(display_id)
            location_name = location.location_name if location else display_name
            
            # Calculate decision parameters
            proximity_score = context.proximity_scores.get(display_id, 0.0)
            distance = context.user_proximity.estimated_distance
            zone = context.user_proximity.proximity_zone
            
            # Determine action (use recommendation from context)
            action = context.recommended_action or ConnectionAction.IGNORE
            
            # Generate reasoning
            reason = self._generate_decision_reason(
                action, distance, zone, proximity_score, location_name
            )
            
            # Calculate overall confidence
            confidence = self._calculate_decision_confidence(
                proximity_score, context.user_proximity.confidence
            )
            
            decision = ConnectionDecision(
                display_id=display_id,
                display_name=location_name,
                action=action,
                confidence=confidence,
                reason=reason,
                proximity_distance=distance,
                proximity_zone=zone,
                user_preference_score=location.connection_priority if location else 0.5,
                metadata={
                    "proximity_score": proximity_score,
                    "device_type": context.user_proximity.device_type,
                    "signal_quality": context.user_proximity.signal_quality,
                    "decision_count": self.decision_count
                }
            )
            
            self.last_decision = decision
            self.decision_history.append(decision)
            
            return decision
            
        except Exception as e:
            self.logger.error(f"[BRIDGE] Error making decision: {e}")
            return None
    
    def _generate_decision_reason(
        self,
        action: ConnectionAction,
        distance: float,
        zone: ProximityZone,
        score: float,
        display_name: str
    ) -> str:
        """Generate human-readable reason for decision"""
        if action == ConnectionAction.AUTO_CONNECT:
            return (f"You are very close to {display_name} ({distance:.1f}m away, "
                   f"{zone.value} zone). High confidence for automatic connection.")
        elif action == ConnectionAction.PROMPT_USER:
            return (f"You are near {display_name} ({distance:.1f}m away, "
                   f"{zone.value} zone). Would you like to connect?")
        else:
            return f"You are too far from {display_name} ({distance:.1f}m away) for automatic connection."
    
    def _calculate_decision_confidence(
        self,
        proximity_score: float,
        proximity_confidence: float
    ) -> float:
        """Calculate overall decision confidence"""
        # Combine proximity score and proximity confidence
        confidence = 0.6 * proximity_score + 0.4 * proximity_confidence
        return round(confidence, 3)
    
    def get_bridge_stats(self) -> Dict:
        """Get bridge statistics for monitoring"""
        return {
            "context_generation_count": self.context_generation_count,
            "decision_count": self.decision_count,
            "display_locations_count": len(self.display_locations),
            "last_context_time": self.last_context.timestamp.isoformat() if self.last_context else None,
            "last_decision": self.last_decision.to_dict() if self.last_decision else None,
            "proximity_service_stats": self.proximity_service.get_service_stats()
        }


# Singleton instance
_proximity_bridge: Optional[ProximityDisplayBridge] = None

def get_proximity_display_bridge(
    proximity_service: Optional[BluetoothProximityService] = None,
    config_path: Optional[str] = None,
    thresholds: Optional[ProximityThresholds] = None
) -> ProximityDisplayBridge:
    """Get singleton ProximityDisplayBridge instance"""
    global _proximity_bridge
    if _proximity_bridge is None:
        _proximity_bridge = ProximityDisplayBridge(proximity_service, config_path, thresholds)
    return _proximity_bridge
