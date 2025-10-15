"""
Proximity-Aware Command Router
================================

Routes vision commands to displays based on user proximity context.
Integrates with VisionCommandHandler and IntelligentOrchestrator.

Features:
- Contextual display selection based on proximity
- Natural language voice responses
- Command routing to nearest/most relevant display
- Proximity context injection into analysis
- Smart fallback when proximity unavailable

Author: Derek Russell
Date: 2025-10-14
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List

from .proximity_display_context import ProximityDisplayContext, ProximityZone
from .proximity_display_bridge import get_proximity_display_bridge

logger = logging.getLogger(__name__)


class ProximityCommandRouter:
    """
    Routes commands to displays based on proximity context
    
    This integrates proximity intelligence with vision command handling,
    enabling contextual display selection and natural language responses.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.bridge = get_proximity_display_bridge()
        
        # Routing statistics
        self.total_routes = 0
        self.proximity_routes = 0
        self.fallback_routes = 0
        
        self.logger.info("[PROXIMITY ROUTER] Initialized")
    
    async def route_command(
        self,
        command: str,
        available_displays: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Route a command to the most appropriate display based on proximity
        
        Args:
            command: User command string
            available_displays: List of available displays (optional)
            
        Returns:
            Routing result with target display and context
        """
        try:
            self.total_routes += 1
            
            # Get proximity context
            context = await self.bridge.get_proximity_display_context(available_displays)
            
            # Check if we have valid proximity data
            if context.user_proximity and context.nearest_display:
                self.proximity_routes += 1
                return await self._route_with_proximity(command, context)
            else:
                self.fallback_routes += 1
                return await self._route_without_proximity(command, context)
                
        except Exception as e:
            self.logger.error(f"[ROUTER] Error routing command: {e}")
            return {
                "success": False,
                "error": str(e),
                "target_display": None
            }
    
    async def _route_with_proximity(
        self,
        command: str,
        context: ProximityDisplayContext
    ) -> Dict[str, Any]:
        """
        Route command with proximity context
        
        Args:
            command: User command
            context: ProximityDisplayContext with valid proximity data
            
        Returns:
            Routing result with proximity-aware response
        """
        nearest_display = context.nearest_display
        proximity_data = context.user_proximity
        display_id = nearest_display.get("display_id")
        display_name = nearest_display.get("name", f"Display {display_id}")
        
        # Get display location for richer context
        location = context.display_locations.get(display_id)
        location_name = location.location_name if location else display_name
        
        # Generate natural language response
        voice_response = self._generate_proximity_voice_response(
            command, proximity_data, location_name, context
        )
        
        return {
            "success": True,
            "target_display": nearest_display,
            "display_id": display_id,
            "display_name": location_name,
            "proximity_context": context.to_dict(),
            "voice_response": voice_response,
            "routing_reason": f"Routed to nearest display based on proximity ({proximity_data.estimated_distance:.1f}m away)",
            "proximity_based": True
        }
    
    async def _route_without_proximity(
        self,
        command: str,
        context: ProximityDisplayContext
    ) -> Dict[str, Any]:
        """
        Route command without proximity context (fallback)
        
        Args:
            command: User command
            context: ProximityDisplayContext (proximity data unavailable)
            
        Returns:
            Routing result with fallback logic
        """
        # Fallback to primary display or first available
        if context.available_displays:
            # Prefer primary display
            primary = next((d for d in context.available_displays if d.get("is_primary")), None)
            target = primary or context.available_displays[0]
            
            display_id = target.get("display_id")
            display_name = target.get("name", f"Display {display_id}")
            
            return {
                "success": True,
                "target_display": target,
                "display_id": display_id,
                "display_name": display_name,
                "proximity_context": None,
                "voice_response": f"Routing to {display_name} (proximity data unavailable)",
                "routing_reason": "Fallback to primary/first display (no proximity data)",
                "proximity_based": False
            }
        else:
            return {
                "success": False,
                "error": "No displays available",
                "target_display": None,
                "proximity_based": False
            }
    
    def _generate_proximity_voice_response(
        self,
        command: str,
        proximity_data: Any,
        display_name: str,
        context: ProximityDisplayContext
    ) -> str:
        """
        Generate natural language voice response based on proximity
        
        Args:
            command: User command
            proximity_data: ProximityData
            display_name: Display location name
            context: Full proximity context
            
        Returns:
            Natural language response string
        """
        distance = proximity_data.estimated_distance
        zone = proximity_data.proximity_zone
        device_type = proximity_data.device_type
        
        # Get display location for richer context
        display_id = context.nearest_display.get("display_id")
        location = context.display_locations.get(display_id)
        zone_name = location.zone if location else "area"
        
        # Generate contextual response based on proximity zone
        if zone == ProximityZone.IMMEDIATE:
            return f"Sir, I see you're right at the {display_name}. Processing your command here."
        
        elif zone == ProximityZone.NEAR:
            return f"Sir, I detect you're near the {display_name} ({distance:.1f} meters away). Routing to this display."
        
        elif zone == ProximityZone.ROOM:
            return f"Sir, you're in the {zone_name} with the {display_name}. I'll show the results here."
        
        elif zone == ProximityZone.FAR:
            return f"Sir, based on your {device_type} location, routing to the nearest display: {display_name}."
        
        else:
            return f"Routing to {display_name} based on your proximity."
    
    async def generate_proximity_acknowledgment(
        self,
        context: ProximityDisplayContext
    ) -> Optional[str]:
        """
        Generate a proactive proximity acknowledgment
        
        Example: "I see you're near the Living Room TV"
        
        Args:
            context: ProximityDisplayContext
            
        Returns:
            Acknowledgment string or None
        """
        if not context.user_proximity or not context.nearest_display:
            return None
        
        proximity_data = context.user_proximity
        nearest_display = context.nearest_display
        display_id = nearest_display.get("display_id")
        
        # Get display location
        location = context.display_locations.get(display_id)
        location_name = location.location_name if location else nearest_display.get("name")
        
        distance = proximity_data.estimated_distance
        zone = proximity_data.proximity_zone
        
        # Only acknowledge if reasonably close
        if zone in [ProximityZone.IMMEDIATE, ProximityZone.NEAR]:
            return f"Sir, I see you're near the {location_name} ({distance:.1f}m away)."
        
        return None
    
    def get_router_stats(self) -> Dict:
        """Get router statistics"""
        return {
            "total_routes": self.total_routes,
            "proximity_routes": self.proximity_routes,
            "fallback_routes": self.fallback_routes,
            "proximity_usage_rate": round(self.proximity_routes / max(self.total_routes, 1), 3)
        }


# Singleton instance
_proximity_router: Optional[ProximityCommandRouter] = None

def get_proximity_command_router() -> ProximityCommandRouter:
    """Get singleton ProximityCommandRouter instance"""
    global _proximity_router
    if _proximity_router is None:
        _proximity_router = ProximityCommandRouter()
    return _proximity_router
