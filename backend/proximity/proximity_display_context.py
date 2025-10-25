"""
Proximity Display Context - Core Data Structures
=================================================

Defines all data structures for the proximity-aware display system.
All dataclasses are designed to be serializable, hashable, and immutable
where appropriate for safe concurrent access.

Key Structures:
- ProximityData: Bluetooth device proximity information
- DisplayLocation: Physical location metadata for displays
- ProximityDisplayContext: Aggregated context for decision-making
- ConnectionDecision: Auto-connection decision with reasoning
- ProximityZone: Spatial zone classification

Author: Derek Russell
Date: 2025-10-14
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from enum import Enum
import json


class ProximityZone(Enum):
    """Spatial proximity zones based on distance thresholds"""
    IMMEDIATE = "immediate"      # 0-1m: Touch distance
    NEAR = "near"                # 1-3m: Conversational distance
    ROOM = "room"                # 3-8m: Same room
    FAR = "far"                  # 8-15m: Adjacent room
    OUT_OF_RANGE = "out_of_range"  # >15m: Too far


class ConnectionState(Enum):
    """Display connection states"""
    CONNECTED = "connected"
    AVAILABLE = "available"
    CONNECTING = "connecting"
    DISCONNECTING = "disconnecting"
    OUT_OF_RANGE = "out_of_range"
    ERROR = "error"


class ConnectionAction(Enum):
    """Possible connection actions"""
    AUTO_CONNECT = "auto_connect"
    PROMPT_USER = "prompt_user"
    IGNORE = "ignore"
    DISCONNECT = "disconnect"


@dataclass
class ProximityData:
    """
    Bluetooth proximity data for a detected device
    
    Attributes:
        device_name: Human-readable device name (e.g., "Derek's Apple Watch")
        device_uuid: Unique Bluetooth identifier
        device_type: Device type (apple_watch, iphone, airpods, etc.)
        rssi: Received Signal Strength Indicator (dBm)
        estimated_distance: Calculated distance in meters
        proximity_zone: Classified proximity zone
        timestamp: When this reading was taken
        confidence: Confidence score (0.0-1.0) for distance estimate
        signal_quality: Signal quality indicator (0.0-1.0)
    """
    device_name: str
    device_uuid: str
    device_type: str
    rssi: int
    estimated_distance: float
    proximity_zone: ProximityZone
    timestamp: datetime
    confidence: float = 1.0
    signal_quality: float = 1.0
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict"""
        return {
            "device_name": self.device_name,
            "device_uuid": self.device_uuid,
            "device_type": self.device_type,
            "rssi": self.rssi,
            "estimated_distance": round(self.estimated_distance, 2),
            "proximity_zone": self.proximity_zone.value,
            "timestamp": self.timestamp.isoformat(),
            "confidence": round(self.confidence, 3),
            "signal_quality": round(self.signal_quality, 3)
        }
    
    @staticmethod
    def classify_zone(distance: float) -> ProximityZone:
        """Classify distance into proximity zone"""
        if distance < 1.0:
            return ProximityZone.IMMEDIATE
        elif distance < 3.0:
            return ProximityZone.NEAR
        elif distance < 8.0:
            return ProximityZone.ROOM
        elif distance < 15.0:
            return ProximityZone.FAR
        else:
            return ProximityZone.OUT_OF_RANGE


@dataclass
class DisplayLocation:
    """
    Physical location metadata for a display
    
    Attributes:
        display_id: Core Graphics display ID
        location_name: Human-readable location (e.g., "Living Room TV")
        zone: Location zone identifier (e.g., "living_room", "office")
        expected_proximity_range: (min_distance, max_distance) in meters
        bluetooth_beacon_uuid: Optional Bluetooth beacon for direct correlation
        position_3d: Optional 3D coordinates (x, y, z) in meters
        auto_connect_enabled: Whether auto-connection is allowed
        connection_priority: Priority score (0.0-1.0) for this display
        tags: Custom tags for filtering/matching
    """
    display_id: int
    location_name: str
    zone: str
    expected_proximity_range: Tuple[float, float]
    bluetooth_beacon_uuid: Optional[str] = None
    position_3d: Optional[Tuple[float, float, float]] = None
    auto_connect_enabled: bool = True
    connection_priority: float = 0.5
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict"""
        return {
            "display_id": self.display_id,
            "location_name": self.location_name,
            "zone": self.zone,
            "expected_proximity_range": list(self.expected_proximity_range),
            "bluetooth_beacon_uuid": self.bluetooth_beacon_uuid,
            "position_3d": list(self.position_3d) if self.position_3d else None,
            "auto_connect_enabled": self.auto_connect_enabled,
            "connection_priority": round(self.connection_priority, 3),
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DisplayLocation':
        """Create from dict"""
        return cls(
            display_id=data["display_id"],
            location_name=data["location_name"],
            zone=data["zone"],
            expected_proximity_range=tuple(data["expected_proximity_range"]),
            bluetooth_beacon_uuid=data.get("bluetooth_beacon_uuid"),
            position_3d=tuple(data["position_3d"]) if data.get("position_3d") else None,
            auto_connect_enabled=data.get("auto_connect_enabled", True),
            connection_priority=data.get("connection_priority", 0.5),
            tags=data.get("tags", [])
        )
    
    def is_in_range(self, distance: float) -> bool:
        """Check if distance is within expected range"""
        min_dist, max_dist = self.expected_proximity_range
        return min_dist <= distance <= max_dist


@dataclass
class ProximityDisplayContext:
    """
    Aggregated context for proximity-aware display decisions
    
    This is the primary context object passed between services and
    used for intelligent display selection and routing.
    
    Attributes:
        user_proximity: Current user proximity data (None if unavailable)
        available_displays: List of all detected displays
        display_locations: Location metadata for displays
        proximity_scores: Calculated proximity scores per display
        nearest_display: Display with highest proximity score
        connection_states: Current connection state per display
        recommended_action: Suggested action based on context
        context_metadata: Additional metadata for logging/debugging
        timestamp: When this context was generated
    """
    user_proximity: Optional[ProximityData]
    available_displays: List[Dict]  # DisplayInfo objects as dicts
    display_locations: Dict[int, DisplayLocation]
    proximity_scores: Dict[int, float]
    nearest_display: Optional[Dict]
    connection_states: Dict[int, ConnectionState]
    recommended_action: Optional[ConnectionAction]
    context_metadata: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict"""
        return {
            "user_proximity": self.user_proximity.to_dict() if self.user_proximity else None,
            "available_displays": self.available_displays,
            "display_locations": {
                str(k): v.to_dict() for k, v in self.display_locations.items()
            },
            "proximity_scores": {
                str(k): round(v, 3) for k, v in self.proximity_scores.items()
            },
            "nearest_display": self.nearest_display,
            "connection_states": {
                str(k): v.value for k, v in self.connection_states.items()
            },
            "recommended_action": self.recommended_action.value if self.recommended_action else None,
            "context_metadata": self.context_metadata,
            "timestamp": self.timestamp.isoformat()
        }
    
    def get_displays_in_zone(self, zone: ProximityZone) -> List[Dict]:
        """Get all displays within a specific proximity zone"""
        if not self.user_proximity:
            return []
        
        displays_in_zone = []
        for display in self.available_displays:
            display_id = display.get("display_id")
            score = self.proximity_scores.get(display_id, 0.0)
            
            # High score indicates closer proximity
            if zone == ProximityZone.IMMEDIATE and score >= 0.8:
                displays_in_zone.append(display)
            elif zone == ProximityZone.NEAR and 0.6 <= score < 0.8:
                displays_in_zone.append(display)
            elif zone == ProximityZone.ROOM and 0.3 <= score < 0.6:
                displays_in_zone.append(display)
            elif zone == ProximityZone.FAR and 0.1 <= score < 0.3:
                displays_in_zone.append(display)
        
        return displays_in_zone


@dataclass
class ConnectionDecision:
    """
    Auto-connection decision with reasoning
    
    Attributes:
        display_id: Target display ID
        display_name: Human-readable display name
        action: Recommended action
        confidence: Confidence score (0.0-1.0)
        reason: Human-readable reason for decision
        proximity_distance: Estimated distance to display
        proximity_zone: Classified proximity zone
        user_preference_score: Learned user preference (if available)
        metadata: Additional decision metadata
    """
    display_id: int
    display_name: str
    action: ConnectionAction
    confidence: float
    reason: str
    proximity_distance: float
    proximity_zone: ProximityZone
    user_preference_score: float = 0.5
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict"""
        return {
            "display_id": self.display_id,
            "display_name": self.display_name,
            "action": self.action.value,
            "confidence": round(self.confidence, 3),
            "reason": self.reason,
            "proximity_distance": round(self.proximity_distance, 2),
            "proximity_zone": self.proximity_zone.value,
            "user_preference_score": round(self.user_preference_score, 3),
            "metadata": self.metadata
        }
    
    @property
    def should_auto_connect(self) -> bool:
        """Check if confidence is high enough for auto-connect"""
        return (
            self.action == ConnectionAction.AUTO_CONNECT and
            self.confidence >= 0.8 and
            self.proximity_zone in [ProximityZone.IMMEDIATE, ProximityZone.NEAR]
        )


@dataclass
class ProximityThresholds:
    """
    Configurable thresholds for proximity-based decisions
    
    All distances in meters, all scores in range [0.0, 1.0]
    """
    immediate_distance: float = 1.0
    near_distance: float = 3.0
    room_distance: float = 8.0
    far_distance: float = 15.0
    auto_connect_distance: float = 1.5
    auto_connect_confidence: float = 0.8
    prompt_user_confidence: float = 0.5
    debounce_time_seconds: float = 3.0
    rssi_smoothing_window: int = 5
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict"""
        return {
            "immediate_distance": self.immediate_distance,
            "near_distance": self.near_distance,
            "room_distance": self.room_distance,
            "far_distance": self.far_distance,
            "auto_connect_distance": self.auto_connect_distance,
            "auto_connect_confidence": self.auto_connect_confidence,
            "prompt_user_confidence": self.prompt_user_confidence,
            "debounce_time_seconds": self.debounce_time_seconds,
            "rssi_smoothing_window": self.rssi_smoothing_window
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ProximityThresholds':
        """Create from dict"""
        return cls(**data)


# Export all for easy importing
__all__ = [
    "ProximityZone",
    "ConnectionState",
    "ConnectionAction",
    "ProximityData",
    "DisplayLocation",
    "ProximityDisplayContext",
    "ConnectionDecision",
    "ProximityThresholds"
]
