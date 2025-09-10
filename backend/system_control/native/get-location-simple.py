#!/usr/bin/env python3
"""
Simple location getter using macOS location services via PyObjC
This should work in Cursor's terminal since Cursor has location permissions
"""

import json
import sys

try:
    from CoreLocation import CLLocationManager, CLGeocoder
    from Foundation import NSRunLoop, NSDate
    import time
    
    class LocationGetter:
        def __init__(self):
            self.location_manager = CLLocationManager.alloc().init()
            self.location_manager.setDelegate_(self)
            self.location_manager.setDesiredAccuracy_(1000)  # 1km accuracy
            self.location = None
            self.timeout = 5.0
            self.start_time = time.time()
            
        def locationManager_didUpdateLocations_(self, manager, locations):
            """Called when location is updated"""
            if locations and len(locations) > 0:
                self.location = locations[0]
                # Stop updates once we have a location
                self.location_manager.stopUpdatingLocation()
                
        def locationManager_didFailWithError_(self, manager, error):
            """Called on location error"""
            print(f"Location error: {error}", file=sys.stderr)
            self.location_manager.stopUpdatingLocation()
            
        def get_location(self):
            """Get current location"""
            # Check authorization
            status = CLLocationManager.authorizationStatus()
            
            if status == 3:  # Authorized
                # Start location updates
                self.location_manager.startUpdatingLocation()
                
                # Run loop until we get location or timeout
                while not self.location and (time.time() - self.start_time) < self.timeout:
                    NSRunLoop.currentRunLoop().runUntilDate_(
                        NSDate.dateWithTimeIntervalSinceNow_(0.1)
                    )
                
                self.location_manager.stopUpdatingLocation()
                
                if self.location:
                    return {
                        "latitude": self.location.coordinate().latitude,
                        "longitude": self.location.coordinate().longitude,
                        "accuracy": self.location.horizontalAccuracy(),
                        "timestamp": str(self.location.timestamp()),
                        "status": "success"
                    }
                else:
                    return {"status": "timeout", "error": "Location request timed out"}
            else:
                status_map = {
                    0: "not_determined",
                    1: "restricted", 
                    2: "denied",
                    3: "authorized",
                    4: "authorized_always"
                }
                return {
                    "status": "error",
                    "error": f"Location access {status_map.get(status, 'unknown')}"
                }
    
    # Create location getter and get location
    getter = LocationGetter()
    result = getter.get_location()
    print(json.dumps(result, indent=2))
    
except ImportError:
    print(json.dumps({
        "status": "error",
        "error": "PyObjC not available. Install with: pip install pyobjc-framework-CoreLocation"
    }))
except Exception as e:
    print(json.dumps({
        "status": "error",
        "error": str(e)
    }))