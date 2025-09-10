import Foundation
import CoreLocation

print("Forcing Terminal to request location access...")

// Create location manager
let manager = CLLocationManager()

// Set delegate to nil to avoid complexity
manager.delegate = nil

// Multiple attempts to trigger location permission
print("Triggering location services...")
manager.startUpdatingLocation()
manager.requestLocation()

// Try to access location directly
if let location = manager.location {
    print("Current location: \(location)")
} else {
    print("No cached location available")
}

// Keep trying for a few seconds
for i in 1...3 {
    print("Attempt \(i)...")
    manager.startUpdatingLocation()
    Thread.sleep(forTimeInterval: 1.0)
}

print("\nDone! Check System Preferences > Security & Privacy > Privacy > Location Services")
print("Terminal should now appear in the list.")
