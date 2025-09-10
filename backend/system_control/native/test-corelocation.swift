import Foundation
import CoreLocation

// Simple test to check location permissions
class LocationTest: NSObject, CLLocationManagerDelegate {
    let locationManager = CLLocationManager()
    
    override init() {
        super.init()
        locationManager.delegate = self
    }
    
    func test() {
        print("Testing location access...")
        print("Authorization status: \(locationManager.authorizationStatus.rawValue)")
        
        switch locationManager.authorizationStatus {
        case .notDetermined:
            print("Status: Not Determined - Permission never requested")
        case .restricted:
            print("Status: Restricted - Location services restricted")
        case .denied:
            print("Status: Denied - User denied location access")
        case .authorized, .authorizedAlways:
            print("Status: Authorized - Can access location")
            
            // Try to get location
            locationManager.requestLocation()
            
            // Check if we have a cached location
            if let location = locationManager.location {
                print("Cached location: \(location.coordinate.latitude), \(location.coordinate.longitude)")
            } else {
                print("No cached location available")
            }
        @unknown default:
            print("Status: Unknown")
        }
    }
    
    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        if let location = locations.first {
            print("Got location: \(location.coordinate.latitude), \(location.coordinate.longitude)")
        }
    }
    
    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        print("Location error: \(error.localizedDescription)")
    }
}

// Run test
let test = LocationTest()
test.test()

// Keep running for a moment to allow callbacks
RunLoop.current.run(until: Date(timeIntervalSinceNow: 2))