#!/usr/bin/env swift

import Foundation
import CoreLocation

class SimpleLocationTest: NSObject, CLLocationManagerDelegate {
    let locationManager = CLLocationManager()
    var gotLocation = false
    
    override init() {
        super.init()
        locationManager.delegate = self
    }
    
    func test() {
        print("Testing location access...")
        
        let status = locationManager.authorizationStatus
        print("Current authorization status: \(statusString(status))")
        
        switch status {
        case .authorizedAlways, .authorizedWhenInUse:
            print("Location authorized! Requesting location...")
            locationManager.requestLocation()
        case .notDetermined:
            print("Location not determined. Terminal needs permission.")
            exit(1)
        case .denied, .restricted:
            print("Location denied. Check System Preferences.")
            exit(1)
        @unknown default:
            print("Unknown status")
            exit(1)
        }
        
        // Wait up to 5 seconds for location
        let deadline = Date().addingTimeInterval(5)
        while !gotLocation && Date() < deadline {
            RunLoop.current.run(until: Date().addingTimeInterval(0.1))
        }
        
        if !gotLocation {
            print("Timeout getting location")
            exit(1)
        }
    }
    
    func statusString(_ status: CLAuthorizationStatus) -> String {
        switch status {
        case .notDetermined: return "notDetermined"
        case .restricted: return "restricted"
        case .denied: return "denied"
        case .authorizedAlways: return "authorizedAlways"
        case .authorizedWhenInUse: return "authorizedWhenInUse"
        @unknown default: return "unknown"
        }
    }
    
    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        if let location = locations.first {
            print("✅ Got location: \(location.coordinate.latitude), \(location.coordinate.longitude)")
            gotLocation = true
            exit(0)
        }
    }
    
    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        print("❌ Location error: \(error)")
        exit(1)
    }
}

let test = SimpleLocationTest()
test.test()