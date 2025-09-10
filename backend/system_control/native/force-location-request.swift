#!/usr/bin/env swift

import CoreLocation
import Foundation

print("🌍 Forcing Location Permission Request")
print("=====================================\n")

class ForceLocationRequest: NSObject, CLLocationManagerDelegate {
    let locationManager = CLLocationManager()
    
    override init() {
        super.init()
        locationManager.delegate = self
        locationManager.desiredAccuracy = kCLLocationAccuracyBest
    }
    
    func request() {
        print("Attempting to trigger location permission dialog...\n")
        
        // Multiple attempts to trigger the permission
        locationManager.startUpdatingLocation()
        locationManager.requestLocation()
        
        // Try to actually get a location
        if let location = locationManager.location {
            print("✅ Got cached location: \(location.coordinate)")
        }
        
        // Check status
        let status = locationManager.authorizationStatus
        print("Status: \(status.rawValue)")
        
        switch status {
        case .notDetermined:
            print("❓ Permission not determined - Terminal needs to be added to Location Services")
            print("\nTry this command in a NEW Terminal window:")
            print("  /usr/bin/python3 -c 'import CoreLocation; CoreLocation.CLLocationManager().requestLocation()'")
        case .restricted:
            print("🚫 Location services are restricted")
        case .denied:
            print("❌ Location access denied")
        case .authorizedAlways:
            print("✅ Location access authorized!")
        @unknown default:
            print("❓ Unknown status")
        }
    }
    
    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        print("\n✅ SUCCESS! Got location update!")
        print("Terminal now has location permission!")
        exit(0)
    }
    
    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        print("\nLocation request failed: \(error.localizedDescription)")
        print("\n📱 Terminal should now be visible in Location Services!")
        print("Go check System Settings > Privacy & Security > Location Services")
    }
}

let requester = ForceLocationRequest()
requester.request()

// Keep running for a bit
RunLoop.current.run(until: Date().addingTimeInterval(3))