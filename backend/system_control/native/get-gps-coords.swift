#!/usr/bin/env swift

import Foundation
import CoreLocation

// Simple GPS coordinate getter
class GPSCoordGetter: NSObject, CLLocationManagerDelegate {
    private let locationManager = CLLocationManager()
    private var gotLocation = false
    
    override init() {
        super.init()
        locationManager.delegate = self
        locationManager.desiredAccuracy = kCLLocationAccuracyBest
    }
    
    func getCoordinates() {
        // Check if we have permission
        let status = locationManager.authorizationStatus
        
        if status == .authorized || status == .authorizedAlways {
            // Request one-time location
            locationManager.requestLocation()
            
            // Wait up to 5 seconds
            let timeout = Date().addingTimeInterval(5)
            while !gotLocation && Date() < timeout {
                RunLoop.current.run(until: Date().addingTimeInterval(0.1))
            }
            
            if !gotLocation {
                print("TIMEOUT")
            }
        } else {
            print("NO_PERMISSION:\(status.rawValue)")
        }
    }
    
    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        if let location = locations.first {
            // Output simple format: lat,lon,accuracy
            print("\(location.coordinate.latitude),\(location.coordinate.longitude),\(location.horizontalAccuracy)")
            gotLocation = true
        }
    }
    
    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        print("ERROR:\(error.localizedDescription)")
        gotLocation = true
    }
}

// Run
let getter = GPSCoordGetter()
getter.getCoordinates()