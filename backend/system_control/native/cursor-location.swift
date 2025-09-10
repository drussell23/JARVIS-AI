#!/usr/bin/env swift

import Foundation
import CoreLocation

// Simple location getter for Cursor environment
class CursorLocationService: NSObject, CLLocationManagerDelegate {
    private let locationManager = CLLocationManager()
    private var completion: ((CLLocation?) -> Void)?
    
    override init() {
        super.init()
        locationManager.delegate = self
        locationManager.desiredAccuracy = kCLLocationAccuracyBest
    }
    
    func getLocation(completion: @escaping (CLLocation?) -> Void) {
        self.completion = completion
        
        // Check if we can get location
        if CLLocationManager.locationServicesEnabled() {
            locationManager.requestLocation()
        } else {
            completion(nil)
        }
    }
    
    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        completion?(locations.first)
    }
    
    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        print("Error: \(error.localizedDescription)", to: &errorStream)
        completion?(nil)
    }
}

// Error stream
var errorStream = FileHandle.standardError
extension FileHandle : TextOutputStream {
    public func write(_ string: String) {
        self.write(string.data(using: .utf8)!)
    }
}

// Main
let service = CursorLocationService()
let semaphore = DispatchSemaphore(value: 0)

service.getLocation { location in
    if let location = location {
        // Output simple format
        print("\(location.coordinate.latitude),\(location.coordinate.longitude)")
    } else {
        print("0,0")
    }
    semaphore.signal()
}

semaphore.wait()
exit(0)