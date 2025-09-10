#!/usr/bin/env swift

import Foundation
import CoreLocation
import AppKit

// Test location permissions and status
class LocationPermissionTester: NSObject, CLLocationManagerDelegate {
    let locationManager = CLLocationManager()
    
    override init() {
        super.init()
        locationManager.delegate = self
        checkLocationServices()
    }
    
    func checkLocationServices() {
        print("🔍 Checking Location Services Status")
        print(String(repeating: "=", count: 50))
        
        // Check if location services are enabled
        let servicesEnabled = CLLocationManager.locationServicesEnabled()
        print("✓ Location Services Enabled: \(servicesEnabled)")
        
        // Check authorization status
        let status = locationManager.authorizationStatus
        print("✓ Authorization Status: \(getStatusString(status)) (code: \(status.rawValue))")
        
        // Check which apps have permission
        print("\n📍 Location Permission Details:")
        print("- App Bundle ID: \(Bundle.main.bundleIdentifier ?? "Unknown")")
        print("- Executable: \(ProcessInfo.processInfo.processName)")
        
        // Try to get current location
        print("\n🌍 Attempting to get location...")
        
        switch status {
        case .authorized, .authorizedAlways:
            print("✅ Permission granted - requesting location")
            locationManager.requestLocation()
        case .notDetermined:
            print("⚠️  Permission not determined - need to request")
            print("\nTo fix this:")
            print("1. Run the JarvisLocationService app")
            print("2. Grant location permission when prompted")
        case .denied:
            print("❌ Permission denied")
            print("\nTo fix this:")
            print("1. Open System Settings > Privacy & Security > Location Services")
            print("2. Find 'Terminal' or 'JarvisLocationService' in the list")
            print("3. Enable the checkbox")
        case .restricted:
            print("🚫 Location access restricted by system")
        @unknown default:
            print("❓ Unknown authorization status")
        }
        
        // Keep running for a moment
        RunLoop.current.run(until: Date(timeIntervalSinceNow: 3))
    }
    
    func getStatusString(_ status: CLAuthorizationStatus) -> String {
        switch status {
        case .notDetermined: return "Not Determined"
        case .restricted: return "Restricted"
        case .denied: return "Denied"
        case .authorized: return "Authorized"
        case .authorizedAlways: return "Authorized Always"
        @unknown default: return "Unknown"
        }
    }
    
    // Delegate methods
    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        if let location = locations.first {
            print("\n✅ Successfully got location!")
            print("   Latitude: \(location.coordinate.latitude)")
            print("   Longitude: \(location.coordinate.longitude)")
            print("   Accuracy: \(location.horizontalAccuracy) meters")
            print("   Timestamp: \(location.timestamp)")
            
            // Try reverse geocoding
            let geocoder = CLGeocoder()
            geocoder.reverseGeocodeLocation(location) { placemarks, error in
                if let placemark = placemarks?.first {
                    print("\n📍 Location Details:")
                    print("   City: \(placemark.locality ?? "Unknown")")
                    print("   Region: \(placemark.administrativeArea ?? "Unknown")")
                    print("   Country: \(placemark.country ?? "Unknown")")
                    print("   Postal Code: \(placemark.postalCode ?? "Unknown")")
                }
                exit(0)
            }
        }
    }
    
    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        print("\n❌ Location error: \(error.localizedDescription)")
        exit(1)
    }
    
    func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        print("\n🔔 Authorization changed to: \(getStatusString(manager.authorizationStatus))")
    }
}

// Run the test
let tester = LocationPermissionTester()
RunLoop.main.run()