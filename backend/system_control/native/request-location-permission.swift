#!/usr/bin/env swift

import CoreLocation
import Foundation

print("🌍 JARVIS Location Permission Request")
print("=====================================")
print("")
print("This script will request location permission for Terminal.")
print("After running this, Terminal will appear in Location Services.")
print("")

class LocationPermissionRequester: NSObject, CLLocationManagerDelegate {
    let locationManager = CLLocationManager()
    
    func requestPermission() {
        locationManager.delegate = self
        
        let status = locationManager.authorizationStatus
        print("Current status: \(statusString(status))")
        
        if status == .notDetermined {
            print("\n📍 Requesting location permission...")
            print("⚠️  IMPORTANT: A system dialog should appear.")
            print("   If no dialog appears, check System Settings.")
            
            // This will trigger the permission request
            locationManager.startUpdatingLocation()
            locationManager.requestLocation()
        } else if status == .authorizedAlways {
            print("✅ Location permission already granted!")
        } else {
            print("❌ Location permission denied or restricted.")
            print("   Please enable in System Settings > Privacy & Security > Location Services")
        }
        
        // Keep the program running to handle the response
        RunLoop.current.run(until: Date().addingTimeInterval(5))
    }
    
    func statusString(_ status: CLAuthorizationStatus) -> String {
        switch status {
        case .notDetermined: return "Not Determined (Terminal not in list)"
        case .restricted: return "Restricted"
        case .denied: return "Denied"
        case .authorizedAlways: return "Authorized Always"
        @unknown default: return "Unknown"
        }
    }
    
    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        if let location = locations.first {
            print("\n✅ Success! Got location: \(location.coordinate.latitude), \(location.coordinate.longitude)")
            print("   Terminal should now appear in Location Services!")
            exit(0)
        }
    }
    
    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        print("\n⚠️  Location error: \(error.localizedDescription)")
        
        if (error as NSError).code == 1 {
            print("\n📱 Terminal should now appear in Location Services!")
            print("   Go to: System Settings > Privacy & Security > Location Services")
            print("   Find 'Terminal' in the list and enable it")
        }
    }
    
    func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        let newStatus = manager.authorizationStatus
        print("\n🔄 Authorization changed to: \(statusString(newStatus))")
        
        if newStatus == .authorizedAlways {
            print("✅ Permission granted! Requesting location...")
            manager.requestLocation()
        }
    }
}

// Run the permission requester
let requester = LocationPermissionRequester()
requester.requestPermission()