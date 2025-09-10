import Foundation
import CoreLocation
import AppKit

// Create a minimal macOS app that requests location permission
class TerminalLocationHelper: NSObject, NSApplicationDelegate, CLLocationManagerDelegate {
    let locationManager = CLLocationManager()
    
    func applicationDidFinishLaunching(_ notification: Notification) {
        // Set the app to run in background
        NSApp.setActivationPolicy(.accessory)
        
        locationManager.delegate = self
        
        print("Requesting location permission for Terminal...")
        
        // This will trigger the system permission dialog
        locationManager.startUpdatingLocation()
        
        // Give it a moment then quit
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
            NSApp.terminate(nil)
        }
    }
    
    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        print("✅ Location permission granted!")
        NSApp.terminate(nil)
    }
    
    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        print("Location error: \(error)")
    }
}

// Create and run the app
let app = NSApplication.shared
let delegate = TerminalLocationHelper()
app.delegate = delegate
app.run()
