#!/usr/bin/env swift

import Foundation
import CoreLocation
import AppKit

// Create a simple macOS app that requests location permission
class LocationPermissionSetup: NSObject, CLLocationManagerDelegate, NSApplicationDelegate {
    let locationManager = CLLocationManager()
    var window: NSWindow!
    
    func applicationDidFinishLaunching(_ notification: Notification) {
        // Create a simple window
        window = NSWindow(
            contentRect: NSRect(x: 100, y: 100, width: 400, height: 200),
            styleMask: [.titled, .closable],
            backing: .buffered,
            defer: false
        )
        
        window.title = "JARVIS Weather Location Permission"
        window.center()
        
        // Create info label
        let label = NSTextField(labelWithString: """
            JARVIS Weather needs location permission.
            
            Click 'Request Permission' and allow location access
            in the system dialog that appears.
            
            Then close this window.
            """)
        label.frame = NSRect(x: 20, y: 80, width: 360, height: 100)
        label.alignment = .center
        
        // Create button
        let button = NSButton(title: "Request Permission", target: self, action: #selector(requestPermission))
        button.frame = NSRect(x: 150, y: 40, width: 100, height: 30)
        
        window.contentView?.addSubview(label)
        window.contentView?.addSubview(button)
        
        window.makeKeyAndOrderFront(nil)
        
        // Check current status
        locationManager.delegate = self
    }
    
    @objc func requestPermission() {
        print("Requesting location permission...")
        
        let status = locationManager.authorizationStatus
        switch status {
        case .notDetermined:
            locationManager.requestWhenInUseAuthorization()
        case .restricted, .denied:
            showAlert(title: "Location Access Denied", 
                     message: "Please enable location access in System Preferences > Security & Privacy > Privacy > Location Services")
        case .authorizedAlways, .authorizedWhenInUse:
            showAlert(title: "Permission Granted", 
                     message: "Location permission is already granted!")
        @unknown default:
            break
        }
    }
    
    func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        let status = manager.authorizationStatus
        switch status {
        case .authorizedAlways, .authorizedWhenInUse:
            showAlert(title: "Success!", 
                     message: "Location permission granted. You can now close this window.")
        case .denied, .restricted:
            showAlert(title: "Permission Denied", 
                     message: "Location access was denied. Please enable it in System Preferences.")
        default:
            break
        }
    }
    
    func showAlert(title: String, message: String) {
        let alert = NSAlert()
        alert.messageText = title
        alert.informativeText = message
        alert.alertStyle = .informational
        alert.addButton(withTitle: "OK")
        alert.runModal()
    }
    
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }
}

// Run as a proper macOS app
let app = NSApplication.shared
let delegate = LocationPermissionSetup()
app.delegate = delegate
app.run()