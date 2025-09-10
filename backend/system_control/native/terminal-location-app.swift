#!/usr/bin/env swift

import Cocoa
import CoreLocation

// Create a proper macOS app with location delegate
@main
class TerminalLocationApp: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    var locationManager: CLLocationManager!
    
    func applicationDidFinishLaunching(_ aNotification: Notification) {
        // Create a simple window
        window = NSWindow(contentRect: NSRect(x: 100, y: 100, width: 480, height: 300),
                          styleMask: [.titled, .closable],
                          backing: .buffered, defer: false)
        window.title = "Enable Terminal Location Access"
        window.center()
        window.makeKeyAndOrderFront(nil)
        
        // Add instructions
        let textView = NSTextView(frame: window.contentView!.bounds)
        textView.string = """
        This app will help Terminal appear in Location Services.
        
        Steps:
        1. Click the 'Request Location' button below
        2. When prompted, click 'Allow'
        3. Check System Preferences > Privacy & Security > Location Services
        4. Terminal should now appear in the list
        5. Enable the checkbox next to Terminal
        
        Note: You may need to run this from a NEW Terminal window.
        """
        textView.isEditable = false
        textView.font = NSFont.systemFont(ofSize: 14)
        
        // Add button
        let button = NSButton(frame: NSRect(x: 190, y: 20, width: 100, height: 30))
        button.title = "Request Location"
        button.bezelStyle = .rounded
        button.target = self
        button.action = #selector(requestLocation)
        
        window.contentView?.addSubview(textView)
        window.contentView?.addSubview(button)
        
        // Initialize location manager
        locationManager = CLLocationManager()
        locationManager.delegate = self
    }
    
    @objc func requestLocation() {
        print("Requesting location...")
        locationManager.startUpdatingLocation()
        locationManager.requestLocation()
    }
}

extension TerminalLocationApp: CLLocationManagerDelegate {
    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        print("✅ Got location! Terminal should now appear in Location Services.")
        
        // Show success alert
        let alert = NSAlert()
        alert.messageText = "Success!"
        alert.informativeText = "Terminal should now appear in Location Services. Check System Preferences > Privacy & Security > Location Services."
        alert.runModal()
    }
    
    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        print("Location error: \(error)")
        
        // Show error alert
        let alert = NSAlert()
        alert.messageText = "Location Request Failed"
        alert.informativeText = "Error: \(error.localizedDescription)\n\nTry opening a new Terminal window and running this again."
        alert.runModal()
    }
    
    func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        print("Authorization status changed: \(manager.authorizationStatus.rawValue)")
    }
}