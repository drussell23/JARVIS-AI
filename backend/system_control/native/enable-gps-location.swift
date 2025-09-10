import Cocoa
import CoreLocation

// App to enable GPS location for JARVIS
class GPSLocationEnabler: NSObject, NSApplicationDelegate, CLLocationManagerDelegate {
    var window: NSWindow!
    var locationManager: CLLocationManager!
    var statusLabel: NSTextField!
    
    func applicationDidFinishLaunching(_ notification: Notification) {
        // Create window
        window = NSWindow(contentRect: NSRect(x: 100, y: 100, width: 500, height: 300),
                          styleMask: [.titled, .closable],
                          backing: .buffered, defer: false)
        window.title = "Enable GPS for JARVIS"
        window.center()
        
        // Create UI
        let contentView = NSView(frame: window.contentView!.bounds)
        
        let titleLabel = NSTextField(labelWithString: "🌍 Enable GPS Location for JARVIS")
        titleLabel.font = NSFont.boldSystemFont(ofSize: 18)
        titleLabel.alignment = .center
        titleLabel.frame = NSRect(x: 50, y: 240, width: 400, height: 30)
        contentView.addSubview(titleLabel)
        
        statusLabel = NSTextField(labelWithString: "Checking location status...")
        statusLabel.alignment = .center
        statusLabel.frame = NSRect(x: 50, y: 150, width: 400, height: 60)
        statusLabel.isEditable = false
        statusLabel.isBezeled = false
        statusLabel.backgroundColor = .clear
        contentView.addSubview(statusLabel)
        
        let requestButton = NSButton(frame: NSRect(x: 150, y: 100, width: 200, height: 35))
        requestButton.title = "Enable GPS Location"
        requestButton.bezelStyle = .rounded
        requestButton.target = self
        requestButton.action = #selector(requestLocation)
        contentView.addSubview(requestButton)
        
        let testButton = NSButton(frame: NSRect(x: 150, y: 50, width: 200, height: 35))
        testButton.title = "Test Location"
        testButton.bezelStyle = .rounded
        testButton.target = self
        testButton.action = #selector(testLocation)
        contentView.addSubview(testButton)
        
        window.contentView = contentView
        window.makeKeyAndOrderFront(nil)
        
        // Initialize location manager
        locationManager = CLLocationManager()
        locationManager.delegate = self
        locationManager.desiredAccuracy = kCLLocationAccuracyBest
        
        checkStatus()
    }
    
    func checkStatus() {
        let enabled = CLLocationManager.locationServicesEnabled()
        let status = locationManager.authorizationStatus
        
        var message = "Location Services: \(enabled ? "✅ Enabled" : "❌ Disabled")\n"
        
        switch status {
        case .notDetermined:
            message += "Permission: ⚠️ Not Requested"
        case .restricted:
            message += "Permission: 🚫 Restricted"
        case .denied:
            message += "Permission: ❌ Denied"
        case .authorized, .authorizedAlways:
            message += "Permission: ✅ Granted"
        @unknown default:
            message += "Permission: ❓ Unknown"
        }
        
        statusLabel.stringValue = message
    }
    
    @objc func requestLocation() {
        // This will trigger the permission dialog
        locationManager.requestWhenInUseAuthorization()
        locationManager.startUpdatingLocation()
    }
    
    @objc func testLocation() {
        locationManager.requestLocation()
    }
    
    // CLLocationManagerDelegate methods
    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        if let location = locations.first {
            let message = """
            ✅ GPS Location Retrieved!
            Latitude: \(location.coordinate.latitude)
            Longitude: \(location.coordinate.longitude)
            Accuracy: \(location.horizontalAccuracy)m
            """
            statusLabel.stringValue = message
            
            // Also output to console for testing
            print("GPS_LOCATION:\(location.coordinate.latitude),\(location.coordinate.longitude)")
        }
    }
    
    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        statusLabel.stringValue = "❌ Error: \(error.localizedDescription)"
    }
    
    func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        checkStatus()
        
        if manager.authorizationStatus == .authorized || 
           manager.authorizationStatus == .authorizedAlways {
            // Automatically get location when authorized
            manager.requestLocation()
        }
    }
}

// Create and run app
let app = NSApplication.shared
let delegate = GPSLocationEnabler()
app.delegate = delegate
app.run()