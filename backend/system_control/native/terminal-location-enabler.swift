import Cocoa
import CoreLocation

// Full macOS app to enable Terminal location permissions
class TerminalLocationEnabler: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    var locationManager: CLLocationManager!
    var statusLabel: NSTextField!
    var requestButton: NSButton!
    
    func applicationDidFinishLaunching(_ aNotification: Notification) {
        // Create app window
        window = NSWindow(contentRect: NSRect(x: 100, y: 100, width: 600, height: 400),
                          styleMask: [.titled, .closable, .miniaturizable],
                          backing: .buffered, defer: false)
        window.title = "Terminal Location Services Enabler"
        window.center()
        
        // Create main view
        let contentView = NSView(frame: window.contentView!.bounds)
        contentView.wantsLayer = true
        
        // Title
        let titleLabel = NSTextField(labelWithString: "Enable Location Services for Terminal")
        titleLabel.font = NSFont.boldSystemFont(ofSize: 18)
        titleLabel.alignment = .center
        titleLabel.frame = NSRect(x: 50, y: 340, width: 500, height: 30)
        contentView.addSubview(titleLabel)
        
        // Instructions
        let instructionsView = NSTextView(frame: NSRect(x: 50, y: 150, width: 500, height: 180))
        instructionsView.string = """
        This app helps Terminal access Location Services on macOS.
        
        Instructions:
        1. Click "Request Location Permission" below
        2. macOS will show a permission dialog - click "Allow"
        3. Open System Settings > Privacy & Security > Location Services
        4. Find "Terminal" in the list and enable the checkbox
        5. If Terminal doesn't appear, try these steps:
           • Close all Terminal windows
           • Open a new Terminal window
           • Run this app again from the new Terminal
        
        Note: This app creates a proper location request that Terminal can use.
        The permission dialog will mention this helper app, but it enables
        location access for the Terminal that launched it.
        """
        instructionsView.isEditable = false
        instructionsView.font = NSFont.systemFont(ofSize: 14)
        instructionsView.backgroundColor = NSColor.clear
        contentView.addSubview(instructionsView)
        
        // Status label
        statusLabel = NSTextField(labelWithString: "Ready to request location permission")
        statusLabel.alignment = .center
        statusLabel.frame = NSRect(x: 50, y: 100, width: 500, height: 30)
        contentView.addSubview(statusLabel)
        
        // Request button
        requestButton = NSButton(frame: NSRect(x: 200, y: 50, width: 200, height: 35))
        requestButton.title = "Request Location Permission"
        requestButton.bezelStyle = .rounded
        requestButton.target = self
        requestButton.action = #selector(requestLocationPermission)
        contentView.addSubview(requestButton)
        
        window.contentView = contentView
        window.makeKeyAndOrderFront(nil)
        
        // Initialize location manager
        locationManager = CLLocationManager()
        locationManager.delegate = self
        
        // Check current authorization status
        checkAuthorizationStatus()
    }
    
    @objc func requestLocationPermission() {
        statusLabel.stringValue = "Requesting location permission..."
        requestButton.isEnabled = false
        
        // Multiple methods to trigger location permission
        locationManager.startUpdatingLocation()
        
        // Also try requestLocation for one-time access
        locationManager.requestLocation()
        
        // Try to get current location
        if let location = locationManager.location {
            statusLabel.stringValue = "✅ Got location! Check System Settings now."
            showSuccessAlert(with: location)
        } else {
            statusLabel.stringValue = "⏳ Waiting for location permission..."
        }
    }
    
    func checkAuthorizationStatus() {
        let status = locationManager.authorizationStatus
        
        switch status {
        case .authorized, .authorizedAlways:
            statusLabel.stringValue = "✅ Location access already authorized!"
            statusLabel.textColor = NSColor.systemGreen
        case .denied:
            statusLabel.stringValue = "❌ Location access denied. Check System Settings."
            statusLabel.textColor = NSColor.systemRed
        case .restricted:
            statusLabel.stringValue = "⚠️ Location access restricted by system."
            statusLabel.textColor = NSColor.systemOrange
        case .notDetermined:
            statusLabel.stringValue = "Ready to request location permission"
            statusLabel.textColor = NSColor.labelColor
        @unknown default:
            statusLabel.stringValue = "Unknown authorization status"
        }
    }
    
    func showSuccessAlert(with location: CLLocation? = nil) {
        let alert = NSAlert()
        alert.messageText = "Location Permission Requested!"
        
        var message = "Next steps:\n\n"
        message += "1. Open System Settings > Privacy & Security > Location Services\n"
        message += "2. Look for 'Terminal' in the list (may need to scroll)\n"
        message += "3. Enable the checkbox next to Terminal\n\n"
        
        if let location = location {
            message += "Current location detected:\n"
            message += "Latitude: \(location.coordinate.latitude)\n"
            message += "Longitude: \(location.coordinate.longitude)"
        }
        
        alert.informativeText = message
        alert.addButton(withTitle: "Open Location Services")
        alert.addButton(withTitle: "OK")
        
        if alert.runModal() == .alertFirstButtonReturn {
            // Open Location Services in System Settings
            if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_LocationServices") {
                NSWorkspace.shared.open(url)
            }
        }
    }
}

// Location Manager Delegate
extension TerminalLocationEnabler: CLLocationManagerDelegate {
    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard let location = locations.first else { return }
        
        statusLabel.stringValue = "✅ Location received! Terminal should now appear in Location Services."
        statusLabel.textColor = NSColor.systemGreen
        requestButton.isEnabled = true
        
        showSuccessAlert(with: location)
        
        // Stop updating to save battery
        locationManager.stopUpdatingLocation()
    }
    
    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        statusLabel.stringValue = "❌ Error: \(error.localizedDescription)"
        statusLabel.textColor = NSColor.systemRed
        requestButton.isEnabled = true
        
        // Show error alert
        let alert = NSAlert()
        alert.messageText = "Location Request Failed"
        alert.informativeText = """
        Error: \(error.localizedDescription)
        
        Troubleshooting:
        1. Make sure Location Services are enabled in System Settings
        2. Try closing all Terminal windows and opening a new one
        3. Run this app again from the new Terminal window
        """
        alert.runModal()
    }
    
    func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        checkAuthorizationStatus()
        
        let status = manager.authorizationStatus
        if status == .authorized {
            statusLabel.stringValue = "✅ Location access authorized! Terminal can now use location services."
            statusLabel.textColor = NSColor.systemGreen
            showSuccessAlert()
        }
    }
    
    func locationManager(_ manager: CLLocationManager, didChangeAuthorization status: CLAuthorizationStatus) {
        checkAuthorizationStatus()
    }
}

// Main entry point
let app = NSApplication.shared
let delegate = TerminalLocationEnabler()
app.delegate = delegate
app.run()