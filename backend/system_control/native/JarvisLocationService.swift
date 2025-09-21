import Foundation
import CoreLocation

// MARK: - Location Response Structure
struct LocationResponse: Codable {
    let latitude: Double
    let longitude: Double
    let accuracy: Double
    let altitude: Double?
    let city: String?
    let region: String?
    let country: String?
    let timestamp: String
    let source: String
    let status: String
    let error: String?
}

// MARK: - Location Service Manager
class JarvisLocationService: NSObject {
    private let locationManager = CLLocationManager()
    private var completion: ((LocationResponse) -> Void)?
    private let geocoder = CLGeocoder()
    private var timeoutTimer: Timer?
    
    override init() {
        super.init()
        locationManager.delegate = self
        locationManager.desiredAccuracy = kCLLocationAccuracyBest
        locationManager.distanceFilter = kCLDistanceFilterNone
    }
    
    func getCurrentLocation(completion: @escaping (LocationResponse) -> Void) {
        self.completion = completion
        
        // Start timeout timer (5 seconds)
        timeoutTimer = Timer.scheduledTimer(withTimeInterval: 5.0, repeats: false) { _ in
            self.handleTimeout()
        }
        
        // Check authorization status
        let status = locationManager.authorizationStatus
        
        switch status {
        case .authorized, .authorizedAlways:
            requestLocation()
            
        case .notDetermined:
            // First time - need to request permission
            // This will trigger the system permission dialog
            locationManager.requestWhenInUseAuthorization()
            
        case .denied:
            completion(LocationResponse(
                latitude: 0,
                longitude: 0,
                accuracy: 0,
                altitude: nil,
                city: nil,
                region: nil,
                country: nil,
                timestamp: ISO8601DateFormatter().string(from: Date()),
                source: "CoreLocation",
                status: "denied",
                error: "Location access denied. Please enable in System Settings > Privacy & Security > Location Services"
            ))
            
        case .restricted:
            completion(LocationResponse(
                latitude: 0,
                longitude: 0,
                accuracy: 0,
                altitude: nil,
                city: nil,
                region: nil,
                country: nil,
                timestamp: ISO8601DateFormatter().string(from: Date()),
                source: "CoreLocation",
                status: "restricted",
                error: "Location access restricted by system"
            ))
            
        @unknown default:
            completion(LocationResponse(
                latitude: 0,
                longitude: 0,
                accuracy: 0,
                altitude: nil,
                city: nil,
                region: nil,
                country: nil,
                timestamp: ISO8601DateFormatter().string(from: Date()),
                source: "CoreLocation",
                status: "unknown",
                error: "Unknown authorization status"
            ))
        }
    }
    
    private func requestLocation() {
        // Try to get cached location first for faster response
        if let location = locationManager.location,
           location.timestamp.timeIntervalSinceNow > -300 { // Less than 5 minutes old
            self.processLocation(location)
        } else {
            // Request fresh location
            locationManager.requestLocation()
        }
    }
    
    private func processLocation(_ location: CLLocation) {
        // Cancel timeout timer
        timeoutTimer?.invalidate()
        timeoutTimer = nil
        
        // Reverse geocode to get city/region/country
        geocoder.reverseGeocodeLocation(location) { placemarks, error in
            var city: String?
            var region: String?
            var country: String?
            
            if let placemark = placemarks?.first {
                city = placemark.locality ?? placemark.subAdministrativeArea
                region = placemark.administrativeArea
                country = placemark.country
            }
            
            let response = LocationResponse(
                latitude: location.coordinate.latitude,
                longitude: location.coordinate.longitude,
                accuracy: location.horizontalAccuracy,
                altitude: location.altitude,
                city: city,
                region: region,
                country: country,
                timestamp: ISO8601DateFormatter().string(from: Date()),
                source: "CoreLocation",
                status: "success",
                error: nil
            )
            
            self.completion?(response)
        }
    }
    
    private func handleTimeout() {
        completion?(LocationResponse(
            latitude: 0,
            longitude: 0,
            accuracy: 0,
            altitude: nil,
            city: nil,
            region: nil,
            country: nil,
            timestamp: ISO8601DateFormatter().string(from: Date()),
            source: "CoreLocation",
            status: "timeout",
            error: "Location request timed out"
        ))
    }
}

// MARK: - CLLocationManagerDelegate
extension JarvisLocationService: CLLocationManagerDelegate {
    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard let location = locations.first else { return }
        processLocation(location)
    }
    
    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        // Cancel timeout timer
        timeoutTimer?.invalidate()
        timeoutTimer = nil
        
        completion?(LocationResponse(
            latitude: 0,
            longitude: 0,
            accuracy: 0,
            altitude: nil,
            city: nil,
            region: nil,
            country: nil,
            timestamp: ISO8601DateFormatter().string(from: Date()),
            source: "CoreLocation",
            status: "error",
            error: error.localizedDescription
        ))
    }
    
    func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        let status = manager.authorizationStatus
        
        switch status {
        case .authorized, .authorizedAlways:
            // Permission granted, now request location
            requestLocation()
            
        case .denied, .restricted:
            // Permission denied
            let errorMsg = status == .denied ? 
                "Location access denied" : 
                "Location access restricted"
            
            completion?(LocationResponse(
                latitude: 0,
                longitude: 0,
                accuracy: 0,
                altitude: nil,
                city: nil,
                region: nil,
                country: nil,
                timestamp: ISO8601DateFormatter().string(from: Date()),
                source: "CoreLocation",
                status: status == .denied ? "denied" : "restricted",
                error: errorMsg
            ))
            
        default:
            break
        }
    }
}

// MARK: - Main Entry Point
class LocationServiceRunner {
    static func run() {
        let service = JarvisLocationService()
        
        service.getCurrentLocation { response in
            // Output JSON to stdout for Python to read
            let encoder = JSONEncoder()
            encoder.outputFormatting = .prettyPrinted
            
            if let data = try? encoder.encode(response),
               let json = String(data: data, encoding: .utf8) {
                print(json)
            } else {
                print("{\"status\":\"error\",\"error\":\"Failed to encode response\"}")
            }
            
            // Exit after providing location
            exit(0)
        }
        
        // Keep RunLoop alive while waiting for location
        RunLoop.current.run()
    }
}

// Start the service
LocationServiceRunner.run()