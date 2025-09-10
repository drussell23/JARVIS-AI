#!/usr/bin/env swift

import Foundation
import CoreLocation
import WeatherKit

// MARK: - Models

struct WeatherResponse: Codable {
    let location: String
    let latitude: Double
    let longitude: Double
    let temperature: Double
    let temperatureFahrenheit: Double
    let apparentTemperature: Double
    let apparentTemperatureFahrenheit: Double
    let condition: String
    let symbolName: String
    let description: String
    let humidity: Double
    let pressure: Double
    let pressureInHg: Double
    let windSpeed: Double
    let windSpeedMph: Double
    let windDirection: String
    let windDirectionDegrees: Double
    let visibility: Double
    let visibilityMiles: Double
    let uvIndex: Int
    let cloudCover: Double
    let isDaylight: Bool
    let sunrise: String?
    let sunset: String?
    let moonPhase: String
    let precipitationChance: Double
    let precipitationIntensity: Double
    let dewPoint: Double
    let dewPointFahrenheit: Double
    let source: String
    let timestamp: String
    let timezone: String
    let alerts: [WeatherAlert]
    let hourlyForecast: [HourlyForecast]?
    let dailyForecast: [DailyForecast]?
    
    private enum CodingKeys: String, CodingKey {
        case location, latitude, longitude, temperature
        case temperatureFahrenheit = "temperature_f"
        case apparentTemperature = "feels_like"
        case apparentTemperatureFahrenheit = "feels_like_f"
        case condition, symbolName, description, humidity, pressure
        case pressureInHg = "pressure_inhg"
        case windSpeed = "wind_speed"
        case windSpeedMph = "wind_speed_mph"
        case windDirection = "wind_direction"
        case windDirectionDegrees = "wind_direction_degrees"
        case visibility
        case visibilityMiles = "visibility_miles"
        case uvIndex = "uv_index"
        case cloudCover = "cloud_cover"
        case isDaylight = "is_daylight"
        case sunrise, sunset
        case moonPhase = "moon_phase"
        case precipitationChance = "precipitation_chance"
        case precipitationIntensity = "precipitation_intensity"
        case dewPoint = "dew_point"
        case dewPointFahrenheit = "dew_point_f"
        case source, timestamp, timezone, alerts
        case hourlyForecast = "hourly_forecast"
        case dailyForecast = "daily_forecast"
    }
}

struct WeatherAlert: Codable {
    let id: String
    let summary: String
    let severity: String
    let urgency: String
    let regions: [String]
    let effectiveDate: String
    let expiresDate: String
    
    private enum CodingKeys: String, CodingKey {
        case id, summary, severity, urgency, regions
        case effectiveDate = "effective_date"
        case expiresDate = "expires_date"
    }
}

struct HourlyForecast: Codable {
    let time: String
    let temperature: Double
    let temperatureFahrenheit: Double
    let condition: String
    let symbolName: String
    let precipitationChance: Double
    
    private enum CodingKeys: String, CodingKey {
        case time, temperature
        case temperatureFahrenheit = "temperature_f"
        case condition, symbolName
        case precipitationChance = "precipitation_chance"
    }
}

struct DailyForecast: Codable {
    let date: String
    let highTemperature: Double
    let highTemperatureFahrenheit: Double
    let lowTemperature: Double
    let lowTemperatureFahrenheit: Double
    let condition: String
    let symbolName: String
    let precipitationChance: Double
    
    private enum CodingKeys: String, CodingKey {
        case date
        case highTemperature = "high_temperature"
        case highTemperatureFahrenheit = "high_temperature_f"
        case lowTemperature = "low_temperature"
        case lowTemperatureFahrenheit = "low_temperature_f"
        case condition, symbolName
        case precipitationChance = "precipitation_chance"
    }
}

struct LocationResponse: Codable {
    let latitude: Double
    let longitude: Double
    let city: String
    let state: String?
    let country: String
    let timezone: String
    let source: String
}

struct ErrorResponse: Codable {
    let error: String
    let message: String
    let code: String
}

// MARK: - Helper for stderr output
var standardError = FileHandle.standardError

extension FileHandle: TextOutputStream {
    public func write(_ string: String) {
        if let data = string.data(using: .utf8) {
            self.write(data)
        }
    }
}

// MARK: - Location Manager

class LocationManager: NSObject, CLLocationManagerDelegate {
    private let locationManager = CLLocationManager()
    private var completion: ((Result<CLLocation, Error>) -> Void)?
    private let semaphore = DispatchSemaphore(value: 0)
    
    override init() {
        super.init()
        locationManager.delegate = self
        locationManager.desiredAccuracy = kCLLocationAccuracyBest
    }
    
    func getCurrentLocation() async throws -> CLLocation {
        // For CLI tools, we need to handle this synchronously
        return try await withCheckedThrowingContinuation { continuation in
            
            DispatchQueue.main.async { [weak self] in
                guard let self = self else {
                    continuation.resume(throwing: LocationError.unknown)
                    return
                }
                
                self.completion = { result in
                    switch result {
                    case .success(let location):
                        continuation.resume(returning: location)
                    case .failure(let error):
                        continuation.resume(throwing: error)
                    }
                }
                
                // Check current authorization status
                let status = self.locationManager.authorizationStatus
                
                print("Location authorization status: \(self.authorizationStatusString(status))", to: &standardError)
                
                switch status {
                case .notDetermined:
                    // For CLI tools, we can't request authorization interactively
                    print("Location permission not determined. Grant permission to Terminal in System Preferences.", to: &standardError)
                    self.completion?(.failure(LocationError.notDetermined))
                    
                case .restricted, .denied:
                    print("Location access denied. Enable for Terminal in System Preferences.", to: &standardError)
                    self.completion?(.failure(LocationError.unauthorized))
                    
                case .authorizedAlways, .authorizedWhenInUse:
                    // Request location with timeout
                    self.locationManager.requestLocation()
                    
                    // Set a timeout for location request
                    DispatchQueue.main.asyncAfter(deadline: .now() + 5.0) { [weak self] in
                        if self?.completion != nil {
                            self?.completion?(.failure(LocationError.timeout))
                            self?.completion = nil
                        }
                    }
                    
                @unknown default:
                    self.completion?(.failure(LocationError.unknown))
                }
            }
        }
    }
    
    private func authorizationStatusString(_ status: CLAuthorizationStatus) -> String {
        switch status {
        case .notDetermined: return "notDetermined"
        case .restricted: return "restricted"
        case .denied: return "denied"
        case .authorizedAlways: return "authorizedAlways"
        case .authorizedWhenInUse: return "authorizedWhenInUse"
        @unknown default: return "unknown"
        }
    }
    
    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        if let location = locations.first {
            completion?(.success(location))
        }
    }
    
    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        completion?(.failure(error))
    }
    
    func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        let status = manager.authorizationStatus
        switch status {
        case .authorizedAlways, .authorizedWhenInUse:
            manager.requestLocation()
        case .denied, .restricted:
            completion?(.failure(LocationError.unauthorized))
        case .notDetermined:
            // For command line tools, we need to handle this differently
            // We can't request authorization in a CLI tool, so we fail gracefully
            completion?(.failure(LocationError.notDetermined))
        @unknown default:
            completion?(.failure(LocationError.unknown))
        }
    }
}

// MARK: - Weather Service

@available(macOS 13.0, *)
class WeatherService {
    private let weatherService: WeatherKit.WeatherService
    private let geocoder = CLGeocoder()
    
    init() {
        self.weatherService = WeatherKit.WeatherService()
    }
    
    func getWeather(for location: CLLocation, includeForecast: Bool = false) async throws -> WeatherResponse {
        // Get weather data with error handling
        let weather: Weather
        do {
            weather = try await weatherService.weather(for: location)
        } catch let error as NSError {
            // Check for WeatherKit specific errors
            if error.domain == "WeatherDaemon.WDSJWTAuthenticatorServiceListener.Errors" || 
               error.code == 1 {
                throw WeatherError.notAuthorized
            } else if error.domain == NSURLErrorDomain {
                throw WeatherError.networkError
            } else {
                throw WeatherError.weatherUnavailable
            }
        }
        
        // Reverse geocode for location name
        let placemarks = try await geocoder.reverseGeocodeLocation(location)
        let placemark = placemarks.first
        
        let cityName = placemark?.locality ?? placemark?.administrativeArea ?? "Unknown Location"
        let state = placemark?.administrativeArea
        let country = placemark?.country ?? ""
        
        var locationName = cityName
        if let state = state, country == "United States" {
            locationName = "\(cityName), \(state)"
        } else if country != cityName {
            locationName = "\(cityName), \(country)"
        }
        
        // Get current weather
        let current = weather.currentWeather
        
        // Temperature conversions
        let tempC = current.temperature.value
        let tempF = tempC * 9/5 + 32
        let feelsLikeC = current.apparentTemperature.value
        let feelsLikeF = feelsLikeC * 9/5 + 32
        let dewPointC = current.dewPoint.value
        let dewPointF = dewPointC * 9/5 + 32
        
        // Wind direction
        let windDir = windDirectionString(from: current.wind.direction.value)
        
        // Moon phase
        let moonPhaseString = moonPhaseDescription(weather.dailyForecast.first?.moon.phase)
        
        // Format sunrise/sunset
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "HH:mm"
        dateFormatter.timeZone = TimeZone.current
        
        let sunrise = weather.dailyForecast.first?.sun.sunrise.map { dateFormatter.string(from: $0) }
        let sunset = weather.dailyForecast.first?.sun.sunset.map { dateFormatter.string(from: $0) }
        
        // Weather alerts - Fixed for actual WeatherKit API
        let alerts = (weather.weatherAlerts ?? []).map { alert in
            WeatherAlert(
                id: UUID().uuidString,  // WeatherKit alerts don't have IDs
                summary: alert.summary,
                severity: severityString(alert.severity),
                urgency: urgencyString(alert.severity), // Use severity for urgency
                regions: [alert.region ?? "Unknown region"],
                effectiveDate: ISO8601DateFormatter().string(from: Date()), // Use current date as fallback
                expiresDate: ISO8601DateFormatter().string(from: Date().addingTimeInterval(86400)) // 24 hours later
            )
        }
        
        // Hourly forecast (next 12 hours)
        var hourlyForecast: [HourlyForecast]? = nil
        if includeForecast {
            let hourFormatter = DateFormatter()
            hourFormatter.dateFormat = "HH:mm"
            
            hourlyForecast = Array(weather.hourlyForecast.prefix(12)).map { hour in
                HourlyForecast(
                    time: hourFormatter.string(from: hour.date),
                    temperature: hour.temperature.value,
                    temperatureFahrenheit: hour.temperature.value * 9/5 + 32,
                    condition: hour.condition.description,
                    symbolName: hour.symbolName,
                    precipitationChance: hour.precipitationChance
                )
            }
        }
        
        // Daily forecast (next 5 days)
        var dailyForecast: [DailyForecast]? = nil
        if includeForecast {
            let dayFormatter = DateFormatter()
            dayFormatter.dateFormat = "EEE, MMM d"
            
            dailyForecast = Array(weather.dailyForecast.prefix(5)).map { day in
                DailyForecast(
                    date: dayFormatter.string(from: day.date),
                    highTemperature: day.highTemperature.value,
                    highTemperatureFahrenheit: day.highTemperature.value * 9/5 + 32,
                    lowTemperature: day.lowTemperature.value,
                    lowTemperatureFahrenheit: day.lowTemperature.value * 9/5 + 32,
                    condition: day.condition.description,
                    symbolName: day.symbolName,
                    precipitationChance: day.precipitationChance
                )
            }
        }
        
        return WeatherResponse(
            location: locationName,
            latitude: location.coordinate.latitude,
            longitude: location.coordinate.longitude,
            temperature: tempC.rounded(),
            temperatureFahrenheit: tempF.rounded(),
            apparentTemperature: feelsLikeC.rounded(),
            apparentTemperatureFahrenheit: feelsLikeF.rounded(),
            condition: current.condition.description,
            symbolName: current.symbolName,
            description: current.condition.description.lowercased(),
            humidity: (current.humidity * 100).rounded(),
            pressure: current.pressure.value,
            pressureInHg: current.pressure.converted(to: .inchesOfMercury).value,
            windSpeed: current.wind.speed.value,
            windSpeedMph: current.wind.speed.converted(to: .milesPerHour).value,
            windDirection: windDir,
            windDirectionDegrees: current.wind.direction.value,
            visibility: current.visibility.value / 1000, // Convert to km
            visibilityMiles: current.visibility.converted(to: .miles).value,
            uvIndex: Int(current.uvIndex.value),
            cloudCover: (current.cloudCover * 100).rounded(),
            isDaylight: current.isDaylight,
            sunrise: sunrise,
            sunset: sunset,
            moonPhase: moonPhaseString,
            precipitationChance: 0.0,  // CurrentWeather doesn't have precipitationChance
            precipitationIntensity: current.precipitationIntensity.value,
            dewPoint: dewPointC.rounded(),
            dewPointFahrenheit: dewPointF.rounded(),
            source: "WeatherKit",
            timestamp: ISO8601DateFormatter().string(from: Date()),
            timezone: TimeZone.current.identifier,
            alerts: alerts,
            hourlyForecast: hourlyForecast,
            dailyForecast: dailyForecast
        )
    }
    
    func getWeatherForCity(_ cityName: String, includeForecast: Bool = false) async throws -> WeatherResponse {
        let locations = try await geocoder.geocodeAddressString(cityName)
        guard let location = locations.first?.location else {
            throw WeatherError.locationNotFound
        }
        
        return try await getWeather(for: location, includeForecast: includeForecast)
    }
    
    private func windDirectionString(from degrees: Double) -> String {
        let directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                         "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        let index = Int((degrees + 11.25) / 22.5) % 16
        return directions[index]
    }
    
    private func moonPhaseDescription(_ phase: MoonPhase?) -> String {
        guard let phase = phase else { return "Unknown" }
        switch phase {
        case .new: return "New Moon"
        case .waxingCrescent: return "Waxing Crescent"
        case .firstQuarter: return "First Quarter"
        case .waxingGibbous: return "Waxing Gibbous"
        case .full: return "Full Moon"
        case .waningGibbous: return "Waning Gibbous"
        case .lastQuarter: return "Last Quarter"
        case .waningCrescent: return "Waning Crescent"
        @unknown default: return "Unknown"
        }
    }
    
    private func severityString(_ severity: WeatherSeverity) -> String {
        switch severity {
        case .minor: return "minor"
        case .moderate: return "moderate"
        case .severe: return "severe"
        case .extreme: return "extreme"
        case .unknown: return "unknown"
        @unknown default: return "unknown"
        }
    }
    
    private func urgencyString(_ severity: WeatherSeverity) -> String {
        // Map severity to urgency since WeatherKit doesn't have urgency
        switch severity {
        case .extreme: return "immediate"
        case .severe: return "immediate"
        case .moderate: return "expected"
        case .minor: return "future"
        case .unknown: return "unknown"
        @unknown default: return "unknown"
        }
    }
}

// MARK: - Errors

enum LocationError: LocalizedError {
    case unauthorized
    case unknown
    case notDetermined
    case timeout
    
    var errorDescription: String? {
        switch self {
        case .unauthorized:
            return "Location access is denied. Please enable location services in System Preferences > Security & Privacy > Privacy > Location Services."
        case .unknown:
            return "An unknown location error occurred."
        case .notDetermined:
            return "Location permission not determined. For command line tools, grant permission to Terminal in System Preferences > Security & Privacy > Privacy > Location Services."
        case .timeout:
            return "Location request timed out. Please ensure location services are enabled."
        }
    }
}

enum WeatherError: LocalizedError {
    case locationNotFound
    case weatherUnavailable
    case notAuthorized
    case networkError
    
    var errorDescription: String? {
        switch self {
        case .locationNotFound:
            return "Location not found. Please check the city name and try again."
        case .weatherUnavailable:
            return "Weather data is temporarily unavailable. Please try again later."
        case .notAuthorized:
            return "This app has not been authorized to use WeatherKit. Please ensure the app is properly signed with WeatherKit entitlements."
        case .networkError:
            return "Network error occurred while fetching weather data. Please check your internet connection."
        }
    }
}

// MARK: - Command Line Interface

func printUsage() {
    print("""
    Usage: jarvis-weather [command] [options]
    
    Commands:
      current              Get weather for current location (default)
      city <name>          Get weather for a specific city
      location             Get current location information
      temperature          Get just the temperature
    
    Options:
      -p, --pretty         Pretty print JSON output
      -f, --forecast       Include hourly and daily forecast
      -v, --version        Show version
    """)
}

// MARK: - Main Program

// Parse command line arguments
let arguments = CommandLine.arguments
let programName = arguments[0]

// Simple argument parsing
func parseArguments() -> (command: String, isPretty: Bool, includeForecast: Bool) {
    let command = arguments.count > 1 ? arguments[1] : "current"
    let isPretty = arguments.contains("--pretty") || arguments.contains("-p")
    let includeForecast = arguments.contains("--forecast") || arguments.contains("-f")
    let showVersion = arguments.contains("--version") || arguments.contains("-v")
    let showHelp = arguments.contains("--help") || arguments.contains("-h")
    
    if showVersion {
        print("jarvis-weather 1.0.0")
        exit(0)
    }
    
    if showHelp {
        printUsage()
        exit(0)
    }
    
    return (command, isPretty, includeForecast)
}

let (command, isPretty, includeForecast) = parseArguments()

// Create JSON encoder
let encoder = JSONEncoder()
if isPretty {
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
}

// Handle commands
switch command {
case "current":
    let semaphore = DispatchSemaphore(value: 0)
    var result: Result<WeatherResponse, Error>?
    
    Task {
        do {
            let locationManager = LocationManager()
            let weatherService = WeatherService()
            
            let location = try await locationManager.getCurrentLocation()
            let weather = try await weatherService.getWeather(for: location, includeForecast: includeForecast)
            result = .success(weather)
        } catch {
            result = .failure(error)
        }
        semaphore.signal()
    }
    
    semaphore.wait()
    
    switch result {
    case .success(let weather):
        let data = try! encoder.encode(weather)
        print(String(data: data, encoding: .utf8)!)
    case .failure(let error):
        let errorResponse = ErrorResponse(
            error: "weather_failed",
            message: error.localizedDescription,
            code: "WEATHER_ERROR"
        )
        let data = try! encoder.encode(errorResponse)
        print(String(data: data, encoding: .utf8)!)
        exit(1)
    case .none:
        exit(1)
    }
    
case "city":
    guard arguments.count > 2 else {
        let error = ErrorResponse(
            error: "missing_city",
            message: "Please provide a city name",
            code: "MISSING_ARGUMENT"
        )
        let data = try! encoder.encode(error)
        print(String(data: data, encoding: .utf8)!)
        exit(1)
    }
    
    // Join all arguments after "city" as the city name
    let cityName = arguments[2...].joined(separator: " ")
        .replacingOccurrences(of: "--pretty", with: "")
        .replacingOccurrences(of: "-p", with: "")
        .replacingOccurrences(of: "--forecast", with: "")
        .replacingOccurrences(of: "-f", with: "")
        .trimmingCharacters(in: .whitespacesAndNewlines)
    
    let semaphore = DispatchSemaphore(value: 0)
    var result: Result<WeatherResponse, Error>?
    
    Task {
        do {
            let weatherService = WeatherService()
            let weather = try await weatherService.getWeatherForCity(cityName, includeForecast: includeForecast)
            result = .success(weather)
        } catch {
            result = .failure(error)
        }
        semaphore.signal()
    }
    
    semaphore.wait()
    
    switch result {
    case .success(let weather):
        let data = try! encoder.encode(weather)
        print(String(data: data, encoding: .utf8)!)
    case .failure(let error):
        let errorResponse = ErrorResponse(
            error: "weather_failed",
            message: error.localizedDescription,
            code: "CITY_NOT_FOUND"
        )
        let data = try! encoder.encode(errorResponse)
        print(String(data: data, encoding: .utf8)!)
        exit(1)
    case .none:
        exit(1)
    }
    
case "location":
    let semaphore = DispatchSemaphore(value: 0)
    var result: Result<LocationResponse, Error>?
    
    Task {
        do {
            let locationManager = LocationManager()
            let geocoder = CLGeocoder()
            
            let location = try await locationManager.getCurrentLocation()
            let placemarks = try await geocoder.reverseGeocodeLocation(location)
            let placemark = placemarks.first
            
            let response = LocationResponse(
                latitude: location.coordinate.latitude,
                longitude: location.coordinate.longitude,
                city: placemark?.locality ?? "Unknown",
                state: placemark?.administrativeArea,
                country: placemark?.country ?? "Unknown",
                timezone: TimeZone.current.identifier,
                source: "CoreLocation"
            )
            result = .success(response)
        } catch {
            result = .failure(error)
        }
        semaphore.signal()
    }
    
    semaphore.wait()
    
    switch result {
    case .success(let location):
        let data = try! encoder.encode(location)
        print(String(data: data, encoding: .utf8)!)
    case .failure(let error):
        let errorResponse = ErrorResponse(
            error: "location_failed",
            message: error.localizedDescription,
            code: "LOCATION_ERROR"
        )
        let data = try! encoder.encode(errorResponse)
        print(String(data: data, encoding: .utf8)!)
        exit(1)
    case .none:
        exit(1)
    }
    
case "temperature":
    let semaphore = DispatchSemaphore(value: 0)
    var result: Result<WeatherResponse, Error>?
    
    Task {
        do {
            let locationManager = LocationManager()
            let weatherService = WeatherService()
            
            let location = try await locationManager.getCurrentLocation()
            let weather = try await weatherService.getWeather(for: location)
            result = .success(weather)
        } catch {
            result = .failure(error)
        }
        semaphore.signal()
    }
    
    semaphore.wait()
    
    switch result {
    case .success(let weather):
        // Output simplified temperature response
        let tempResponse = [
            "temperature": weather.temperature,
            "temperature_f": weather.temperatureFahrenheit,
            "feels_like": weather.apparentTemperature,
            "feels_like_f": weather.apparentTemperatureFahrenheit,
            "location": weather.location
        ] as [String : Any]
        let data = try! JSONSerialization.data(withJSONObject: tempResponse, options: isPretty ? [.prettyPrinted, .sortedKeys] : [])
        print(String(data: data, encoding: .utf8)!)
    case .failure(let error):
        let errorResponse = ErrorResponse(
            error: "temperature_failed",
            message: error.localizedDescription,
            code: "TEMPERATURE_ERROR"
        )
        let data = try! encoder.encode(errorResponse)
        print(String(data: data, encoding: .utf8)!)
        exit(1)
    case .none:
        exit(1)
    }
    
default:
    printUsage()
    exit(0)
}

// Keep the program running until async tasks complete
RunLoop.main.run()