#!/usr/bin/env swift

import Foundation
import CoreLocation
import WeatherKit

// MARK: - Models (same as before)
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
    let windSpeed: Double
    let windSpeedMph: Double
    let windDirection: String
    let visibility: Double
    let uvIndex: Int
    let cloudCover: Double
    let isDaylight: Bool
    let source: String
    let timestamp: String
    let timezone: String
    
    private enum CodingKeys: String, CodingKey {
        case location, latitude, longitude, temperature
        case temperatureFahrenheit = "temperature_f"
        case apparentTemperature = "feels_like"
        case apparentTemperatureFahrenheit = "feels_like_f"
        case condition, symbolName, description, humidity, pressure
        case windSpeed = "wind_speed"
        case windSpeedMph = "wind_speed_mph"
        case windDirection = "wind_direction"
        case visibility
        case uvIndex = "uv_index"
        case cloudCover = "cloud_cover"
        case isDaylight = "is_daylight"
        case source, timestamp, timezone
    }
}

struct ErrorResponse: Codable {
    let error: String
    let message: String
    let code: String
}

// MARK: - IP-based Location Service
class IPLocationService {
    func getCurrentLocation() async throws -> CLLocation {
        // Try multiple IP geolocation services
        let services = [
            "http://ip-api.com/json/?fields=status,lat,lon,city,regionName,country",
            "https://ipapi.co/json/",
            "https://ipinfo.io/json"
        ]
        
        for serviceURL in services {
            do {
                let url = URL(string: serviceURL)!
                let (data, _) = try await URLSession.shared.data(from: url)
                
                if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    var lat: Double?
                    var lon: Double?
                    
                    // Handle different response formats
                    if serviceURL.contains("ip-api.com") {
                        lat = json["lat"] as? Double
                        lon = json["lon"] as? Double
                    } else if serviceURL.contains("ipapi.co") {
                        lat = json["latitude"] as? Double
                        lon = json["longitude"] as? Double
                    } else if serviceURL.contains("ipinfo.io") {
                        if let loc = json["loc"] as? String {
                            let parts = loc.split(separator: ",")
                            if parts.count == 2 {
                                lat = Double(parts[0])
                                lon = Double(parts[1])
                            }
                        }
                    }
                    
                    if let lat = lat, let lon = lon {
                        return CLLocation(latitude: lat, longitude: lon)
                    }
                }
            } catch {
                // Try next service
                continue
            }
        }
        
        throw NSError(domain: "IPLocation", code: 1, userInfo: [NSLocalizedDescriptionKey: "Could not determine location from IP"])
    }
}

// MARK: - Hybrid Weather Service
@available(macOS 13.0, *)
class HybridWeatherService {
    private let weatherService = WeatherService()
    private let geocoder = CLGeocoder()
    private let ipLocationService = IPLocationService()
    
    func getCurrentWeather() async throws -> WeatherResponse {
        // Try to get location from IP (no permissions needed)
        let location = try await ipLocationService.getCurrentLocation()
        
        // Get weather for that location
        return try await getWeather(for: location)
    }
    
    func getWeather(for location: CLLocation) async throws -> WeatherResponse {
        // Get weather data
        let weather = try await weatherService.weather(for: location)
        
        // Get location name
        let placemarks = try await geocoder.reverseGeocodeLocation(location)
        let placemark = placemarks.first
        
        let cityName = placemark?.locality ?? placemark?.administrativeArea ?? "Unknown Location"
        let state = placemark?.administrativeArea
        let country = placemark?.country ?? ""
        
        var locationName = cityName
        if let state = state, country == "United States" {
            locationName = "\(cityName), \(state)"
        } else if !country.isEmpty && country != cityName {
            locationName = "\(cityName), \(country)"
        }
        
        // Build response
        let current = weather.currentWeather
        
        let tempC = current.temperature.value
        let tempF = tempC * 9/5 + 32
        let feelsLikeC = current.apparentTemperature.value
        let feelsLikeF = feelsLikeC * 9/5 + 32
        
        let windDirections = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                             "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        let windIndex = Int((current.wind.direction.value + 11.25) / 22.5) % 16
        let windDir = windDirections[windIndex]
        
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
            windSpeed: current.wind.speed.value,
            windSpeedMph: current.wind.speed.converted(to: .milesPerHour).value,
            windDirection: windDir,
            visibility: current.visibility.value / 1000,
            uvIndex: Int(current.uvIndex.value),
            cloudCover: (current.cloudCover * 100).rounded(),
            isDaylight: current.isDaylight,
            source: "WeatherKit",
            timestamp: ISO8601DateFormatter().string(from: Date()),
            timezone: TimeZone.current.identifier
        )
    }
    
    func getWeatherForCity(_ cityName: String) async throws -> WeatherResponse {
        let locations = try await geocoder.geocodeAddressString(cityName)
        guard let location = locations.first?.location else {
            throw NSError(domain: "Weather", code: 404, userInfo: [NSLocalizedDescriptionKey: "Location not found"])
        }
        
        return try await getWeather(for: location)
    }
}

// MARK: - Main Program
let arguments = CommandLine.arguments
let command = arguments.count > 1 ? arguments[1] : "current"
let isPretty = arguments.contains("--pretty") || arguments.contains("-p")

let encoder = JSONEncoder()
if isPretty {
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
}

Task { @MainActor in
    do {
        let weatherService = HybridWeatherService()
        
        switch command {
        case "current":
            let weather = try await weatherService.getCurrentWeather()
            let data = try encoder.encode(weather)
            print(String(data: data, encoding: .utf8)!)
            exit(0)
            
        case "city":
            guard arguments.count > 2 else {
                let error = ErrorResponse(error: "missing_city", message: "Please provide a city name", code: "MISSING_ARGUMENT")
                let data = try encoder.encode(error)
                print(String(data: data, encoding: .utf8)!)
                exit(1)
            }
            
            let cityName = arguments[2...].joined(separator: " ")
                .replacingOccurrences(of: "--pretty", with: "")
                .replacingOccurrences(of: "-p", with: "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            
            let weather = try await weatherService.getWeatherForCity(cityName)
            let data = try encoder.encode(weather)
            print(String(data: data, encoding: .utf8)!)
            exit(0)
            
        default:
            print("Usage: jarvis-weather [current|city <name>] [--pretty]")
            exit(0)
        }
    } catch {
        let errorResponse = ErrorResponse(
            error: "weather_failed",
            message: error.localizedDescription,
            code: "WEATHER_ERROR"
        )
        let data = try encoder.encode(errorResponse)
        print(String(data: data, encoding: .utf8)!)
        exit(1)
    }
}

RunLoop.main.run()