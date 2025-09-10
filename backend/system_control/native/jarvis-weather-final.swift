#!/usr/bin/env swift

import Foundation
import CoreLocation

// Since WeatherKit requires app signing and permissions, 
// we'll use a combination of approaches:
// 1. IP-based location (no permissions needed)
// 2. Free weather APIs (no keys needed)
// 3. System integration where possible

struct WeatherResponse: Codable {
    let location: String
    let latitude: Double
    let longitude: Double
    let temperature: Double
    let temperatureFahrenheit: Double
    let apparentTemperature: Double
    let apparentTemperatureFahrenheit: Double
    let condition: String
    let description: String
    let humidity: Double
    let pressure: Double
    let windSpeed: Double
    let windSpeedMph: Double
    let windDirection: String
    let visibility: Double
    let uvIndex: Int
    let cloudCover: Double
    let source: String
    let timestamp: String
    
    private enum CodingKeys: String, CodingKey {
        case location, latitude, longitude, temperature
        case temperatureFahrenheit = "temperature_f"
        case apparentTemperature = "feels_like"
        case apparentTemperatureFahrenheit = "feels_like_f"
        case condition, description, humidity, pressure
        case windSpeed = "wind_speed"
        case windSpeedMph = "wind_speed_mph"
        case windDirection = "wind_direction"
        case visibility
        case uvIndex = "uv_index"
        case cloudCover = "cloud_cover"
        case source, timestamp
    }
}

class WeatherService {
    // Get location from IP
    func getIPLocation() async throws -> (Double, Double, String) {
        let url = URL(string: "https://ipapi.co/json/")!
        let (data, _) = try await URLSession.shared.data(from: url)
        
        if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
           let lat = json["latitude"] as? Double,
           let lon = json["longitude"] as? Double,
           let city = json["city"] as? String {
            
            let region = json["region"] as? String ?? ""
            let country = json["country_name"] as? String ?? ""
            
            var location = city
            if !region.isEmpty && country == "United States" {
                location = "\(city), \(region)"
            } else if !country.isEmpty {
                location = "\(city), \(country)"
            }
            
            return (lat, lon, location)
        }
        
        throw NSError(domain: "Location", code: 1, userInfo: [NSLocalizedDescriptionKey: "Could not determine location"])
    }
    
    // Get weather from wttr.in
    func getWeather(lat: Double, lon: Double, location: String) async throws -> WeatherResponse {
        let url = URL(string: "https://wttr.in/\(lat),\(lon)?format=j1")!
        let (data, _) = try await URLSession.shared.data(from: url)
        
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let current = (json["current_condition"] as? [[String: Any]])?.first else {
            throw NSError(domain: "Weather", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid weather data"])
        }
        
        // Parse weather data
        let tempC = Double(current["temp_C"] as? String ?? "0") ?? 0
        let tempF = Double(current["temp_F"] as? String ?? "0") ?? tempC * 9/5 + 32
        let feelsLikeC = Double(current["FeelsLikeC"] as? String ?? "0") ?? tempC
        let feelsLikeF = Double(current["FeelsLikeF"] as? String ?? "0") ?? feelsLikeC * 9/5 + 32
        
        let condition = (current["weatherDesc"] as? [[String: Any]])?.first?["value"] as? String ?? "Unknown"
        
        return WeatherResponse(
            location: location,
            latitude: lat,
            longitude: lon,
            temperature: tempC,
            temperatureFahrenheit: tempF,
            apparentTemperature: feelsLikeC,
            apparentTemperatureFahrenheit: feelsLikeF,
            condition: condition,
            description: condition.lowercased(),
            humidity: Double(current["humidity"] as? String ?? "0") ?? 0,
            pressure: Double(current["pressure"] as? String ?? "1013") ?? 1013,
            windSpeed: Double(current["windspeedKmph"] as? String ?? "0") ?? 0,
            windSpeedMph: Double(current["windspeedMiles"] as? String ?? "0") ?? 0,
            windDirection: current["winddir16Point"] as? String ?? "N",
            visibility: Double(current["visibility"] as? String ?? "10") ?? 10,
            uvIndex: Int(current["uvIndex"] as? String ?? "0") ?? 0,
            cloudCover: Double(current["cloudcover"] as? String ?? "0") ?? 0,
            source: "wttr.in",
            timestamp: ISO8601DateFormatter().string(from: Date())
        )
    }
    
    func getCurrentWeather() async throws -> WeatherResponse {
        let (lat, lon, location) = try await getIPLocation()
        return try await getWeather(lat: lat, lon: lon, location: location)
    }
    
    func getCityWeather(_ city: String) async throws -> WeatherResponse {
        let geocoder = CLGeocoder()
        let placemarks = try await geocoder.geocodeAddressString(city)
        
        guard let placemark = placemarks.first,
              let location = placemark.location else {
            throw NSError(domain: "Weather", code: 404, userInfo: [NSLocalizedDescriptionKey: "City not found"])
        }
        
        let cityName = placemark.locality ?? city
        let region = placemark.administrativeArea ?? ""
        let country = placemark.country ?? ""
        
        var locationName = cityName
        if !region.isEmpty && country == "United States" {
            locationName = "\(cityName), \(region)"
        } else if !country.isEmpty && country != cityName {
            locationName = "\(cityName), \(country)"
        }
        
        return try await getWeather(
            lat: location.coordinate.latitude,
            lon: location.coordinate.longitude,
            location: locationName
        )
    }
}

// Main program
let args = CommandLine.arguments
let command = args.count > 1 ? args[1] : "current"
let isPretty = args.contains("--pretty") || args.contains("-p")

let encoder = JSONEncoder()
if isPretty {
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
}

Task {
    do {
        let service = WeatherService()
        let weather: WeatherResponse
        
        switch command {
        case "current":
            weather = try await service.getCurrentWeather()
            
        case "city":
            guard args.count > 2 else {
                print(#"{"error":"missing_city","message":"Please provide a city name"}"#)
                exit(1)
            }
            let city = args[2...].joined(separator: " ")
                .replacingOccurrences(of: "--pretty", with: "")
                .replacingOccurrences(of: "-p", with: "")
                .trimmingCharacters(in: .whitespaces)
            weather = try await service.getCityWeather(city)
            
        default:
            print("Usage: jarvis-weather [current|city <name>] [--pretty]")
            exit(0)
        }
        
        let data = try encoder.encode(weather)
        print(String(data: data, encoding: .utf8)!)
        exit(0)
        
    } catch {
        let errorDict = [
            "error": "weather_failed",
            "message": error.localizedDescription,
            "code": "WEATHER_ERROR"
        ]
        if let data = try? encoder.encode(errorDict) {
            print(String(data: data, encoding: .utf8)!)
        }
        exit(1)
    }
}

RunLoop.main.run()