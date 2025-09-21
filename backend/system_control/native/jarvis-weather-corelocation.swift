import Foundation
import CoreLocation
import AppKit

// Weather response structure
struct WeatherResponse: Codable {
    let location: String
    let temperature: Int
    let temperature_f: Int
    let condition: String
    let humidity: Int
    let wind_speed: Double
    let wind_speed_mph: Double
    let wind_direction: String
    let pressure: Int
    let visibility: Double
    let uv_index: Int
    let feels_like: Int
    let feels_like_f: Int
    let cloud_cover: Int
    let description: String
    let latitude: Double
    let longitude: Double
    let timestamp: String
    let source: String
}

// Location Manager for getting current location
class LocationManager: NSObject, CLLocationManagerDelegate {
    private let locationManager = CLLocationManager()
    private var completion: ((CLLocation?) -> Void)?
    
    override init() {
        super.init()
        locationManager.delegate = self
    }
    
    func getCurrentLocation(completion: @escaping (CLLocation?) -> Void) {
        self.completion = completion
        
        // Check authorization status
        let status = locationManager.authorizationStatus
        
        switch status {
        case .authorized, .authorizedAlways:
            // We have permission, get location
            locationManager.requestLocation()
        case .notDetermined:
            // Need to request permission - shouldn't happen if TerminalLocationEnabler worked
            print("Location permission not determined. Please run TerminalLocationEnabler first.", to: &errorStream)
            completion(nil)
        case .denied, .restricted:
            print("Location access denied. Please enable in System Settings.", to: &errorStream)
            completion(nil)
        @unknown default:
            print("Unknown location authorization status.", to: &errorStream)
            completion(nil)
        }
    }
    
    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        if let location = locations.first {
            completion?(location)
        }
    }
    
    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        print("Location error: \(error.localizedDescription)", to: &errorStream)
        completion?(nil)
    }
}

// Weather service using wttr.in API
class WeatherService {
    func fetchWeather(latitude: Double, longitude: Double) async throws -> WeatherResponse {
        let url = URL(string: "https://wttr.in/\(latitude),\(longitude)?format=j1")!
        
        let (data, response) = try await URLSession.shared.data(from: url)
        
        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            throw NSError(domain: "WeatherAPI", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to fetch weather data"])
        }
        
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let current = json["current_condition"] as! [[String: Any]]
        let currentCondition = current[0]
        
        // Get location info
        let nearestArea = json["nearest_area"] as! [[String: Any]]
        let area = nearestArea[0]
        let cityInfo = area["areaName"] as! [[String: String]]
        let countryInfo = area["country"] as! [[String: String]]
        let city = cityInfo[0]["value"]!
        let country = countryInfo[0]["value"]!
        
        // Parse weather data
        let tempC = Int((currentCondition["temp_C"] as! NSString).integerValue)
        let tempF = Int((currentCondition["temp_F"] as! NSString).integerValue)
        let humidity = Int((currentCondition["humidity"] as! NSString).integerValue)
        let windSpeedKmph = Double((currentCondition["windspeedKmph"] as! NSString).doubleValue)
        let windSpeedMph = Double((currentCondition["windspeedMiles"] as! NSString).doubleValue)
        let windDir = currentCondition["winddir16Point"] as! String
        let pressure = Int((currentCondition["pressure"] as! NSString).integerValue)
        let visibility = Double((currentCondition["visibility"] as! NSString).doubleValue)
        let uvIndex = Int((currentCondition["uvIndex"] as! NSString).integerValue)
        let feelsLikeC = Int((currentCondition["FeelsLikeC"] as! NSString).integerValue)
        let feelsLikeF = Int((currentCondition["FeelsLikeF"] as! NSString).integerValue)
        let cloudCover = Int((currentCondition["cloudcover"] as! NSString).integerValue)
        
        let weatherDesc = currentCondition["weatherDesc"] as! [[String: String]]
        let description = weatherDesc[0]["value"]!
        
        return WeatherResponse(
            location: "\(city), \(country)",
            temperature: tempC,
            temperature_f: tempF,
            condition: description,
            humidity: humidity,
            wind_speed: windSpeedKmph,
            wind_speed_mph: windSpeedMph,
            wind_direction: windDir,
            pressure: pressure,
            visibility: visibility,
            uv_index: uvIndex,
            feels_like: feelsLikeC,
            feels_like_f: feelsLikeF,
            cloud_cover: cloudCover,
            description: description.lowercased(),
            latitude: latitude,
            longitude: longitude,
            timestamp: ISO8601DateFormatter().string(from: Date()),
            source: "CoreLocation + wttr.in"
        )
    }
    
    func fetchWeatherByCity(_ city: String) async throws -> WeatherResponse {
        let encodedCity = city.addingPercentEncoding(withAllowedCharacters: .urlHostAllowed)!
        let url = URL(string: "https://wttr.in/\(encodedCity)?format=j1")!
        
        let (data, response) = try await URLSession.shared.data(from: url)
        
        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            throw NSError(domain: "WeatherAPI", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to fetch weather data"])
        }
        
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let current = json["current_condition"] as! [[String: Any]]
        let currentCondition = current[0]
        
        // Get location info
        let nearestArea = json["nearest_area"] as! [[String: Any]]
        let area = nearestArea[0]
        let latitude = Double(area["latitude"] as! String)!
        let longitude = Double(area["longitude"] as! String)!
        let cityInfo = area["areaName"] as! [[String: String]]
        let countryInfo = area["country"] as! [[String: String]]
        let cityName = cityInfo[0]["value"]!
        let country = countryInfo[0]["value"]!
        
        // Parse weather data (same as above)
        let tempC = Int((currentCondition["temp_C"] as! NSString).integerValue)
        let tempF = Int((currentCondition["temp_F"] as! NSString).integerValue)
        let humidity = Int((currentCondition["humidity"] as! NSString).integerValue)
        let windSpeedKmph = Double((currentCondition["windspeedKmph"] as! NSString).doubleValue)
        let windSpeedMph = Double((currentCondition["windspeedMiles"] as! NSString).doubleValue)
        let windDir = currentCondition["winddir16Point"] as! String
        let pressure = Int((currentCondition["pressure"] as! NSString).integerValue)
        let visibility = Double((currentCondition["visibility"] as! NSString).doubleValue)
        let uvIndex = Int((currentCondition["uvIndex"] as! NSString).integerValue)
        let feelsLikeC = Int((currentCondition["FeelsLikeC"] as! NSString).integerValue)
        let feelsLikeF = Int((currentCondition["FeelsLikeF"] as! NSString).integerValue)
        let cloudCover = Int((currentCondition["cloudcover"] as! NSString).integerValue)
        
        let weatherDesc = currentCondition["weatherDesc"] as! [[String: String]]
        let description = weatherDesc[0]["value"]!
        
        return WeatherResponse(
            location: "\(cityName), \(country)",
            temperature: tempC,
            temperature_f: tempF,
            condition: description,
            humidity: humidity,
            wind_speed: windSpeedKmph,
            wind_speed_mph: windSpeedMph,
            wind_direction: windDir,
            pressure: pressure,
            visibility: visibility,
            uv_index: uvIndex,
            feels_like: feelsLikeC,
            feels_like_f: feelsLikeF,
            cloud_cover: cloudCover,
            description: description.lowercased(),
            latitude: latitude,
            longitude: longitude,
            timestamp: ISO8601DateFormatter().string(from: Date()),
            source: "CoreLocation + wttr.in"
        )
    }
}

// Error output stream
var errorStream = FileHandle.standardError

extension FileHandle : TextOutputStream {
    public func write(_ string: String) {
        self.write(string.data(using: .utf8)!)
    }
}

// Main program
enum JarvisWeatherCoreLocation {
    static func main() async {
        let args = CommandLine.arguments
        let command = args.count > 1 ? args[1] : "current"
        
        do {
            let weatherService = WeatherService()
            let weather: WeatherResponse
            
            if command == "current" {
                // Use CoreLocation to get current location
                let locationManager = LocationManager()
                
                // Use async/await pattern instead of semaphore
                let location = await withCheckedContinuation { continuation in
                    locationManager.getCurrentLocation { location in
                        continuation.resume(returning: location)
                    }
                }
                
                guard let location = location else {
                    print("Failed to get current location", to: &errorStream)
                    exit(1)
                }
                
                weather = try await weatherService.fetchWeather(
                    latitude: location.coordinate.latitude,
                    longitude: location.coordinate.longitude
                )
            } else if command == "city" && args.count > 2 {
                let city = args[2...].joined(separator: " ")
                weather = try await weatherService.fetchWeatherByCity(city)
            } else {
                print("Usage: jarvis-weather-corelocation [current | city <city name>]", to: &errorStream)
                exit(1)
            }
            
            // Output JSON
            let encoder = JSONEncoder()
            let jsonData = try encoder.encode(weather)
            print(String(data: jsonData, encoding: .utf8)!)
            
        } catch {
            print("Error: \(error.localizedDescription)", to: &errorStream)
            exit(1)
        }
    }
}

// Entry point
Task {
    await JarvisWeatherCoreLocation.main()
}