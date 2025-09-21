#!/usr/bin/env swift

import Foundation
import CoreLocation
import WeatherKit
import os.log

// Custom Logger
@available(macOS 10.12, *)
let logger = Logger(subsystem: "com.jarvis.weather", category: "WeatherProvider")

// Weather Provider for JARVIS using macOS WeatherKit
@available(macOS 13.0, *)
class WeatherProvider: NSObject {
    private let weatherService = WeatherService.shared
    private var locationManager: CLLocationManager?
    private var currentLocation: CLLocation?
    private let geocoder = CLGeocoder()
    
    // Cache configuration
    private var weatherCache: [String: (weather: Weather, timestamp: Date)] = [:]
    private let cacheExpiration: TimeInterval = 300 // 5 minutes
    
    override init() {
        super.init()
        setupLocationServices()
    }
    
    private func setupLocationServices() {
        locationManager = CLLocationManager()
        locationManager?.delegate = self
        locationManager?.desiredAccuracy = kCLLocationAccuracyKilometer
        
        // Request location authorization
        if CLLocationManager.authorizationStatus() == .notDetermined {
            locationManager?.requestWhenInUseAuthorization()
        }
    }
    
    // MARK: - Public Methods
    
    func getWeatherData(completion: @escaping (String?) -> Void) {
        Task {
            do {
                let location = try await getCurrentLocation()
                let weather = try await fetchWeather(for: location)
                let formattedData = formatWeatherForJSON(weather: weather, location: location)
                completion(formattedData)
            } catch {
                logger.error("Failed to get weather: \(error.localizedDescription)")
                completion(nil)
            }
        }
    }
    
    func getWeatherForCity(city: String, completion: @escaping (String?) -> Void) {
        Task {
            do {
                let location = try await getLocationForCity(city)
                let weather = try await fetchWeather(for: location)
                let formattedData = formatWeatherForJSON(weather: weather, location: location, cityName: city)
                completion(formattedData)
            } catch {
                logger.error("Failed to get weather for city \(city): \(error.localizedDescription)")
                completion(nil)
            }
        }
    }
    
    // MARK: - Location Methods
    
    private func getCurrentLocation() async throws -> CLLocation {
        // Check if we have a recent location
        if let location = currentLocation,
           Date().timeIntervalSince(location.timestamp) < 3600 { // 1 hour
            return location
        }
        
        // Get current location from system
        guard let locationManager = locationManager else {
            throw WeatherError.locationServicesDisabled
        }
        
        // Use IP-based location as fallback
        if CLLocationManager.authorizationStatus() != .authorized {
            return try await getLocationFromIP()
        }
        
        // Get precise location
        locationManager.startUpdatingLocation()
        
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                self.locationManager?.stopUpdatingLocation()
                
                if let location = self.currentLocation {
                    continuation.resume(returning: location)
                } else {
                    // Fallback to IP location
                    Task {
                        do {
                            let ipLocation = try await self.getLocationFromIP()
                            continuation.resume(returning: ipLocation)
                        } catch {
                            continuation.resume(throwing: error)
                        }
                    }
                }
            }
        }
    }
    
    private func getLocationFromIP() async throws -> CLLocation {
        // Use IP geolocation API
        let url = URL(string: "https://ipapi.co/json/")!
        let (data, _) = try await URLSession.shared.data(from: url)
        
        struct IPLocation: Decodable {
            let latitude: Double
            let longitude: Double
            let city: String
            let region: String
            let country_name: String
        }
        
        let ipInfo = try JSONDecoder().decode(IPLocation.self, from: data)
        return CLLocation(latitude: ipInfo.latitude, longitude: ipInfo.longitude)
    }
    
    private func getLocationForCity(_ city: String) async throws -> CLLocation {
        let placemarks = try await geocoder.geocodeAddressString(city)
        guard let location = placemarks.first?.location else {
            throw WeatherError.cityNotFound
        }
        return location
    }
    
    // MARK: - Weather Fetching
    
    private func fetchWeather(for location: CLLocation) async throws -> Weather {
        // Check cache first
        let cacheKey = "\(location.coordinate.latitude),\(location.coordinate.longitude)"
        if let cached = weatherCache[cacheKey],
           Date().timeIntervalSince(cached.timestamp) < cacheExpiration {
            logger.info("Returning cached weather data")
            return cached.weather
        }
        
        // Fetch fresh weather data
        let weather = try await weatherService.weather(for: location)
        
        // Update cache
        weatherCache[cacheKey] = (weather, Date())
        
        return weather
    }
    
    // MARK: - Formatting
    
    private func formatWeatherForJSON(weather: Weather, location: CLLocation, cityName: String? = nil) -> String? {
        var weatherData: [String: Any] = [:]
        
        // Get city name if not provided
        let finalCityName = cityName ?? getCityName(from: location)
        
        // Current weather
        if let current = weather.currentWeather {
            weatherData["location"] = finalCityName
            weatherData["temperature"] = Int(current.temperature.value)
            weatherData["temperatureUnit"] = current.temperature.unit.symbol
            weatherData["feelsLike"] = Int(current.apparentTemperature.value)
            weatherData["description"] = current.condition.description
            weatherData["condition"] = getConditionString(current.condition)
            weatherData["humidity"] = Int(current.humidity * 100)
            weatherData["windSpeed"] = round(current.wind.speed.value * 10) / 10
            weatherData["windSpeedUnit"] = current.wind.speed.unit.symbol
            weatherData["windDirection"] = current.wind.compassDirection.abbreviation
            weatherData["pressure"] = Int(current.pressure.value)
            weatherData["pressureUnit"] = current.pressure.unit.symbol
            weatherData["visibility"] = round(current.visibility.value / 1000 * 10) / 10 // Convert to km
            weatherData["uvIndex"] = current.uvIndex.value
            weatherData["cloudCover"] = Int(current.cloudCover * 100)
            weatherData["dewPoint"] = Int(current.dewPoint.value)
            weatherData["isDaylight"] = current.isDaylight
            weatherData["timestamp"] = ISO8601DateFormatter().string(from: Date())
        }
        
        // Daily forecast summary
        if let dailyForecast = weather.dailyForecast.forecast.first {
            weatherData["todayHigh"] = Int(dailyForecast.highTemperature.value)
            weatherData["todayLow"] = Int(dailyForecast.lowTemperature.value)
            weatherData["sunrise"] = formatTime(dailyForecast.sun.sunrise)
            weatherData["sunset"] = formatTime(dailyForecast.sun.sunset)
            weatherData["moonPhase"] = dailyForecast.moon.phase.description
            
            if let precipChance = dailyForecast.precipitationChance {
                weatherData["precipitationChance"] = Int(precipChance * 100)
            }
            
            if let precipAmount = dailyForecast.precipitationAmount {
                weatherData["precipitationAmount"] = round(precipAmount.value * 10) / 10
                weatherData["precipitationUnit"] = precipAmount.unit.symbol
            }
        }
        
        // Hourly forecast for next 6 hours
        var hourlyData: [[String: Any]] = []
        for (index, hour) in weather.hourlyForecast.forecast.prefix(6).enumerated() {
            var hourData: [String: Any] = [:]
            hourData["hour"] = formatTime(hour.date)
            hourData["temperature"] = Int(hour.temperature.value)
            hourData["condition"] = getConditionString(hour.condition)
            hourData["precipitationChance"] = Int((hour.precipitationChance ?? 0) * 100)
            hourlyData.append(hourData)
        }
        weatherData["hourlyForecast"] = hourlyData
        
        // Weather alerts if any
        if !weather.weatherAlerts.isEmpty {
            var alerts: [[String: String]] = []
            for alert in weather.weatherAlerts {
                alerts.append([
                    "summary": alert.summary,
                    "severity": alert.severity.description,
                    "source": alert.source
                ])
            }
            weatherData["alerts"] = alerts
        }
        
        // Intelligent insights
        weatherData["insights"] = generateWeatherInsights(weather: weather)
        
        // Convert to JSON
        do {
            let jsonData = try JSONSerialization.data(withJSONObject: weatherData, options: [.prettyPrinted])
            return String(data: jsonData, encoding: .utf8)
        } catch {
            logger.error("Failed to serialize weather data: \(error)")
            return nil
        }
    }
    
    private func getCityName(from location: CLLocation) -> String {
        // This would ideally use reverse geocoding, but for now return a default
        return "Current Location"
    }
    
    private func formatTime(_ date: Date?) -> String {
        guard let date = date else { return "N/A" }
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm"
        return formatter.string(from: date)
    }
    
    private func getConditionString(_ condition: WeatherCondition) -> String {
        switch condition {
        case .clear: return "clear"
        case .cloudy: return "cloudy"
        case .rain: return "rain"
        case .snow: return "snow"
        case .sleet: return "sleet"
        case .hail: return "hail"
        case .thunderstorms: return "thunderstorms"
        case .tropicalStorm: return "tropical storm"
        case .hurricane: return "hurricane"
        case .smoky: return "smoky"
        case .foggy: return "foggy"
        case .haze: return "hazy"
        case .windy: return "windy"
        case .frigid: return "frigid"
        case .hot: return "hot"
        case .freezingRain: return "freezing rain"
        case .heavyRain: return "heavy rain"
        case .heavySnow: return "heavy snow"
        case .blowingDust: return "blowing dust"
        case .drizzle: return "drizzle"
        case .flurries: return "flurries"
        case .mostlyClear: return "mostly clear"
        case .mostlyCloudy: return "mostly cloudy"
        case .partlyCloudy: return "partly cloudy"
        case .scatteredThunderstorms: return "scattered thunderstorms"
        case .strongStorms: return "strong storms"
        case .sunShowers: return "sun showers"
        case .wintryMix: return "wintry mix"
        default: return "unknown"
        }
    }
    
    private func generateWeatherInsights(weather: Weather) -> [String] {
        var insights: [String] = []
        
        guard let current = weather.currentWeather else { return insights }
        
        // Temperature insights
        let temp = current.temperature.converted(to: .celsius).value
        let feelsLike = current.apparentTemperature.converted(to: .celsius).value
        
        if temp > 30 {
            insights.append("It's quite hot today. Stay hydrated and seek shade when possible.")
        } else if temp > 25 {
            insights.append("Warm weather today. Perfect for outdoor activities.")
        } else if temp < 0 {
            insights.append("Freezing temperatures. Bundle up and watch for icy conditions.")
        } else if temp < 10 {
            insights.append("Cool weather. A light jacket would be advisable.")
        }
        
        // Feels like difference
        if abs(feelsLike - temp) > 5 {
            if feelsLike > temp {
                insights.append("High humidity makes it feel warmer than the actual temperature.")
            } else {
                insights.append("Wind chill makes it feel cooler than the actual temperature.")
            }
        }
        
        // Wind insights
        let windSpeed = current.wind.speed.converted(to: .kilometersPerHour).value
        if windSpeed > 40 {
            insights.append("Strong winds today. Secure loose items and be cautious outdoors.")
        } else if windSpeed > 25 {
            insights.append("Breezy conditions. Good day for flying kites!")
        }
        
        // UV insights
        if current.uvIndex.value >= 8 {
            insights.append("Very high UV index. Sunscreen and protective clothing essential.")
        } else if current.uvIndex.value >= 6 {
            insights.append("High UV levels. Don't forget sunscreen if going outside.")
        }
        
        // Visibility insights
        let visibility = current.visibility.converted(to: .kilometers).value
        if visibility < 1 {
            insights.append("Poor visibility. Drive carefully and use headlights.")
        }
        
        // Precipitation insights
        if let dailyForecast = weather.dailyForecast.forecast.first {
            if let precipChance = dailyForecast.precipitationChance, precipChance > 0.7 {
                insights.append("High chance of precipitation. Keep an umbrella handy.")
            }
        }
        
        return insights
    }
}

// MARK: - CLLocationManagerDelegate

@available(macOS 13.0, *)
extension WeatherProvider: CLLocationManagerDelegate {
    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        if let location = locations.last {
            currentLocation = location
            logger.info("Location updated: \(location.coordinate.latitude), \(location.coordinate.longitude)")
        }
    }
    
    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        logger.error("Location manager failed: \(error.localizedDescription)")
    }
    
    func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        logger.info("Location authorization changed: \(manager.authorizationStatus.rawValue)")
    }
}

// MARK: - Error Types

enum WeatherError: LocalizedError {
    case locationServicesDisabled
    case cityNotFound
    case weatherKitUnavailable
    
    var errorDescription: String? {
        switch self {
        case .locationServicesDisabled:
            return "Location services are disabled. Please enable them in System Settings."
        case .cityNotFound:
            return "Could not find the specified city."
        case .weatherKitUnavailable:
            return "WeatherKit is not available on this system."
        }
    }
}

// MARK: - Main Entry Point

@available(macOS 13.0, *)
@main
struct WeatherProviderApp {
    static func main() async {
        let provider = WeatherProvider()
        
        // Parse command line arguments
        let args = CommandLine.arguments
        
        if args.count > 1 {
            let command = args[1]
            
            switch command {
            case "current":
                // Get current location weather
                provider.getWeatherData { json in
                    if let json = json {
                        print(json)
                    } else {
                        print("{\"error\": \"Failed to get weather data\"}")
                    }
                    exit(0)
                }
                
            case "city":
                // Get weather for specific city
                if args.count > 2 {
                    let city = args[2...].joined(separator: " ")
                    provider.getWeatherForCity(city: city) { json in
                        if let json = json {
                            print(json)
                        } else {
                            print("{\"error\": \"Failed to get weather for \(city)\"}")
                        }
                        exit(0)
                    }
                } else {
                    print("{\"error\": \"City name required\"}")
                    exit(1)
                }
                
            default:
                print("{\"error\": \"Unknown command: \(command)\"}")
                exit(1)
            }
            
            // Keep running until completion
            RunLoop.current.run()
        } else {
            // Default to current weather
            provider.getWeatherData { json in
                if let json = json {
                    print(json)
                } else {
                    print("{\"error\": \"Failed to get weather data\"}")
                }
                exit(0)
            }
            
            RunLoop.current.run()
        }
    }
}

// For older macOS versions, provide a fallback
if #available(macOS 13.0, *) {
    // Main app runs here
} else {
    print("{\"error\": \"WeatherKit requires macOS 13.0 or later. Please use the API-based weather service.\"}")
    exit(1)
}