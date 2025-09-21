#!/usr/bin/env swift

import Foundation
import WeatherKit
import CoreLocation

print("Testing WeatherKit access...")

// Simple test location (Toronto)
let location = CLLocation(latitude: 43.6532, longitude: -79.3832)

Task {
    do {
        print("Creating WeatherService...")
        let service = WeatherService()
        
        print("Requesting weather...")
        let weather = try await service.weather(for: location)
        
        print("Success! Temperature: \(weather.currentWeather.temperature.value)°C")
        print("Condition: \(weather.currentWeather.condition.description)")
        exit(0)
    } catch {
        print("Error: \(error)")
        print("Error type: \(type(of: error))")
        
        if let nsError = error as NSError? {
            print("Domain: \(nsError.domain)")
            print("Code: \(nsError.code)")
            print("User Info: \(nsError.userInfo)")
        }
        exit(1)
    }
}

RunLoop.main.run()