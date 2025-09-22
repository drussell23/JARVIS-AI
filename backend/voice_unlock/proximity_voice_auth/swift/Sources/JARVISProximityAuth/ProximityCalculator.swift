//
//  ProximityCalculator.swift
//  JARVISProximityAuth
//
//  Calculates proximity confidence from Bluetooth signals
//

import Foundation
import CoreBluetooth

public class ProximityCalculator {
    
    // MARK: - Types
    
    public struct ProximityMetrics {
        public let distance: Double
        public let confidence: Double
        public let signalQuality: Double
        public let isReliable: Bool
    }
    
    // MARK: - Properties
    
    private let txPower: Double = -59.0  // Calibrated transmit power at 1m
    private let pathLossExponent: Double = 2.0  // For indoor environment
    
    // Signal quality thresholds
    private let excellentRSSI = -50
    private let goodRSSI = -60
    private let fairRSSI = -70
    private let poorRSSI = -80
    
    // MARK: - Public Methods
    
    /// Calculate proximity metrics from RSSI value
    public func calculateMetrics(rssi: Int) -> ProximityMetrics {
        let distance = calculateDistance(rssi: rssi)
        let confidence = calculateConfidence(rssi: rssi, distance: distance)
        let signalQuality = calculateSignalQuality(rssi: rssi)
        let isReliable = rssi > poorRSSI && signalQuality > 50.0
        
        return ProximityMetrics(
            distance: distance,
            confidence: confidence,
            signalQuality: signalQuality,
            isReliable: isReliable
        )
    }
    
    /// Calculate distance from RSSI using path loss model
    private func calculateDistance(rssi: Int) -> Double {
        // Path loss formula: RSSI = TxPower - 10 * n * log10(distance)
        // Rearranged: distance = 10^((TxPower - RSSI) / (10 * n))
        
        let distance = pow(10.0, (txPower - Double(rssi)) / (10.0 * pathLossExponent))
        
        // Apply smoothing for more stable readings
        return min(10.0, max(0.1, distance))  // Clamp between 0.1m and 10m
    }
    
    /// Calculate confidence score (0-100) based on RSSI and distance
    private func calculateConfidence(rssi: Int, distance: Double) -> Double {
        var confidence = 0.0
        
        // Base confidence on signal strength
        if rssi >= excellentRSSI {
            confidence = 95.0
        } else if rssi >= goodRSSI {
            confidence = 85.0
        } else if rssi >= fairRSSI {
            confidence = 70.0
        } else if rssi >= poorRSSI {
            confidence = 50.0
        } else {
            confidence = 20.0
        }
        
        // Adjust based on distance
        if distance <= 1.0 {
            confidence = min(100.0, confidence + 10.0)  // Very close
        } else if distance <= 3.0 {
            // Target range - confidence remains unchanged
        } else if distance <= 5.0 {
            confidence = max(0.0, confidence - 10.0)  // Getting far
        } else {
            confidence = max(0.0, confidence - 20.0)  // Too far
        }
        
        return confidence
    }
    
    /// Calculate signal quality percentage
    private func calculateSignalQuality(rssi: Int) -> Double {
        // Map RSSI to quality percentage
        // -30 dBm = 100%, -90 dBm = 0%
        let minRSSI = -90.0
        let maxRSSI = -30.0
        
        let quality = (Double(rssi) - minRSSI) / (maxRSSI - minRSSI) * 100.0
        return min(100.0, max(0.0, quality))
    }
    
    /// Apply Kalman filter for smoothing distance estimates
    public func smoothDistance(_ newDistance: Double, 
                             previousDistance: Double?,
                             processNoise: Double = 0.1,
                             measurementNoise: Double = 1.0) -> Double {
        
        guard let previous = previousDistance else {
            return newDistance
        }
        
        // Simple exponential smoothing
        let alpha = 0.3  // Smoothing factor
        return alpha * newDistance + (1 - alpha) * previous
    }
}