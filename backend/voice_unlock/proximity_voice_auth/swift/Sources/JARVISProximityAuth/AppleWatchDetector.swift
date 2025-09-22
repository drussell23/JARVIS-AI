//
//  AppleWatchDetector.swift
//  JARVISProximityAuth
//
//  Detects Apple Watch proximity using Bluetooth LE
//

import Foundation
import CoreBluetooth
import os.log

/// Apple Watch proximity detection service
public class AppleWatchDetector: NSObject {
    
    // MARK: - Properties
    
    private var centralManager: CBCentralManager?
    private var detectedDevices: [UUID: WatchDevice] = [:]
    private let logger = Logger(subsystem: "com.jarvis.proximity", category: "AppleWatchDetector")
    
    /// Callback for proximity updates
    public var proximityUpdateHandler: ((ProximityStatus) -> Void)?
    
    /// Current proximity status
    public private(set) var currentStatus = ProximityStatus.unknown
    
    // MARK: - Types
    
    public struct ProximityStatus {
        public let isNearby: Bool
        public let confidence: Double
        public let distance: Double?
        public let deviceInfo: WatchDevice?
        public let timestamp: Date
        
        public static let unknown = ProximityStatus(
            isNearby: false,
            confidence: 0,
            distance: nil,
            deviceInfo: nil,
            timestamp: Date()
        )
    }
    
    public struct WatchDevice {
        public let identifier: UUID
        public let name: String?
        public let rssi: Int
        public let lastSeen: Date
        public let isAppleWatch: Bool
    }
    
    // MARK: - Initialization
    
    public override init() {
        super.init()
    }
    
    // MARK: - Public Methods
    
    /// Start detecting Apple Watch proximity
    public func startDetection() {
        logger.info("Starting Apple Watch proximity detection")
        
        centralManager = CBCentralManager(
            delegate: self,
            queue: DispatchQueue(label: "com.jarvis.proximity.bluetooth", qos: .userInitiated)
        )
    }
    
    /// Stop detection
    public func stopDetection() {
        logger.info("Stopping Apple Watch proximity detection")
        
        centralManager?.stopScan()
        centralManager = nil
        detectedDevices.removeAll()
        currentStatus = .unknown
    }
    
    /// Calculate proximity score from RSSI
    private func calculateProximity(rssi: Int) -> (distance: Double, confidence: Double) {
        // Typical values for Apple Watch BLE
        let txPower = -59.0  // Calibrated transmit power at 1 meter
        let n = 2.0          // Path loss exponent for indoor environment
        
        // Calculate distance using path loss formula
        let distance = pow(10, (txPower - Double(rssi)) / (10.0 * n))
        
        // Calculate confidence based on signal strength
        let confidence: Double
        if rssi >= -50 {
            confidence = 95.0  // Very close
        } else if rssi >= -60 {
            confidence = 85.0  // Close
        } else if rssi >= -70 {
            confidence = 70.0  // Medium
        } else if rssi >= -80 {
            confidence = 50.0  // Far
        } else {
            confidence = 20.0  // Very far
        }
        
        return (distance, confidence)
    }
    
    /// Check if device is likely an Apple Watch
    private func isLikelyAppleWatch(_ peripheral: CBPeripheral) -> Bool {
        // Check device name patterns
        if let name = peripheral.name {
            let watchPatterns = ["Watch", "⌚️", "Apple Watch"]
            for pattern in watchPatterns {
                if name.contains(pattern) {
                    return true
                }
            }
        }
        
        // Check service UUIDs if available
        if let services = peripheral.services {
            for service in services {
                // Apple Watch specific service UUIDs
                let watchServiceUUIDs = [
                    CBUUID(string: "180D"), // Heart Rate Service
                    CBUUID(string: "180A"), // Device Information Service
                    CBUUID(string: "180F"), // Battery Service
                ]
                
                if watchServiceUUIDs.contains(service.uuid) {
                    return true
                }
            }
        }
        
        return false
    }
    
    /// Update proximity status
    private func updateProximityStatus() {
        // Find the closest Apple Watch
        let now = Date()
        let recentThreshold = 5.0 // seconds
        
        let recentWatches = detectedDevices.values.filter { device in
            device.isAppleWatch &&
            now.timeIntervalSince(device.lastSeen) < recentThreshold
        }
        
        if let closestWatch = recentWatches.max(by: { $0.rssi < $1.rssi }) {
            let (distance, confidence) = calculateProximity(rssi: closestWatch.rssi)
            
            let status = ProximityStatus(
                isNearby: distance <= 3.0, // Within 3 meters
                confidence: confidence,
                distance: distance,
                deviceInfo: closestWatch,
                timestamp: now
            )
            
            currentStatus = status
            proximityUpdateHandler?(status)
            
            logger.debug("Proximity updated: distance=\(distance)m, confidence=\(confidence)%")
        } else {
            // No Apple Watch detected
            let status = ProximityStatus(
                isNearby: false,
                confidence: 0,
                distance: nil,
                deviceInfo: nil,
                timestamp: now
            )
            
            currentStatus = status
            proximityUpdateHandler?(status)
        }
    }
    
    // MARK: - Public Methods
    
    public func getProximityStatus() -> (isNearby: Bool, confidence: Double, distance: Double) {
        let devices = Array(detectedDevices.values)
        
        guard !devices.isEmpty else {
            return (false, 0.0, 10.0)
        }
        
        // Find the closest device based on RSSI
        let closestDevice = devices.max { $0.rssi < $1.rssi }!
        
        // Calculate metrics for the closest device
        let calculator = ProximityCalculator()
        let metrics = calculator.calculateMetrics(rssi: closestDevice.rssi)
        
        return (
            isNearby: metrics.distance <= 3.0,
            confidence: metrics.confidence,
            distance: metrics.distance
        )
    }
    
    public func getDetectedDevices() -> [WatchDevice] {
        return Array(detectedDevices.values)
    }
}

// MARK: - CBCentralManagerDelegate

extension AppleWatchDetector: CBCentralManagerDelegate {
    
    public func centralManagerDidUpdateState(_ central: CBCentralManager) {
        switch central.state {
        case .poweredOn:
            logger.info("Bluetooth powered on, starting scan")
            
            // Start scanning for all devices
            // Note: In production, you might want to scan for specific service UUIDs
            central.scanForPeripherals(
                withServices: nil,
                options: [
                    CBCentralManagerScanOptionAllowDuplicatesKey: true
                ]
            )
            
        case .poweredOff:
            logger.warning("Bluetooth powered off")
            stopDetection()
            
        case .resetting:
            logger.warning("Bluetooth resetting")
            
        case .unauthorized:
            logger.error("Bluetooth unauthorized")
            
        case .unknown:
            logger.warning("Bluetooth state unknown")
            
        case .unsupported:
            logger.error("Bluetooth not supported")
            
        @unknown default:
            logger.error("Unknown Bluetooth state")
        }
    }
    
    public func centralManager(_ central: CBCentralManager, 
                             didDiscover peripheral: CBPeripheral,
                             advertisementData: [String : Any],
                             rssi RSSI: NSNumber) {
        
        let rssi = RSSI.intValue
        
        // Ignore very weak signals
        guard rssi > -90 else { return }
        
        // Check if this might be an Apple Watch
        let isWatch = isLikelyAppleWatch(peripheral)
        
        // Create or update device record
        let device = WatchDevice(
            identifier: peripheral.identifier,
            name: peripheral.name,
            rssi: rssi,
            lastSeen: Date(),
            isAppleWatch: isWatch
        )
        
        detectedDevices[peripheral.identifier] = device
        
        if isWatch {
            logger.debug("Apple Watch detected: \(peripheral.name ?? "Unknown"), RSSI: \(rssi)")
        }
        
        // Update proximity status
        updateProximityStatus()
    }
}