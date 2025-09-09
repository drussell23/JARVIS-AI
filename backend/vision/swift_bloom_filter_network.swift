#!/usr/bin/env swift

//
//  Swift Bloom Filter Network for JARVIS Vision System
//  Optimized for native macOS UI element tracking and deduplication
//  
//  Architecture:
//  - Native Swift performance with memory safety
//  - Integration with macOS Accessibility APIs
//  - UI element fingerprinting and tracking
//  - Memory-efficient hierarchical filtering
//

import Foundation
import Cocoa
import simd
import Accelerate

// MARK: - Bloom Filter Level Enumeration

@objc public enum BloomFilterLevel: Int, CaseIterable {
    case global = 0
    case regional = 1
    case element = 2
    
    public var name: String {
        switch self {
        case .global: return "GLOBAL"
        case .regional: return "REGIONAL"
        case .element: return "ELEMENT"
        }
    }
}

// MARK: - Performance Metrics

@objc public class BloomFilterMetrics: NSObject {
    @objc public private(set) var totalInsertions: UInt64 = 0
    @objc public private(set) var totalQueries: UInt64 = 0
    @objc public private(set) var probableHits: UInt64 = 0
    @objc public private(set) var confirmedHits: UInt64 = 0
    @objc public private(set) var falsePositives: UInt64 = 0
    @objc public private(set) var resetCount: UInt32 = 0
    @objc public private(set) var lastReset: Date = Date()
    
    private let queue = DispatchQueue(label: "bloom.metrics", qos: .utility)
    
    // MARK: - Calculated Properties
    
    @objc public var falsePositiveRate: Double {
        guard probableHits > 0 else { return 0.0 }
        return Double(falsePositives) / Double(probableHits)
    }
    
    @objc public var hitRate: Double {
        guard totalQueries > 0 else { return 0.0 }
        return Double(probableHits) / Double(totalQueries)
    }
    
    // MARK: - Thread-safe Updates
    
    func recordInsertion() {
        queue.async {
            self.totalInsertions += 1
        }
    }
    
    func recordQuery() {
        queue.async {
            self.totalQueries += 1
        }
    }
    
    func recordProbableHit() {
        queue.async {
            self.probableHits += 1
        }
    }
    
    func recordConfirmedHit() {
        queue.async {
            self.confirmedHits += 1
        }
    }
    
    func recordFalsePositive() {
        queue.async {
            self.falsePositives += 1
        }
    }
    
    func recordReset() {
        queue.async {
            self.resetCount += 1
            self.lastReset = Date()
        }
    }
    
    @objc public func reset() {
        queue.async {
            let oldResetCount = self.resetCount
            self.totalInsertions = 0
            self.totalQueries = 0
            self.probableHits = 0
            self.confirmedHits = 0
            self.falsePositives = 0
            self.resetCount = oldResetCount + 1
            self.lastReset = Date()
        }
    }
    
    @objc public func getSnapshot() -> [String: Any] {
        return queue.sync {
            return [
                "totalInsertions": totalInsertions,
                "totalQueries": totalQueries,
                "probableHits": probableHits,
                "confirmedHits": confirmedHits,
                "falsePositives": falsePositives,
                "resetCount": resetCount,
                "lastReset": lastReset,
                "hitRate": hitRate,
                "falsePositiveRate": falsePositiveRate
            ]
        }
    }
}

// MARK: - Swift Adaptive Bloom Filter

@objc public class SwiftAdaptiveBloomFilter: NSObject {
    private let sizeMB: Double
    private let sizeBits: Int
    private let expectedElements: Int
    private let maxFalsePositiveRate: Double
    private let level: BloomFilterLevel
    
    // Bit array using UInt64 for performance
    private var bitArray: [UInt64]
    private let bitArrayLength: Int
    private let queue = DispatchQueue(label: "bloom.filter", qos: .userInteractive)
    
    // Hash parameters
    private let numHashes: Int
    private let hashSeeds: [UInt64]
    
    // Metrics and state
    @objc public let metrics: BloomFilterMetrics
    private let saturationThreshold: Double = 0.8
    
    // MARK: - Initialization
    
    @objc public init(sizeMB: Double, 
                      expectedElements: Int, 
                      maxFalsePositiveRate: Double = 0.01, 
                      level: BloomFilterLevel) {
        self.sizeMB = sizeMB
        self.sizeBits = Int(sizeMB * 1024 * 1024 * 8) // Convert MB to bits
        self.expectedElements = expectedElements
        self.maxFalsePositiveRate = maxFalsePositiveRate
        self.level = level
        
        // Calculate optimal number of hash functions based on level
        switch level {
        case .global:
            self.numHashes = 10  // 10 hash functions as per spec
        case .regional:
            self.numHashes = 7   // 7 hash functions as per spec
        case .element:
            self.numHashes = 5   // 5 hash functions as per spec
        }
        
        // Initialize bit array aligned to UInt64 boundaries
        self.bitArrayLength = (self.sizeBits + 63) / 64
        self.bitArray = Array(repeating: 0, count: self.bitArrayLength)
        
        // Generate diverse hash seeds using prime numbers
        var seeds: [UInt64] = []
        var seed: UInt64 = 2654435761 // Large prime
        for i in 0..<self.numHashes {
            seeds.append(seed &* UInt64(i + 1))
            seed = seed &* 1664525 &+ 1013904223 // Linear congruential generator
        }
        self.hashSeeds = seeds
        
        self.metrics = BloomFilterMetrics()
        
        super.init()
        
        NSLog("SwiftBloomFilter initialized: %@, %.1fMB, %d hash functions, %d expected elements", 
              level.name, sizeMB, numHashes, expectedElements)
    }
    
    // MARK: - Hash Functions
    
    private func fastHash(_ data: Data, seed: UInt64) -> Int {
        // Use SipHash for cryptographically strong hashing
        var hasher = SipHasher(key0: seed, key1: seed &+ 1)
        data.withUnsafeBytes { bytes in
            hasher.update(bytes)
        }
        return Int(hasher.finalize() % UInt64(sizeBits))
    }
    
    private func multiHash(_ data: Data) -> [Int] {
        return hashSeeds.map { seed in
            fastHash(data, seed: seed)
        }
    }
    
    // MARK: - Core Operations
    
    @objc public func add(_ data: Data) -> Bool {
        return queue.sync {
            let positions = multiHash(data)
            
            // Set bits using atomic operations
            for position in positions {
                let wordIndex = position / 64
                let bitIndex = position % 64
                let mask: UInt64 = 1 << bitIndex
                
                guard wordIndex < bitArrayLength else { continue }
                bitArray[wordIndex] |= mask
            }
            
            metrics.recordInsertion()
            return true
        }
    }
    
    @objc public func contains(_ data: Data) -> Bool {
        metrics.recordQuery()
        
        return queue.sync {
            let positions = multiHash(data)
            
            // Check all positions
            for position in positions {
                let wordIndex = position / 64
                let bitIndex = position % 64
                let mask: UInt64 = 1 << bitIndex
                
                guard wordIndex < bitArrayLength else { return false }
                
                if (bitArray[wordIndex] & mask) == 0 {
                    return false // Definitely not in set
                }
            }
            
            // All positions are set - probably in set
            metrics.recordProbableHit()
            return true
        }
    }
    
    @objc public func reset() {
        queue.sync {
            for i in 0..<bitArrayLength {
                bitArray[i] = 0
            }
            metrics.recordReset()
        }
        
        NSLog("Reset %@ bloom filter", level.name)
    }
    
    @objc public func estimateSaturation() -> Double {
        return queue.sync {
            // Sample bits for performance (checking every 64th word)
            let sampleSize = min(1000, bitArrayLength)
            let step = max(1, bitArrayLength / sampleSize)
            
            var setBits: UInt64 = 0
            var totalBits: UInt64 = 0
            
            for i in stride(from: 0, to: bitArrayLength, by: step) {
                let word = bitArray[i]
                setBits += UInt64(word.nonzeroBitCount)
                totalBits += 64
            }
            
            guard totalBits > 0 else { return 0.0 }
            return Double(setBits) / Double(totalBits)
        }
    }
    
    @objc public func getMemoryUsage() -> [String: Any] {
        let actualMB = Double(bitArrayLength * MemoryLayout<UInt64>.size) / (1024 * 1024)
        let saturation = estimateSaturation()
        
        return [
            "allocated_mb": sizeMB,
            "actual_mb": actualMB,
            "saturation_level": saturation,
            "level": level.name,
            "metrics": metrics.getSnapshot()
        ]
    }
}

// MARK: - Swift Bloom Filter Network

@objc public class SwiftBloomFilterNetwork: NSObject {
    private let globalFilter: SwiftAdaptiveBloomFilter
    private let regionalFilters: [SwiftAdaptiveBloomFilter]  // 4 quadrant filters
    private let elementFilter: SwiftAdaptiveBloomFilter
    
    private let networkMetrics: NSMutableDictionary
    private let enableHierarchicalChecking: Bool
    private let queue = DispatchQueue(label: "bloom.network", qos: .userInteractive)
    
    // MARK: - Initialization
    
    @objc public init(globalSizeMB: Double = 4.0,     // 4MB for global (spec: 4MB)
                      regionalSizeMB: Double = 1.0,    // 1MB × 4 for regional (spec: 1MB×4)
                      elementSizeMB: Double = 2.0,     // 2MB for element (spec: 2MB)
                      enableHierarchicalChecking: Bool = true) {
        
        self.globalFilter = SwiftAdaptiveBloomFilter(
            sizeMB: globalSizeMB,
            expectedElements: 100000,  // Increased for global scope
            level: .global
        )
        
        // Create 4 regional filters (one per quadrant) - 1MB each
        self.regionalFilters = (0..<4).map { _ in
            SwiftAdaptiveBloomFilter(
                sizeMB: regionalSizeMB,
                expectedElements: 5000,  // Elements per quadrant
                level: .regional
            )
        }
        
        self.elementFilter = SwiftAdaptiveBloomFilter(
            sizeMB: elementSizeMB,
            expectedElements: 20000,  // Increased for element scope
            level: .element
        )
        
        self.enableHierarchicalChecking = enableHierarchicalChecking
        
        // Initialize network metrics
        self.networkMetrics = NSMutableDictionary()
        self.networkMetrics["totalChecks"] = NSNumber(value: 0)
        self.networkMetrics["globalHits"] = NSNumber(value: 0)
        self.networkMetrics["regionalHits"] = NSNumber(value: 0)
        self.networkMetrics["elementHits"] = NSNumber(value: 0)
        self.networkMetrics["totalMisses"] = NSNumber(value: 0)
        self.networkMetrics["hierarchicalShortcuts"] = NSNumber(value: 0)
        
        super.init()
        
        NSLog("SwiftBloomFilterNetwork initialized: Total %.1fMB allocated", 
              globalSizeMB + regionalSizeMB + elementSizeMB)
    }
    
    // MARK: - Key Generation
    
    private func generateElementKey(elementData: [String: Any], context: [String: Any]? = nil) -> Data {
        // Create consistent key from element data
        var keyString = ""
        
        // Sort keys for consistent hashing
        let sortedKeys = elementData.keys.sorted()
        for key in sortedKeys {
            if let value = elementData[key] {
                keyString += "\(key):\(value)|"
            }
        }
        
        // Add context if provided
        if let context = context {
            let sortedContextKeys = context.keys.sorted()
            for key in sortedContextKeys {
                if let value = context[key] {
                    keyString += "ctx_\(key):\(value)|"
                }
            }
        }
        
        return keyString.data(using: .utf8) ?? Data()
    }
    
    // MARK: - Network Operations
    
    @objc public func checkAndAdd(elementData: [String: Any], 
                                  context: [String: Any]? = nil, 
                                  checkLevel: BloomFilterLevel = .global) -> (Bool, BloomFilterLevel) {
        
        queue.sync {
            incrementNetworkMetric("totalChecks")
            
            let elementKey = generateElementKey(elementData: elementData, context: context)
            
            if enableHierarchicalChecking {
                // Hierarchical checking with short-circuit
                
                // Check Global first (most comprehensive)
                if globalFilter.contains(elementKey) {
                    incrementNetworkMetric("globalHits")
                    incrementNetworkMetric("hierarchicalShortcuts")
                    return (true, .global)
                }
                
                // Check Regional filters if appropriate
                if checkLevel == .regional || checkLevel == .element {
                    // Check all quadrant filters
                    for regionalFilter in regionalFilters {
                        if regionalFilter.contains(elementKey) {
                            incrementNetworkMetric("regionalHits")
                            // Promote to global for future shortcuts
                            globalFilter.add(elementKey)
                            return (true, .regional)
                        }
                    }
                }
                
                // Check Element if appropriate
                if checkLevel == .element {
                    if elementFilter.contains(elementKey) {
                        incrementNetworkMetric("elementHits")
                        // Promote to higher levels
                        // Add to all regional filters for simplicity
                        for regionalFilter in regionalFilters {
                            regionalFilter.add(elementKey)
                        }
                        globalFilter.add(elementKey)
                        return (true, .element)
                    }
                }
            } else {
                // Direct level checking
                let targetFilter = getFilterForLevel(checkLevel)
                if targetFilter.contains(elementKey) {
                    incrementNetworkMetric("\(checkLevel.name.lowercased())Hits")
                    return (true, checkLevel)
                }
            }
            
            // Not found - add to appropriate levels
            incrementNetworkMetric("totalMisses")
            addToAppropriateLevel(elementKey, level: checkLevel)
            
            return (false, checkLevel)
        }
    }
    
    private func getFilterForLevel(_ level: BloomFilterLevel, quadrant: Int? = nil) -> [SwiftAdaptiveBloomFilter] {
        switch level {
        case .global: return [globalFilter]
        case .regional: 
            if let q = quadrant, q >= 0 && q < 4 {
                return [regionalFilters[q]]
            }
            return regionalFilters
        case .element: return [elementFilter]
        }
    }
    
    private func addToAppropriateLevel(_ elementKey: Data, level: BloomFilterLevel) {
        switch level {
        case .global:
            globalFilter.add(elementKey)
        case .regional:
            // Add to all regional filters for simplicity
            for regionalFilter in regionalFilters {
                regionalFilter.add(elementKey)
            }
        case .element:
            elementFilter.add(elementKey)
        }
    }
    
    private func incrementNetworkMetric(_ key: String) {
        if let current = networkMetrics[key] as? NSNumber {
            networkMetrics[key] = NSNumber(value: current.uint64Value + 1)
        } else {
            networkMetrics[key] = NSNumber(value: 1)
        }
    }
    
    // MARK: - Statistics and Management
    
    @objc public func getNetworkStats() -> [String: Any] {
        return queue.sync {
            var stats: [String: Any] = [:]
            
            // Network metrics
            for (key, value) in networkMetrics {
                if let key = key as? String, let value = value as? NSNumber {
                    stats[key] = value.uint64Value
                }
            }
            
            // Individual filter stats
            stats["global"] = globalFilter.getMemoryUsage()
            // Regional filter stats
            for (idx, regionalFilter) in regionalFilters.enumerated() {
                stats["regional_q\(idx)"] = regionalFilter.getMemoryUsage()
            }
            stats["element"] = elementFilter.getMemoryUsage()
            
            // Calculated metrics
            let totalChecks = stats["totalChecks"] as? UInt64 ?? 0
            let totalHits = (stats["globalHits"] as? UInt64 ?? 0) +
                           (stats["regionalHits"] as? UInt64 ?? 0) +
                           (stats["elementHits"] as? UInt64 ?? 0)
            
            if totalChecks > 0 {
                stats["overallHitRate"] = Double(totalHits) / Double(totalChecks)
                let shortcuts = stats["hierarchicalShortcuts"] as? UInt64 ?? 0
                stats["hierarchicalEfficiency"] = Double(shortcuts) / Double(totalChecks)
            }
            
            let regionalTotalMB = regionalFilters.reduce(0.0) { $0 + $1.sizeMB }
            stats["totalMemoryMB"] = globalFilter.sizeMB + regionalTotalMB + elementFilter.sizeMB
            
            return stats
        }
    }
    
    @objc public func resetNetwork(level: BloomFilterLevel? = nil) {
        queue.sync {
            if let level = level {
                let filters = getFilterForLevel(level)
                for filter in filters {
                    filter.reset()
                }
                NSLog("Reset %@ bloom filter(s)", level.name)
            } else {
                // Reset all levels
                globalFilter.reset()
                for regionalFilter in regionalFilters {
                    regionalFilter.reset()
                }
                elementFilter.reset()
                
                // Reset network metrics
                for key in networkMetrics.allKeys {
                    networkMetrics[key] = NSNumber(value: 0)
                }
                
                NSLog("Reset entire Swift Bloom Filter Network")
            }
        }
    }
    
    @objc public func optimizeNetwork() {
        queue.sync {
            // Check saturation levels and reset if needed
            let globalSat = globalFilter.estimateSaturation()
            let regionalSats = regionalFilters.map { $0.estimateSaturation() }
            let elementSat = elementFilter.estimateSaturation()
            
            if globalSat > 0.85 {
                globalFilter.reset()
                NSLog("Optimized: Reset global filter (saturation: %.2f%%)", globalSat * 100)
            }
            for (idx, regionalFilter) in regionalFilters.enumerated() {
                if regionalSats[idx] > 0.85 {
                    regionalFilter.reset()
                    NSLog("Optimized: Reset regional filter %d (saturation: %.2f%%)", idx, regionalSats[idx] * 100)
                }
            }
            if elementSat > 0.85 {
                elementFilter.reset()
                NSLog("Optimized: Reset element filter (saturation: %.2f%%)", elementSat * 100)
            }
        }
    }
}

// MARK: - macOS UI Element Integration

@objc public class SwiftUIElementTracker: NSObject {
    private let bloomNetwork: SwiftBloomFilterNetwork
    private let accessibilityQueue = DispatchQueue(label: "ui.accessibility", qos: .userInteractive)
    
    @objc public init(bloomNetwork: SwiftBloomFilterNetwork) {
        self.bloomNetwork = bloomNetwork
        super.init()
    }
    
    @objc public func isUIElementDuplicate(_ element: AXUIElement, 
                                          windowContext: [String: Any]? = nil) -> Bool {
        // Extract element properties
        guard let elementData = extractElementData(element) else { return false }
        
        let (isDuplicate, _) = bloomNetwork.checkAndAdd(
            elementData: elementData,
            context: windowContext,
            checkLevel: .element
        )
        
        return isDuplicate
    }
    
    @objc public func isWindowRegionDuplicate(_ region: CGRect, 
                                             windowInfo: [String: Any]? = nil) -> Bool {
        let regionData: [String: Any] = [
            "x": region.origin.x,
            "y": region.origin.y,
            "width": region.size.width,
            "height": region.size.height,
            "area": region.size.width * region.size.height
        ]
        
        let (isDuplicate, _) = bloomNetwork.checkAndAdd(
            elementData: regionData,
            context: windowInfo,
            checkLevel: .regional
        )
        
        return isDuplicate
    }
    
    private func extractElementData(_ element: AXUIElement) -> [String: Any]? {
        var elementData: [String: Any] = [:]
        
        // Extract basic properties
        if let role = getAXValue(element, attribute: kAXRoleAttribute) as? String {
            elementData["role"] = role
        }
        
        if let title = getAXValue(element, attribute: kAXTitleAttribute) as? String {
            elementData["title"] = title
        }
        
        if let value = getAXValue(element, attribute: kAXValueAttribute) {
            elementData["value"] = String(describing: value)
        }
        
        if let position = getAXValue(element, attribute: kAXPositionAttribute) as? CGPoint {
            elementData["x"] = position.x
            elementData["y"] = position.y
        }
        
        if let size = getAXValue(element, attribute: kAXSizeAttribute) as? CGSize {
            elementData["width"] = size.width
            elementData["height"] = size.height
        }
        
        return elementData.isEmpty ? nil : elementData
    }
    
    private func getAXValue(_ element: AXUIElement, attribute: CFString) -> Any? {
        var value: CFTypeRef?
        let result = AXUIElementCopyAttributeValue(element, attribute, &value)
        return result == .success ? value : nil
    }
}

// MARK: - SipHasher Implementation

private struct SipHasher {
    private var v0: UInt64
    private var v1: UInt64
    private var v2: UInt64
    private var v3: UInt64
    private var messageLength: UInt64 = 0
    private var buffer: [UInt8] = []
    
    init(key0: UInt64, key1: UInt64) {
        v0 = 0x736f6d6570736575 ^ key0
        v1 = 0x646f72616e646f6d ^ key1
        v2 = 0x6c7967656e657261 ^ key0
        v3 = 0x7465646279746573 ^ key1
    }
    
    mutating func update<T>(_ bytes: T) where T: Collection, T.Element == UInt8 {
        buffer.append(contentsOf: bytes)
        messageLength += UInt64(bytes.count)
        
        // Process complete 8-byte chunks
        while buffer.count >= 8 {
            let chunk = buffer.prefix(8)
            buffer.removeFirst(8)
            
            var m: UInt64 = 0
            for (i, byte) in chunk.enumerated() {
                m |= UInt64(byte) << (i * 8)
            }
            
            compress(m)
        }
    }
    
    mutating func finalize() -> UInt64 {
        // Process remaining bytes
        var m: UInt64 = messageLength << 56
        for (i, byte) in buffer.enumerated() {
            m |= UInt64(byte) << (i * 8)
        }
        
        compress(m)
        
        // Finalization
        v2 ^= 0xff
        for _ in 0..<4 {
            sipRound()
        }
        
        return v0 ^ v1 ^ v2 ^ v3
    }
    
    private mutating func compress(_ m: UInt64) {
        v3 ^= m
        for _ in 0..<2 {
            sipRound()
        }
        v0 ^= m
    }
    
    private mutating func sipRound() {
        v0 = v0 &+ v1
        v1 = rotateLeft(v1, by: 13)
        v1 ^= v0
        v0 = rotateLeft(v0, by: 32)
        
        v2 = v2 &+ v3
        v3 = rotateLeft(v3, by: 16)
        v3 ^= v2
        
        v0 = v0 &+ v3
        v3 = rotateLeft(v3, by: 21)
        v3 ^= v0
        
        v2 = v2 &+ v1
        v1 = rotateLeft(v1, by: 17)
        v1 ^= v2
        v2 = rotateLeft(v2, by: 32)
    }
    
    private func rotateLeft(_ value: UInt64, by amount: Int) -> UInt64 {
        return (value << amount) | (value >> (64 - amount))
    }
}

// MARK: - Global Singleton

@objc public class SwiftBloomFilterNetworkSingleton: NSObject {
    @objc public static let shared = SwiftBloomFilterNetworkSingleton()
    @objc public let network: SwiftBloomFilterNetwork
    
    private override init() {
        self.network = SwiftBloomFilterNetwork(
            globalSizeMB: 4.0,
            regionalSizeMB: 1.0,
            elementSizeMB: 2.0
        )
        super.init()
    }
    
    @objc public func resetNetwork() {
        network.resetNetwork()
    }
}