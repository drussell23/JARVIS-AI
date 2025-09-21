//
//  SemanticCacheSwift.swift
//  JARVIS Vision Semantic Cache
//
//  Purpose: Native macOS cache integration with Core ML and system caching
//

import Foundation
import CoreML
import Vision
import Accelerate
import CryptoKit
import Combine

/// Cache priority levels
enum CachePriority: Int {
    case low = 0
    case medium = 1
    case high = 2
    case critical = 3
}

/// Semantic cache entry
class SemanticCacheEntry {
    let key: String
    let value: Data
    var embedding: [Float]?
    var context: [String: Any]?
    let timestamp: Date
    var lastAccess: Date
    var accessCount: Int
    let ttlSeconds: TimeInterval
    let priority: CachePriority
    
    var sizeBytes: Int {
        var size = value.count
        if let embedding = embedding {
            size += embedding.count * MemoryLayout<Float>.size
        }
        return size
    }
    
    init(key: String, value: Data, embedding: [Float]? = nil,
         context: [String: Any]? = nil, ttlSeconds: TimeInterval,
         priority: CachePriority = .medium) {
        self.key = key
        self.value = value
        self.embedding = embedding
        self.context = context
        self.timestamp = Date()
        self.lastAccess = Date()
        self.accessCount = 0
        self.ttlSeconds = ttlSeconds
        self.priority = priority
    }
    
    func isExpired() -> Bool {
        return Date().timeIntervalSince(timestamp) > ttlSeconds
    }
    
    func access() {
        lastAccess = Date()
        accessCount += 1
    }
    
    func calculateValueScore() -> Double {
        let ageFactor = 1.0 / (1.0 + Date().timeIntervalSince(timestamp) / 3600)
        let accessFactor = min(Double(accessCount) / 10.0, 1.0)
        let recencyFactor = 1.0 / (1.0 + Date().timeIntervalSince(lastAccess) / 600)
        let priorityFactor = Double(priority.rawValue + 1) / 4.0
        
        return ageFactor * 0.2 + accessFactor * 0.3 + recencyFactor * 0.3 + priorityFactor * 0.2
    }
}

/// High-performance embedding generator using Core ML
class EmbeddingGenerator {
    private var model: MLModel?
    private let queue = DispatchQueue(label: "com.jarvis.embedding", qos: .userInitiated)
    
    init() {
        loadModel()
    }
    
    private func loadModel() {
        // In production, load actual Core ML model
        // For now, use Vision framework's text recognition features
    }
    
    func generateEmbedding(for text: String, completion: @escaping ([Float]?) -> Void) {
        queue.async {
            // Use Vision framework for feature extraction
            if #available(macOS 13.0, *) {
                self.generateVisionEmbedding(text, completion: completion)
            } else {
                // Fallback to hash-based embedding
                let embedding = self.generateHashEmbedding(text)
                completion(embedding)
            }
        }
    }
    
    @available(macOS 13.0, *)
    private func generateVisionEmbedding(_ text: String, completion: @escaping ([Float]?) -> Void) {
        // Create image from text for Vision processing
        guard let image = textToImage(text) else {
            completion(nil)
            return
        }
        
        let request = VNFeaturePrintObservationRequest { request, error in
            guard error == nil,
                  let observations = request.results as? [VNFeaturePrintObservation],
                  let featurePrint = observations.first else {
                completion(nil)
                return
            }
            
            // Convert feature print to embedding
            var embedding = [Float]()
            
            do {
                let data = try featurePrint.data(version: .version1)
                data.withUnsafeBytes { bytes in
                    let floats = bytes.bindMemory(to: Float.self)
                    embedding = Array(floats)
                }
                
                // Normalize
                var length: Float = 0
                vDSP_svesq(embedding, 1, &length, vDSP_Length(embedding.count))
                length = sqrtf(length)
                if length > 0 {
                    var normalizedEmbedding = [Float](repeating: 0, count: embedding.count)
                    var divisor = length
                    vDSP_vsdiv(embedding, 1, &divisor, &normalizedEmbedding, 1, vDSP_Length(embedding.count))
                    embedding = normalizedEmbedding
                }
                
                completion(embedding)
            } catch {
                completion(nil)
            }
        }
        
        let handler = VNImageRequestHandler(cgImage: image, options: [:])
        try? handler.perform([request])
    }
    
    private func textToImage(_ text: String) -> CGImage? {
        let attributes: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: 16),
            .foregroundColor: NSColor.black
        ]
        
        let size = (text as NSString).size(withAttributes: attributes)
        let rect = CGRect(origin: .zero, size: size)
        
        guard let context = CGContext(
            data: nil,
            width: Int(size.width),
            height: Int(size.height),
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }
        
        context.setFillColor(CGColor.white)
        context.fill(rect)
        
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(cgContext: context, flipped: false)
        (text as NSString).draw(in: rect, withAttributes: attributes)
        NSGraphicsContext.restoreGraphicsState()
        
        return context.makeImage()
    }
    
    private func generateHashEmbedding(_ text: String) -> [Float] {
        // Generate embedding from hash
        let hash = SHA256.hash(data: text.data(using: .utf8)!)
        var embedding = [Float]()
        
        for byte in hash {
            embedding.append(Float(byte) / 255.0)
        }
        
        // Expand to desired dimension (384)
        while embedding.count < 384 {
            embedding.append(embedding[embedding.count % 32])
        }
        
        return Array(embedding.prefix(384))
    }
}

/// Accelerate-optimized similarity computation
class SimilarityComputer {
    static func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return 0 }
        
        var dotProduct: Float = 0
        var normA: Float = 0
        var normB: Float = 0
        
        // Use Accelerate for SIMD operations
        vDSP_dotpr(a, 1, b, 1, &dotProduct, vDSP_Length(a.count))
        vDSP_svesq(a, 1, &normA, vDSP_Length(a.count))
        vDSP_svesq(b, 1, &normB, vDSP_Length(b.count))
        
        normA = sqrtf(normA)
        normB = sqrtf(normB)
        
        return (normA > 0 && normB > 0) ? dotProduct / (normA * normB) : 0
    }
    
    static func euclideanDistance(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return Float.infinity }
        
        var diff = [Float](repeating: 0, count: a.count)
        var distance: Float = 0
        
        // a - b
        vDSP_vsub(b, 1, a, 1, &diff, 1, vDSP_Length(a.count))
        // sum of squares
        vDSP_svesq(diff, 1, &distance, vDSP_Length(a.count))
        
        return sqrtf(distance)
    }
}

/// Native macOS semantic cache implementation
@objc class SemanticCacheSwift: NSObject {
    private let cacheQueue = DispatchQueue(label: "com.jarvis.cache", attributes: .concurrent)
    private let embeddingGenerator = EmbeddingGenerator()
    
    // Cache layers
    private var l1ExactCache: [String: SemanticCacheEntry] = [:]
    private var l2SemanticCache: [String: SemanticCacheEntry] = [:]
    private var l3ContextualCache: [String: SemanticCacheEntry] = [:]
    
    // LSH buckets for semantic search
    private var lshBuckets: [[String]] = Array(repeating: [], count: 256)
    
    // Cache statistics
    private var stats = CacheStatistics()
    
    // Configuration
    private let maxCacheSizeMB: Int = 250
    private let similarityThreshold: Float = 0.85
    
    // Memory pressure monitoring
    private var memoryPressureSource: DispatchSourceMemoryPressure?
    
    override init() {
        super.init()
        setupMemoryPressureMonitoring()
    }
    
    private func setupMemoryPressureMonitoring() {
        memoryPressureSource = DispatchSource.makeMemoryPressureSource(
            eventMask: [.warning, .critical],
            queue: cacheQueue
        )
        
        memoryPressureSource?.setEventHandler { [weak self] in
            self?.handleMemoryPressure()
        }
        
        memoryPressureSource?.resume()
    }
    
    private func handleMemoryPressure() {
        // Evict low-value entries
        evictLowValueEntries(targetReduction: 0.3)
    }
    
    // MARK: - L1 Exact Cache
    
    @objc func getL1(key: String) -> Data? {
        var result: Data?
        
        cacheQueue.sync {
            if let entry = l1ExactCache[key], !entry.isExpired() {
                entry.access()
                result = entry.value
                stats.l1Hits += 1
            } else {
                stats.l1Misses += 1
            }
        }
        
        return result
    }
    
    @objc func putL1(key: String, value: Data, ttlSeconds: Double = 30) {
        cacheQueue.async(flags: .barrier) {
            let entry = SemanticCacheEntry(
                key: key,
                value: value,
                ttlSeconds: ttlSeconds,
                priority: .high
            )
            
            self.l1ExactCache[key] = entry
            self.evictIfNeeded()
        }
    }
    
    // MARK: - L2 Semantic Cache
    
    @objc func getL2Semantic(queryText: String, completion: @escaping (Data?, Float) -> Void) {
        embeddingGenerator.generateEmbedding(for: queryText) { [weak self] queryEmbedding in
            guard let self = self, let queryEmbedding = queryEmbedding else {
                completion(nil, 0)
                return
            }
            
            self.cacheQueue.sync {
                let candidates = self.getCandidatesFromLSH(embedding: queryEmbedding)
                
                var bestMatch: SemanticCacheEntry?
                var bestSimilarity: Float = 0
                
                for candidateKey in candidates {
                    if let entry = self.l2SemanticCache[candidateKey],
                       !entry.isExpired(),
                       let entryEmbedding = entry.embedding {
                        
                        let similarity = SimilarityComputer.cosineSimilarity(
                            queryEmbedding,
                            entryEmbedding
                        )
                        
                        if similarity > bestSimilarity && similarity >= self.similarityThreshold {
                            bestSimilarity = similarity
                            bestMatch = entry
                        }
                    }
                }
                
                if let match = bestMatch {
                    match.access()
                    self.stats.l2Hits += 1
                    completion(match.value, bestSimilarity)
                } else {
                    self.stats.l2Misses += 1
                    completion(nil, 0)
                }
            }
        }
    }
    
    @objc func putL2Semantic(key: String, value: Data, text: String, ttlSeconds: Double = 300) {
        embeddingGenerator.generateEmbedding(for: text) { [weak self] embedding in
            guard let self = self, let embedding = embedding else { return }
            
            self.cacheQueue.async(flags: .barrier) {
                let entry = SemanticCacheEntry(
                    key: key,
                    value: value,
                    embedding: embedding,
                    ttlSeconds: ttlSeconds,
                    priority: .medium
                )
                
                self.l2SemanticCache[key] = entry
                self.addToLSH(key: key, embedding: embedding)
                self.evictIfNeeded()
            }
        }
    }
    
    // MARK: - LSH Operations
    
    private func getCandidatesFromLSH(embedding: [Float]) -> [String] {
        let hashes = computeLSHHashes(embedding: embedding)
        var candidates = Set<String>()
        
        for hash in hashes {
            let bucketIndex = Int(hash) % lshBuckets.count
            candidates.formUnion(lshBuckets[bucketIndex])
        }
        
        return Array(candidates)
    }
    
    private func addToLSH(key: String, embedding: [Float]) {
        let hashes = computeLSHHashes(embedding: embedding)
        
        for hash in hashes {
            let bucketIndex = Int(hash) % lshBuckets.count
            if !lshBuckets[bucketIndex].contains(key) {
                lshBuckets[bucketIndex].append(key)
            }
        }
    }
    
    private func computeLSHHashes(embedding: [Float], numHashes: Int = 4) -> [UInt32] {
        var hashes = [UInt32]()
        
        // Simple random projection LSH
        for i in 0..<numHashes {
            var hash: UInt32 = 0
            let offset = i * 8
            
            for j in 0..<min(8, embedding.count - offset) {
                if embedding[offset + j] > 0 {
                    hash |= (1 << j)
                }
            }
            
            hashes.append(hash)
        }
        
        return hashes
    }
    
    // MARK: - Memory Management
    
    private func evictIfNeeded() {
        let currentSizeMB = calculateCurrentCacheSizeMB()
        
        if currentSizeMB > Double(maxCacheSizeMB) {
            evictLowValueEntries(targetReduction: 0.2)
        }
    }
    
    private func evictLowValueEntries(targetReduction: Double) {
        var allEntries: [(String, SemanticCacheEntry, Double)] = []
        
        // Collect all entries with value scores
        for (key, entry) in l1ExactCache {
            allEntries.append((key, entry, entry.calculateValueScore()))
        }
        
        for (key, entry) in l2SemanticCache {
            allEntries.append((key, entry, entry.calculateValueScore()))
        }
        
        for (key, entry) in l3ContextualCache {
            allEntries.append((key, entry, entry.calculateValueScore()))
        }
        
        // Sort by value score (ascending)
        allEntries.sort { $0.2 < $1.2 }
        
        // Calculate target eviction size
        let currentSize = allEntries.reduce(0) { $0 + $1.1.sizeBytes }
        let targetEviction = Int(Double(currentSize) * targetReduction)
        
        var evictedSize = 0
        
        for (key, entry, _) in allEntries {
            if evictedSize >= targetEviction {
                break
            }
            
            // Remove from appropriate cache
            l1ExactCache.removeValue(forKey: key)
            l2SemanticCache.removeValue(forKey: key)
            l3ContextualCache.removeValue(forKey: key)
            
            evictedSize += entry.sizeBytes
            stats.evictions += 1
        }
    }
    
    private func calculateCurrentCacheSizeMB() -> Double {
        var totalBytes = 0
        
        totalBytes += l1ExactCache.values.reduce(0) { $0 + $1.sizeBytes }
        totalBytes += l2SemanticCache.values.reduce(0) { $0 + $1.sizeBytes }
        totalBytes += l3ContextualCache.values.reduce(0) { $0 + $1.sizeBytes }
        
        return Double(totalBytes) / (1024 * 1024)
    }
    
    // MARK: - Statistics
    
    @objc func getStatistics() -> [String: Any] {
        return cacheQueue.sync {
            [
                "l1_hits": stats.l1Hits,
                "l1_misses": stats.l1Misses,
                "l1_hit_rate": stats.l1HitRate,
                "l2_hits": stats.l2Hits,
                "l2_misses": stats.l2Misses,
                "l2_hit_rate": stats.l2HitRate,
                "l3_hits": stats.l3Hits,
                "l3_misses": stats.l3Misses,
                "l3_hit_rate": stats.l3HitRate,
                "total_evictions": stats.evictions,
                "cache_size_mb": calculateCurrentCacheSizeMB(),
                "l1_entries": l1ExactCache.count,
                "l2_entries": l2SemanticCache.count,
                "l3_entries": l3ContextualCache.count
            ]
        }
    }
    
    @objc func clearAllCaches() {
        cacheQueue.async(flags: .barrier) {
            self.l1ExactCache.removeAll()
            self.l2SemanticCache.removeAll()
            self.l3ContextualCache.removeAll()
            self.lshBuckets = Array(repeating: [], count: 256)
            self.stats = CacheStatistics()
        }
    }
}

/// Cache statistics
private struct CacheStatistics {
    var l1Hits: Int = 0
    var l1Misses: Int = 0
    var l2Hits: Int = 0
    var l2Misses: Int = 0
    var l3Hits: Int = 0
    var l3Misses: Int = 0
    var evictions: Int = 0
    
    var l1HitRate: Double {
        let total = l1Hits + l1Misses
        return total > 0 ? Double(l1Hits) / Double(total) : 0
    }
    
    var l2HitRate: Double {
        let total = l2Hits + l2Misses
        return total > 0 ? Double(l2Hits) / Double(total) : 0
    }
    
    var l3HitRate: Double {
        let total = l3Hits + l3Misses
        return total > 0 ? Double(l3Hits) / Double(total) : 0
    }
}

/// Predictive cache warming
@objc class PredictiveCacheWarmer: NSObject {
    private let cache: SemanticCacheSwift
    private let predictionQueue = DispatchQueue(label: "com.jarvis.prediction", qos: .background)
    private var accessHistory: [(String, Date)] = []
    private var patternSequences: [String: [String]] = [:]
    private let maxHistory = 1000
    
    init(cache: SemanticCacheSwift) {
        self.cache = cache
        super.init()
    }
    
    @objc func recordAccess(_ query: String) {
        predictionQueue.async { [weak self] in
            guard let self = self else { return }
            
            self.accessHistory.append((query, Date()))
            
            // Maintain max history
            if self.accessHistory.count > self.maxHistory {
                self.accessHistory.removeFirst(self.accessHistory.count - self.maxHistory)
            }
            
            // Update patterns
            if self.accessHistory.count >= 2 {
                let prevQuery = self.accessHistory[self.accessHistory.count - 2].0
                
                if self.patternSequences[prevQuery] == nil {
                    self.patternSequences[prevQuery] = []
                }
                self.patternSequences[prevQuery]!.append(query)
            }
            
            // Trigger prediction
            self.predictAndWarm(currentQuery: query)
        }
    }
    
    private func predictAndWarm(currentQuery: String) {
        guard let nextQueries = patternSequences[currentQuery] else { return }
        
        // Count frequencies
        var frequencies: [String: Int] = [:]
        for query in nextQueries {
            frequencies[query, default: 0] += 1
        }
        
        // Sort by frequency
        let predictions = frequencies.sorted { $0.value > $1.value }
        
        // Pre-warm top predictions
        for (query, count) in predictions.prefix(3) {
            let confidence = Double(count) / Double(nextQueries.count)
            
            if confidence > 0.5 {
                // This would trigger pre-computation
                print("Pre-warming cache for predicted query: \(query) (confidence: \(confidence))")
            }
        }
    }
}