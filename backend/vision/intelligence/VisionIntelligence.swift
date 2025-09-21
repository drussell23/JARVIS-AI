import Foundation
import Vision
import CoreML
import CoreImage
import AppKit
import Accelerate

/// Dynamic Visual State Intelligence for macOS
@objc public class VisionIntelligence: NSObject {
    
    // MARK: - Types
    
    @objc public enum ApplicationStateType: Int {
        case unknown = 0
        case idle
        case active
        case loading
        case error
        case modal
        case transition
        case custom
    }
    
    @objc public class VisualSignature: NSObject {
        @objc public let signatureId: String
        @objc public var confidence: Float = 0.0
        @objc public var occurrenceCount: Int = 0
        @objc public var lastSeen: Date?
        
        var colorPatterns: [String: Any] = [:]
        var structuralPatterns: [String: Any] = [:]
        var textPatterns: [String] = []
        var visualFeatures: [[Float]] = []
        
        @objc public init(signatureId: String) {
            self.signatureId = signatureId
            super.init()
        }
        
        @objc public func updateConfidence(matchScore: Float) {
            // Exponential moving average
            confidence = (confidence * 0.9) + (matchScore * 0.1)
            occurrenceCount += 1
            lastSeen = Date()
        }
    }
    
    @objc public class ApplicationState: NSObject {
        @objc public let stateId: String
        @objc public var stateType: ApplicationStateType
        @objc public var observationCount: Int = 0
        @objc public var lastObserved: Date?
        
        var signatures: [VisualSignature] = []
        var transitionsTo: [String: Float] = [:]
        var transitionsFrom: [String: Float] = [:]
        var durationStats: [String: Float] = [:]
        
        @objc public init(stateId: String, stateType: ApplicationStateType) {
            self.stateId = stateId
            self.stateType = stateType
            super.init()
        }
        
        @objc public func addTransition(toState: String?, fromState: String?) {
            if let toState = toState {
                transitionsTo[toState] = (transitionsTo[toState] ?? 0) + 1
            }
            if let fromState = fromState {
                transitionsFrom[fromState] = (transitionsFrom[fromState] ?? 0) + 1
            }
            normalizeTransitions()
        }
        
        private func normalizeTransitions() {
            // Convert counts to probabilities
            let normalizeDict = { (dict: inout [String: Float]) in
                let total = dict.values.reduce(0, +)
                if total > 0 {
                    for key in dict.keys {
                        dict[key]! /= total
                    }
                }
            }
            normalizeDict(&transitionsTo)
            normalizeDict(&transitionsFrom)
        }
    }
    
    // MARK: - Properties
    
    private let visionQueue = DispatchQueue(label: "com.jarvis.vision.intelligence", qos: .userInitiated)
    private var textRecognitionRequest: VNRecognizeTextRequest?
    private var contourRequest: VNDetectContoursRequest?
    private var saliencyRequest: VNGenerateAttentionBasedSaliencyImageRequest?
    
    private var applicationStates: [String: [String: ApplicationState]] = [:] // [appId: [stateId: state]]
    private var stateDetectors: [StateDetectorProtocol] = []
    private let learningEnabled = true
    
    // MARK: - Initialization
    
    @objc public override init() {
        super.init()
        setupVisionRequests()
        initializeDetectors()
    }
    
    private func setupVisionRequests() {
        // Text recognition for UI understanding
        textRecognitionRequest = VNRecognizeTextRequest { [weak self] request, error in
            self?.processTextObservations(request.results as? [VNRecognizedTextObservation])
        }
        textRecognitionRequest?.recognitionLevel = .accurate
        textRecognitionRequest?.usesLanguageCorrection = true
        
        // Contour detection for UI structure
        contourRequest = VNDetectContoursRequest { [weak self] request, error in
            self?.processContourObservations(request.results as? [VNContoursObservation])
        }
        contourRequest?.contrastAdjustment = 1.0
        contourRequest?.detectsDarkOnLight = true
        
        // Saliency for focus detection
        saliencyRequest = VNGenerateAttentionBasedSaliencyImageRequest { [weak self] request, error in
            self?.processSaliencyObservations(request.results as? [VNSaliencyImageObservation])
        }
    }
    
    private func initializeDetectors() {
        stateDetectors = [
            PatternBasedStateDetector(),
            ColorBasedStateDetector(),
            StructuralStateDetector()
        ]
    }
    
    // MARK: - Public API
    
    @objc public func analyzeScreenshot(_ image: NSImage, 
                                       appIdentifier: String,
                                       completion: @escaping (NSDictionary) -> Void) {
        visionQueue.async { [weak self] in
            guard let self = self,
                  let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
                completion(["error": "Failed to process image"])
                return
            }
            
            // Extract visual features
            let features = self.extractVisualFeatures(from: cgImage)
            
            // Create observation
            let observation = StateObservation(
                visualFeatures: features,
                image: cgImage,
                timestamp: Date()
            )
            
            // Detect state
            let (detectedState, confidence) = self.detectState(
                observation: observation,
                appId: appIdentifier
            )
            
            // Handle state
            self.handleStateDetection(
                stateId: detectedState,
                confidence: confidence,
                observation: observation,
                appId: appIdentifier
            )
            
            // Return results
            let result: [String: Any] = [
                "state_id": detectedState ?? "unknown",
                "confidence": confidence,
                "features": features,
                "app_id": appIdentifier,
                "timestamp": Date().timeIntervalSince1970
            ]
            
            DispatchQueue.main.async {
                completion(result as NSDictionary)
            }
        }
    }
    
    @objc public func getStateInsights(for appId: String) -> NSDictionary {
        guard let states = applicationStates[appId] else {
            return ["message": "No states tracked for this application"]
        }
        
        let insights: [String: Any] = [
            "total_states": states.count,
            "states": states.map { (stateId, state) in
                [
                    "id": stateId,
                    "type": state.stateType.rawValue,
                    "observations": state.observationCount,
                    "last_seen": state.lastObserved?.timeIntervalSince1970 ?? 0
                ]
            }
        ]
        
        return insights as NSDictionary
    }
    
    // MARK: - Visual Feature Extraction
    
    private func extractVisualFeatures(from image: CGImage) -> [String: Any] {
        var features: [String: Any] = [:]
        
        // Color analysis
        features["color_histogram"] = computeColorHistogram(image)
        features["dominant_colors"] = extractDominantColors(image)
        
        // Structural analysis
        features["edge_density"] = computeEdgeDensity(image)
        features["complexity_score"] = computeComplexityScore(image)
        
        // Run Vision requests
        let requestHandler = VNImageRequestHandler(cgImage: image, options: [:])
        
        do {
            var requests: [VNRequest] = []
            if let textReq = textRecognitionRequest { requests.append(textReq) }
            if let contourReq = contourRequest { requests.append(contourReq) }
            if let saliencyReq = saliencyRequest { requests.append(saliencyReq) }
            
            try requestHandler.perform(requests)
        } catch {
            print("Vision request failed: \(error)")
        }
        
        return features
    }
    
    private func computeColorHistogram(_ image: CGImage) -> [Float] {
        guard let pixelData = image.dataProvider?.data else { return [] }
        let data: UnsafePointer<UInt8> = CFDataGetBytePtr(pixelData)
        
        var histogram = Array(repeating: Float(0), count: 256)
        let pixelCount = image.width * image.height
        
        // Simple grayscale histogram for now
        for i in 0..<pixelCount {
            let offset = i * 4
            let r = Float(data[offset])
            let g = Float(data[offset + 1])
            let b = Float(data[offset + 2])
            let gray = Int((r + g + b) / 3.0)
            histogram[gray] += 1.0
        }
        
        // Normalize
        let total = Float(pixelCount)
        return histogram.map { $0 / total }
    }
    
    private func extractDominantColors(_ image: CGImage) -> [[Float]] {
        // K-means clustering for dominant colors
        // Simplified implementation - returns top 5 colors
        var dominantColors: [[Float]] = []
        
        // This would implement proper k-means clustering
        // For now, return placeholder
        dominantColors.append([0.2, 0.3, 0.4]) // Example RGB values
        
        return dominantColors
    }
    
    private func computeEdgeDensity(_ image: CGImage) -> Float {
        // Sobel edge detection
        guard let grayscaleImage = convertToGrayscale(image) else { return 0.0 }
        
        // Apply Sobel operator (simplified)
        // In production, use Accelerate framework for convolution
        let width = grayscaleImage.width
        let height = grayscaleImage.height
        
        // Placeholder - actual implementation would compute edge map
        return 0.5
    }
    
    private func computeComplexityScore(_ image: CGImage) -> Float {
        // Compute visual complexity based on:
        // - Edge density
        // - Color variance
        // - Texture patterns
        // - Element count
        
        // Placeholder implementation
        return 0.75
    }
    
    private func convertToGrayscale(_ image: CGImage) -> CGImage? {
        let width = image.width
        let height = image.height
        
        let colorSpace = CGColorSpaceCreateDeviceGray()
        let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        )
        
        context?.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return context?.makeImage()
    }
    
    // MARK: - State Detection
    
    private func detectState(observation: StateObservation, appId: String) -> (String?, Float) {
        var bestMatch: (String?, Float) = (nil, 0.0)
        
        for detector in stateDetectors {
            let (stateId, confidence) = detector.detectState(
                observation: observation,
                knownStates: applicationStates[appId] ?? [:]
            )
            
            if confidence > bestMatch.1 {
                bestMatch = (stateId, confidence)
            }
        }
        
        return bestMatch
    }
    
    private func handleStateDetection(stateId: String?,
                                    confidence: Float,
                                    observation: StateObservation,
                                    appId: String) {
        // Initialize app states if needed
        if applicationStates[appId] == nil {
            applicationStates[appId] = [:]
        }
        
        guard let stateId = stateId else {
            // Potentially new state
            if learningEnabled && confidence < 0.5 {
                discoverNewState(observation: observation, appId: appId)
            }
            return
        }
        
        // Update existing state
        if var state = applicationStates[appId]?[stateId] {
            state.observationCount += 1
            state.lastObserved = Date()
            applicationStates[appId]?[stateId] = state
            
            // Learn from observation
            for detector in stateDetectors {
                detector.learn(from: observation, stateId: stateId)
            }
        } else if confidence > 0.7 {
            // New state with high confidence
            let newState = ApplicationState(
                stateId: stateId,
                stateType: inferStateType(from: observation)
            )
            applicationStates[appId]?[stateId] = newState
        }
    }
    
    private func discoverNewState(observation: StateObservation, appId: String) {
        // Generate unique state ID
        let stateId = generateStateId(from: observation)
        
        // Create new state
        let newState = ApplicationState(
            stateId: stateId,
            stateType: inferStateType(from: observation)
        )
        
        applicationStates[appId]?[stateId] = newState
        
        // Train detectors on new state
        for detector in stateDetectors {
            detector.learn(from: observation, stateId: stateId)
        }
    }
    
    private func generateStateId(from observation: StateObservation) -> String {
        // Generate ID based on visual features
        let features = observation.visualFeatures
        let timestamp = Int(observation.timestamp.timeIntervalSince1970)
        return "state_\(timestamp)_\(UUID().uuidString.prefix(8))"
    }
    
    private func inferStateType(from observation: StateObservation) -> ApplicationStateType {
        // Infer type based on visual characteristics
        // This would use ML or heuristics
        return .custom
    }
    
    // MARK: - Vision Request Processing
    
    private func processTextObservations(_ observations: [VNRecognizedTextObservation]?) {
        guard let observations = observations else { return }
        
        // Extract text content for state understanding
        let recognizedText = observations.compactMap { observation in
            observation.topCandidates(1).first?.string
        }
        
        // Store for state detection
        // This would be integrated into the observation
    }
    
    private func processContourObservations(_ observations: [VNContoursObservation]?) {
        guard let observations = observations else { return }
        
        // Analyze UI structure from contours
        // This helps identify buttons, panels, dialogs, etc.
    }
    
    private func processSaliencyObservations(_ observations: [VNSaliencyImageObservation]?) {
        guard let observations = observations else { return }
        
        // Identify focus areas and important UI elements
    }
}

// MARK: - Supporting Types

private struct StateObservation {
    let visualFeatures: [String: Any]
    let image: CGImage
    let timestamp: Date
    var textContent: [String] = []
    var structuralElements: [[String: Any]] = []
}

private protocol StateDetectorProtocol {
    func detectState(observation: StateObservation, 
                    knownStates: [String: VisionIntelligence.ApplicationState]) -> (String?, Float)
    func learn(from observation: StateObservation, stateId: String)
}

private class PatternBasedStateDetector: StateDetectorProtocol {
    private var learnedPatterns: [String: [[String: Any]]] = [:]
    
    func detectState(observation: StateObservation, 
                    knownStates: [String: VisionIntelligence.ApplicationState]) -> (String?, Float) {
        // Pattern matching implementation
        return (nil, 0.0)
    }
    
    func learn(from observation: StateObservation, stateId: String) {
        // Store patterns for future matching
    }
}

private class ColorBasedStateDetector: StateDetectorProtocol {
    func detectState(observation: StateObservation, 
                    knownStates: [String: VisionIntelligence.ApplicationState]) -> (String?, Float) {
        // Color-based state detection
        return (nil, 0.0)
    }
    
    func learn(from observation: StateObservation, stateId: String) {
        // Learn color patterns
    }
}

private class StructuralStateDetector: StateDetectorProtocol {
    func detectState(observation: StateObservation, 
                    knownStates: [String: VisionIntelligence.ApplicationState]) -> (String?, Float) {
        // Structure-based state detection
        return (nil, 0.0)
    }
    
    func learn(from observation: StateObservation, stateId: String) {
        // Learn structural patterns
    }
}