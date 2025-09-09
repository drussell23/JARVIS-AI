//
//  PredictiveEngineSwift.swift
//  JARVIS Vision Predictive Engine
//
//  Purpose: Native macOS predictive pre-computation with Core ML integration
//

import Foundation
import CoreML
import Accelerate
import AppKit
import Combine
import os.log

private let logger = OSLog(subsystem: "com.jarvis.vision", category: "PredictiveEngine")

/// State vector for Markov chain
struct StateVector: Hashable, Codable {
    let appId: String
    let appState: String
    var userAction: String?
    var timeContext: String?
    var goalContext: String?
    var workflowPhase: String?
    var confidence: Double = 1.0
    var metadata: [String: String] = [:]
    
    /// Calculate similarity with another state
    func similarity(to other: StateVector) -> Double {
        var score = 0.0
        
        // Weighted comparison
        if appId == other.appId { score += 0.3 }
        if appState == other.appState { score += 0.25 }
        if userAction == other.userAction { score += 0.15 }
        if timeContext == other.timeContext { score += 0.1 }
        if goalContext == other.goalContext { score += 0.15 }
        if workflowPhase == other.workflowPhase { score += 0.05 }
        
        return score
    }
}

/// Transition in the Markov chain
class StateTransition {
    let fromState: StateVector
    let toState: StateVector
    var count: Int = 1
    var temporalWeight: Double = 1.0
    var lastObserved = Date()
    
    var probability: Double {
        Double(count) * (0.7 + 0.3 * temporalWeight)
    }
    
    init(from: StateVector, to: StateVector) {
        self.fromState = from
        self.toState = to
    }
}

/// Sparse transition matrix implementation
class TransitionMatrix {
    private var stateIndex: [StateVector: Int] = [:]
    private var indexToState: [Int: StateVector] = [:]
    private var nextIndex = 0
    
    // Sparse storage for transitions
    private var transitions: [Int: [Int: StateTransition]] = [:]
    private let queue = DispatchQueue(label: "com.jarvis.transitionMatrix", attributes: .concurrent)
    
    /// Add or get state index
    func addState(_ state: StateVector) -> Int {
        return queue.sync(flags: .barrier) {
            if let idx = stateIndex[state] {
                return idx
            }
            
            let idx = nextIndex
            stateIndex[state] = idx
            indexToState[idx] = state
            nextIndex += 1
            
            return idx
        }
    }
    
    /// Record state transition
    func addTransition(from: StateVector, to: StateVector, temporalFactor: Double = 1.0) {
        let fromIdx = addState(from)
        let toIdx = addState(to)
        
        queue.async(flags: .barrier) {
            if self.transitions[fromIdx] == nil {
                self.transitions[fromIdx] = [:]
            }
            
            if let existing = self.transitions[fromIdx]?[toIdx] {
                existing.count += 1
                existing.temporalWeight = 0.9 * existing.temporalWeight + 0.1 * temporalFactor
                existing.lastObserved = Date()
            } else {
                let transition = StateTransition(from: from, to: to)
                transition.temporalWeight = temporalFactor
                self.transitions[fromIdx]?[toIdx] = transition
            }
        }
    }
    
    /// Get top-k predictions for a state
    func getPredictions(for state: StateVector, topK: Int = 5) -> [(StateVector, Double, Double)] {
        return queue.sync {
            guard let stateIdx = stateIndex[state],
                  let stateTransitions = transitions[stateIdx] else {
                return []
            }
            
            // Calculate total probability mass
            let totalProb = stateTransitions.values.reduce(0.0) { $0 + $1.probability }
            
            guard totalProb > 0 else { return [] }
            
            // Sort by probability and get top-k
            let predictions = stateTransitions.values
                .sorted { $0.probability > $1.probability }
                .prefix(topK)
                .compactMap { transition -> (StateVector, Double, Double)? in
                    guard let nextState = indexToState[stateIndex[transition.toState] ?? -1] else {
                        return nil
                    }
                    
                    let probability = transition.probability / totalProb
                    let confidence = min(1.0, Double(transition.count) / 10.0)
                    
                    return (nextState, probability, confidence)
                }
            
            return Array(predictions)
        }
    }
}

/// Prediction task for speculative execution
class PredictionTask: Comparable {
    let id: String
    let state: StateVector
    let predictedStates: [(StateVector, Double)]
    let priority: Double
    let deadline: Date?
    let createdAt = Date()
    var result: Any?
    var status = "pending"
    
    init(state: StateVector, predictions: [(StateVector, Double)], priority: Double) {
        self.id = UUID().uuidString
        self.state = state
        self.predictedStates = predictions
        self.priority = priority
        self.deadline = Date().addingTimeInterval(30) // 30 second deadline
    }
    
    static func < (lhs: PredictionTask, rhs: PredictionTask) -> Bool {
        lhs.priority < rhs.priority
    }
    
    static func == (lhs: PredictionTask, rhs: PredictionTask) -> Bool {
        lhs.id == rhs.id
    }
}

/// macOS-specific prediction executor
class MacOSPredictionExecutor {
    private let queue = OperationQueue()
    private let workspace = NSWorkspace.shared
    
    init() {
        queue.maxConcurrentOperationCount = 4
        queue.qualityOfService = .userInitiated
    }
    
    /// Execute prediction based on state type
    func executePrediction(_ task: PredictionTask, completion: @escaping (Data?) -> Void) {
        queue.addOperation {
            let result: Data?
            
            switch task.state.appId {
            case "com.apple.Safari", "com.google.Chrome":
                result = self.predictBrowserAction(task)
            case "com.microsoft.VSCode", "com.apple.dt.Xcode":
                result = self.predictEditorAction(task)
            case "com.apple.finder":
                result = self.predictFinderAction(task)
            default:
                result = self.predictGenericAction(task)
            }
            
            DispatchQueue.main.async {
                completion(result)
            }
        }
    }
    
    private func predictBrowserAction(_ task: PredictionTask) -> Data? {
        // Predict browser navigation, search, etc.
        guard let nextState = task.predictedStates.first else { return nil }
        
        let prediction: [String: Any] = [
            "type": "browser_navigation",
            "currentUrl": task.state.metadata["url"] ?? "",
            "predictedAction": nextState.0.userAction ?? "navigate",
            "suggestions": getPredictedUrls(for: task.state),
            "confidence": nextState.1
        ]
        
        return try? JSONSerialization.data(withJSONObject: prediction)
    }
    
    private func predictEditorAction(_ task: PredictionTask) -> Data? {
        // Predict code completion, file navigation, etc.
        guard let nextState = task.predictedStates.first else { return nil }
        
        let prediction: [String: Any] = [
            "type": "editor_action",
            "currentFile": task.state.metadata["file"] ?? "",
            "predictedAction": nextState.0.userAction ?? "edit",
            "suggestions": getCodeSuggestions(for: task.state),
            "confidence": nextState.1
        ]
        
        return try? JSONSerialization.data(withJSONObject: prediction)
    }
    
    private func predictFinderAction(_ task: PredictionTask) -> Data? {
        // Predict file operations
        guard let nextState = task.predictedStates.first else { return nil }
        
        let prediction: [String: Any] = [
            "type": "file_operation",
            "currentPath": task.state.metadata["path"] ?? "",
            "predictedAction": nextState.0.userAction ?? "navigate",
            "suggestedPaths": getRecentPaths(),
            "confidence": nextState.1
        ]
        
        return try? JSONSerialization.data(withJSONObject: prediction)
    }
    
    private func predictGenericAction(_ task: PredictionTask) -> Data? {
        guard let nextState = task.predictedStates.first else { return nil }
        
        let prediction: [String: Any] = [
            "type": "generic",
            "appId": task.state.appId,
            "predictedAction": nextState.0.userAction ?? "unknown",
            "confidence": nextState.1
        ]
        
        return try? JSONSerialization.data(withJSONObject: prediction)
    }
    
    private func getPredictedUrls(for state: StateVector) -> [String] {
        // In production, would use browsing history and patterns
        return ["https://github.com", "https://stackoverflow.com", "https://docs.swift.org"]
    }
    
    private func getCodeSuggestions(for state: StateVector) -> [String] {
        // In production, would use language server and context
        return ["function completion", "variable suggestion", "import statement"]
    }
    
    private func getRecentPaths() -> [String] {
        // Get recent document paths
        let fileManager = FileManager.default
        let documentsUrl = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first
        let downloadsUrl = fileManager.urls(for: .downloadsDirectory, in: .userDomainMask).first
        
        return [
            documentsUrl?.path ?? "~/Documents",
            downloadsUrl?.path ?? "~/Downloads",
            "~/Desktop"
        ]
    }
}

/// Application state tracker using NSWorkspace
class ApplicationStateTracker {
    private let workspace = NSWorkspace.shared
    private var activeAppObserver: NSObjectProtocol?
    private var windowObserver: NSObjectProtocol?
    
    var stateChangeHandler: ((StateVector) -> Void)?
    
    init() {
        startTracking()
    }
    
    deinit {
        stopTracking()
    }
    
    private func startTracking() {
        // Track active application changes
        activeAppObserver = workspace.notificationCenter.addObserver(
            forName: NSWorkspace.didActivateApplicationNotification,
            object: nil,
            queue: .main
        ) { [weak self] notification in
            self?.handleAppActivation(notification)
        }
        
        // Track window focus changes
        windowObserver = NotificationCenter.default.addObserver(
            forName: NSWindow.didBecomeKeyNotification,
            object: nil,
            queue: .main
        ) { [weak self] notification in
            self?.handleWindowFocus(notification)
        }
    }
    
    private func stopTracking() {
        if let observer = activeAppObserver {
            workspace.notificationCenter.removeObserver(observer)
        }
        if let observer = windowObserver {
            NotificationCenter.default.removeObserver(observer)
        }
    }
    
    private func handleAppActivation(_ notification: Notification) {
        guard let app = notification.userInfo?[NSWorkspace.applicationUserInfoKey] as? NSRunningApplication else {
            return
        }
        
        let state = StateVector(
            appId: app.bundleIdentifier ?? "unknown",
            appState: "activated",
            userAction: "switch_app",
            timeContext: getTimeContext(),
            metadata: [
                "appName": app.localizedName ?? "",
                "pid": String(app.processIdentifier)
            ]
        )
        
        stateChangeHandler?(state)
    }
    
    private func handleWindowFocus(_ notification: Notification) {
        guard let window = notification.object as? NSWindow,
              let app = NSRunningApplication.current.bundleIdentifier else {
            return
        }
        
        let state = StateVector(
            appId: app,
            appState: "window_focused",
            userAction: "focus_window",
            timeContext: getTimeContext(),
            metadata: [
                "windowTitle": window.title,
                "windowFrame": NSStringFromRect(window.frame)
            ]
        )
        
        stateChangeHandler?(state)
    }
    
    private func getTimeContext() -> String {
        let hour = Calendar.current.component(.hour, from: Date())
        
        switch hour {
        case 6..<12: return "morning"
        case 12..<17: return "afternoon"
        case 17..<22: return "evening"
        default: return "night"
        }
    }
}

/// Main predictive engine for macOS
@objc class PredictiveEngineSwift: NSObject {
    private let transitionMatrix = TransitionMatrix()
    private let executor = MacOSPredictionExecutor()
    private let stateTracker = ApplicationStateTracker()
    
    private var currentState: StateVector?
    private var stateHistory: [StateVector] = []
    private let historyLimit = 100
    
    // Result cache
    private let resultCache = NSCache<NSString, NSData>()
    
    // Statistics
    private var stats = PredictionStats()
    
    override init() {
        super.init()
        setupStateTracking()
        resultCache.countLimit = 1000
    }
    
    private func setupStateTracking() {
        stateTracker.stateChangeHandler = { [weak self] newState in
            self?.updateState(newState)
        }
    }
    
    /// Update current state and generate predictions
    @objc func updateState(_ newState: StateVector) {
        // Record transition
        if let current = currentState {
            let temporalFactor = calculateTemporalFactor()
            transitionMatrix.addTransition(from: current, to: newState, temporalFactor: temporalFactor)
        }
        
        // Update state
        currentState = newState
        stateHistory.append(newState)
        if stateHistory.count > historyLimit {
            stateHistory.removeFirst()
        }
        
        // Generate predictions
        generatePredictions(for: newState)
    }
    
    private func calculateTemporalFactor() -> Double {
        guard stateHistory.count >= 2 else { return 1.0 }
        
        // Calculate time between last two states
        let recentStates = stateHistory.suffix(2)
        // In production, would track timestamps properly
        return 0.95 // Decay factor
    }
    
    private func generatePredictions(for state: StateVector) {
        let predictions = transitionMatrix.getPredictions(for: state, topK: 5)
        
        for (nextState, probability, confidence) in predictions {
            if confidence >= 0.7 {
                // Create and execute prediction task
                let task = PredictionTask(
                    state: state,
                    predictions: [(nextState, probability)],
                    priority: probability * confidence
                )
                
                executePrediction(task)
                stats.predictionsMade += 1
            }
        }
    }
    
    private func executePrediction(_ task: PredictionTask) {
        executor.executePrediction(task) { [weak self] result in
            guard let self = self, let data = result else { return }
            
            // Cache result
            let cacheKey = "\(task.state.hashValue)-\(task.predictedStates.first?.0.hashValue ?? 0)" as NSString
            self.resultCache.setObject(data as NSData, forKey: cacheKey)
            
            self.stats.predictionsExecuted += 1
            
            os_log(.info, log: logger, "Executed prediction: %{public}@", task.id)
        }
    }
    
    /// Get cached prediction result
    @objc func getCachedResult(current: StateVector, target: StateVector) -> Data? {
        let cacheKey = "\(current.hashValue)-\(target.hashValue)" as NSString
        
        if let cached = resultCache.object(forKey: cacheKey) {
            stats.cacheHits += 1
            return cached as Data
        }
        
        stats.cacheMisses += 1
        return nil
    }
    
    /// Get prediction statistics
    @objc func getStatistics() -> [String: Any] {
        return [
            "predictions_made": stats.predictionsMade,
            "predictions_executed": stats.predictionsExecuted,
            "cache_hits": stats.cacheHits,
            "cache_misses": stats.cacheMisses,
            "cache_hit_rate": stats.cacheHitRate,
            "state_count": transitionMatrix.stateIndex.count,
            "history_size": stateHistory.count
        ]
    }
    
    /// Manual state update for Python integration
    @objc func updateStateManual(
        appId: String,
        appState: String,
        userAction: String?,
        timeContext: String?,
        goalContext: String?,
        workflowPhase: String?
    ) {
        let state = StateVector(
            appId: appId,
            appState: appState,
            userAction: userAction,
            timeContext: timeContext,
            goalContext: goalContext,
            workflowPhase: workflowPhase
        )
        
        updateState(state)
    }
    
    /// Get predictions for current state
    @objc func getCurrentPredictions() -> [[String: Any]] {
        guard let current = currentState else { return [] }
        
        let predictions = transitionMatrix.getPredictions(for: current, topK: 5)
        
        return predictions.map { (state, probability, confidence) in
            [
                "app_id": state.appId,
                "app_state": state.appState,
                "user_action": state.userAction ?? "",
                "probability": probability,
                "confidence": confidence
            ]
        }
    }
}

/// Statistics tracking
private struct PredictionStats {
    var predictionsMade: Int = 0
    var predictionsExecuted: Int = 0
    var cacheHits: Int = 0
    var cacheMisses: Int = 0
    
    var cacheHitRate: Double {
        let total = cacheHits + cacheMisses
        return total > 0 ? Double(cacheHits) / Double(total) : 0.0
    }
}

/// Core ML integration for advanced predictions
@available(macOS 10.15, *)
class MLPredictionEnhancer {
    private var model: MLModel?
    
    init() {
        loadModel()
    }
    
    private func loadModel() {
        // In production, load actual Core ML model for state prediction
        // For now, placeholder
    }
    
    func enhancePredictions(_ states: [(StateVector, Double)]) -> [(StateVector, Double, [String: Any])] {
        // Would use Core ML to add context and enhance predictions
        return states.map { (state, prob) in
            (state, prob, [:])
        }
    }
}