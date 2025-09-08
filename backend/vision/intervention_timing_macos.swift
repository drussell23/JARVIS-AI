import Foundation
import Cocoa
import Combine

/// Natural intervention timing detector for macOS
class NaturalInterventionTiming {
    
    // MARK: - Types
    
    enum ActivityType {
        case typing
        case reading
        case coding
        case debugging
        case browsing
        case communicating
        case idle
        case transitioning
    }
    
    struct UserActivity {
        let type: ActivityType
        let startTime: Date
        var endTime: Date?
        let applicationName: String
        let confidence: Double
        
        var duration: TimeInterval {
            return (endTime ?? Date()).timeIntervalSince(startTime)
        }
    }
    
    struct TimingOpportunity {
        let timestamp: Date
        let score: Double
        let reason: String
        let activityContext: ActivityType
        let cognitiveLoad: Double
    }
    
    struct KeyboardActivity {
        let timestamp: Date
        let keyCount: Int
        let backspaceCount: Int
        let pauseDuration: TimeInterval
    }
    
    struct MouseActivity {
        let timestamp: Date
        let distance: Double
        let clicks: Int
        let scrollAmount: Double
        let velocity: Double
    }
    
    // MARK: - Properties
    
    private let activityQueue = DispatchQueue(label: "com.jarvis.intervention.timing", qos: .userInitiated)
    private var activityHistory: [UserActivity] = []
    private var keyboardBuffer: [KeyboardActivity] = []
    private var mouseBuffer: [MouseActivity] = []
    private let maxHistorySize = 100
    
    // Monitoring state
    private var eventMonitor: Any?
    private var mouseMonitor: Any?
    private var lastKeyPress: Date?
    private var lastMouseMove: Date?
    private var currentActivity: UserActivity?
    
    // Timing detection
    private var naturalBreakThreshold: TimeInterval = 5.0 // seconds
    private var taskBoundaryIndicators = Set<String>()
    private var cognitiveLoadEstimator = CognitiveLoadEstimator()
    
    // Publishers
    private let timingOpportunityPublisher = PassthroughSubject<TimingOpportunity, Never>()
    var timingOpportunities: AnyPublisher<TimingOpportunity, Never> {
        timingOpportunityPublisher.eraseToAnyPublisher()
    }
    
    // MARK: - Initialization
    
    init() {
        setupTaskBoundaryIndicators()
        startMonitoring()
    }
    
    deinit {
        stopMonitoring()
    }
    
    // MARK: - Setup
    
    private func setupTaskBoundaryIndicators() {
        // Common task boundary indicators
        taskBoundaryIndicators = [
            "cmd+s",        // Save
            "cmd+w",        // Close window/tab
            "cmd+q",        // Quit app
            "cmd+tab",      // Switch app
            "cmd+space",    // Spotlight
            "cmd+shift+[",  // Previous tab
            "cmd+shift+]",  // Next tab
            "return",       // Enter (sometimes indicates completion)
            "cmd+return",   // Submit/Execute
        ]
    }
    
    // MARK: - Monitoring
    
    private func startMonitoring() {
        // Keyboard monitoring
        eventMonitor = NSEvent.addGlobalMonitorForEvents(
            matching: [.keyDown, .keyUp],
            handler: { [weak self] event in
                self?.handleKeyboardEvent(event)
            }
        )
        
        // Mouse monitoring
        mouseMonitor = NSEvent.addGlobalMonitorForEvents(
            matching: [.mouseMoved, .leftMouseDown, .rightMouseDown, .scrollWheel],
            handler: { [weak self] event in
                self?.handleMouseEvent(event)
            }
        )
        
        // Application monitoring
        NSWorkspace.shared.notificationCenter.addObserver(
            self,
            selector: #selector(applicationDidActivate(_:)),
            name: NSWorkspace.didActivateApplicationNotification,
            object: nil
        )
        
        // Periodic analysis
        Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            self?.analyzeTimingOpportunity()
        }
    }
    
    private func stopMonitoring() {
        if let monitor = eventMonitor {
            NSEvent.removeMonitor(monitor)
        }
        if let monitor = mouseMonitor {
            NSEvent.removeMonitor(monitor)
        }
        NSWorkspace.shared.notificationCenter.removeObserver(self)
    }
    
    // MARK: - Event Handlers
    
    private func handleKeyboardEvent(_ event: NSEvent) {
        activityQueue.async {
            let now = Date()
            
            // Calculate pause duration
            let pauseDuration = self.lastKeyPress.map { now.timeIntervalSince($0) } ?? 0
            self.lastKeyPress = now
            
            // Track backspaces
            let isBackspace = event.keyCode == 51 // Backspace key code
            
            // Create keyboard activity
            let activity = KeyboardActivity(
                timestamp: now,
                keyCount: 1,
                backspaceCount: isBackspace ? 1 : 0,
                pauseDuration: pauseDuration
            )
            
            self.keyboardBuffer.append(activity)
            
            // Maintain buffer size
            if self.keyboardBuffer.count > 1000 {
                self.keyboardBuffer.removeFirst(100)
            }
            
            // Check for task boundary
            if let characters = event.charactersIgnoringModifiers {
                let modifiers = event.modifierFlags
                let keyCombo = self.getKeyCombo(characters: characters, modifiers: modifiers)
                
                if self.taskBoundaryIndicators.contains(keyCombo) {
                    self.detectTaskBoundary(keyCombo: keyCombo)
                }
            }
            
            // Update current activity
            self.updateCurrentActivity(type: .typing)
        }
    }
    
    private func handleMouseEvent(_ event: NSEvent) {
        activityQueue.async {
            let now = Date()
            
            // Calculate mouse metrics
            let velocity = self.lastMouseMove.map { 
                event.deltaX / now.timeIntervalSince($0) 
            } ?? 0
            self.lastMouseMove = now
            
            let activity = MouseActivity(
                timestamp: now,
                distance: sqrt(pow(event.deltaX, 2) + pow(event.deltaY, 2)),
                clicks: event.clickCount,
                scrollAmount: abs(event.scrollingDeltaY),
                velocity: abs(velocity)
            )
            
            self.mouseBuffer.append(activity)
            
            // Maintain buffer size
            if self.mouseBuffer.count > 1000 {
                self.mouseBuffer.removeFirst(100)
            }
        }
    }
    
    @objc private func applicationDidActivate(_ notification: Notification) {
        guard let app = notification.userInfo?[NSWorkspace.applicationUserInfoKey] as? NSRunningApplication,
              let appName = app.localizedName else { return }
        
        activityQueue.async {
            // End current activity
            if var current = self.currentActivity {
                current.endTime = Date()
                self.activityHistory.append(current)
            }
            
            // Start new activity
            let activityType = self.inferActivityType(for: appName)
            self.currentActivity = UserActivity(
                type: activityType,
                startTime: Date(),
                applicationName: appName,
                confidence: 0.8
            )
            
            // Maintain history size
            if self.activityHistory.count > self.maxHistorySize {
                self.activityHistory.removeFirst()
            }
            
            // This is a natural break point
            self.detectNaturalBreak(reason: "Application switch to \(appName)")
        }
    }
    
    // MARK: - Analysis
    
    private func analyzeTimingOpportunity() {
        activityQueue.async {
            let now = Date()
            
            // Check for idle time (natural break)
            if let lastKey = self.lastKeyPress,
               let lastMouse = self.lastMouseMove {
                let keyIdle = now.timeIntervalSince(lastKey)
                let mouseIdle = now.timeIntervalSince(lastMouse)
                let idleTime = min(keyIdle, mouseIdle)
                
                if idleTime > self.naturalBreakThreshold {
                    let opportunity = TimingOpportunity(
                        timestamp: now,
                        score: min(idleTime / 30.0, 1.0), // Max score at 30s idle
                        reason: "Natural break - idle for \(Int(idleTime))s",
                        activityContext: .idle,
                        cognitiveLoad: 0.1
                    )
                    self.timingOpportunityPublisher.send(opportunity)
                }
            }
            
            // Analyze cognitive load
            let cognitiveLoad = self.calculateCognitiveLoad()
            
            // Low cognitive load is good for intervention
            if cognitiveLoad < 0.3 {
                let opportunity = TimingOpportunity(
                    timestamp: now,
                    score: 1.0 - cognitiveLoad,
                    reason: "Low cognitive load detected",
                    activityContext: self.currentActivity?.type ?? .idle,
                    cognitiveLoad: cognitiveLoad
                )
                self.timingOpportunityPublisher.send(opportunity)
            }
        }
    }
    
    private func calculateCognitiveLoad() -> Double {
        // Analyze recent keyboard activity
        let recentKeyboard = keyboardBuffer.suffix(100)
        let typingSpeed = Double(recentKeyboard.count) / 10.0 // Keys per 10 seconds
        let backspaceRate = Double(recentKeyboard.filter { $0.backspaceCount > 0 }.count) / 
                           Double(max(recentKeyboard.count, 1))
        
        // Analyze recent mouse activity
        let recentMouse = mouseBuffer.suffix(100)
        let mouseIntensity = recentMouse.reduce(0.0) { $0 + $1.velocity } / 
                            Double(max(recentMouse.count, 1))
        
        // Combine factors
        let keyboardLoad = min(typingSpeed / 100.0, 1.0) + backspaceRate * 2.0
        let mouseLoad = min(mouseIntensity / 1000.0, 1.0)
        
        // Current activity factor
        let activityFactor: Double
        switch currentActivity?.type {
        case .debugging:
            activityFactor = 0.8
        case .coding:
            activityFactor = 0.7
        case .typing:
            activityFactor = 0.5
        case .reading:
            activityFactor = 0.3
        case .idle:
            activityFactor = 0.1
        default:
            activityFactor = 0.4
        }
        
        // Weighted combination
        let load = (keyboardLoad * 0.4 + mouseLoad * 0.2 + activityFactor * 0.4)
        return min(max(load, 0.0), 1.0)
    }
    
    // MARK: - Detection Methods
    
    private func detectNaturalBreak(reason: String) {
        let opportunity = TimingOpportunity(
            timestamp: Date(),
            score: 0.9,
            reason: reason,
            activityContext: currentActivity?.type ?? .transitioning,
            cognitiveLoad: calculateCognitiveLoad()
        )
        timingOpportunityPublisher.send(opportunity)
    }
    
    private func detectTaskBoundary(keyCombo: String) {
        let reason: String
        switch keyCombo {
        case "cmd+s":
            reason = "File saved - potential task completion"
        case "cmd+w":
            reason = "Window/tab closed - task boundary"
        case "cmd+tab":
            reason = "Application switch - context change"
        case "cmd+return":
            reason = "Command executed - action completed"
        default:
            reason = "Task boundary indicator: \(keyCombo)"
        }
        
        let opportunity = TimingOpportunity(
            timestamp: Date(),
            score: 0.8,
            reason: reason,
            activityContext: currentActivity?.type ?? .transitioning,
            cognitiveLoad: calculateCognitiveLoad()
        )
        timingOpportunityPublisher.send(opportunity)
    }
    
    // MARK: - Helper Methods
    
    private func getKeyCombo(characters: String, modifiers: NSEvent.ModifierFlags) -> String {
        var combo = ""
        
        if modifiers.contains(.command) { combo += "cmd+" }
        if modifiers.contains(.shift) { combo += "shift+" }
        if modifiers.contains(.option) { combo += "opt+" }
        if modifiers.contains(.control) { combo += "ctrl+" }
        
        combo += characters.lowercased()
        return combo
    }
    
    private func inferActivityType(for appName: String) -> ActivityType {
        let lowercased = appName.lowercased()
        
        if lowercased.contains("xcode") || lowercased.contains("visual studio") || 
           lowercased.contains("intellij") || lowercased.contains("sublime") {
            return .coding
        } else if lowercased.contains("slack") || lowercased.contains("teams") || 
                  lowercased.contains("mail") || lowercased.contains("messages") {
            return .communicating
        } else if lowercased.contains("safari") || lowercased.contains("chrome") || 
                  lowercased.contains("firefox") {
            return .browsing
        } else if lowercased.contains("terminal") || lowercased.contains("console") {
            return .debugging
        } else if lowercased.contains("preview") || lowercased.contains("reader") || 
                  lowercased.contains("books") {
            return .reading
        } else {
            return .typing
        }
    }
    
    private func updateCurrentActivity(type: ActivityType) {
        if currentActivity == nil || currentActivity?.type != type {
            // End current activity
            if var current = currentActivity {
                current.endTime = Date()
                activityHistory.append(current)
            }
            
            // Start new activity
            currentActivity = UserActivity(
                type: type,
                startTime: Date(),
                applicationName: NSWorkspace.shared.frontmostApplication?.localizedName ?? "Unknown",
                confidence: 0.7
            )
        }
    }
    
    // MARK: - Public API
    
    func getCurrentActivity() -> UserActivity? {
        return currentActivity
    }
    
    func getRecentActivities(count: Int = 10) -> [UserActivity] {
        return Array(activityHistory.suffix(count))
    }
    
    func getCognitiveLoad() -> Double {
        return calculateCognitiveLoad()
    }
    
    func getIdleTime() -> TimeInterval {
        let now = Date()
        let keyIdle = lastKeyPress.map { now.timeIntervalSince($0) } ?? 0
        let mouseIdle = lastMouseMove.map { now.timeIntervalSince($0) } ?? 0
        return min(keyIdle, mouseIdle)
    }
}

// MARK: - Cognitive Load Estimator

private class CognitiveLoadEstimator {
    private var errorPatterns: [Date] = []
    private var rapidActionPatterns: [Date] = []
    
    func addErrorPattern() {
        errorPatterns.append(Date())
        // Keep only recent patterns
        let cutoff = Date().addingTimeInterval(-300) // 5 minutes
        errorPatterns = errorPatterns.filter { $0 > cutoff }
    }
    
    func addRapidAction() {
        rapidActionPatterns.append(Date())
        let cutoff = Date().addingTimeInterval(-60) // 1 minute
        rapidActionPatterns = rapidActionPatterns.filter { $0 > cutoff }
    }
    
    func getErrorRate() -> Double {
        return Double(errorPatterns.count) / 300.0 // Errors per 5 minutes
    }
    
    func getRapidActionRate() -> Double {
        return Double(rapidActionPatterns.count) / 60.0 // Rapid actions per minute
    }
}