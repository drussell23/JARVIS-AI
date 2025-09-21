import Foundation
import Cocoa
import Vision
import CoreML

/// Native macOS anomaly detection using system APIs
class MacOSAnomalyDetector {
    
    // MARK: - Types
    
    enum VisualAnomaly {
        case unexpectedWindow(NSWindow)
        case errorDialog(NSAlert)
        case unusualLayout(NSView)
        case missingElements([String])
        case performanceIssue(String)
    }
    
    enum SystemAnomaly {
        case highCPU(Double)
        case highMemory(Double)
        case networkIssue(String)
        case appCrash(String)
        case permissionDenied(String)
    }
    
    struct AnomalyEvent {
        let id: String
        let timestamp: Date
        let type: String
        let severity: Int
        let description: String
        let evidence: [String: Any]
        let suggestedActions: [String]
    }
    
    // MARK: - Properties
    
    private var windowObserver: NSWindowDidBecomeKeyNotificationObserver?
    private var workspaceObserver: NSWorkspaceNotificationObserver?
    private let anomalyQueue = DispatchQueue(label: "com.jarvis.anomaly.detection", qos: .userInitiated)
    private var detectedAnomalies: [AnomalyEvent] = []
    private let maxAnomalyHistory = 100
    
    // Baseline tracking
    private var windowBaseline: Set<String> = []
    private var normalCPURange: ClosedRange<Double> = 0...70
    private var normalMemoryRange: ClosedRange<Double> = 0...80
    
    // MARK: - Initialization
    
    init() {
        setupObservers()
        establishBaseline()
    }
    
    deinit {
        removeObservers()
    }
    
    // MARK: - Setup
    
    private func setupObservers() {
        // Window notifications
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(windowDidBecomeKey(_:)),
            name: NSWindow.didBecomeKeyNotification,
            object: nil
        )
        
        // Application notifications
        NSWorkspace.shared.notificationCenter.addObserver(
            self,
            selector: #selector(applicationDidLaunch(_:)),
            name: NSWorkspace.didLaunchApplicationNotification,
            object: nil
        )
        
        NSWorkspace.shared.notificationCenter.addObserver(
            self,
            selector: #selector(applicationDidTerminate(_:)),
            name: NSWorkspace.didTerminateApplicationNotification,
            object: nil
        )
        
        // System notifications
        DistributedNotificationCenter.default().addObserver(
            self,
            selector: #selector(systemWillSleep(_:)),
            name: NSNotification.Name("com.apple.screenIsLocked"),
            object: nil
        )
    }
    
    private func removeObservers() {
        NotificationCenter.default.removeObserver(self)
        NSWorkspace.shared.notificationCenter.removeObserver(self)
        DistributedNotificationCenter.default().removeObserver(self)
    }
    
    // MARK: - Baseline Establishment
    
    private func establishBaseline() {
        anomalyQueue.async {
            // Capture current window state
            self.updateWindowBaseline()
            
            // Monitor system resources for baseline
            self.monitorSystemResourcesForBaseline()
        }
    }
    
    private func updateWindowBaseline() {
        DispatchQueue.main.async {
            let windows = NSApplication.shared.windows
            self.windowBaseline = Set(windows.compactMap { $0.title })
        }
    }
    
    private func monitorSystemResourcesForBaseline() {
        // Sample CPU and memory over time
        var cpuSamples: [Double] = []
        var memorySamples: [Double] = []
        
        let timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in
            let cpu = self.getCurrentCPUUsage()
            let memory = self.getCurrentMemoryUsage()
            
            cpuSamples.append(cpu)
            memorySamples.append(memory)
            
            // After 60 samples, calculate baseline
            if cpuSamples.count >= 60 {
                let avgCPU = cpuSamples.reduce(0, +) / Double(cpuSamples.count)
                let stdCPU = self.standardDeviation(cpuSamples)
                self.normalCPURange = (avgCPU - 2 * stdCPU)...(avgCPU + 2 * stdCPU)
                
                let avgMemory = memorySamples.reduce(0, +) / Double(memorySamples.count)
                let stdMemory = self.standardDeviation(memorySamples)
                self.normalMemoryRange = (avgMemory - 2 * stdMemory)...(avgMemory + 2 * stdMemory)
            }
        }
        
        // Stop after baseline period
        DispatchQueue.main.asyncAfter(deadline: .now() + 60) {
            timer.invalidate()
        }
    }
    
    // MARK: - Real-time Detection
    
    func detectVisualAnomalies() -> [VisualAnomaly] {
        var anomalies: [VisualAnomaly] = []
        
        DispatchQueue.main.sync {
            // Check for unexpected windows
            let currentWindows = NSApplication.shared.windows
            for window in currentWindows {
                if let title = window.title, !title.isEmpty {
                    // Check for error indicators
                    if containsErrorIndicators(title) {
                        anomalies.append(.errorDialog(NSAlert()))
                    }
                    
                    // Check for unexpected popups
                    if isUnexpectedPopup(window) {
                        anomalies.append(.unexpectedWindow(window))
                    }
                }
            }
            
            // Check for unusual layouts
            if let keyWindow = NSApplication.shared.keyWindow {
                if let contentView = keyWindow.contentView {
                    if hasUnusualLayout(contentView) {
                        anomalies.append(.unusualLayout(contentView))
                    }
                }
            }
        }
        
        return anomalies
    }
    
    func detectSystemAnomalies() -> [SystemAnomaly] {
        var anomalies: [SystemAnomaly] = []
        
        // Check CPU usage
        let cpuUsage = getCurrentCPUUsage()
        if !normalCPURange.contains(cpuUsage) && cpuUsage > normalCPURange.upperBound {
            anomalies.append(.highCPU(cpuUsage))
        }
        
        // Check memory usage
        let memoryUsage = getCurrentMemoryUsage()
        if !normalMemoryRange.contains(memoryUsage) && memoryUsage > normalMemoryRange.upperBound {
            anomalies.append(.highMemory(memoryUsage))
        }
        
        // Check for crashed applications
        if let crashedApps = checkForCrashedApplications() {
            for app in crashedApps {
                anomalies.append(.appCrash(app))
            }
        }
        
        return anomalies
    }
    
    // MARK: - Detection Helpers
    
    private func containsErrorIndicators(_ text: String) -> Bool {
        let errorKeywords = ["error", "failed", "exception", "crash", "unable", 
                           "invalid", "denied", "refused", "timeout"]
        let lowercasedText = text.lowercased()
        return errorKeywords.contains { lowercasedText.contains($0) }
    }
    
    private func isUnexpectedPopup(_ window: NSWindow) -> Bool {
        // Check window style and behavior
        let isModal = window.styleMask.contains(.docModalWindow)
        let isAlert = window.level == .alert || window.level == .popUpMenu
        let isSmallWindow = window.frame.width < 600 && window.frame.height < 400
        
        // Check if it appeared suddenly
        let isNew = !windowBaseline.contains(window.title ?? "")
        
        return (isModal || isAlert || isSmallWindow) && isNew
    }
    
    private func hasUnusualLayout(_ view: NSView) -> Bool {
        // Check for overlapping subviews
        let subviews = view.subviews
        var overlapCount = 0
        
        for i in 0..<subviews.count {
            for j in (i+1)..<subviews.count {
                if subviews[i].frame.intersects(subviews[j].frame) {
                    overlapCount += 1
                }
            }
        }
        
        // Unusual if many overlaps
        return overlapCount > 5
    }
    
    // MARK: - System Monitoring
    
    private func getCurrentCPUUsage() -> Double {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_,
                         task_flavor_t(MACH_TASK_BASIC_INFO),
                         $0,
                         &count)
            }
        }
        
        if result == KERN_SUCCESS {
            return Double(info.resident_size) / Double(1024 * 1024 * 1024) * 100
        }
        
        return 0
    }
    
    private func getCurrentMemoryUsage() -> Double {
        let task = mach_task_self_
        var info = vm_statistics_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics_data_t>.size) / 4
        
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics(mach_host_self(),
                              HOST_VM_INFO,
                              $0,
                              &count)
            }
        }
        
        if result == KERN_SUCCESS {
            let total = Double(info.wire_count + info.active_count + info.inactive_count + info.free_count)
            let used = Double(info.wire_count + info.active_count)
            return (used / total) * 100
        }
        
        return 0
    }
    
    private func checkForCrashedApplications() -> [String]? {
        // Check system log for recent crashes
        let task = Process()
        task.launchPath = "/usr/bin/log"
        task.arguments = ["show", "--predicate", "eventMessage contains 'crashed'", 
                         "--last", "1m", "--style", "json"]
        
        let pipe = Pipe()
        task.standardOutput = pipe
        
        do {
            try task.run()
            task.waitUntilExit()
            
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            if let output = String(data: data, encoding: .utf8) {
                // Parse crash logs
                return parseCrashLogs(output)
            }
        } catch {
            print("Failed to check crash logs: \(error)")
        }
        
        return nil
    }
    
    private func parseCrashLogs(_ logs: String) -> [String] {
        // Simple parsing - in production use proper JSON parsing
        var crashedApps: [String] = []
        
        let lines = logs.split(separator: "\n")
        for line in lines {
            if line.contains("crashed") {
                // Extract app name from log line
                if let appName = extractAppName(from: String(line)) {
                    crashedApps.append(appName)
                }
            }
        }
        
        return crashedApps
    }
    
    private func extractAppName(from logLine: String) -> String? {
        // Simple extraction - improve for production
        let components = logLine.split(separator: " ")
        for component in components {
            if component.hasSuffix(".app") {
                return String(component)
            }
        }
        return nil
    }
    
    // MARK: - Notification Handlers
    
    @objc private func windowDidBecomeKey(_ notification: Notification) {
        guard let window = notification.object as? NSWindow else { return }
        
        anomalyQueue.async {
            // Check if this is an unexpected window
            if self.isUnexpectedPopup(window) {
                self.recordAnomaly(
                    type: "unexpected_window",
                    severity: 2,
                    description: "Unexpected window appeared: \(window.title ?? "Untitled")"
                )
            }
        }
    }
    
    @objc private func applicationDidLaunch(_ notification: Notification) {
        guard let app = notification.userInfo?[NSWorkspace.applicationUserInfoKey] as? NSRunningApplication else { return }
        
        anomalyQueue.async {
            // Check if this is an unexpected launch
            if self.isUnexpectedAppLaunch(app) {
                self.recordAnomaly(
                    type: "unexpected_launch",
                    severity: 1,
                    description: "Unexpected application launched: \(app.localizedName ?? "Unknown")"
                )
            }
        }
    }
    
    @objc private func applicationDidTerminate(_ notification: Notification) {
        guard let app = notification.userInfo?[NSWorkspace.applicationUserInfoKey] as? NSRunningApplication else { return }
        
        anomalyQueue.async {
            // Check if this was a crash
            if app.isTerminated && !app.isFinishedLaunching {
                self.recordAnomaly(
                    type: "app_crash",
                    severity: 3,
                    description: "Application crashed: \(app.localizedName ?? "Unknown")"
                )
            }
        }
    }
    
    @objc private func systemWillSleep(_ notification: Notification) {
        // Save current state before sleep
        anomalyQueue.async {
            self.updateWindowBaseline()
        }
    }
    
    private func isUnexpectedAppLaunch(_ app: NSRunningApplication) -> Bool {
        // Check if app is in a whitelist of expected apps
        let expectedApps = ["Finder", "Dock", "SystemUIServer", "WindowServer"]
        if let appName = app.localizedName {
            return !expectedApps.contains(appName)
        }
        return true
    }
    
    // MARK: - Recording and Reporting
    
    private func recordAnomaly(type: String, severity: Int, description: String, 
                              evidence: [String: Any] = [:]) {
        let anomaly = AnomalyEvent(
            id: UUID().uuidString,
            timestamp: Date(),
            type: type,
            severity: severity,
            description: description,
            evidence: evidence,
            suggestedActions: generateSuggestedActions(for: type)
        )
        
        detectedAnomalies.append(anomaly)
        
        // Keep history size limited
        if detectedAnomalies.count > maxAnomalyHistory {
            detectedAnomalies.removeFirst()
        }
        
        // Notify observers
        notifyAnomalyDetected(anomaly)
    }
    
    private func generateSuggestedActions(for type: String) -> [String] {
        switch type {
        case "unexpected_window":
            return ["Close the window if safe", "Check for malware", "Review recent installations"]
        case "app_crash":
            return ["Restart the application", "Check crash logs", "Update the application"]
        case "high_cpu":
            return ["Check Activity Monitor", "Close unnecessary apps", "Restart if needed"]
        case "high_memory":
            return ["Free up memory", "Check for memory leaks", "Restart applications"]
        default:
            return ["Investigate the anomaly", "Monitor for recurrence"]
        }
    }
    
    private func notifyAnomalyDetected(_ anomaly: AnomalyEvent) {
        // Post notification for other components
        NotificationCenter.default.post(
            name: Notification.Name("JARVISAnomalyDetected"),
            object: self,
            userInfo: [
                "anomaly": anomaly,
                "timestamp": anomaly.timestamp,
                "severity": anomaly.severity
            ]
        )
    }
    
    // MARK: - Utility Functions
    
    private func standardDeviation(_ values: [Double]) -> Double {
        let mean = values.reduce(0, +) / Double(values.count)
        let variance = values.reduce(0) { $0 + pow($1 - mean, 2) } / Double(values.count)
        return sqrt(variance)
    }
    
    // MARK: - Public API
    
    func getRecentAnomalies(count: Int = 10) -> [AnomalyEvent] {
        return Array(detectedAnomalies.suffix(count))
    }
    
    func clearAnomalyHistory() {
        anomalyQueue.async {
            self.detectedAnomalies.removeAll()
        }
    }
    
    func exportAnomalies() -> Data? {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        
        do {
            return try encoder.encode(detectedAnomalies)
        } catch {
            print("Failed to encode anomalies: \(error)")
            return nil
        }
    }
}

// MARK: - Notification Observer Types

private class NSWindowDidBecomeKeyNotificationObserver {
    let observer: Any
    init(observer: Any) {
        self.observer = observer
    }
}

private class NSWorkspaceNotificationObserver {
    let observer: Any
    init(observer: Any) {
        self.observer = observer
    }
}