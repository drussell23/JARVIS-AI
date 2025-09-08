#!/usr/bin/env swift

//
// Workflow Automation for Pattern Engine
// Native macOS event capture and workflow automation
//

import Foundation
import Cocoa
import Combine
import os.log

/// System event types for workflow tracking
enum SystemEventType: String, Codable {
    case appLaunch = "app_launch"
    case appSwitch = "app_switch"
    case appQuit = "app_quit"
    case fileOpen = "file_open"
    case fileSave = "file_save"
    case windowFocus = "window_focus"
    case workspaceChange = "workspace_change"
    case systemWake = "system_wake"
    case systemSleep = "system_sleep"
    case networkChange = "network_change"
    case deviceConnect = "device_connect"
    case userActivity = "user_activity"
}

/// Captured system event
struct SystemEvent: Codable {
    let eventId: String
    let eventType: SystemEventType
    let timestamp: Date
    let applicationName: String?
    let applicationBundleId: String?
    let windowTitle: String?
    let filePath: String?
    let metadata: [String: String]
    
    init(type: SystemEventType, app: NSRunningApplication? = nil, window: String? = nil, file: String? = nil) {
        self.eventId = UUID().uuidString
        self.eventType = type
        self.timestamp = Date()
        self.applicationName = app?.localizedName
        self.applicationBundleId = app?.bundleIdentifier
        self.windowTitle = window
        self.filePath = file
        self.metadata = [:]
    }
}

/// Workflow automation action
struct AutomationAction: Codable {
    let actionId: String
    let actionType: String
    let targetApp: String?
    let parameters: [String: String]
    let delay: TimeInterval
}

/// Workflow automation engine
class WorkflowAutomation: NSObject {
    private let logger = Logger(subsystem: "com.jarvis.workflow", category: "automation")
    private let eventSubject = PassthroughSubject<SystemEvent, Never>()
    private var eventBuffer: [SystemEvent] = []
    private let maxBufferSize = 1000
    
    // Observers
    private var appObserver: NSObjectProtocol?
    private var workspaceObserver: NSObjectProtocol?
    private var systemObserver: NSObjectProtocol?
    private var fileObserver: NSObjectProtocol?
    
    // Automation
    private var automationQueue = DispatchQueue(label: "com.jarvis.workflow.automation", qos: .userInitiated)
    private var scheduledAutomations: [String: DispatchWorkItem] = [:]
    
    override init() {
        super.init()
        setupEventObservers()
    }
    
    deinit {
        removeEventObservers()
    }
    
    // MARK: - Event Observation
    
    private func setupEventObservers() {
        let workspace = NSWorkspace.shared
        let nc = workspace.notificationCenter
        let dnc = DistributedNotificationCenter.default()
        
        // App launch/quit observers
        appObserver = nc.addObserver(
            forName: NSWorkspace.didLaunchApplicationNotification,
            object: nil,
            queue: .main
        ) { [weak self] notification in
            if let app = notification.userInfo?[NSWorkspace.applicationUserInfoKey] as? NSRunningApplication {
                self?.recordEvent(SystemEvent(type: .appLaunch, app: app))
            }
        }
        
        nc.addObserver(
            forName: NSWorkspace.didTerminateApplicationNotification,
            object: nil,
            queue: .main
        ) { [weak self] notification in
            if let app = notification.userInfo?[NSWorkspace.applicationUserInfoKey] as? NSRunningApplication {
                self?.recordEvent(SystemEvent(type: .appQuit, app: app))
            }
        }
        
        // App activation observer
        nc.addObserver(
            forName: NSWorkspace.didActivateApplicationNotification,
            object: nil,
            queue: .main
        ) { [weak self] notification in
            if let app = notification.userInfo?[NSWorkspace.applicationUserInfoKey] as? NSRunningApplication {
                self?.recordEvent(SystemEvent(type: .appSwitch, app: app))
            }
        }
        
        // System wake/sleep
        nc.addObserver(
            forName: NSWorkspace.didWakeNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.recordEvent(SystemEvent(type: .systemWake))
        }
        
        nc.addObserver(
            forName: NSWorkspace.willSleepNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.recordEvent(SystemEvent(type: .systemSleep))
        }
        
        // File system events (simplified - production would use FSEvents)
        dnc.addObserver(
            forName: NSNotification.Name("com.apple.finder.fileOpen"),
            object: nil,
            queue: .main
        ) { [weak self] notification in
            if let path = notification.userInfo?["path"] as? String {
                self?.recordEvent(SystemEvent(type: .fileOpen, file: path))
            }
        }
        
        // Accessibility API for window titles (requires permissions)
        if AXIsProcessTrusted() {
            setupAccessibilityObservers()
        }
    }
    
    private func setupAccessibilityObservers() {
        // Monitor focused window changes
        NSWorkspace.shared.notificationCenter.addObserver(
            self,
            selector: #selector(focusedWindowChanged),
            name: NSWorkspace.didActivateApplicationNotification,
            object: nil
        )
    }
    
    @objc private func focusedWindowChanged(_ notification: Notification) {
        guard let app = notification.userInfo?[NSWorkspace.applicationUserInfoKey] as? NSRunningApplication,
              let appElement = AXUIElementCreateApplication(app.processIdentifier) else {
            return
        }
        
        var windowRef: CFTypeRef?
        let result = AXUIElementCopyAttributeValue(appElement, kAXFocusedWindowAttribute as CFString, &windowRef)
        
        if result == .success, let window = windowRef {
            var titleRef: CFTypeRef?
            let titleResult = AXUIElementCopyAttributeValue(window as! AXUIElement, kAXTitleAttribute as CFString, &titleRef)
            
            if titleResult == .success, let title = titleRef as? String {
                recordEvent(SystemEvent(type: .windowFocus, app: app, window: title))
            }
        }
    }
    
    private func removeEventObservers() {
        if let observer = appObserver {
            NotificationCenter.default.removeObserver(observer)
        }
        // Remove other observers...
    }
    
    private func recordEvent(_ event: SystemEvent) {
        eventBuffer.append(event)
        if eventBuffer.count > maxBufferSize {
            eventBuffer.removeFirst()
        }
        
        eventSubject.send(event)
        logger.debug("Recorded event: \(event.eventType.rawValue) for \(event.applicationName ?? "system")")
    }
    
    // MARK: - Event Retrieval
    
    func getRecentEvents(count: Int = 100) -> [SystemEvent] {
        return Array(eventBuffer.suffix(count))
    }
    
    func getEventsJSON() -> String? {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        
        do {
            let data = try encoder.encode(getRecentEvents())
            return String(data: data, encoding: .utf8)
        } catch {
            logger.error("Failed to encode events: \(error)")
            return nil
        }
    }
    
    var eventPublisher: AnyPublisher<SystemEvent, Never> {
        eventSubject.eraseToAnyPublisher()
    }
    
    // MARK: - Workflow Automation
    
    func executeAutomation(_ actions: [AutomationAction]) {
        automationQueue.async { [weak self] in
            for action in actions {
                self?.executeAction(action)
                if action.delay > 0 {
                    Thread.sleep(forTimeInterval: action.delay)
                }
            }
        }
    }
    
    private func executeAction(_ action: AutomationAction) {
        logger.info("Executing action: \(action.actionType)")
        
        switch action.actionType {
        case "launch_app":
            if let bundleId = action.targetApp {
                launchApplication(bundleId: bundleId)
            }
            
        case "switch_app":
            if let bundleId = action.targetApp {
                activateApplication(bundleId: bundleId)
            }
            
        case "open_file":
            if let path = action.parameters["path"] {
                openFile(at: path)
            }
            
        case "open_url":
            if let urlString = action.parameters["url"],
               let url = URL(string: urlString) {
                NSWorkspace.shared.open(url)
            }
            
        case "keystroke":
            if let keys = action.parameters["keys"] {
                simulateKeystroke(keys)
            }
            
        case "notification":
            if let title = action.parameters["title"],
               let message = action.parameters["message"] {
                showNotification(title: title, message: message)
            }
            
        default:
            logger.warning("Unknown action type: \(action.actionType)")
        }
    }
    
    private func launchApplication(bundleId: String) {
        if let url = NSWorkspace.shared.urlForApplication(withBundleIdentifier: bundleId) {
            NSWorkspace.shared.openApplication(at: url, configuration: NSWorkspace.OpenConfiguration())
        }
    }
    
    private func activateApplication(bundleId: String) {
        let apps = NSWorkspace.shared.runningApplications
        if let app = apps.first(where: { $0.bundleIdentifier == bundleId }) {
            app.activate(options: .activateIgnoringOtherApps)
        }
    }
    
    private func openFile(at path: String) {
        let url = URL(fileURLWithPath: path)
        NSWorkspace.shared.open(url)
    }
    
    private func simulateKeystroke(_ keys: String) {
        // Requires accessibility permissions
        // Implementation would use CGEvent API
        logger.debug("Simulating keystroke: \(keys)")
    }
    
    private func showNotification(title: String, message: String) {
        let notification = NSUserNotification()
        notification.title = title
        notification.informativeText = message
        notification.soundName = NSUserNotificationDefaultSoundName
        
        NSUserNotificationCenter.default.deliver(notification)
    }
    
    // MARK: - Pattern Matching
    
    func matchEventSequence(_ pattern: [String]) -> Bool {
        let recentEvents = getRecentEvents(count: pattern.count * 2)
        let eventTypes = recentEvents.map { $0.eventType.rawValue }
        
        // Simple subsequence matching
        var patternIndex = 0
        for eventType in eventTypes {
            if patternIndex < pattern.count && eventType == pattern[patternIndex] {
                patternIndex += 1
            }
        }
        
        return patternIndex == pattern.count
    }
    
    func getEventSequence(windowSize: Int = 10) -> [String] {
        return getRecentEvents(count: windowSize).map { $0.eventType.rawValue }
    }
}

// MARK: - Python Integration

@_cdecl("workflow_automation_create")
public func workflow_automation_create() -> UnsafeMutableRawPointer {
    let automation = WorkflowAutomation()
    return Unmanaged.passRetained(automation).toOpaque()
}

@_cdecl("workflow_automation_get_events")
public func workflow_automation_get_events(_ automationPtr: UnsafeMutableRawPointer) -> UnsafePointer<CChar>? {
    let automation = Unmanaged<WorkflowAutomation>.fromOpaque(automationPtr).takeUnretainedValue()
    
    guard let json = automation.getEventsJSON() else {
        return nil
    }
    
    return strdup(json)
}

@_cdecl("workflow_automation_execute")
public func workflow_automation_execute(_ automationPtr: UnsafeMutableRawPointer, _ actionsJSON: UnsafePointer<CChar>) {
    let automation = Unmanaged<WorkflowAutomation>.fromOpaque(automationPtr).takeUnretainedValue()
    
    guard let jsonString = String(cString: actionsJSON).data(using: .utf8) else {
        return
    }
    
    do {
        let actions = try JSONDecoder().decode([AutomationAction].self, from: jsonString)
        automation.executeAutomation(actions)
    } catch {
        print("Failed to decode actions: \(error)")
    }
}

@_cdecl("workflow_automation_free_string")
public func workflow_automation_free_string(_ str: UnsafePointer<CChar>) {
    free(UnsafeMutablePointer(mutating: str))
}

@_cdecl("workflow_automation_destroy")
public func workflow_automation_destroy(_ automationPtr: UnsafeMutableRawPointer) {
    Unmanaged<WorkflowAutomation>.fromOpaque(automationPtr).release()
}

// MARK: - CLI Interface

if CommandLine.arguments.count > 1 {
    let command = CommandLine.arguments[1]
    
    switch command {
    case "monitor":
        print("[WorkflowAutomation] Starting event monitoring...")
        let automation = WorkflowAutomation()
        
        var cancellable = automation.eventPublisher.sink { event in
            print("[\(event.timestamp)] \(event.eventType.rawValue): \(event.applicationName ?? "system")")
        }
        
        RunLoop.main.run()
        
    case "test":
        let automation = WorkflowAutomation()
        
        // Test automation
        let actions = [
            AutomationAction(
                actionId: "1",
                actionType: "notification",
                targetApp: nil,
                parameters: ["title": "Workflow Test", "message": "Automation is working!"],
                delay: 0
            )
        ]
        
        automation.executeAutomation(actions)
        Thread.sleep(forTimeInterval: 1.0)
        
    default:
        print("Usage: \(CommandLine.arguments[0]) [monitor|test]")
    }
}