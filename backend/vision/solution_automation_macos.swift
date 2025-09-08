import Foundation
import Cocoa
import Accessibility
import Combine

/// Native macOS solution automation for executing captured solutions
class SolutionAutomation: NSObject {
    
    // MARK: - Types
    
    enum AutomationError: Error {
        case accessibilityNotEnabled
        case applicationNotFound(String)
        case elementNotFound(String)
        case actionFailed(String)
        case timeout
    }
    
    struct AutomationStep {
        let action: String
        let target: String?
        let parameters: [String: Any]
        let waitCondition: String?
        let timeout: TimeInterval
        let verification: String?
    }
    
    struct AutomationResult {
        let success: Bool
        let stepIndex: Int
        let message: String
        let duration: TimeInterval
        let screenshot: NSImage?
    }
    
    struct ApplicationContext {
        let bundleIdentifier: String
        let processId: pid_t
        let window: AXUIElement?
        let isActive: Bool
    }
    
    // MARK: - Properties
    
    private let actionQueue = DispatchQueue(label: "com.jarvis.solution.automation", qos: .userInitiated)
    private var currentApplication: ApplicationContext?
    private var executionHistory: [AutomationResult] = []
    private let maxHistorySize = 100
    
    // Accessibility helpers
    private var systemWideElement: AXUIElement {
        return AXUIElementCreateSystemWide()
    }
    
    // Publishers for monitoring
    private let executionPublisher = PassthroughSubject<AutomationResult, Never>()
    var executionResults: AnyPublisher<AutomationResult, Never> {
        executionPublisher.eraseToAnyPublisher()
    }
    
    // MARK: - Initialization
    
    override init() {
        super.init()
        checkAccessibilityPermissions()
    }
    
    // MARK: - Accessibility
    
    private func checkAccessibilityPermissions() {
        let trusted = AXIsProcessTrustedWithOptions(
            [kAXTrustedCheckOptionPrompt.takeRetainedValue() as String: true] as CFDictionary
        )
        
        if !trusted {
            print("⚠️ Accessibility permissions not granted. Some automation features may not work.")
        }
    }
    
    // MARK: - Application Management
    
    func activateApplication(bundleId: String) throws -> ApplicationContext {
        guard let app = NSWorkspace.shared.runningApplications.first(where: {
            $0.bundleIdentifier == bundleId
        }) else {
            // Try to launch the application
            guard let launched = NSWorkspace.shared.launchApplication(
                withBundleIdentifier: bundleId,
                options: [],
                additionalEventParamDescriptor: nil,
                launchIdentifier: nil
            ) else {
                throw AutomationError.applicationNotFound(bundleId)
            }
            
            // Wait for launch
            Thread.sleep(forTimeInterval: 2.0)
            
            guard let app = NSWorkspace.shared.runningApplications.first(where: {
                $0.bundleIdentifier == bundleId
            }) else {
                throw AutomationError.applicationNotFound(bundleId)
            }
            
            return try activateRunningApp(app)
        }
        
        return try activateRunningApp(app)
    }
    
    private func activateRunningApp(_ app: NSRunningApplication) throws -> ApplicationContext {
        // Activate the application
        app.activate(options: .activateIgnoringOtherApps)
        
        // Get accessibility element
        let appElement = AXUIElementCreateApplication(app.processIdentifier)
        
        // Get main window
        var windowValue: CFTypeRef?
        let result = AXUIElementCopyAttributeValue(
            appElement,
            kAXMainWindowAttribute as CFString,
            &windowValue
        )
        
        let window = (result == .success) ? (windowValue as! AXUIElement?) : nil
        
        return ApplicationContext(
            bundleIdentifier: app.bundleIdentifier ?? "unknown",
            processId: app.processIdentifier,
            window: window,
            isActive: app.isActive
        )
    }
    
    // MARK: - Solution Execution
    
    func executeSolution(
        steps: [AutomationStep],
        targetApp: String? = nil,
        captureScreenshots: Bool = true
    ) async throws -> [AutomationResult] {
        var results: [AutomationResult] = []
        
        // Activate target application if specified
        if let appId = targetApp {
            currentApplication = try activateApplication(bundleId: appId)
            
            // Give app time to become ready
            try await Task.sleep(nanoseconds: 500_000_000) // 0.5 seconds
        }
        
        // Execute each step
        for (index, step) in steps.enumerated() {
            let startTime = Date()
            
            do {
                // Execute the step
                let success = try await executeStep(step)
                
                // Capture screenshot if requested
                let screenshot = captureScreenshots ? captureScreen() : nil
                
                // Verify if needed
                if let verification = step.verification {
                    let verified = try await verifyCondition(verification)
                    if !verified {
                        throw AutomationError.actionFailed("Verification failed: \(verification)")
                    }
                }
                
                let result = AutomationResult(
                    success: success,
                    stepIndex: index,
                    message: "Executed \(step.action)",
                    duration: Date().timeIntervalSince(startTime),
                    screenshot: screenshot
                )
                
                results.append(result)
                executionPublisher.send(result)
                
                // Wait if specified
                if let waitCondition = step.waitCondition {
                    try await waitForCondition(waitCondition, timeout: step.timeout)
                }
                
            } catch {
                let result = AutomationResult(
                    success: false,
                    stepIndex: index,
                    message: "Failed: \(error.localizedDescription)",
                    duration: Date().timeIntervalSince(startTime),
                    screenshot: captureScreenshots ? captureScreen() : nil
                )
                
                results.append(result)
                executionPublisher.send(result)
                
                // Stop execution on failure
                throw error
            }
        }
        
        // Store in history
        executionHistory.append(contentsOf: results)
        if executionHistory.count > maxHistorySize {
            executionHistory.removeFirst(executionHistory.count - maxHistorySize)
        }
        
        return results
    }
    
    // MARK: - Step Execution
    
    private func executeStep(_ step: AutomationStep) async throws -> Bool {
        switch step.action.lowercased() {
        case "click":
            return try await performClick(step)
            
        case "type", "input":
            return try await performTypeText(step)
            
        case "key", "keypress":
            return try await performKeyPress(step)
            
        case "menu":
            return try await performMenuAction(step)
            
        case "drag":
            return try await performDrag(step)
            
        case "scroll":
            return try await performScroll(step)
            
        case "wait":
            return try await performWait(step)
            
        case "focus":
            return try await performFocus(step)
            
        case "screenshot":
            return performScreenshot(step)
            
        default:
            // Try to execute as AppleScript
            return try await executeAppleScript(step)
        }
    }
    
    // MARK: - Actions
    
    private func performClick(_ step: AutomationStep) async throws -> Bool {
        guard let target = step.target else {
            throw AutomationError.actionFailed("Click requires a target")
        }
        
        // Find element
        guard let element = try findElement(matching: target) else {
            throw AutomationError.elementNotFound(target)
        }
        
        // Get position
        var positionValue: CFTypeRef?
        AXUIElementCopyAttributeValue(
            element,
            kAXPositionAttribute as CFString,
            &positionValue
        )
        
        guard let positionRef = positionValue else {
            throw AutomationError.actionFailed("Could not get element position")
        }
        
        var position = CGPoint.zero
        AXValueGetValue(positionRef as! AXValue, .cgPoint, &position)
        
        // Get size for center calculation
        var sizeValue: CFTypeRef?
        AXUIElementCopyAttributeValue(
            element,
            kAXSizeAttribute as CFString,
            &sizeValue
        )
        
        if let sizeRef = sizeValue {
            var size = CGSize.zero
            AXValueGetValue(sizeRef as! AXValue, .cgSize, &size)
            position.x += size.width / 2
            position.y += size.height / 2
        }
        
        // Perform click
        let clickCount = step.parameters["count"] as? Int ?? 1
        let button = step.parameters["button"] as? String ?? "left"
        
        let mouseButton: CGMouseButton = button == "right" ? .right : .left
        
        for _ in 0..<clickCount {
            let downEvent = CGEvent(
                mouseEventSource: nil,
                mouseType: mouseButton == .right ? .rightMouseDown : .leftMouseDown,
                mouseCursorPosition: position,
                mouseButton: mouseButton
            )
            downEvent?.post(tap: .cghidEventTap)
            
            let upEvent = CGEvent(
                mouseEventSource: nil,
                mouseType: mouseButton == .right ? .rightMouseUp : .leftMouseUp,
                mouseCursorPosition: position,
                mouseButton: mouseButton
            )
            upEvent?.post(tap: .cghidEventTap)
            
            if clickCount > 1 {
                try await Task.sleep(nanoseconds: 100_000_000) // 0.1s between clicks
            }
        }
        
        return true
    }
    
    private func performTypeText(_ step: AutomationStep) async throws -> Bool {
        let text = step.parameters["text"] as? String ?? step.target ?? ""
        let clearFirst = step.parameters["clear"] as? Bool ?? false
        
        // Clear field if requested
        if clearFirst {
            // Cmd+A to select all
            let selectAllEvent = CGEvent(keyboardEventSource: nil, virtualKey: 0x00, keyDown: true)
            selectAllEvent?.flags = .maskCommand
            selectAllEvent?.post(tap: .cghidEventTap)
            
            // Delete
            let deleteEvent = CGEvent(keyboardEventSource: nil, virtualKey: 0x33, keyDown: true)
            deleteEvent?.post(tap: .cghidEventTap)
        }
        
        // Type the text
        let source = CGEventSource(stateID: .hidSystemState)
        
        for character in text {
            if let event = CGEvent(keyboardEventSource: source, virtualKey: 0, keyDown: true) {
                event.keyboardSetUnicodeString(stringLength: 1, unicodeString: [character.utf16.first!])
                event.post(tap: .cghidEventTap)
            }
            
            // Small delay between keystrokes
            try await Task.sleep(nanoseconds: 50_000_000) // 0.05s
        }
        
        return true
    }
    
    private func performKeyPress(_ step: AutomationStep) async throws -> Bool {
        guard let keyCode = step.parameters["keyCode"] as? Int ?? 
                           keyCodeForKey(step.target ?? "") else {
            throw AutomationError.actionFailed("Invalid key code")
        }
        
        let modifiers = step.parameters["modifiers"] as? [String] ?? []
        var flags = CGEventFlags()
        
        // Set modifier flags
        for modifier in modifiers {
            switch modifier.lowercased() {
            case "cmd", "command":
                flags.insert(.maskCommand)
            case "shift":
                flags.insert(.maskShift)
            case "alt", "option":
                flags.insert(.maskAlternate)
            case "ctrl", "control":
                flags.insert(.maskControl)
            default:
                break
            }
        }
        
        // Create and post key event
        let keyDown = CGEvent(keyboardEventSource: nil, virtualKey: CGKeyCode(keyCode), keyDown: true)
        keyDown?.flags = flags
        keyDown?.post(tap: .cghidEventTap)
        
        let keyUp = CGEvent(keyboardEventSource: nil, virtualKey: CGKeyCode(keyCode), keyDown: false)
        keyUp?.flags = flags
        keyUp?.post(tap: .cghidEventTap)
        
        return true
    }
    
    private func performMenuAction(_ step: AutomationStep) async throws -> Bool {
        guard let menuPath = step.target else {
            throw AutomationError.actionFailed("Menu action requires a path")
        }
        
        let menuItems = menuPath.split(separator: ">").map { $0.trimmingCharacters(in: .whitespaces) }
        
        guard let app = currentApplication,
              let appElement = AXUIElementCreateApplication(app.processId) as AXUIElement? else {
            throw AutomationError.actionFailed("No active application")
        }
        
        // Get menu bar
        var menuBarValue: CFTypeRef?
        AXUIElementCopyAttributeValue(
            appElement,
            kAXMenuBarAttribute as CFString,
            &menuBarValue
        )
        
        guard let menuBar = menuBarValue as? AXUIElement else {
            throw AutomationError.elementNotFound("Menu bar")
        }
        
        // Navigate menu hierarchy
        var currentMenu: AXUIElement? = menuBar
        
        for itemName in menuItems {
            guard let menu = currentMenu,
                  let item = try findMenuItem(in: menu, named: itemName) else {
                throw AutomationError.elementNotFound("Menu item: \(itemName)")
            }
            
            // Click the menu item
            AXUIElementPerformAction(item, kAXPressAction as CFString)
            
            // Get submenu if not the last item
            if itemName != menuItems.last {
                try await Task.sleep(nanoseconds: 200_000_000) // 0.2s
                
                var submenuValue: CFTypeRef?
                AXUIElementCopyAttributeValue(
                    item,
                    kAXMenuItemMarkCharAttribute as CFString,
                    &submenuValue
                )
                currentMenu = submenuValue as? AXUIElement
            }
        }
        
        return true
    }
    
    private func performWait(_ step: AutomationStep) async throws -> Bool {
        let duration = step.parameters["duration"] as? TimeInterval ?? 
                      TimeInterval(step.target ?? "1") ?? 1.0
        
        try await Task.sleep(nanoseconds: UInt64(duration * 1_000_000_000))
        return true
    }
    
    private func performScreenshot(_ step: AutomationStep) -> Bool {
        if let screenshot = captureScreen() {
            // Save to specified path if provided
            if let path = step.parameters["path"] as? String {
                let url = URL(fileURLWithPath: path)
                if let tiffData = screenshot.tiffRepresentation,
                   let bitmap = NSBitmapImageRep(data: tiffData),
                   let pngData = bitmap.representation(using: .png, properties: [:]) {
                    try? pngData.write(to: url)
                }
            }
            return true
        }
        return false
    }
    
    // MARK: - Helper Methods
    
    private func findElement(matching description: String) throws -> AXUIElement? {
        guard let app = currentApplication else {
            throw AutomationError.applicationNotFound("No active application")
        }
        
        let appElement = AXUIElementCreateApplication(app.processId)
        
        // Try different search strategies
        
        // 1. By title/label
        if let element = findElementByAttribute(
            in: appElement,
            attribute: kAXTitleAttribute as CFString,
            value: description
        ) {
            return element
        }
        
        // 2. By identifier
        if let element = findElementByAttribute(
            in: appElement,
            attribute: kAXIdentifierAttribute as CFString,
            value: description
        ) {
            return element
        }
        
        // 3. By role and title
        if description.contains(":") {
            let parts = description.split(separator: ":", maxSplits: 1)
            let role = String(parts[0])
            let title = String(parts[1])
            
            if let element = findElementByRoleAndTitle(
                in: appElement,
                role: role,
                title: title
            ) {
                return element
            }
        }
        
        return nil
    }
    
    private func findElementByAttribute(
        in parent: AXUIElement,
        attribute: CFString,
        value: String
    ) -> AXUIElement? {
        var children: CFTypeRef?
        let result = AXUIElementCopyAttributeValue(
            parent,
            kAXChildrenAttribute as CFString,
            &children
        )
        
        guard result == .success,
              let childArray = children as? [AXUIElement] else {
            return nil
        }
        
        for child in childArray {
            var attrValue: CFTypeRef?
            let attrResult = AXUIElementCopyAttributeValue(child, attribute, &attrValue)
            
            if attrResult == .success,
               let stringValue = attrValue as? String,
               stringValue == value {
                return child
            }
            
            // Recursive search
            if let found = findElementByAttribute(in: child, attribute: attribute, value: value) {
                return found
            }
        }
        
        return nil
    }
    
    private func findElementByRoleAndTitle(
        in parent: AXUIElement,
        role: String,
        title: String
    ) -> AXUIElement? {
        var children: CFTypeRef?
        let result = AXUIElementCopyAttributeValue(
            parent,
            kAXChildrenAttribute as CFString,
            &children
        )
        
        guard result == .success,
              let childArray = children as? [AXUIElement] else {
            return nil
        }
        
        for child in childArray {
            var roleValue: CFTypeRef?
            var titleValue: CFTypeRef?
            
            AXUIElementCopyAttributeValue(child, kAXRoleAttribute as CFString, &roleValue)
            AXUIElementCopyAttributeValue(child, kAXTitleAttribute as CFString, &titleValue)
            
            if let childRole = roleValue as? String,
               let childTitle = titleValue as? String,
               childRole == role && childTitle == title {
                return child
            }
            
            // Recursive search
            if let found = findElementByRoleAndTitle(in: child, role: role, title: title) {
                return found
            }
        }
        
        return nil
    }
    
    private func findMenuItem(in menu: AXUIElement, named name: String) throws -> AXUIElement? {
        var children: CFTypeRef?
        let result = AXUIElementCopyAttributeValue(
            menu,
            kAXChildrenAttribute as CFString,
            &children
        )
        
        guard result == .success,
              let items = children as? [AXUIElement] else {
            return nil
        }
        
        for item in items {
            var titleValue: CFTypeRef?
            AXUIElementCopyAttributeValue(
                item,
                kAXTitleAttribute as CFString,
                &titleValue
            )
            
            if let title = titleValue as? String,
               title == name {
                return item
            }
        }
        
        return nil
    }
    
    private func keyCodeForKey(_ key: String) -> Int? {
        // Map common keys to key codes
        let keyMap: [String: Int] = [
            "return": 0x24,
            "enter": 0x24,
            "tab": 0x30,
            "space": 0x31,
            "delete": 0x33,
            "escape": 0x35,
            "esc": 0x35,
            "command": 0x37,
            "cmd": 0x37,
            "shift": 0x38,
            "option": 0x3A,
            "alt": 0x3A,
            "control": 0x3B,
            "ctrl": 0x3B,
            "up": 0x7E,
            "down": 0x7D,
            "left": 0x7B,
            "right": 0x7C,
            "f1": 0x7A,
            "f2": 0x78,
            "f3": 0x63,
            "f4": 0x76,
            "f5": 0x60,
            "f6": 0x61,
            "f7": 0x62,
            "f8": 0x64,
            "f9": 0x65,
            "f10": 0x6D,
            "f11": 0x67,
            "f12": 0x6F
        ]
        
        return keyMap[key.lowercased()]
    }
    
    private func captureScreen() -> NSImage? {
        if let screen = NSScreen.main {
            let rect = screen.frame
            
            if let imageRef = CGWindowListCreateImage(
                rect,
                .optionOnScreenOnly,
                kCGNullWindowID,
                .bestResolution
            ) {
                return NSImage(cgImage: imageRef, size: rect.size)
            }
        }
        
        return nil
    }
    
    private func waitForCondition(
        _ condition: String,
        timeout: TimeInterval
    ) async throws {
        let startTime = Date()
        
        while Date().timeIntervalSince(startTime) < timeout {
            if try await verifyCondition(condition) {
                return
            }
            try await Task.sleep(nanoseconds: 100_000_000) // 0.1s
        }
        
        throw AutomationError.timeout
    }
    
    private func verifyCondition(_ condition: String) async throws -> Bool {
        // Simple element existence check
        if condition.hasPrefix("exists:") {
            let elementDesc = String(condition.dropFirst(7))
            return try findElement(matching: elementDesc) != nil
        }
        
        // Window title check
        if condition.hasPrefix("window:") {
            let expectedTitle = String(condition.dropFirst(7))
            if let app = currentApplication,
               let window = app.window {
                var titleValue: CFTypeRef?
                AXUIElementCopyAttributeValue(
                    window,
                    kAXTitleAttribute as CFString,
                    &titleValue
                )
                if let title = titleValue as? String {
                    return title.contains(expectedTitle)
                }
            }
        }
        
        // Default to true
        return true
    }
    
    private func executeAppleScript(_ step: AutomationStep) async throws -> Bool {
        guard let script = step.parameters["script"] as? String ?? step.target else {
            throw AutomationError.actionFailed("No script provided")
        }
        
        let appleScript = NSAppleScript(source: script)
        var error: NSDictionary?
        
        appleScript?.executeAndReturnError(&error)
        
        if let error = error {
            throw AutomationError.actionFailed("AppleScript error: \(error)")
        }
        
        return true
    }
    
    // MARK: - Public API
    
    func getExecutionHistory() -> [AutomationResult] {
        return executionHistory
    }
    
    func clearHistory() {
        executionHistory.removeAll()
    }
    
    func performDrag(_ step: AutomationStep) async throws -> Bool {
        // Implementation for drag operations
        // This would involve mouse down, move, and up events
        return true
    }
    
    func performScroll(_ step: AutomationStep) async throws -> Bool {
        let direction = step.parameters["direction"] as? String ?? "down"
        let amount = step.parameters["amount"] as? Int ?? 5
        
        // Create scroll events
        let scrollEvent = CGEvent(
            scrollWheelEvent2Source: nil,
            units: .line,
            wheelCount: 1,
            wheel1: Int32(direction == "down" ? -amount : amount),
            wheel2: 0,
            wheel3: 0
        )
        
        scrollEvent?.post(tap: .cghidEventTap)
        return true
    }
    
    func performFocus(_ step: AutomationStep) async throws -> Bool {
        guard let target = step.target,
              let element = try findElement(matching: target) else {
            throw AutomationError.elementNotFound(step.target ?? "")
        }
        
        AXUIElementSetAttributeValue(
            element,
            kAXFocusedAttribute as CFString,
            kCFBooleanTrue
        )
        
        return true
    }
}

// MARK: - Solution Capture

class SolutionCapture {
    /// Capture user actions as a solution
    static func startCapturing(
        for duration: TimeInterval = 60,
        completion: @escaping ([AutomationStep]) -> Void
    ) {
        // This would use event monitors to capture user actions
        // For now, returning a placeholder
        
        DispatchQueue.main.asyncAfter(deadline: .now() + duration) {
            completion([])
        }
    }
}

// MARK: - Solution Verification

extension SolutionAutomation {
    func verifySolution(
        steps: [AutomationStep],
        expectedOutcome: String
    ) async throws -> Bool {
        // Execute solution
        let results = try await executeSolution(steps: steps)
        
        // Check if all steps succeeded
        let allSucceeded = results.allSatisfy { $0.success }
        
        // Verify expected outcome
        let outcomeVerified = try await verifyCondition(expectedOutcome)
        
        return allSucceeded && outcomeVerified
    }
}