import Foundation
import AppKit
import CoreServices
import IOKit
import IOKit.audio
import Network
import Security

/// Comprehensive System Control module for JARVIS
/// Provides secure, robust system operations with permission management
public class SystemControl {
    
    // MARK: - Types
    
    public enum SystemOperation {
        case appLifecycle(AppOperation)
        case systemPreference(PreferenceOperation)
        case fileSystem(FileOperation)
        case clipboard(ClipboardOperation)
        
        var requiresConfirmation: Bool {
            switch self {
            case .appLifecycle(let op):
                return op.isDestructive
            case .systemPreference(let op):
                return op.isSystemWide
            case .fileSystem(let op):
                return op.isDestructive
            case .clipboard:
                return false
            }
        }
    }
    
    public enum AppOperation {
        case launch(bundleIdentifier: String)
        case close(bundleIdentifier: String, force: Bool = false)
        case minimize(bundleIdentifier: String)
        case maximize(bundleIdentifier: String)
        case hide(bundleIdentifier: String)
        case switchTo(bundleIdentifier: String)
        case listRunning
        case getInfo(bundleIdentifier: String)
        
        var isDestructive: Bool {
            switch self {
            case .close(_, let force):
                return force
            default:
                return false
            }
        }
    }
    
    public enum PreferenceOperation {
        case setVolume(Float)
        case setBrightness(Float)
        case toggleWiFi(Bool)
        case toggleBluetooth(Bool)
        case setDoNotDisturb(Bool)
        case getDarkMode
        case setDarkMode(Bool)
        case getSystemInfo
        
        var isSystemWide: Bool {
            switch self {
            case .toggleWiFi, .toggleBluetooth, .setDoNotDisturb:
                return true
            default:
                return false
            }
        }
    }
    
    public enum FileOperation {
        case copy(from: URL, to: URL)
        case move(from: URL, to: URL)
        case delete(URL)
        case createDirectory(URL)
        case search(query: String, in: URL)
        case getInfo(URL)
        case setPermissions(URL, permissions: Int)
        
        var isDestructive: Bool {
            switch self {
            case .move, .delete, .setPermissions:
                return true
            default:
                return false
            }
        }
    }
    
    public enum ClipboardOperation {
        case read
        case write(String)
        case clear
        case getHistory
    }
    
    public struct OperationResult {
        public let success: Bool
        public let operation: SystemOperation
        public let result: Any?
        public let error: SystemControlError?
        public let executionTime: TimeInterval
        public let auditLog: AuditLogEntry
    }
    
    public struct AuditLogEntry {
        public let timestamp: Date
        public let operation: String
        public let user: String
        public let result: Bool
        public let details: [String: Any]
    }
    
    public enum SystemControlError: Error, LocalizedError {
        case permissionDenied(String)
        case operationFailed(String)
        case invalidParameter(String)
        case timeout(String)
        case notFound(String)
        case securityViolation(String)
        
        public var errorDescription: String? {
            switch self {
            case .permissionDenied(let detail):
                return "Permission denied: \(detail)"
            case .operationFailed(let detail):
                return "Operation failed: \(detail)"
            case .invalidParameter(let detail):
                return "Invalid parameter: \(detail)"
            case .timeout(let detail):
                return "Operation timed out: \(detail)"
            case .notFound(let detail):
                return "Not found: \(detail)"
            case .securityViolation(let detail):
                return "Security violation: \(detail)"
            }
        }
    }
    
    // MARK: - Properties
    
    private let permissionManager: PermissionManager
    private let errorHandler: ErrorHandler
    private let auditLogger: AuditLogger
    private let operationQueue: OperationQueue
    private let retryPolicy: RetryPolicy
    
    // MARK: - Initialization
    
    public init() {
        self.permissionManager = PermissionManager()
        self.errorHandler = ErrorHandler()
        self.auditLogger = AuditLogger()
        self.operationQueue = OperationQueue()
        self.operationQueue.name = "com.jarvis.systemcontrol"
        self.operationQueue.maxConcurrentOperationCount = 4
        self.retryPolicy = RetryPolicy()
    }
    
    // MARK: - Public Methods
    
    /// Execute a system operation with permission checking and error handling
    public func execute(_ operation: SystemOperation) async throws -> OperationResult {
        let startTime = Date()
        
        // Check permissions
        guard await permissionManager.checkPermission(for: operation) else {
            let error = SystemControlError.permissionDenied("Insufficient permissions for \(operation)")
            logOperation(operation, success: false, error: error, duration: 0)
            throw error
        }
        
        // Execute with retry and error handling
        do {
            let result = try await retryPolicy.execute {
                try await self.performOperation(operation)
            }
            
            let duration = Date().timeIntervalSince(startTime)
            let auditEntry = logOperation(operation, success: true, error: nil, duration: duration)
            
            return OperationResult(
                success: true,
                operation: operation,
                result: result,
                error: nil,
                executionTime: duration,
                auditLog: auditEntry
            )
        } catch {
            let duration = Date().timeIntervalSince(startTime)
            let controlError = errorHandler.handle(error, for: operation)
            let auditEntry = logOperation(operation, success: false, error: controlError, duration: duration)
            
            throw controlError
        }
    }
    
    // MARK: - App Lifecycle Management
    
    private func performAppOperation(_ operation: AppOperation) async throws -> Any? {
        switch operation {
        case .launch(let bundleId):
            return try launchApp(bundleId)
            
        case .close(let bundleId, let force):
            return try closeApp(bundleId, force: force)
            
        case .minimize(let bundleId):
            return try minimizeApp(bundleId)
            
        case .maximize(let bundleId):
            return try maximizeApp(bundleId)
            
        case .hide(let bundleId):
            return try hideApp(bundleId)
            
        case .switchTo(let bundleId):
            return try switchToApp(bundleId)
            
        case .listRunning:
            return getRunningApps()
            
        case .getInfo(let bundleId):
            return try getAppInfo(bundleId)
        }
    }
    
    private func launchApp(_ bundleIdentifier: String) throws -> Bool {
        guard let url = NSWorkspace.shared.urlForApplication(withBundleIdentifier: bundleIdentifier) else {
            throw SystemControlError.notFound("Application not found: \(bundleIdentifier)")
        }
        
        let configuration = NSWorkspace.OpenConfiguration()
        configuration.activates = true
        
        let semaphore = DispatchSemaphore(value: 0)
        var launchError: Error?
        
        NSWorkspace.shared.openApplication(at: url, configuration: configuration) { app, error in
            launchError = error
            semaphore.signal()
        }
        
        semaphore.wait()
        
        if let error = launchError {
            throw SystemControlError.operationFailed("Failed to launch app: \(error.localizedDescription)")
        }
        
        return true
    }
    
    private func closeApp(_ bundleIdentifier: String, force: Bool) throws -> Bool {
        let apps = NSWorkspace.shared.runningApplications.filter { $0.bundleIdentifier == bundleIdentifier }
        
        guard !apps.isEmpty else {
            throw SystemControlError.notFound("App not running: \(bundleIdentifier)")
        }
        
        for app in apps {
            let terminated = force ? app.forceTerminate() : app.terminate()
            if !terminated {
                throw SystemControlError.operationFailed("Failed to close app: \(bundleIdentifier)")
            }
        }
        
        return true
    }
    
    private func minimizeApp(_ bundleIdentifier: String) throws -> Bool {
        guard let app = NSWorkspace.shared.runningApplications.first(where: { $0.bundleIdentifier == bundleIdentifier }) else {
            throw SystemControlError.notFound("App not running: \(bundleIdentifier)")
        }
        
        app.hide()
        return true
    }
    
    private func maximizeApp(_ bundleIdentifier: String) throws -> Bool {
        guard let app = NSWorkspace.shared.runningApplications.first(where: { $0.bundleIdentifier == bundleIdentifier }) else {
            throw SystemControlError.notFound("App not running: \(bundleIdentifier)")
        }
        
        app.unhide()
        app.activate(options: .activateIgnoringOtherApps)
        return true
    }
    
    private func hideApp(_ bundleIdentifier: String) throws -> Bool {
        guard let app = NSWorkspace.shared.runningApplications.first(where: { $0.bundleIdentifier == bundleIdentifier }) else {
            throw SystemControlError.notFound("App not running: \(bundleIdentifier)")
        }
        
        app.hide()
        return true
    }
    
    private func switchToApp(_ bundleIdentifier: String) throws -> Bool {
        guard let app = NSWorkspace.shared.runningApplications.first(where: { $0.bundleIdentifier == bundleIdentifier }) else {
            throw SystemControlError.notFound("App not running: \(bundleIdentifier)")
        }
        
        app.activate(options: [.activateAllWindows, .activateIgnoringOtherApps])
        return true
    }
    
    private func getRunningApps() -> [[String: Any]] {
        return NSWorkspace.shared.runningApplications
            .filter { $0.activationPolicy == .regular }
            .map { app in
                [
                    "name": app.localizedName ?? "Unknown",
                    "bundleIdentifier": app.bundleIdentifier ?? "",
                    "processIdentifier": app.processIdentifier,
                    "isActive": app.isActive,
                    "isHidden": app.isHidden
                ]
            }
    }
    
    private func getAppInfo(_ bundleIdentifier: String) throws -> [String: Any] {
        guard let url = NSWorkspace.shared.urlForApplication(withBundleIdentifier: bundleIdentifier) else {
            throw SystemControlError.notFound("Application not found: \(bundleIdentifier)")
        }
        
        let bundle = Bundle(url: url)
        let running = NSWorkspace.shared.runningApplications.first { $0.bundleIdentifier == bundleIdentifier }
        
        return [
            "bundleIdentifier": bundleIdentifier,
            "name": bundle?.object(forInfoDictionaryKey: "CFBundleName") ?? "Unknown",
            "version": bundle?.object(forInfoDictionaryKey: "CFBundleShortVersionString") ?? "Unknown",
            "path": url.path,
            "isRunning": running != nil,
            "processId": running?.processIdentifier ?? -1
        ]
    }
    
    // MARK: - System Preferences
    
    private func performPreferenceOperation(_ operation: PreferenceOperation) async throws -> Any? {
        switch operation {
        case .setVolume(let level):
            return try setSystemVolume(level)
            
        case .setBrightness(let level):
            return try setDisplayBrightness(level)
            
        case .toggleWiFi(let enabled):
            return try toggleWiFi(enabled)
            
        case .toggleBluetooth(let enabled):
            return try toggleBluetooth(enabled)
            
        case .setDoNotDisturb(let enabled):
            return try setDoNotDisturb(enabled)
            
        case .getDarkMode:
            return isDarkModeEnabled()
            
        case .setDarkMode(let enabled):
            return try setDarkMode(enabled)
            
        case .getSystemInfo:
            return getSystemInfo()
        }
    }
    
    private func setSystemVolume(_ level: Float) throws -> Bool {
        // Implementation using CoreAudio
        // This is a simplified version - full implementation would use AudioHardware APIs
        let script = "set volume output volume \(Int(level * 100))"
        return try executeAppleScript(script)
    }
    
    private func setDisplayBrightness(_ level: Float) throws -> Bool {
        // Implementation using IOKit
        // This requires additional entitlements in production
        let script = "tell application \"System Events\" to set brightness to \(level)"
        return try executeAppleScript(script)
    }
    
    private func toggleWiFi(_ enabled: Bool) throws -> Bool {
        let script = """
        do shell script "networksetup -setairportpower en0 \(enabled ? "on" : "off")" with administrator privileges
        """
        return try executeAppleScript(script)
    }
    
    private func toggleBluetooth(_ enabled: Bool) throws -> Bool {
        // This would require BluetoothManager framework
        throw SystemControlError.operationFailed("Bluetooth control not implemented")
    }
    
    private func setDoNotDisturb(_ enabled: Bool) throws -> Bool {
        // This requires notification center manipulation
        throw SystemControlError.operationFailed("DND control not implemented")
    }
    
    private func isDarkModeEnabled() -> Bool {
        return NSApp.appearance?.name == .darkAqua
    }
    
    private func setDarkMode(_ enabled: Bool) throws -> Bool {
        let script = """
        tell application "System Events"
            tell appearance preferences
                set dark mode to \(enabled ? "true" : "false")
            end tell
        end tell
        """
        return try executeAppleScript(script)
    }
    
    private func getSystemInfo() -> [String: Any] {
        let processInfo = ProcessInfo.processInfo
        return [
            "macOSVersion": processInfo.operatingSystemVersionString,
            "processorCount": processInfo.processorCount,
            "physicalMemory": processInfo.physicalMemory,
            "systemUptime": processInfo.systemUptime,
            "darkMode": isDarkModeEnabled()
        ]
    }
    
    // MARK: - File Operations
    
    private func performFileOperation(_ operation: FileOperation) async throws -> Any? {
        switch operation {
        case .copy(let from, let to):
            return try copyFile(from: from, to: to)
            
        case .move(let from, let to):
            return try moveFile(from: from, to: to)
            
        case .delete(let url):
            return try deleteFile(url)
            
        case .createDirectory(let url):
            return try createDirectory(url)
            
        case .search(let query, let inURL):
            return try searchFiles(query: query, in: inURL)
            
        case .getInfo(let url):
            return try getFileInfo(url)
            
        case .setPermissions(let url, let permissions):
            return try setFilePermissions(url, permissions: permissions)
        }
    }
    
    private func copyFile(from source: URL, to destination: URL) throws -> Bool {
        try FileManager.default.copyItem(at: source, to: destination)
        return true
    }
    
    private func moveFile(from source: URL, to destination: URL) throws -> Bool {
        try FileManager.default.moveItem(at: source, to: destination)
        return true
    }
    
    private func deleteFile(_ url: URL) throws -> Bool {
        try FileManager.default.trashItem(at: url, resultingItemURL: nil)
        return true
    }
    
    private func createDirectory(_ url: URL) throws -> Bool {
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return true
    }
    
    private func searchFiles(query: String, in directory: URL) throws -> [URL] {
        let enumerator = FileManager.default.enumerator(at: directory,
                                                       includingPropertiesForKeys: [.nameKey],
                                                       options: [.skipsHiddenFiles])
        
        var results: [URL] = []
        while let url = enumerator?.nextObject() as? URL {
            if url.lastPathComponent.localizedCaseInsensitiveContains(query) {
                results.append(url)
            }
        }
        
        return results
    }
    
    private func getFileInfo(_ url: URL) throws -> [String: Any] {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        return [
            "path": url.path,
            "name": url.lastPathComponent,
            "size": attributes[.size] ?? 0,
            "creationDate": attributes[.creationDate] ?? Date(),
            "modificationDate": attributes[.modificationDate] ?? Date(),
            "isDirectory": (attributes[.type] as? FileAttributeType) == .typeDirectory
        ]
    }
    
    private func setFilePermissions(_ url: URL, permissions: Int) throws -> Bool {
        try FileManager.default.setAttributes([.posixPermissions: permissions], ofItemAtPath: url.path)
        return true
    }
    
    // MARK: - Clipboard Operations
    
    private func performClipboardOperation(_ operation: ClipboardOperation) async throws -> Any? {
        switch operation {
        case .read:
            return readClipboard()
            
        case .write(let text):
            return writeClipboard(text)
            
        case .clear:
            return clearClipboard()
            
        case .getHistory:
            return getClipboardHistory()
        }
    }
    
    private func readClipboard() -> String? {
        return NSPasteboard.general.string(forType: .string)
    }
    
    private func writeClipboard(_ text: String) -> Bool {
        NSPasteboard.general.clearContents()
        return NSPasteboard.general.setString(text, forType: .string)
    }
    
    private func clearClipboard() -> Bool {
        NSPasteboard.general.clearContents()
        return true
    }
    
    private func getClipboardHistory() -> [String] {
        // This would require maintaining a clipboard history
        // For now, return current clipboard only
        if let current = readClipboard() {
            return [current]
        }
        return []
    }
    
    // MARK: - Helper Methods
    
    private func performOperation(_ operation: SystemOperation) async throws -> Any? {
        switch operation {
        case .appLifecycle(let appOp):
            return try await performAppOperation(appOp)
            
        case .systemPreference(let prefOp):
            return try await performPreferenceOperation(prefOp)
            
        case .fileSystem(let fileOp):
            return try await performFileOperation(fileOp)
            
        case .clipboard(let clipOp):
            return try await performClipboardOperation(clipOp)
        }
    }
    
    private func executeAppleScript(_ script: String) throws -> Bool {
        let appleScript = NSAppleScript(source: script)
        var error: NSDictionary?
        
        appleScript?.executeAndReturnError(&error)
        
        if let error = error {
            throw SystemControlError.operationFailed("AppleScript error: \(error)")
        }
        
        return true
    }
    
    private func logOperation(_ operation: SystemOperation, success: Bool, error: Error?, duration: TimeInterval) -> AuditLogEntry {
        let entry = AuditLogEntry(
            timestamp: Date(),
            operation: String(describing: operation),
            user: NSUserName(),
            result: success,
            details: [
                "duration": duration,
                "error": error?.localizedDescription ?? "none"
            ]
        )
        
        auditLogger.log(entry)
        return entry
    }
}

// MARK: - Supporting Components

class PermissionManager {
    func checkPermission(for operation: SystemOperation) async -> Bool {
        // Check if operation requires user confirmation
        if operation.requiresConfirmation {
            // In production, this would show a confirmation dialog
            // For now, we'll assume permission is granted
            return true
        }
        
        // Check system permissions
        switch operation {
        case .fileSystem:
            return checkFileSystemPermission()
        case .systemPreference:
            return checkSystemPreferencePermission()
        default:
            return true
        }
    }
    
    private func checkFileSystemPermission() -> Bool {
        // Check if we have file system access
        return true // Simplified for now
    }
    
    private func checkSystemPreferencePermission() -> Bool {
        // Check if we have system preference access
        return true // Simplified for now
    }
}

class ErrorHandler {
    func handle(_ error: Error, for operation: SystemOperation) -> SystemControlError {
        // Convert generic errors to SystemControlError
        if let controlError = error as? SystemControlError {
            return controlError
        }
        
        // Map common errors
        if (error as NSError).code == NSFileNoSuchFileError {
            return .notFound("File not found")
        }
        
        if (error as NSError).code == NSFileWriteNoPermissionError {
            return .permissionDenied("No write permission")
        }
        
        return .operationFailed(error.localizedDescription)
    }
}

class AuditLogger {
    private let logQueue = DispatchQueue(label: "com.jarvis.audit", attributes: .concurrent)
    private var logs: [AuditLogEntry] = []
    
    func log(_ entry: AuditLogEntry) {
        logQueue.async(flags: .barrier) {
            self.logs.append(entry)
            // In production, this would persist to disk
        }
    }
    
    func getLogs() -> [AuditLogEntry] {
        logQueue.sync {
            return logs
        }
    }
}

class RetryPolicy {
    private let maxRetries = 3
    private let baseDelay: TimeInterval = 0.1
    
    func execute<T>(_ operation: () async throws -> T) async throws -> T {
        var lastError: Error?
        
        for attempt in 0..<maxRetries {
            do {
                return try await operation()
            } catch {
                lastError = error
                
                // Don't retry certain errors
                if let controlError = error as? SystemControlError {
                    switch controlError {
                    case .permissionDenied, .securityViolation, .invalidParameter:
                        throw error
                    default:
                        break
                    }
                }
                
                // Exponential backoff
                if attempt < maxRetries - 1 {
                    let delay = baseDelay * pow(2.0, Double(attempt))
                    try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
                }
            }
        }
        
        throw lastError ?? SystemControlError.operationFailed("Unknown error")
    }
}