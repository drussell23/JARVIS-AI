import Foundation
import Security
import LocalAuthentication
import ServiceManagement

/// Comprehensive security and permission management for system operations
public class SecurityManager {
    
    // MARK: - Types
    
    public enum Permission {
        case accessibility
        case fullDiskAccess
        case automation(bundleIdentifier: String)
        case screenRecording
        case microphone
        case camera
        case contacts
        case calendars
        case reminders
        case photos
        case location
        case systemEvents
        
        var description: String {
            switch self {
            case .accessibility:
                return "Accessibility access to control system UI"
            case .fullDiskAccess:
                return "Full Disk Access to read all files"
            case .automation(let app):
                return "Automation permission for \(app)"
            case .screenRecording:
                return "Screen recording permission"
            case .microphone:
                return "Microphone access"
            case .camera:
                return "Camera access"
            case .contacts:
                return "Contacts access"
            case .calendars:
                return "Calendar access"
            case .reminders:
                return "Reminders access"
            case .photos:
                return "Photos library access"
            case .location:
                return "Location services"
            case .systemEvents:
                return "System Events automation"
            }
        }
    }
    
    public struct PermissionStatus {
        public let permission: Permission
        public let isGranted: Bool
        public let requiresSystemPrompt: Bool
        public let canRequestAccess: Bool
        public let instructions: String?
    }
    
    public struct SecurityPolicy {
        public let requireAuthentication: Bool
        public let requireConfirmation: Bool
        public let allowedOperations: Set<String>
        public let deniedOperations: Set<String>
        public let auditingEnabled: Bool
        public let maxRetries: Int
        
        public static let `default` = SecurityPolicy(
            requireAuthentication: false,
            requireConfirmation: true,
            allowedOperations: [],
            deniedOperations: [],
            auditingEnabled: true,
            maxRetries: 3
        )
        
        public static let strict = SecurityPolicy(
            requireAuthentication: true,
            requireConfirmation: true,
            allowedOperations: [],
            deniedOperations: [],
            auditingEnabled: true,
            maxRetries: 1
        )
    }
    
    public struct AuthenticationResult {
        public let success: Bool
        public let method: AuthenticationMethod
        public let error: Error?
        public let timestamp: Date
    }
    
    public enum AuthenticationMethod {
        case none
        case password
        case biometric
        case systemDialog
    }
    
    // MARK: - Properties
    
    private static let shared = SecurityManager()
    private var policy: SecurityPolicy = .default
    private let authContext = LAContext()
    private let auditLogger = AuditLogger()
    private var permissionCache: [String: PermissionStatus] = [:]
    private let cacheQueue = DispatchQueue(label: "com.jarvis.security.cache", attributes: .concurrent)
    
    // MARK: - Permission Checking
    
    public static func checkPermission(_ permission: Permission) -> PermissionStatus {
        // Check cache first
        let cacheKey = permission.cacheKey
        if let cached = shared.getCachedPermission(cacheKey) {
            return cached
        }
        
        let status: PermissionStatus
        
        switch permission {
        case .accessibility:
            status = checkAccessibilityPermission()
            
        case .fullDiskAccess:
            status = checkFullDiskAccess()
            
        case .automation(let bundleId):
            status = checkAutomationPermission(for: bundleId)
            
        case .screenRecording:
            status = checkScreenRecordingPermission()
            
        case .microphone:
            status = checkMicrophonePermission()
            
        case .camera:
            status = checkCameraPermission()
            
        case .contacts:
            status = checkContactsPermission()
            
        case .calendars:
            status = checkCalendarPermission()
            
        case .reminders:
            status = checkRemindersPermission()
            
        case .photos:
            status = checkPhotosPermission()
            
        case .location:
            status = checkLocationPermission()
            
        case .systemEvents:
            status = checkSystemEventsPermission()
        }
        
        // Cache the result
        shared.setCachedPermission(cacheKey, status: status)
        
        return status
    }
    
    // MARK: - Permission Requests
    
    public static func requestPermission(_ permission: Permission, completion: @escaping (Bool) -> Void) {
        switch permission {
        case .accessibility:
            requestAccessibilityPermission()
            completion(false) // User must grant manually
            
        case .fullDiskAccess:
            openFullDiskAccessPreferences()
            completion(false) // User must grant manually
            
        case .automation(let bundleId):
            requestAutomationPermission(for: bundleId, completion: completion)
            
        case .screenRecording:
            requestScreenRecordingPermission(completion: completion)
            
        case .microphone:
            requestMicrophonePermission(completion: completion)
            
        case .camera:
            requestCameraPermission(completion: completion)
            
        case .contacts:
            requestContactsPermission(completion: completion)
            
        case .calendars:
            requestCalendarPermission(completion: completion)
            
        case .reminders:
            requestRemindersPermission(completion: completion)
            
        case .photos:
            requestPhotosPermission(completion: completion)
            
        case .location:
            requestLocationPermission(completion: completion)
            
        case .systemEvents:
            requestSystemEventsPermission(completion: completion)
        }
    }
    
    // MARK: - Authentication
    
    public static func authenticate(reason: String, completion: @escaping (AuthenticationResult) -> Void) {
        let context = LAContext()
        var error: NSError?
        
        // Check if biometric authentication is available
        if context.canEvaluatePolicy(.deviceOwnerAuthenticationWithBiometrics, error: &error) {
            // Use biometric authentication
            context.evaluatePolicy(.deviceOwnerAuthenticationWithBiometrics, localizedReason: reason) { success, error in
                DispatchQueue.main.async {
                    completion(AuthenticationResult(
                        success: success,
                        method: .biometric,
                        error: error,
                        timestamp: Date()
                    ))
                }
            }
        } else if context.canEvaluatePolicy(.deviceOwnerAuthentication, error: &error) {
            // Fall back to password
            context.evaluatePolicy(.deviceOwnerAuthentication, localizedReason: reason) { success, error in
                DispatchQueue.main.async {
                    completion(AuthenticationResult(
                        success: success,
                        method: .password,
                        error: error,
                        timestamp: Date()
                    ))
                }
            }
        } else {
            // No authentication available
            completion(AuthenticationResult(
                success: true,
                method: .none,
                error: nil,
                timestamp: Date()
            ))
        }
    }
    
    // MARK: - Policy Management
    
    public static func setSecurityPolicy(_ policy: SecurityPolicy) {
        shared.policy = policy
    }
    
    public static func getSecurityPolicy() -> SecurityPolicy {
        return shared.policy
    }
    
    public static func validateOperation(_ operation: String) -> Bool {
        let policy = shared.policy
        
        // Check denied list first
        if policy.deniedOperations.contains(operation) {
            return false
        }
        
        // If allowed list is empty, allow all non-denied
        if policy.allowedOperations.isEmpty {
            return true
        }
        
        // Check allowed list
        return policy.allowedOperations.contains(operation)
    }
    
    // MARK: - Specific Permission Implementations
    
    private static func checkAccessibilityPermission() -> PermissionStatus {
        let trusted = AXIsProcessTrusted()
        
        return PermissionStatus(
            permission: .accessibility,
            isGranted: trusted,
            requiresSystemPrompt: false,
            canRequestAccess: true,
            instructions: trusted ? nil : "Grant access in System Preferences > Security & Privacy > Privacy > Accessibility"
        )
    }
    
    private static func checkFullDiskAccess() -> PermissionStatus {
        // Check by attempting to read a protected directory
        let protectedPath = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Safari/Bookmarks.plist")
        
        let hasAccess = FileManager.default.isReadableFile(atPath: protectedPath.path)
        
        return PermissionStatus(
            permission: .fullDiskAccess,
            isGranted: hasAccess,
            requiresSystemPrompt: false,
            canRequestAccess: false,
            instructions: hasAccess ? nil : "Grant access in System Preferences > Security & Privacy > Privacy > Full Disk Access"
        )
    }
    
    private static func checkAutomationPermission(for bundleId: String) -> PermissionStatus {
        // This would require sending an Apple Event to check
        // For now, we'll return a basic status
        return PermissionStatus(
            permission: .automation(bundleIdentifier: bundleId),
            isGranted: false,
            requiresSystemPrompt: true,
            canRequestAccess: true,
            instructions: nil
        )
    }
    
    private static func checkScreenRecordingPermission() -> PermissionStatus {
        // Check by attempting to capture a small area
        if let _ = CGWindowListCreateImage(
            CGRect(x: 0, y: 0, width: 1, height: 1),
            .optionOnScreenOnly,
            kCGNullWindowID,
            [.bestResolution]
        ) {
            return PermissionStatus(
                permission: .screenRecording,
                isGranted: true,
                requiresSystemPrompt: false,
                canRequestAccess: false,
                instructions: nil
            )
        }
        
        return PermissionStatus(
            permission: .screenRecording,
            isGranted: false,
            requiresSystemPrompt: false,
            canRequestAccess: false,
            instructions: "Grant access in System Preferences > Security & Privacy > Privacy > Screen Recording"
        )
    }
    
    private static func checkMicrophonePermission() -> PermissionStatus {
        if #available(macOS 10.14, *) {
            switch AVCaptureDevice.authorizationStatus(for: .audio) {
            case .authorized:
                return PermissionStatus(
                    permission: .microphone,
                    isGranted: true,
                    requiresSystemPrompt: false,
                    canRequestAccess: false,
                    instructions: nil
                )
            case .denied, .restricted:
                return PermissionStatus(
                    permission: .microphone,
                    isGranted: false,
                    requiresSystemPrompt: false,
                    canRequestAccess: false,
                    instructions: "Grant access in System Preferences > Security & Privacy > Privacy > Microphone"
                )
            case .notDetermined:
                return PermissionStatus(
                    permission: .microphone,
                    isGranted: false,
                    requiresSystemPrompt: true,
                    canRequestAccess: true,
                    instructions: nil
                )
            @unknown default:
                return PermissionStatus(
                    permission: .microphone,
                    isGranted: false,
                    requiresSystemPrompt: false,
                    canRequestAccess: false,
                    instructions: "Unknown permission status"
                )
            }
        }
        
        return PermissionStatus(
            permission: .microphone,
            isGranted: true,
            requiresSystemPrompt: false,
            canRequestAccess: false,
            instructions: nil
        )
    }
    
    private static func checkCameraPermission() -> PermissionStatus {
        if #available(macOS 10.14, *) {
            switch AVCaptureDevice.authorizationStatus(for: .video) {
            case .authorized:
                return PermissionStatus(
                    permission: .camera,
                    isGranted: true,
                    requiresSystemPrompt: false,
                    canRequestAccess: false,
                    instructions: nil
                )
            case .denied, .restricted:
                return PermissionStatus(
                    permission: .camera,
                    isGranted: false,
                    requiresSystemPrompt: false,
                    canRequestAccess: false,
                    instructions: "Grant access in System Preferences > Security & Privacy > Privacy > Camera"
                )
            case .notDetermined:
                return PermissionStatus(
                    permission: .camera,
                    isGranted: false,
                    requiresSystemPrompt: true,
                    canRequestAccess: true,
                    instructions: nil
                )
            @unknown default:
                return PermissionStatus(
                    permission: .camera,
                    isGranted: false,
                    requiresSystemPrompt: false,
                    canRequestAccess: false,
                    instructions: "Unknown permission status"
                )
            }
        }
        
        return PermissionStatus(
            permission: .camera,
            isGranted: true,
            requiresSystemPrompt: false,
            canRequestAccess: false,
            instructions: nil
        )
    }
    
    // MARK: - Permission Request Implementations
    
    private static func requestAccessibilityPermission() {
        let options = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true] as CFDictionary
        AXIsProcessTrustedWithOptions(options)
    }
    
    private static func openFullDiskAccessPreferences() {
        let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_AllFiles")!
        NSWorkspace.shared.open(url)
    }
    
    private static func requestMicrophonePermission(completion: @escaping (Bool) -> Void) {
        if #available(macOS 10.14, *) {
            AVCaptureDevice.requestAccess(for: .audio) { granted in
                DispatchQueue.main.async {
                    completion(granted)
                }
            }
        } else {
            completion(true)
        }
    }
    
    private static func requestCameraPermission(completion: @escaping (Bool) -> Void) {
        if #available(macOS 10.14, *) {
            AVCaptureDevice.requestAccess(for: .video) { granted in
                DispatchQueue.main.async {
                    completion(granted)
                }
            }
        } else {
            completion(true)
        }
    }
    
    private static func requestScreenRecordingPermission(completion: @escaping (Bool) -> Void) {
        // Screen recording permission can't be requested programmatically
        // Open System Preferences
        let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture")!
        NSWorkspace.shared.open(url)
        completion(false)
    }
    
    // Placeholder implementations for other permissions
    private static func checkContactsPermission() -> PermissionStatus {
        return PermissionStatus(
            permission: .contacts,
            isGranted: false,
            requiresSystemPrompt: true,
            canRequestAccess: true,
            instructions: nil
        )
    }
    
    private static func checkCalendarPermission() -> PermissionStatus {
        return PermissionStatus(
            permission: .calendars,
            isGranted: false,
            requiresSystemPrompt: true,
            canRequestAccess: true,
            instructions: nil
        )
    }
    
    private static func checkRemindersPermission() -> PermissionStatus {
        return PermissionStatus(
            permission: .reminders,
            isGranted: false,
            requiresSystemPrompt: true,
            canRequestAccess: true,
            instructions: nil
        )
    }
    
    private static func checkPhotosPermission() -> PermissionStatus {
        return PermissionStatus(
            permission: .photos,
            isGranted: false,
            requiresSystemPrompt: true,
            canRequestAccess: true,
            instructions: nil
        )
    }
    
    private static func checkLocationPermission() -> PermissionStatus {
        return PermissionStatus(
            permission: .location,
            isGranted: false,
            requiresSystemPrompt: true,
            canRequestAccess: true,
            instructions: nil
        )
    }
    
    private static func checkSystemEventsPermission() -> PermissionStatus {
        return checkAutomationPermission(for: "com.apple.systemevents")
    }
    
    private static func requestAutomationPermission(for bundleId: String, completion: @escaping (Bool) -> Void) {
        // This would trigger the automation permission dialog
        completion(false)
    }
    
    private static func requestContactsPermission(completion: @escaping (Bool) -> Void) {
        completion(false)
    }
    
    private static func requestCalendarPermission(completion: @escaping (Bool) -> Void) {
        completion(false)
    }
    
    private static func requestRemindersPermission(completion: @escaping (Bool) -> Void) {
        completion(false)
    }
    
    private static func requestPhotosPermission(completion: @escaping (Bool) -> Void) {
        completion(false)
    }
    
    private static func requestLocationPermission(completion: @escaping (Bool) -> Void) {
        completion(false)
    }
    
    private static func requestSystemEventsPermission(completion: @escaping (Bool) -> Void) {
        requestAutomationPermission(for: "com.apple.systemevents", completion: completion)
    }
    
    // MARK: - Cache Management
    
    private func getCachedPermission(_ key: String) -> PermissionStatus? {
        return cacheQueue.sync {
            permissionCache[key]
        }
    }
    
    private func setCachedPermission(_ key: String, status: PermissionStatus) {
        cacheQueue.async(flags: .barrier) {
            self.permissionCache[key] = status
        }
    }
    
    public static func clearPermissionCache() {
        shared.cacheQueue.async(flags: .barrier) {
            shared.permissionCache.removeAll()
        }
    }
}

// MARK: - Extensions

extension SecurityManager.Permission {
    var cacheKey: String {
        switch self {
        case .accessibility:
            return "accessibility"
        case .fullDiskAccess:
            return "fullDiskAccess"
        case .automation(let bundleId):
            return "automation_\(bundleId)"
        case .screenRecording:
            return "screenRecording"
        case .microphone:
            return "microphone"
        case .camera:
            return "camera"
        case .contacts:
            return "contacts"
        case .calendars:
            return "calendars"
        case .reminders:
            return "reminders"
        case .photos:
            return "photos"
        case .location:
            return "location"
        case .systemEvents:
            return "systemEvents"
        }
    }
}

// MARK: - Required imports

import AVFoundation