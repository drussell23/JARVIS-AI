#!/usr/bin/swift
//
// WindowIntelligenceService.swift
// Enhanced window metadata collection for JARVIS multi-space awareness
//

import Foundation
import AppKit
import CoreGraphics
import Quartz

// MARK: - Data Models

struct SpaceIdentifier: Codable {
    let id: Int
    let uuid: String
    let displayID: Int
}

struct EnhancedWindowInfo: Codable {
    let windowID: Int
    let appName: String
    let windowTitle: String
    let processID: Int
    
    // Space information
    let spaceID: Int?
    let spaceUUID: String?
    
    // Position and state
    let bounds: CGRect
    let isMinimized: Bool
    let isFullscreen: Bool
    let isFocused: Bool
    let layer: Int
    let alpha: Float
    
    // Enhanced metadata
    let bundleIdentifier: String?
    let documentPath: String?
    let documentModified: Date?
    let windowCreated: Date?
    
    // Additional context
    let isOnActiveSpace: Bool
    let sharingState: String?  // "none", "camera", "screen"
    let memoryUsage: Int?
}

struct SpaceInfo: Codable {
    let spaceID: Int
    let spaceUUID: String
    let displayID: Int
    let isCurrentSpace: Bool
    let windowCount: Int
    let hasFocus: Bool
    let spaceType: String  // "normal", "fullscreen", "dashboard"
    let spaceIndex: Int    // Position in space list
}

struct WorkspaceSnapshot: Codable {
    let timestamp: Date
    let currentSpaceID: Int
    let spaces: [SpaceInfo]
    let windows: [EnhancedWindowInfo]
    let spaceWindowMap: [Int: [Int]]  // spaceID -> [windowIDs]
    let focusedWindowID: Int?
    let activeAppBundleID: String?
}

// MARK: - Main Service

class WindowIntelligenceService: NSObject {
    
    static let shared = WindowIntelligenceService()
    
    private var lastSnapshot: WorkspaceSnapshot?
    private let queue = DispatchQueue(label: "com.jarvis.window-intelligence", attributes: .concurrent)
    
    // MARK: - Public API
    
    @objc func getAllWindowsAcrossSpaces() -> Data? {
        let snapshot = gatherWorkspaceSnapshot()
        
        do {
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            return try encoder.encode(snapshot)
        } catch {
            print("Failed to encode workspace snapshot: \(error)")
            return nil
        }
    }
    
    @objc func getWindowThumbnails(windowIDs: [Int]) -> [Int: Data] {
        var thumbnails: [Int: Data] = [:]
        
        for windowID in windowIDs {
            if let thumbnail = captureWindowThumbnail(windowID: windowID) {
                thumbnails[windowID] = thumbnail
            }
        }
        
        return thumbnails
    }
    
    @objc func trackSpaceChanges(callback: @escaping (Int, Int) -> Void) {
        // Monitor space changes
        NSWorkspace.shared.notificationCenter.addObserver(
            self,
            selector: #selector(activeSpaceDidChange),
            name: NSWorkspace.activeSpaceDidChangeNotification,
            object: nil
        )
        
        // Store callback for notifications
        self.spaceChangeCallback = callback
    }
    
    // MARK: - Private Implementation
    
    private var spaceChangeCallback: ((Int, Int) -> Void)?
    private var currentSpaceID: Int = 1
    
    private func gatherWorkspaceSnapshot() -> WorkspaceSnapshot {
        // Get all windows
        let windowList = CGWindowListCopyWindowInfo(
            [.optionAll, .excludeDesktopElements],
            kCGNullWindowID
        ) as? [[String: Any]] ?? []
        
        // Get current space info
        let currentSpace = getCurrentSpaceInfo()
        
        // Get all spaces
        let spaces = getAllSpaces()
        
        // Process windows
        var windows: [EnhancedWindowInfo] = []
        var spaceWindowMap: [Int: [Int]] = [:]
        
        for windowDict in windowList {
            if let window = processWindow(windowDict, currentSpaceID: currentSpace.0) {
                windows.append(window)
                
                // Map to space
                let spaceID = window.spaceID ?? currentSpace.0
                if spaceWindowMap[spaceID] == nil {
                    spaceWindowMap[spaceID] = []
                }
                spaceWindowMap[spaceID]?.append(window.windowID)
            }
        }
        
        // Get focused window
        let focusedApp = NSWorkspace.shared.frontmostApplication
        let focusedWindowID = windows.first(where: { 
            $0.processID == focusedApp?.processIdentifier ?? -1 && $0.isFocused 
        })?.windowID
        
        return WorkspaceSnapshot(
            timestamp: Date(),
            currentSpaceID: currentSpace.0,
            spaces: spaces,
            windows: windows,
            spaceWindowMap: spaceWindowMap,
            focusedWindowID: focusedWindowID,
            activeAppBundleID: focusedApp?.bundleIdentifier
        )
    }
    
    private func processWindow(_ windowDict: [String: Any], currentSpaceID: Int) -> EnhancedWindowInfo? {
        // Extract basic info
        guard let windowID = windowDict["kCGWindowNumber"] as? Int,
              let appName = windowDict["kCGWindowOwnerName"] as? String,
              let processID = windowDict["kCGWindowOwnerPID"] as? Int,
              let boundsDict = windowDict["kCGWindowBounds"] as? [String: Any],
              let alpha = windowDict["kCGWindowAlpha"] as? Float else {
            return nil
        }
        
        // Skip system windows
        let skipApps = ["Window Server", "SystemUIServer", "Dock", "Control Center", 
                       "Notification Center", "Wallpaper", "ScreenSaverEngine"]
        if skipApps.contains(appName) || alpha == 0 {
            return nil
        }
        
        // Get bounds
        let bounds = CGRect(
            x: boundsDict["X"] as? CGFloat ?? 0,
            y: boundsDict["Y"] as? CGFloat ?? 0,
            width: boundsDict["Width"] as? CGFloat ?? 0,
            height: boundsDict["Height"] as? CGFloat ?? 0
        )
        
        // Skip tiny windows
        if bounds.width < 50 || bounds.height < 50 {
            return nil
        }
        
        // Get window properties
        let windowTitle = windowDict["kCGWindowName"] as? String ?? ""
        let layer = windowDict["kCGWindowLayer"] as? Int ?? 0
        let isOnscreen = windowDict["kCGWindowIsOnscreen"] as? Bool ?? false
        
        // Determine space (heuristic - not perfect)
        let isOnActiveSpace = isOnscreen && layer == 0
        let spaceID = isOnActiveSpace ? currentSpaceID : estimateSpaceID(windowDict)
        
        // Get app info
        let app = NSRunningApplication(processIdentifier: pid_t(processID))
        let bundleID = app?.bundleIdentifier
        
        // Check if minimized
        let isMinimized = !isOnscreen && windowDict["kCGWindowStoreType"] as? Int == 2
        
        // Check if fullscreen
        let screenBounds = NSScreen.main?.frame ?? .zero
        let isFullscreen = bounds.width >= screenBounds.width * 0.95 && 
                          bounds.height >= screenBounds.height * 0.95
        
        // Determine focus
        let isFocused = app == NSWorkspace.shared.frontmostApplication && layer == 0
        
        // Try to extract document info from title
        let documentPath = extractDocumentPath(from: windowTitle, appName: appName)
        
        return EnhancedWindowInfo(
            windowID: windowID,
            appName: appName,
            windowTitle: windowTitle,
            processID: processID,
            spaceID: spaceID,
            spaceUUID: nil,  // Would need private APIs
            bounds: bounds,
            isMinimized: isMinimized,
            isFullscreen: isFullscreen,
            isFocused: isFocused,
            layer: layer,
            alpha: alpha,
            bundleIdentifier: bundleID,
            documentPath: documentPath,
            documentModified: nil,
            windowCreated: nil,
            isOnActiveSpace: isOnActiveSpace,
            sharingState: detectSharingState(processID: processID),
            memoryUsage: getProcessMemory(processID: processID)
        )
    }
    
    private func getCurrentSpaceInfo() -> (Int, String) {
        // This is a simplified version - in reality you'd use private APIs
        // or more sophisticated detection
        return (1, "space-1")
    }
    
    private func getAllSpaces() -> [SpaceInfo] {
        // Simplified - would use private APIs for real implementation
        var spaces: [SpaceInfo] = []
        
        // Estimate based on Mission Control state
        let missionControlRunning = NSRunningApplication.runningApplications(
            withBundleIdentifier: "com.apple.dock"
        ).first?.isActive ?? false
        
        // Default to 4 spaces (common setup)
        for i in 1...4 {
            spaces.append(SpaceInfo(
                spaceID: i,
                spaceUUID: "space-\(i)",
                displayID: 0,
                isCurrentSpace: i == 1,  // Simplified
                windowCount: 0,  // Will be updated
                hasFocus: i == 1,
                spaceType: "normal",
                spaceIndex: i - 1
            ))
        }
        
        return spaces
    }
    
    private func estimateSpaceID(_ windowDict: [String: Any]) -> Int {
        // Heuristic to guess space based on window properties
        // In production, you'd use actual space detection
        
        let isOnscreen = windowDict["kCGWindowIsOnscreen"] as? Bool ?? false
        let layer = windowDict["kCGWindowLayer"] as? Int ?? 0
        
        if !isOnscreen && layer == 0 {
            // Likely on another space
            return 2  // Default to space 2
        }
        
        return 1
    }
    
    private func extractDocumentPath(from title: String, appName: String) -> String? {
        // Common patterns for document paths in window titles
        
        // VSCode pattern: "filename — folder"
        if appName.contains("Code") && title.contains("—") {
            let parts = title.split(separator: "—")
            if let firstPart = parts.first {
                return String(firstPart).trimmingCharacters(in: .whitespaces)
            }
        }
        
        // Terminal pattern: "bash — /path/to/directory"
        if appName == "Terminal" && title.contains("—") {
            let parts = title.split(separator: "—")
            if parts.count > 1 {
                return String(parts[1]).trimmingCharacters(in: .whitespaces)
            }
        }
        
        // Generic file path detection
        if title.contains("/") || title.contains(".") {
            // Might be a file path
            return title
        }
        
        return nil
    }
    
    private func captureWindowThumbnail(windowID: Int) -> Data? {
        // Create small thumbnail of window
        guard let windowImage = CGWindowListCreateImage(
            .null,
            .optionIncludingWindow,
            CGWindowID(windowID),
            [.boundsIgnoreFraming, .nominalResolution]
        ) else {
            return nil
        }
        
        // Convert to smaller thumbnail
        let maxSize: CGFloat = 200
        let scale = min(maxSize / CGFloat(windowImage.width), 
                       maxSize / CGFloat(windowImage.height))
        
        let thumbnailSize = CGSize(
            width: CGFloat(windowImage.width) * scale,
            height: CGFloat(windowImage.height) * scale
        )
        
        // Create NSImage and resize
        let nsImage = NSImage(cgImage: windowImage, size: thumbnailSize)
        
        // Convert to JPEG data
        guard let tiffData = nsImage.tiffRepresentation,
              let bitmap = NSBitmapImageRep(data: tiffData),
              let jpegData = bitmap.representation(using: .jpeg, properties: [.compressionFactor: 0.7]) else {
            return nil
        }
        
        return jpegData
    }
    
    private func detectSharingState(processID: Int) -> String {
        // Check if app is sharing screen or camera
        // This is a simplified check - real implementation would query system
        
        // Check for common video conferencing apps
        if let app = NSRunningApplication(processIdentifier: pid_t(processID)) {
            let videoApps = ["zoom.us", "Skype", "Microsoft Teams", "Slack", "Discord"]
            
            if let bundleID = app.bundleIdentifier,
               videoApps.contains(where: { bundleID.contains($0) }) {
                // Could be sharing - would need to check actual state
                return "possible"
            }
        }
        
        return "none"
    }
    
    private func getProcessMemory(processID: Int) -> Int? {
        // Get memory usage for process
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        
        var task: mach_port_t = 0
        let result = task_for_pid(mach_task_self_, pid_t(processID), &task)
        
        guard result == KERN_SUCCESS else {
            return nil
        }
        
        let infoResult = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(task, 
                         task_flavor_t(MACH_TASK_BASIC_INFO),
                         $0,
                         &count)
            }
        }
        
        guard infoResult == KERN_SUCCESS else {
            return nil
        }
        
        return Int(info.resident_size)
    }
    
    @objc private func activeSpaceDidChange() {
        // Handle space change notification
        let newSpace = getCurrentSpaceInfo()
        
        if newSpace.0 != currentSpaceID {
            spaceChangeCallback?(currentSpaceID, newSpace.0)
            currentSpaceID = newSpace.0
        }
    }
}

// MARK: - Python Bridge Functions

@_cdecl("get_all_windows_json") // Exposed to Python
public func getAllWindowsJSON() -> UnsafePointer<CChar>? {
    guard let data = WindowIntelligenceService.shared.getAllWindowsAcrossSpaces(),
          let string = String(data: data, encoding: .utf8) else {
        return nil
    }
    
    return (string as NSString).utf8String
}

@_cdecl("get_window_thumbnails") // Exposed to Python 
public func getWindowThumbnails(windowIDs: UnsafePointer<Int32>, count: Int32) -> UnsafePointer<CChar>? {
    var ids: [Int] = []
    for i in 0..<Int(count) {
        ids.append(Int(windowIDs[i]))
    }
    
    let thumbnails = WindowIntelligenceService.shared.getWindowThumbnails(windowIDs: ids)
    
    // Convert to JSON with base64 encoded images
    var result: [String: String] = [:]
    for (windowID, data) in thumbnails {
        result[String(windowID)] = data.base64EncodedString()
    }
    
    do {
        let jsonData = try JSONSerialization.data(withJSONObject: result)
        guard let string = String(data: jsonData, encoding: .utf8) else {
            return nil
        }
        return (string as NSString).utf8String
    } catch {
        return nil
    }
}

// Compile with:
// swiftc -emit-library -o WindowIntelligenceService.dylib WindowIntelligenceService.swift