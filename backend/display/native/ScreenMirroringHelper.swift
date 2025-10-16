#!/usr/bin/env swift
/**
 * Screen Mirroring Helper
 * =======================
 *
 * Simple helper to trigger macOS screen mirroring via Control Center.
 * Uses CGEvent for precise mouse clicks to Control Center icon and device selection.
 *
 * Approach:
 * 1. Find Screen Mirroring icon in Control Center
 * 2. Click it to open menu
 * 3. Find and click the target device in the menu
 *
 * Author: Derek Russell
 * Date: 2025-10-16
 * Version: 1.0
 */

import Cocoa
import ApplicationServices
import Foundation

// MARK: - Logging

func log(_ message: String, level: String = "INFO") {
    let timestamp = DateFormatter.localizedString(from: Date(), dateStyle: .none, timeStyle: .medium)
    print("[\(timestamp)] [\(level)] \(message)")
}

// MARK: - Screen Utilities

func getScreenFrame() -> CGRect {
    guard let screen = NSScreen.main else {
        return CGRect.zero
    }
    return screen.frame
}

// MARK: - Mouse Click Utilities

func clickAtLocation(_ point: CGPoint) {
    log("Clicking at (\(point.x), \(point.y))")

    // Create mouse down event
    let mouseDown = CGEvent(
        mouseEventSource: nil,
        mouseType: .leftMouseDown,
        mouseCursorPosition: point,
        mouseButton: .left
    )

    // Create mouse up event
    let mouseUp = CGEvent(
        mouseEventSource: nil,
        mouseType: .leftMouseUp,
        mouseCursorPosition: point,
        mouseButton: .left
    )

    // Post events
    mouseDown?.post(tap: .cghidEventTap)
    Thread.sleep(forTimeInterval: 0.05)
    mouseUp?.post(tap: .cghidEventTap)
}

// MARK: - Control Center Location

func findControlCenterIcon() -> CGPoint? {
    log("Searching for Control Center icon...")

    let screen = getScreenFrame()

    // Control Center is in the top-right corner of the menu bar
    // On macOS, the menu bar is at the top of the screen
    // Control Center icon is typically ~100-150 pixels from the right edge

    // Estimate: right edge - 100 pixels, top + 12 pixels (center of menu bar)
    let estimatedX = screen.maxX - 100
    let estimatedY = screen.maxY - 12

    log("Control Center estimated position: (\(estimatedX), \(estimatedY))")

    return CGPoint(x: estimatedX, y: estimatedY)
}

// MARK: - Find UI Element by Title

func findElementWithTitle(_ title: String, in element: AXUIElement, depth: Int = 0, maxDepth: Int = 10) -> AXUIElement? {
    if depth > maxDepth {
        return nil
    }

    // Get element role for logging
    var roleValue: AnyObject?
    _ = AXUIElementCopyAttributeValue(element, kAXRoleAttribute as CFString, &roleValue)
    let role = roleValue as? String ?? "unknown"

    // Get element title
    var titleValue: AnyObject?
    let titleResult = AXUIElementCopyAttributeValue(element, kAXTitleAttribute as CFString, &titleValue)

    if titleResult == .success, let elementTitle = titleValue as? String {
        if depth < 3 {  // Only log first few levels to avoid spam
            log("Found element: role=\(role), title='\(elementTitle)'", level: "DEBUG")
        }
        if elementTitle.lowercased().contains(title.lowercased()) {
            log("✅ Matched element with title: '\(elementTitle)'")
            return element
        }
    }

    // Get element description (some menu items use this instead)
    var descValue: AnyObject?
    let descResult = AXUIElementCopyAttributeValue(element, kAXDescriptionAttribute as CFString, &descValue)

    if descResult == .success, let elementDesc = descValue as? String {
        if depth < 3 {
            log("Found element: role=\(role), description='\(elementDesc)'", level: "DEBUG")
        }
        if elementDesc.lowercased().contains(title.lowercased()) {
            log("✅ Matched element with description: '\(elementDesc)'")
            return element
        }
    }

    // Get element value (checkboxes might use this)
    var valueValue: AnyObject?
    let valueResult = AXUIElementCopyAttributeValue(element, kAXValueAttribute as CFString, &valueValue)

    if valueResult == .success, let elementValue = valueValue as? String {
        if depth < 3 {
            log("Found element: role=\(role), value='\(elementValue)'", level: "DEBUG")
        }
        if elementValue.lowercased().contains(title.lowercased()) {
            log("✅ Matched element with value: '\(elementValue)'")
            return element
        }
    }

    // Search children
    var childrenValue: AnyObject?
    let childrenResult = AXUIElementCopyAttributeValue(element, kAXChildrenAttribute as CFString, &childrenValue)

    if childrenResult == .success, let children = childrenValue as? [AXUIElement] {
        for child in children {
            if let found = findElementWithTitle(title, in: child, depth: depth + 1, maxDepth: maxDepth) {
                return found
            }
        }
    }

    return nil
}

// MARK: - Get Element Position

func getElementPosition(_ element: AXUIElement) -> CGPoint? {
    var positionValue: AnyObject?
    let result = AXUIElementCopyAttributeValue(element, kAXPositionAttribute as CFString, &positionValue)

    if result == .success, let axValue = positionValue {
        var point = CGPoint.zero
        if AXValueGetValue(axValue as! AXValue, .cgPoint, &point) {
            return point
        }
    }

    return nil
}

func getElementSize(_ element: AXUIElement) -> CGSize? {
    var sizeValue: AnyObject?
    let result = AXUIElementCopyAttributeValue(element, kAXSizeAttribute as CFString, &sizeValue)

    if result == .success, let axValue = sizeValue {
        var size = CGSize.zero
        if AXValueGetValue(axValue as! AXValue, .cgSize, &size) {
            return size
        }
    }

    return nil
}

func getElementCenter(_ element: AXUIElement) -> CGPoint? {
    guard let position = getElementPosition(element),
          let size = getElementSize(element) else {
        return nil
    }

    return CGPoint(
        x: position.x + size.width / 2,
        y: position.y + size.height / 2
    )
}

// MARK: - Screen Mirroring Logic

func connectToDevice(_ deviceName: String) -> Bool {
    log("Starting screen mirroring to '\(deviceName)'...")

    // Step 1: Click Control Center icon
    guard let controlCenterPoint = findControlCenterIcon() else {
        log("Failed to locate Control Center icon", level: "ERROR")
        return false
    }

    log("Clicking Control Center icon...")
    clickAtLocation(controlCenterPoint)

    // Wait for Control Center to open
    Thread.sleep(forTimeInterval: 1.0)

    // Step 2: Look for Screen Mirroring menu
    log("Searching for Screen Mirroring menu...")

    let systemWideElement = AXUIElementCreateSystemWide()

    // Search for "Screen Mirroring" text
    if let screenMirroringElement = findElementWithTitle("Screen Mirroring", in: systemWideElement) {
        log("Found Screen Mirroring element")

        if let position = getElementCenter(screenMirroringElement) {
            log("Clicking Screen Mirroring menu...")
            clickAtLocation(position)

            // Wait for menu to expand
            Thread.sleep(forTimeInterval: 0.5)
        }
    }

    // Step 3: Search for device in menu
    log("Searching for device '\(deviceName)' in menu...")

    // Wait a bit for menu to fully render
    Thread.sleep(forTimeInterval: 0.5)

    if let deviceElement = findElementWithTitle(deviceName, in: systemWideElement) {
        log("Found device '\(deviceName)'")

        if let position = getElementCenter(deviceElement) {
            log("Clicking device '\(deviceName)'...")
            clickAtLocation(position)

            log("✅ Screen mirroring initiated to '\(deviceName)'")
            return true
        } else {
            log("Could not determine device position", level: "ERROR")
            return false
        }
    } else {
        log("Device '\(deviceName)' not found in menu", level: "ERROR")
        return false
    }
}

// MARK: - Main Entry Point

func printUsage() {
    print("""
    Usage: ScreenMirroringHelper <device_name>

    Example:
        ScreenMirroringHelper "Living Room TV"
    """)
}

// Check accessibility permissions
func checkAccessibilityPermissions() -> Bool {
    let options = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true]
    let trusted = AXIsProcessTrustedWithOptions(options as CFDictionary)

    if !trusted {
        log("⚠️  Accessibility permissions not granted", level: "WARN")
        log("Please grant accessibility permissions in System Settings > Privacy & Security > Accessibility", level: "WARN")
        return false
    }

    return true
}

// Main
if CommandLine.arguments.count < 2 {
    printUsage()
    exit(1)
}

if !checkAccessibilityPermissions() {
    log("Cannot proceed without accessibility permissions", level: "ERROR")
    exit(1)
}

let deviceName = CommandLine.arguments[1]

log("Screen Mirroring Helper started")
log("Target device: \(deviceName)")

let success = connectToDevice(deviceName)

if success {
    log("✅ Screen mirroring connection successful")
    exit(0)
} else {
    log("❌ Screen mirroring connection failed", level: "ERROR")
    exit(1)
}
