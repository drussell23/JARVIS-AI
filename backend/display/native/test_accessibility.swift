#!/usr/bin/swift
/**
 * Test Accessibility API access to menu bar
 */

import Foundation
import ApplicationServices
import Cocoa

func checkAccessibilityPermissions() -> Bool {
    return AXIsProcessTrusted()
}

func listMenuBarItems() {
    let systemWide = AXUIElementCreateSystemWide()
    
    var menuBar: AnyObject?
    let result = AXUIElementCopyAttributeValue(
        systemWide,
        kAXMenuBarAttribute as CFString,
        &menuBar
    )
    
    guard result == .success, let menuBarElement = menuBar as! AXUIElement? else {
        print("❌ Cannot access menu bar: \(result.rawValue)")
        return
    }
    
    print("✅ Menu bar accessible")
    
    var children: AnyObject?
    let childResult = AXUIElementCopyAttributeValue(
        menuBarElement,
        kAXChildrenAttribute as CFString,
        &children
    )
    
    guard childResult == .success, let menuBarItems = children as? [AXUIElement] else {
        print("❌ Cannot get menu bar items: \(childResult.rawValue)")
        return
    }
    
    print("\n📋 Found \(menuBarItems.count) menu bar items:\n")
    
    for (index, item) in menuBarItems.enumerated() {
        var description: AnyObject?
        AXUIElementCopyAttributeValue(item, kAXDescriptionAttribute as CFString, &description)
        
        var title: AnyObject?
        AXUIElementCopyAttributeValue(item, kAXTitleAttribute as CFString, &title)
        
        var role: AnyObject?
        AXUIElementCopyAttributeValue(item, kAXRoleAttribute as CFString, &role)
        
        let descStr = description as? String ?? "(no description)"
        let titleStr = title as? String ?? "(no title)"
        let roleStr = role as? String ?? "(no role)"
        
        print("[\(index)] \(titleStr)")
        print("    Description: \(descStr)")
        print("    Role: \(roleStr)")
        
        // Check if this might be Screen Mirroring
        if descStr.lowercased().contains("screen") || 
           descStr.lowercased().contains("mirror") ||
           descStr.lowercased().contains("display") ||
           titleStr.lowercased().contains("screen") ||
           titleStr.lowercased().contains("mirror") {
            print("    ⭐ POTENTIAL SCREEN MIRRORING ITEM!")
        }
        print()
    }
}

// Main
print(String(repeating: "=", count: 60))
print("Accessibility API Test for Screen Mirroring")
print(String(repeating: "=", count: 60))
print()

if checkAccessibilityPermissions() {
    print("✅ Accessibility permissions granted\n")
    listMenuBarItems()
} else {
    print("❌ Accessibility permissions NOT granted")
    print("Grant permissions in: System Settings → Privacy & Security → Accessibility")
}
