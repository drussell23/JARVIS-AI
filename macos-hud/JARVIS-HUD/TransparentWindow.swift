//
//  TransparentWindow.swift
//  JARVIS-HUD
//
//  Transparent, borderless NSWindow for JARVIS HUD overlay
//  Based on PRD requirements for macOS integration
//

import SwiftUI
import AppKit

/// Custom NSWindow for transparent, always-on-top HUD overlay
class TransparentWindow: NSWindow {

    init() {
        super.init(
            contentRect: NSRect(x: 0, y: 0, width: 600, height: 400),
            styleMask: [.borderless, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )

        // PRD Requirement: Transparent, borderless window
        self.isOpaque = false
        self.backgroundColor = .clear
        self.hasShadow = false
        self.titlebarAppearsTransparent = true
        self.titleVisibility = .hidden

        // PRD Requirement: Always on top (.screenSaver or .statusBar level)
        self.level = .statusBar

        // PRD Requirement: Join all Spaces (Mission Control compatible)
        self.collectionBehavior = [.canJoinAllSpaces, .stationary, .fullScreenAuxiliary]

        // PRD Requirement: Don't steal focus from other windows
        self.isMovableByWindowBackground = false
        self.ignoresMouseEvents = false // Can be toggled for click-through

        // Center on screen
        self.center()
    }

    /// Make window appear in center of screen
    override func center() {
        if let screen = NSScreen.main {
            let screenRect = screen.frame
            let windowRect = self.frame
            let x = (screenRect.width - windowRect.width) / 2
            let y = (screenRect.height - windowRect.height) / 2
            self.setFrameOrigin(NSPoint(x: x, y: y))
        }
    }

    /// Enable click-through mode (for non-interactive HUD display)
    func enableClickThrough() {
        self.ignoresMouseEvents = true
    }

    /// Disable click-through mode (for interactive HUD)
    func disableClickThrough() {
        self.ignoresMouseEvents = false
    }

    /// Show HUD with fade-in animation
    func showHUD() {
        self.orderFrontRegardless()
        self.animator().alphaValue = 1.0
    }

    /// Hide HUD with fade-out animation
    func hideHUD() {
        NSAnimationContext.runAnimationGroup({ context in
            context.duration = 0.3
            self.animator().alphaValue = 0.0
        }, completionHandler: {
            self.orderOut(nil)
        })
    }
}

/// SwiftUI wrapper for TransparentWindow
struct TransparentWindowAccessor: NSViewRepresentable {

    @Binding var window: TransparentWindow?

    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        DispatchQueue.main.async {
            self.window = view.window as? TransparentWindow
        }
        return view
    }

    func updateNSView(_ nsView: NSView, context: Context) {}
}
