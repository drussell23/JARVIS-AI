//
//  JARVISApp.swift
//  JARVIS-HUD
//
//  Main app entry point and window configuration
//

import SwiftUI

@main
struct JARVISApp: App {

    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    var body: some Scene {
        WindowGroup {
            HUDView()
                .background(WindowAccessor())
        }
        .windowStyle(.hiddenTitleBar)
        .commands {
            // Remove default menu items if needed
            CommandGroup(replacing: .newItem) {}
        }
    }
}

/// App delegate for macOS-specific configuration
class AppDelegate: NSObject, NSApplicationDelegate {

    var hudWindow: TransparentWindow?

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Create and configure transparent window
        hudWindow = TransparentWindow()

        // Set the content view
        if let window = hudWindow {
            window.contentView = NSHostingView(rootView: HUDView())
            window.showHUD()
        }

        // Hide from Dock (optional - can be enabled later)
        // NSApp.setActivationPolicy(.accessory)

        // Setup auto-hide timer
        setupAutoHideTimer()
    }

    /// Auto-hide HUD after 10 seconds of inactivity (PRD requirement)
    private func setupAutoHideTimer() {
        // Implementation for auto-hide logic
        // Will be connected to Python backend for activity detection
    }
}

/// Window accessor to configure the NSWindow
struct WindowAccessor: NSViewRepresentable {

    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        DispatchQueue.main.async {
            if let window = view.window {
                // Configure window as transparent overlay
                window.isOpaque = false
                window.backgroundColor = .clear
                window.level = .statusBar
                window.collectionBehavior = [.canJoinAllSpaces, .stationary]
            }
        }
        return view
    }

    func updateNSView(_ nsView: NSView, context: Context) {}
}
