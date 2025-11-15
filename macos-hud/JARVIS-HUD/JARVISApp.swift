//
//  JARVISApp.swift
//  JARVIS-HUD
//
//  Main app entry point and window configuration
//

import SwiftUI
import AppKit

@main
struct JARVISApp: App {

    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    @StateObject private var appState = AppState()

    var body: some Scene {
        WindowGroup {
            ClickThroughContainer {
                Group {
                    if appState.isLoadingComplete {
                        HUDView()
                            .background(WindowAccessor())
                    } else {
                        LoadingHUDView {
                            appState.isLoadingComplete = true
                        }
                        .background(WindowAccessor())
                    }
                }
            }
        }
        .windowStyle(.hiddenTitleBar)
        .commands {
            // Remove default menu items if needed
            CommandGroup(replacing: .newItem) {}
        }
    }
}

/// App state manager
class AppState: ObservableObject {
    @Published var isLoadingComplete: Bool = false
}

/// App delegate for macOS-specific configuration
class AppDelegate: NSObject, NSApplicationDelegate {

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Configure all windows as TRUE HOLOGRAPHIC overlays
        // Completely invisible window frame - only JARVIS UI elements visible
        for window in NSApplication.shared.windows {
            // Make window completely transparent - NO traditional window at all
            window.isOpaque = false
            window.backgroundColor = .clear
            window.hasShadow = false

            // Remove ALL window chrome and borders
            window.titlebarAppearsTransparent = true
            window.titleVisibility = .hidden
            window.styleMask = [.borderless, .fullSizeContentView]

            // Floating overlay - always on top, works in all Spaces
            window.level = .floating  // Changed from .statusBar to .floating for better behavior
            window.collectionBehavior = [.canJoinAllSpaces, .stationary, .fullScreenAuxiliary, .ignoresCycle]

            // CRITICAL: Enable TRUE click-through
            // The ClickThroughHostingView will handle selective event capture
            window.ignoresMouseEvents = false

            // Make window non-activating (doesn't steal focus)
            if let panel = window as? NSPanel {
                panel.isFloatingPanel = true
                panel.becomesKeyOnlyIfNeeded = true
            }

            // Cover entire screen as transparent overlay
            if let screen = NSScreen.main {
                window.setFrame(screen.frame, display: true)
            }
        }

        // Hide from Dock and app switcher (pure floating overlay)
        NSApp.setActivationPolicy(.accessory)

        // Don't activate the app when launched
        NSApp.activate(ignoringOtherApps: false)
    }
}

/// Window accessor to configure the NSWindow
struct WindowAccessor: NSViewRepresentable {

    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        DispatchQueue.main.async {
            if let window = view.window {
                // Make window COMPLETELY INVISIBLE - no chrome, no shadow, no borders
                window.isOpaque = false
                window.backgroundColor = .clear
                window.hasShadow = false
                window.titlebarAppearsTransparent = true
                window.titleVisibility = .hidden
                window.styleMask.insert(.borderless)
                window.styleMask.insert(.fullSizeContentView)
                window.level = .statusBar
                window.collectionBehavior = [.canJoinAllSpaces, .stationary, .fullScreenAuxiliary]

                // CRITICAL: Allow window to become key for keyboard input
                // But still enable click-through for transparent areas
                window.ignoresMouseEvents = false

                // Cover entire screen
                if let screen = NSScreen.main {
                    window.setFrame(screen.frame, display: true)
                }
            }
        }
        return view
    }

    func updateNSView(_ nsView: NSView, context: Context) {}
}

/// Custom hosting view with precise click-through hit testing
class ClickThroughHostingView<Content: View>: NSHostingView<Content> {

    override func hitTest(_ point: NSPoint) -> NSView? {
        // Get default hit test result
        guard let hitView = super.hitTest(point) else {
            return nil
        }

        // If we hit ourselves (the hosting view), pass through to desktop
        if hitView == self {
            return nil
        }

        // Check if hit an interactive element
        if isInteractive(hitView) {
            return hitView
        }

        // Pass through to desktop
        return nil
    }

    private func isInteractive(_ view: NSView) -> Bool {
        // Capture clicks on interactive elements
        return view is NSButton ||
               view is NSControl ||
               (view is NSTextField && (view as! NSTextField).isEditable)
    }
}

/// SwiftUI wrapper for click-through hosting view
struct ClickThroughContainer<Content: View>: NSViewRepresentable {
    let content: Content

    init(@ViewBuilder content: () -> Content) {
        self.content = content()
    }

    func makeNSView(context: Context) -> ClickThroughHostingView<Content> {
        let hostingView = ClickThroughHostingView(rootView: content)
        hostingView.layer?.backgroundColor = .clear
        return hostingView
    }

    func updateNSView(_ nsView: ClickThroughHostingView<Content>, context: Context) {
        nsView.rootView = content
    }
}
