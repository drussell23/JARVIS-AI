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
            window.level = .floating
            window.collectionBehavior = [.canJoinAllSpaces, .stationary, .fullScreenAuxiliary, .ignoresCycle]

            // CRITICAL: Make window FULLY click-through by default
            // This allows all clicks to pass to desktop/windows below
            window.ignoresMouseEvents = true

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

        // Start mouse tracking to enable selective event capture
        startMouseTracking()
    }

    /// Track mouse position to enable events only over interactive UI elements
    private func startMouseTracking() {
        NSEvent.addGlobalMonitorForEvents(matching: [.mouseMoved, .leftMouseDown]) { event in
            // Check if mouse is over an interactive element
            for window in NSApplication.shared.windows {
                if let contentView = window.contentView {
                    let mouseLocation = NSEvent.mouseLocation
                    let windowPoint = window.convertPoint(fromScreen: mouseLocation)

                    if let hitView = contentView.hitTest(windowPoint) {
                        // Check if we hit an interactive element
                        if self.isInteractiveElement(hitView) {
                            // Enable mouse events for this window
                            window.ignoresMouseEvents = false
                            return
                        }
                    }
                }
                // No interactive element - keep click-through enabled
                window.ignoresMouseEvents = true
            }
        }
    }

    private func isInteractiveElement(_ view: NSView) -> Bool {
        return view is NSButton ||
               (view is NSTextField && (view as! NSTextField).isEditable) ||
               (view is NSControl && !(view is NSTextField))
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

/// Custom hosting view with AGGRESSIVE click-through hit testing
/// Only captures clicks on actual interactive UI elements
class ClickThroughHostingView<Content: View>: NSHostingView<Content> {

    override func hitTest(_ point: NSPoint) -> NSView? {
        // Get default hit test result from SwiftUI
        guard let hitView = super.hitTest(point) else {
            // Nothing hit - definitely pass through
            return nil
        }

        // If we hit the hosting view itself, it's empty space - pass through
        if hitView == self {
            return nil
        }

        // Check if we hit an actual interactive element
        // Walk up the view hierarchy to find interactive controls
        var currentView: NSView? = hitView
        while currentView != nil {
            if isInteractive(currentView!) {
                return currentView
            }
            currentView = currentView?.superview

            // Stop at hosting view
            if currentView == self {
                break
            }
        }

        // Not an interactive element - pass through to desktop
        return nil
    }

    private func isInteractive(_ view: NSView) -> Bool {
        // Only capture these specific interactive elements:

        // Buttons
        if view is NSButton {
            return true
        }

        // Editable text fields
        if let textField = view as? NSTextField, textField.isEditable {
            return true
        }

        // Other interactive controls
        if view is NSControl && !(view is NSTextField) {
            return true
        }

        // Everything else passes through (text labels, images, spacers, etc.)
        return false
    }

    /// Allow window to become key for keyboard input when needed
    override var acceptsFirstResponder: Bool {
        return true
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
