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
                        HUDView(onQuit: appDelegate.quitHUD)
                            .background(WindowAccessor())
                            .environmentObject(appState)  // Inject AppState for PythonBridge access
                    } else {
                        LoadingHUDView {
                            appState.isLoadingComplete = true
                        }
                        .background(WindowAccessor())
                        .environmentObject(appState)  // Inject AppState for PythonBridge access
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

    // CRITICAL: PythonBridge MUST persist across LoadingHUDView → HUDView transition
    // If each view creates its own @StateObject PythonBridge, the WebSocket connection
    // is destroyed when transitioning from Loading to HUD view!
    @Published var pythonBridge: PythonBridge

    init() {
        // Initialize PythonBridge once at app launch
        self.pythonBridge = PythonBridge()

        print("🔗 AppState initialized with shared PythonBridge")
        print("   Backend WS: \(pythonBridge.websocketURL)")
        print("   Backend HTTP: \(pythonBridge.apiBaseURL)")

        // Connect to backend immediately when app launches
        pythonBridge.connect()
    }
}

/// App delegate for macOS-specific configuration
class AppDelegate: NSObject, NSApplicationDelegate {

    private var isHUDVisible = true
    private var localEventMonitor: Any?

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

            // Floating overlay - ALWAYS visible across ALL Spaces
            window.level = .floating
            window.collectionBehavior = [
                .canJoinAllSpaces,      // Appears in ALL Mission Control Spaces
                .stationary,             // Stays in place when switching Spaces
                .fullScreenAuxiliary,    // Works alongside full-screen apps
                .ignoresCycle            // Not in Cmd+Tab switcher
            ]

            // CRITICAL: Prevent window from being hidden when switching Spaces
            window.hidesOnDeactivate = false

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

        // Setup keyboard shortcut for toggling HUD visibility (Cmd+\)
        setupKeyboardShortcuts()
    }

    /// Track mouse position to enable events only over interactive UI elements
    private func startMouseTracking() {
        NSEvent.addGlobalMonitorForEvents(matching: [.mouseMoved, .leftMouseDown]) { event in
            // Only track if HUD is visible
            guard self.isHUDVisible else { return }

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

    /// Setup keyboard shortcuts for HUD control
    private func setupKeyboardShortcuts() {
        // Use GLOBAL event monitor since HUD is a floating overlay without focus
        // This ensures shortcuts work even when HUD doesn't have keyboard focus
        NSEvent.addGlobalMonitorForEvents(matching: .keyDown) { event in
            let modifiers = event.modifierFlags

            // Cmd+\ (backslash key code is 42) - Toggle HUD visibility
            if modifiers.contains(.command) && event.keyCode == 42 {
                self.toggleHUDVisibility()
                return // Global monitors can't consume events
            }

            // Cmd+Q+O - Quit HUD application (both keys must be held with Cmd)
            // This is a three-key combination: Cmd + Q + O
            if modifiers.contains(.command) {
                // Check if Q (12) and O (31) are both pressed simultaneously
                let qPressed = CGEventSource.keyState(.hidSystemState, key: CGKeyCode(12))
                let oPressed = CGEventSource.keyState(.hidSystemState, key: CGKeyCode(31))

                if qPressed && oPressed {
                    print("🛑 Cmd+Q+O detected - Quitting JARVIS HUD")
                    self.quitHUD()
                    return
                }
            }
        }

        // Also add local monitor for when app has focus (better responsiveness)
        localEventMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { event in
            let modifiers = event.modifierFlags

            // Cmd+\ (backslash key code is 42) - Toggle HUD visibility
            if modifiers.contains(.command) && event.keyCode == 42 {
                self.toggleHUDVisibility()
                return nil // Consume the event
            }

            // Cmd+Q+O - Quit HUD application
            if modifiers.contains(.command) {
                let qPressed = CGEventSource.keyState(.hidSystemState, key: CGKeyCode(12))
                let oPressed = CGEventSource.keyState(.hidSystemState, key: CGKeyCode(31))

                if qPressed && oPressed {
                    print("🛑 Cmd+Q+O detected - Quitting JARVIS HUD")
                    self.quitHUD()
                    return nil
                }
            }

            return event
        }

        print("⌨️ Keyboard shortcuts registered:")
        print("   • Cmd+\\ → Toggle HUD visibility")
        print("   • Cmd+Q+O → Quit JARVIS HUD")
    }

    /// Toggle HUD visibility with smooth fade animation
    /// HUD persists across all Spaces - only hidden/shown via this toggle or quit
    private func toggleHUDVisibility() {
        isHUDVisible.toggle()

        print("🎯 Toggling HUD visibility: \(isHUDVisible ? "SHOW" : "HIDE")")
        print("   HUD will remain \(isHUDVisible ? "visible" : "hidden") across ALL Spaces")

        for window in NSApplication.shared.windows {
            if isHUDVisible {
                // Show HUD with fade-in animation
                window.alphaValue = 0.0
                window.orderFront(nil)

                // Ensure window remains in all Spaces when shown
                window.collectionBehavior = [.canJoinAllSpaces, .stationary, .fullScreenAuxiliary, .ignoresCycle]

                NSAnimationContext.runAnimationGroup({ context in
                    context.duration = 0.3
                    context.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
                    window.animator().alphaValue = 1.0
                })

                print("✅ HUD shown with fade-in animation (visible in all Spaces)")
            } else {
                // Hide HUD with fade-out animation
                // IMPORTANT: Keep window in all Spaces even when hidden
                // This ensures it's ready to show immediately when toggled back
                NSAnimationContext.runAnimationGroup({ context in
                    context.duration = 0.3
                    context.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
                    window.animator().alphaValue = 0.0
                }, completionHandler: {
                    window.orderOut(nil)
                })

                print("🚫 HUD hidden with fade-out animation (but still present in all Spaces)")
            }
        }
    }

    /// Quit HUD with graceful fade-out animation
    func quitHUD() {
        print("👋 Shutting down JARVIS HUD gracefully...")

        // Fade out all windows with animation
        for window in NSApplication.shared.windows {
            NSAnimationContext.runAnimationGroup({ context in
                context.duration = 0.5
                context.timingFunction = CAMediaTimingFunction(name: .easeIn)
                window.animator().alphaValue = 0.0
            }, completionHandler: {
                // After fade out completes, quit the application
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                    print("✅ JARVIS HUD shutdown complete")
                    NSApplication.shared.terminate(nil)
                }
            })
        }
    }

    private func isInteractiveElement(_ view: NSView) -> Bool {
        return view is NSButton ||
               (view is NSTextField && (view as! NSTextField).isEditable) ||
               (view is NSControl && !(view is NSTextField))
    }

    func applicationWillTerminate(_ notification: Notification) {
        // Clean up event monitor
        if let monitor = localEventMonitor {
            NSEvent.removeMonitor(monitor)
        }
        print("🔌 JARVIS HUD terminated")
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
