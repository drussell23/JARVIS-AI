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

/// App state manager - Central coordination of all HUD systems
class AppState: ObservableObject {
    @Published var isLoadingComplete: Bool = false

    // CRITICAL: PythonBridge MUST persist across LoadingHUDView → HUDView transition
    // If each view creates its own @StateObject PythonBridge, the WebSocket connection
    // is destroyed when transitioning from Loading to HUD view!
    @Published var pythonBridge: PythonBridge

    // 🎤 JARVIS Voice System - Text-to-Speech output
    @Published var voiceManager: VoiceManager

    // 👁️ Vision System - Screen analysis and AI vision
    @Published var visionManager: VisionManager

    @MainActor
    init() {
        print(String(repeating: "=", count: 80))
        print("🔗 AppState.init() STARTED")
        print("   Initializing integrated HUD systems...")

        // Initialize PythonBridge once at app launch
        print("   1️⃣ Creating PythonBridge...")
        let bridge = PythonBridge()
        self.pythonBridge = bridge
        print("      ✓ PythonBridge created")

        // Initialize VoiceManager with backend URL
        print("   2️⃣ Creating VoiceManager...")
        self.voiceManager = VoiceManager(apiBaseURL: bridge.apiBaseURL)
        print("      ✓ VoiceManager created")

        // Initialize VisionManager with backend URL
        print("   3️⃣ Creating VisionManager...")
        self.visionManager = VisionManager(apiBaseURL: bridge.apiBaseURL)
        print("      ✓ VisionManager created")

        // Now print backend URLs
        print("      Backend WS: \(bridge.websocketURL)")
        print("      Backend HTTP: \(bridge.apiBaseURL)")

        // Connect to backend immediately when app launches
        print("   🔌 Calling pythonBridge.connect() to initiate WebSocket connection...")
        pythonBridge.connect()
        print("      ✓ connect() call completed (connection may still be establishing)")

        // Link VoiceManager and VisionManager to PythonBridge for WebSocket events
        setupBridgeIntegration()

        print("🔗 AppState.init() COMPLETED - All systems initialized")
        print(String(repeating: "=", count: 80))
    }

    /// Setup bidirectional integration between PythonBridge and managers
    private func setupBridgeIntegration() {
        print("🔗 Setting up PythonBridge ↔ Manager integration...")

        // Inject managers into PythonBridge for WebSocket event handling
        pythonBridge.voiceManager = voiceManager
        pythonBridge.visionManager = visionManager

        print("   ✓ Integration complete")
    }
}

/// App delegate for macOS-specific configuration
class AppDelegate: NSObject, NSApplicationDelegate {

    private var isHUDVisible = true
    private var localEventMonitor: Any?

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Configure all windows with semi-transparent black background
        for window in NSApplication.shared.windows {
            // Semi-transparent black background
            window.isOpaque = false
            window.backgroundColor = NSColor.black.withAlphaComponent(0.7)
            window.hasShadow = true

            // Remove title bar but keep window frame
            window.titlebarAppearsTransparent = true
            window.titleVisibility = .hidden
            window.styleMask = [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView]

            // Floating window - ALWAYS visible across ALL Spaces
            window.level = .floating
            window.collectionBehavior = [
                .canJoinAllSpaces,      // Appears in ALL Mission Control Spaces
                .stationary,             // Stays in place when switching Spaces
                .fullScreenAuxiliary,    // Works alongside full-screen apps
                .ignoresCycle            // Not in Cmd+Tab switcher
            ]

            // CRITICAL: Prevent window from being hidden when switching Spaces
            window.hidesOnDeactivate = false

            // Window is interactive (not click-through)
            window.ignoresMouseEvents = false

            // Make window non-activating (doesn't steal focus)
            if let panel = window as? NSPanel {
                panel.isFloatingPanel = true
                panel.becomesKeyOnlyIfNeeded = true
            }

            // Set a reasonable window size and position
            let windowWidth: CGFloat = 800
            let windowHeight: CGFloat = 600
            if let screen = NSScreen.main {
                let screenFrame = screen.frame
                let x = (screenFrame.width - windowWidth) / 2
                let y = (screenFrame.height - windowHeight) / 2
                window.setFrame(NSRect(x: x, y: y, width: windowWidth, height: windowHeight), display: true)
            }

            // 🚀 CRITICAL: Make window visible IMMEDIATELY on launch
            // This ensures user sees loading screen from 1-100%
            window.alphaValue = 1.0  // Fully opaque
            window.orderFrontRegardless()  // Bring to front regardless of other windows
            window.makeKeyAndOrderFront(nil)  // Make it the key window

            print("🪟 Window configured for immediate visibility:")
            print("   isVisible: \(window.isVisible)")
            print("   alphaValue: \(window.alphaValue)")
            print("   level: \(window.level.rawValue)")
            print("   frame: \(window.frame)")
        }

        // Hide from Dock and app switcher (pure floating overlay)
        NSApp.setActivationPolicy(.accessory)

        // Activate the app to bring window to front
        NSApp.activate(ignoringOtherApps: true)

        print("✅ App activation policy: accessory, activated: ignoringOtherApps")

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

            // Cmd+O (key code 31) - Toggle HUD visibility
            if modifiers.contains(.command) && event.keyCode == 31 {
                self.toggleHUDVisibility()
                return // Global monitors can't consume events
            }

            // Cmd+Q (key code 12) - Quit HUD application
            if modifiers.contains(.command) && event.keyCode == 12 {
                print("🛑 Cmd+Q detected - Quitting JARVIS HUD")
                self.quitHUD()
                return
            }
        }

        // Also add local monitor for when app has focus (better responsiveness)
        localEventMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { event in
            let modifiers = event.modifierFlags

            // Cmd+O (key code 31) - Toggle HUD visibility
            if modifiers.contains(.command) && event.keyCode == 31 {
                self.toggleHUDVisibility()
                return nil // Consume the event
            }

            // Cmd+Q (key code 12) - Quit HUD application
            if modifiers.contains(.command) && event.keyCode == 12 {
                print("🛑 Cmd+Q detected - Quitting JARVIS HUD")
                self.quitHUD()
                return nil
            }

            return event
        }

        print("⌨️ Keyboard shortcuts registered:")
        print("   • Cmd+O → Toggle HUD visibility (show/hide)")
        print("   • Cmd+Q → Quit JARVIS HUD")
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
                // Semi-transparent black background
                window.isOpaque = false
                window.backgroundColor = NSColor.black.withAlphaComponent(0.7)
                window.hasShadow = true
                window.titlebarAppearsTransparent = true
                window.titleVisibility = .hidden
                window.styleMask = [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView]
                window.level = .statusBar
                window.collectionBehavior = [.canJoinAllSpaces, .stationary, .fullScreenAuxiliary]

                // Window is interactive
                window.ignoresMouseEvents = false

                // Set a reasonable window size and position
                let windowWidth: CGFloat = 800
                let windowHeight: CGFloat = 600
                if let screen = NSScreen.main {
                    let screenFrame = screen.frame
                    let x = (screenFrame.width - windowWidth) / 2
                    let y = (screenFrame.height - windowHeight) / 2
                    window.setFrame(NSRect(x: x, y: y, width: windowWidth, height: windowHeight), display: true)
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
