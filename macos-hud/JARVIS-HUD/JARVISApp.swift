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
    @StateObject private var appState = AppState()

    var body: some Scene {
        WindowGroup {
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
        // Configure all windows as COMPLETELY INVISIBLE overlays
        // Only the JARVIS UI elements will be visible
        for window in NSApplication.shared.windows {
            // Make window completely transparent
            window.isOpaque = false
            window.backgroundColor = .clear
            window.hasShadow = false

            // Remove all window chrome (title bar, borders, everything)
            window.titlebarAppearsTransparent = true
            window.titleVisibility = .hidden
            window.styleMask.insert(.borderless)
            window.styleMask.insert(.fullSizeContentView)

            // Always on top, works in all Spaces
            window.level = .statusBar
            window.collectionBehavior = [.canJoinAllSpaces, .stationary, .fullScreenAuxiliary]

            // Cover entire screen (full-screen transparent overlay)
            if let screen = NSScreen.main {
                window.setFrame(screen.frame, display: true)
            }
        }

        // Hide from Dock (floating overlay, not a regular app)
        NSApp.setActivationPolicy(.accessory)
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
