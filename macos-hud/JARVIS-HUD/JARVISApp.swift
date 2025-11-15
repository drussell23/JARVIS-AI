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
        // Configure all windows as transparent overlays
        for window in NSApplication.shared.windows {
            window.isOpaque = false
            window.backgroundColor = .clear
            window.level = .statusBar
            window.collectionBehavior = [.canJoinAllSpaces, .stationary]
        }

        // Hide from Dock (optional - can be enabled later)
        // NSApp.setActivationPolicy(.accessory)
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
