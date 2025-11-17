//
//  HUDView.swift
//  JARVIS-HUD
//
//  Main HUD interface matching JARVIS web app design exactly
//  Based on web app screenshots - full-screen layout with arc reactor
//

import SwiftUI

/// Main HUD state
enum HUDState {
    case offline
    case listening
    case processing
    case speaking
    case idle
}

/// Transcript message
struct TranscriptMessage: Identifiable, Equatable {
    let id = UUID()
    let speaker: String // "YOU" or "JARVIS"
    let text: String
    let timestamp: Date

    // Equatable conformance (required for onChange)
    static func == (lhs: TranscriptMessage, rhs: TranscriptMessage) -> Bool {
        lhs.id == rhs.id
    }
}

/// Main JARVIS HUD View - Matches web app exactly
struct HUDView: View {

    @State private var hudState: HUDState = .offline
    @State private var transcriptMessages: [TranscriptMessage] = []
    @State private var statusText: String = "SYSTEM OFFLINE - START BACKEND"
    @State private var commandText: String = ""
    var onQuit: (() -> Void)? = nil  // Callback to quit HUD

    // Use shared PythonBridge from AppState (persisted from LoadingHUDView)
    @EnvironmentObject var appState: AppState

    // Convenience accessor for cleaner code
    private var pythonBridge: PythonBridge {
        appState.pythonBridge
    }

    var body: some View {
        ZStack {
            // FULLY TRANSPARENT GLASS - NO BLUR
            // Pure transparency so desktop shows through completely
            Color.clear
                .ignoresSafeArea()

            VStack(spacing: 0) {

                // Top section: J.A.R.V.I.S. title
                VStack(spacing: 8) {
                    Text("J.A.R.V.I.S.")
                        .font(.system(size: 72, weight: .bold, design: .monospaced))
                        .tracking(20) // Letter spacing
                        .foregroundColor(.jarvisGreen)
                        .shadow(color: Color.jarvisGreenGlow(opacity: 0.8), radius: 20)
                        .shadow(color: Color.jarvisGreenGlow(opacity: 0.6), radius: 40)

                    Text("JUST A RATHER VERY INTELLIGENT SYSTEM")
                        .font(.system(size: 12, weight: .medium, design: .monospaced))
                        .tracking(4)
                        .foregroundColor(.jarvisGreen)
                        .shadow(color: Color.jarvisGreenGlow(opacity: 0.6), radius: 10)
                }
                .padding(.top, 60)

                Spacer()

                // Center: Arc Reactor (matches web app 420px container)
                ArcReactorView(state: hudState, onQuit: onQuit)
                    .frame(width: 440, height: 440)

                Spacer()

                // Status message below reactor
                VStack(spacing: 8) {
                    Text(statusText)
                        .font(.system(size: 14, weight: .bold, design: .monospaced))
                        .tracking(3)
                        .foregroundColor(.jarvisGreen)
                        .shadow(color: Color.jarvisGreenGlow(opacity: 0.6), radius: 10)

                    // 🚀 Detailed connection state from Universal Client
                    if !pythonBridge.detailedConnectionState.isEmpty && pythonBridge.detailedConnectionState != statusText {
                        HStack(spacing: 6) {
                            ConnectionStateIndicator(state: pythonBridge.connectionStatus)

                            Text(pythonBridge.detailedConnectionState)
                                .font(.system(size: 10, weight: .medium, design: .monospaced))
                                .foregroundColor(.white.opacity(0.5))
                                .tracking(1)
                        }
                    }

                    // Server version and capabilities (when connected)
                    if pythonBridge.connectionStatus == .connected, !pythonBridge.serverVersion.isEmpty, pythonBridge.serverVersion != "unknown" {
                        Text("v\(pythonBridge.serverVersion) • \(pythonBridge.serverCapabilities.joined(separator: ", "))")
                            .font(.system(size: 9, weight: .regular, design: .monospaced))
                            .foregroundColor(.white.opacity(0.3))
                            .tracking(1)
                    }
                }
                .padding(.bottom, 30)

                // Transcript section
                TranscriptView(messages: transcriptMessages)
                    .frame(height: 180)
                    .padding(.horizontal, 60)
                    .padding(.bottom, 20)

                // Bottom: Command input (matching web app)
                HStack(spacing: 15) {
                    TextField("Type a command to JARVIS...", text: $commandText)
                        .textFieldStyle(PlainTextFieldStyle())
                        .font(.system(size: 14, design: .monospaced))
                        .foregroundColor(.white.opacity(0.7))
                        .padding()
                        .background(
                            RoundedRectangle(cornerRadius: 8)
                                .fill(Color.white.opacity(0.05))
                                .overlay(
                                    RoundedRectangle(cornerRadius: 8)
                                        .stroke(Color.jarvisGreen.opacity(0.3), lineWidth: 1)
                                )
                        )

                    Button(action: sendCommand) {
                        Text("SEND")
                            .font(.system(size: 13, weight: .bold, design: .monospaced))
                            .foregroundColor(.black)
                            .padding(.horizontal, 30)
                            .padding(.vertical, 12)
                            .background(
                                RoundedRectangle(cornerRadius: 8)
                                    .fill(Color.jarvisGreen)
                                    .shadow(color: Color.jarvisGreenGlow(), radius: 15)
                            )
                    }
                    .buttonStyle(PlainButtonStyle())
                }
                .padding(.horizontal, 60)
                .padding(.bottom, 40)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .onAppear {
            print("✅ HUDView appeared - using shared PythonBridge")
            print("   Connection status: \(pythonBridge.connectionStatus)")
            updateStatusFromConnection()
        }
        .onChange(of: pythonBridge.connectionStatus) { newStatus in
            updateStatusFromConnection()
        }
        .onChange(of: pythonBridge.hudState) { newState in
            hudState = newState
        }
        .onChange(of: pythonBridge.transcriptMessages) { newMessages in
            transcriptMessages = newMessages
        }
    }

    private func updateStatusFromConnection() {
        switch pythonBridge.connectionStatus {
        case .connected:
            statusText = "SYSTEM ONLINE - CONNECTED TO BACKEND"
        case .connecting:
            statusText = "CONNECTING TO BACKEND..."
        case .disconnected:
            statusText = "SYSTEM OFFLINE - BACKEND DISCONNECTED"
        case .error:
            statusText = "CONNECTION ERROR - RETRYING..."
        }
    }

    private func sendCommand() {
        // Handle command sending via PythonBridge
        if !commandText.isEmpty {
            // Add to local transcript immediately for responsive UI
            let message = TranscriptMessage(speaker: "YOU", text: commandText, timestamp: Date())
            transcriptMessages.append(message)

            // Send to backend via HTTP API
            Task {
                do {
                    try await pythonBridge.sendCommand(commandText)
                    print("✅ Command sent to backend: \(commandText)")
                } catch {
                    print("❌ Failed to send command: \(error)")
                    statusText = "ERROR: Failed to send command"
                }
            }

            commandText = ""
        }
    }
}

/// Transcript display component - matches web app styling
struct TranscriptView: View {

    let messages: [TranscriptMessage]

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            ForEach(messages) { message in
                VStack(alignment: .leading, spacing: 4) {
                    Text(message.speaker + ":")
                        .font(.system(size: 11, weight: .bold, design: .monospaced))
                        .foregroundColor(.white.opacity(0.5))
                        .tracking(1)

                    Text(message.text)
                        .font(.system(size: 13, design: .monospaced))
                        .foregroundColor(.white.opacity(0.8))
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }
}

/// 🚀 Connection State Indicator - Visual status indicator
struct ConnectionStateIndicator: View {
    let state: PythonBridge.ConnectionStatus

    var body: some View {
        Circle()
            .fill(indicatorColor)
            .frame(width: 8, height: 8)
            .shadow(color: indicatorColor.opacity(0.8), radius: 4)
            .overlay(
                Circle()
                    .stroke(indicatorColor.opacity(0.3), lineWidth: 1)
                    .scaleEffect(state == .connecting ? 1.5 : 1.0)
                    .opacity(state == .connecting ? 0 : 0.6)
                    .animation(.easeInOut(duration: 1.0).repeatForever(autoreverses: false), value: state)
            )
    }

    private var indicatorColor: Color {
        switch state {
        case .connected:
            return .jarvisGreen
        case .connecting:
            return .yellow
        case .disconnected:
            return .red.opacity(0.6)
        case .error:
            return .red
        }
    }
}

#Preview {
    HUDView()
}
