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
struct TranscriptMessage: Identifiable {
    let id = UUID()
    let speaker: String // "YOU" or "JARVIS"
    let text: String
    let timestamp: Date
}

/// Main JARVIS HUD View - Matches web app exactly
struct HUDView: View {

    @State private var hudState: HUDState = .offline
    @State private var transcriptMessages: [TranscriptMessage] = []
    @State private var statusText: String = "SYSTEM OFFLINE - START BACKEND"
    @State private var commandText: String = ""

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
                ArcReactorView(state: hudState)
                    .frame(width: 440, height: 440)

                Spacer()

                // Status message below reactor
                Text(statusText)
                    .font(.system(size: 14, weight: .bold, design: .monospaced))
                    .tracking(3)
                    .foregroundColor(.jarvisGreen)
                    .shadow(color: Color.jarvisGreenGlow(opacity: 0.6), radius: 10)
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
            addDemoMessages()
        }
    }

    private func sendCommand() {
        // Handle command sending
        if !commandText.isEmpty {
            transcriptMessages.append(
                TranscriptMessage(speaker: "YOU", text: commandText, timestamp: Date())
            )
            commandText = ""
        }
    }

    /// Add demo messages
    private func addDemoMessages() {
        transcriptMessages = [
            TranscriptMessage(speaker: "YOU", text: "unlock my screen", timestamp: Date()),
            TranscriptMessage(speaker: "JARVIS", text: "Screen unlocked by Derek J. Russell", timestamp: Date())
        ]
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

#Preview {
    HUDView()
}
