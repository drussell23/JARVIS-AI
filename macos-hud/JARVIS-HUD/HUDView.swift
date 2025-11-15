//
//  HUDView.swift
//  JARVIS-HUD
//
//  Main HUD interface with transcript display and status
//  Matches JARVIS web app design with neon green terminal aesthetic
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
    let speaker: String // "USER" or "JARVIS"
    let text: String
    let timestamp: Date
}

/// Main JARVIS HUD View
struct HUDView: View {

    @State private var hudState: HUDState = .idle
    @State private var transcriptMessages: [TranscriptMessage] = []
    @State private var statusText: String = "SYSTEM ONLINE"
    @State private var isVisible: Bool = true

    var body: some View {
        ZStack {
            // Transparent background with subtle gradient
            LinearGradient(
                colors: [
                    Color.jarvisBlack.opacity(0.95),
                    Color.jarvisBlackBlue.opacity(0.85)
                ],
                startPoint: .top,
                endPoint: .bottom
            )
            .ignoresSafeArea()

            // Main HUD content
            VStack(spacing: 20) {

                // Top status bar
                HStack {
                    Text("J.A.R.V.I.S.")
                        .font(.custom(JARVISTheme.Fonts.display, size: 18))
                        .foregroundColor(.jarvisGreen)
                        .shadow(color: Color.jarvisGreenGlow(), radius: 10)

                    Spacer()

                    Text(statusText)
                        .font(.custom(JARVISTheme.Fonts.monospace, size: 12))
                        .foregroundColor(statusColor)
                        .shadow(color: statusColor.opacity(0.6), radius: 8)
                }
                .padding(.horizontal, 30)
                .padding(.top, 20)

                // Center pulse animation
                if hudState == .listening || hudState == .processing {
                    if hudState == .listening {
                        JARVISPulseCyanView()
                            .frame(height: 150)
                    } else {
                        JARVISPulseView(isActive: true)
                            .frame(height: 200)
                    }
                } else {
                    JARVISPulseView(isActive: hudState == .speaking)
                        .frame(height: 200)
                }

                // Transcript display
                TranscriptView(messages: transcriptMessages)
                    .frame(maxHeight: 150)
                    .padding(.horizontal, 30)

                Spacer()
            }

            // Border glow effect
            RoundedRectangle(cornerRadius: 20)
                .stroke(
                    LinearGradient(
                        colors: [Color.jarvisGreen, Color.jarvisCyan],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    ),
                    lineWidth: 2
                )
                .shadow(color: Color.jarvisGreenGlow(), radius: 15)
                .padding(10)
        }
        .frame(width: 600, height: 400)
        .clipShape(RoundedRectangle(cornerRadius: 20))
        .onAppear {
            // Demo messages
            addDemoMessages()
        }
    }

    /// Color for status text based on state
    private var statusColor: Color {
        switch hudState {
        case .offline:
            return .jarvisError
        case .listening:
            return .jarvisCyan
        case .processing:
            return .jarvisWarning
        case .speaking:
            return .jarvisGreen
        case .idle:
            return .jarvisSuccess
        }
    }

    /// Add demo messages for preview
    private func addDemoMessages() {
        transcriptMessages = [
            TranscriptMessage(speaker: "USER", text: "Open Safari and search for dogs", timestamp: Date()),
            TranscriptMessage(speaker: "JARVIS", text: "Opening Safari...", timestamp: Date()),
            TranscriptMessage(speaker: "JARVIS", text: "Searching for dogs", timestamp: Date())
        ]
    }
}

/// Transcript display component
struct TranscriptView: View {

    let messages: [TranscriptMessage]

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 8) {
                ForEach(messages) { message in
                    TranscriptMessageRow(message: message)
                        .transition(.opacity.combined(with: .move(edge: .bottom)))
                }
            }
        }
    }
}

/// Individual transcript message row
struct TranscriptMessageRow: View {

    let message: TranscriptMessage

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            // Speaker label
            Text(message.speaker + ":")
                .font(.custom(JARVISTheme.Fonts.monospace, size: 11))
                .foregroundColor(speakerColor)
                .frame(width: 80, alignment: .trailing)

            // Message text
            Text(message.text)
                .font(.custom(JARVISTheme.Fonts.monospace, size: 11))
                .foregroundColor(.jarvisGreen)
                .shadow(color: Color.jarvisGreenGlow(opacity: 0.4), radius: 5)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.vertical, 4)
    }

    private var speakerColor: Color {
        message.speaker == "USER" ? .jarvisCyan : .jarvisGreen
    }
}

#Preview {
    ZStack {
        Color.black.ignoresSafeArea()
        HUDView()
    }
}
