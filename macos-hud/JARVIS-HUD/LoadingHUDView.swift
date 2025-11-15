//
//  LoadingHUDView.swift
//  JARVIS-HUD
//
//  Loading screen matching loading.html exactly
//  With matrix transition effect from loading-manager.js
//

import SwiftUI

/// Loading screen state
enum LoadingState {
    case initializing
    case loading(progress: Int, message: String)
    case complete
    case failed(error: String)
}

/// Loading HUD View matching loading.html
struct LoadingHUDView: View {

    @State private var loadingState: LoadingState = .initializing
    @State private var progress: CGFloat = 0
    @State private var logoScale: CGFloat = 1.0
    @State private var showMatrixTransition: Bool = false
    @StateObject private var pythonBridge = PythonBridge()

    var onComplete: () -> Void

    var body: some View {
        ZStack {
            // Semi-transparent black background (50% opacity) - user can see desktop
            Color.black.opacity(0.5).ignoresSafeArea()

            if !showMatrixTransition {
                // Main loading content
                VStack(spacing: 60) {

                    Spacer()

                    // Logo
                    VStack(spacing: 10) {
                        Text("J.A.R.V.I.S.")
                            .font(.system(size: 80, weight: .black, design: .monospaced))
                            .foregroundStyle(
                                LinearGradient(
                                    colors: [.jarvisGreen, .jarvisGreenDark],
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                )
                            )
                            .shadow(color: Color.jarvisGreenGlow(opacity: 0.5), radius: 30)
                            .scaleEffect(logoScale)

                        Text(statusSubtitle)
                            .font(.system(size: 21, weight: .medium, design: .monospaced))
                            .foregroundColor(.jarvisGreenDark)
                            .tracking(4.8) // 0.3em letter spacing
                            .textCase(.uppercase)
                            .opacity(0.9)
                    }

                    // Arc Reactor
                    ArcReactorView(state: .idle)
                        .frame(width: 300, height: 300)

                    // Status message
                    Text(statusMessage)
                        .font(.system(size: 16, weight: .semibold, design: .monospaced))
                        .foregroundColor(.jarvisGreen)
                        .tracking(1.2)
                        .textCase(.uppercase)

                    // Progress bar
                    VStack(spacing: 10) {
                        GeometryReader { geometry in
                            ZStack(alignment: .leading) {
                                // Background
                                RoundedRectangle(cornerRadius: 10)
                                    .fill(Color.white.opacity(0.1))
                                    .frame(height: 8)

                                // Progress
                                RoundedRectangle(cornerRadius: 10)
                                    .fill(
                                        LinearGradient(
                                            colors: progressGradient,
                                            startPoint: .leading,
                                            endPoint: .trailing
                                        )
                                    )
                                    .frame(width: geometry.size.width * (progress / 100), height: 8)
                                    .shadow(color: progressShadowColor, radius: progressShadowRadius)
                            }
                        }
                        .frame(height: 8)

                        // Percentage
                        Text("\(Int(progress))%")
                            .font(.system(size: 18, weight: .bold, design: .monospaced))
                            .foregroundColor(.jarvisGreen)
                    }
                    .frame(maxWidth: 500)
                    .padding(.horizontal, 40)

                    Spacer()
                }
                .padding()
            } else {
                // Matrix transition
                MatrixTransitionView()
            }
        }
        .onAppear {
            startLoadingAnimation()
            // Connect to backend for real-time progress
            pythonBridge.connect()
        }
        .onChange(of: pythonBridge.loadingProgress) { newProgress in
            // Update progress from backend in real-time
            withAnimation(.easeInOut(duration: 0.3)) {
                progress = CGFloat(newProgress)
                loadingState = .loading(progress: newProgress, message: pythonBridge.loadingMessage)
            }
        }
        .onChange(of: pythonBridge.loadingComplete) { isComplete in
            // Only transition when backend explicitly signals completion
            if isComplete {
                print("🎯 Backend signaled completion - starting transition animation")
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                    playCompletionAnimation()
                }
            }
        }
    }

    // MARK: - Computed Properties

    private var statusSubtitle: String {
        switch loadingState {
        case .initializing:
            return "INITIALIZING"
        case .loading:
            return "LOADING"
        case .complete:
            return "SYSTEM READY"
        case .failed:
            return "INITIALIZATION FAILED"
        }
    }

    private var statusMessage: String {
        switch loadingState {
        case .initializing:
            return "Initializing JARVIS..."
        case .loading(_, let message):
            return message
        case .complete:
            return "JARVIS is online!"
        case .failed(let error):
            return error
        }
    }

    private var progressGradient: [Color] {
        if progress >= 75 {
            return [.jarvisGreen, .jarvisGreen]
        } else if progress >= 50 {
            return [.jarvisGreenDark, .jarvisGreen]
        } else if progress >= 25 {
            return [.jarvisGreen, .jarvisGreenDark]
        } else {
            return [.jarvisGreenDark, .jarvisGreenDark]
        }
    }

    private var progressShadowColor: Color {
        progress >= 75 ? Color.jarvisGreenGlow(opacity: 0.8) : Color.jarvisGreenGlow(opacity: 0.4)
    }

    private var progressShadowRadius: CGFloat {
        progress >= 75 ? 20 : 10
    }

    // MARK: - Animations

    private func startLoadingAnimation() {
        // Logo pulse
        withAnimation(
            Animation.easeInOut(duration: 2.0)
                .repeatForever(autoreverses: true)
        ) {
            logoScale = 1.05
        }

        // Simulate loading progress
        simulateLoading()
    }

    private func simulateLoading() {
        // Fallback simulation - only runs if backend doesn't connect within 15 seconds
        // (Gives backend more time to start up, especially on slower machines)
        DispatchQueue.main.asyncAfter(deadline: .now() + 15.0) {
            // Only use simulated loading if no real progress received
            if progress == 0 {
                print("⚠️ No backend progress after 15s, using fallback simulation")

                let stages = [
                    (10, "Loading core systems..."),
                    (25, "Initializing AI modules..."),
                    (40, "Connecting to backend..."),
                    (60, "Loading voice recognition..."),
                    (80, "Finalizing setup..."),
                    (100, "Complete!")
                ]

                for (index, stage) in stages.enumerated() {
                    let delay = Double(index + 1) * 0.8
                    DispatchQueue.main.asyncAfter(deadline: .now() + delay) {
                        withAnimation(.easeInOut(duration: 0.5)) {
                            progress = CGFloat(stage.0)
                            loadingState = .loading(progress: stage.0, message: stage.1)
                        }

                        if stage.0 == 100 {
                            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                                playCompletionAnimation()
                            }
                        }
                    }
                }
            }
        }
    }

    private func playCompletionAnimation() {
        loadingState = .complete

        // Wait a moment then show matrix transition
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.6) {
            withAnimation(.easeOut(duration: 0.8)) {
                showMatrixTransition = true
            }

            // Complete after matrix animation
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.5) {
                onComplete()
            }
        }
    }
}

/// Matrix code rain transition effect
struct MatrixTransitionView: View {

    @State private var columns: [[MatrixCharacter]] = []
    @State private var opacity: Double = 0.3

    var body: some View {
        GeometryReader { geometry in
            ZStack {
                Color.black.ignoresSafeArea()

                Canvas { context, size in
                    for column in columns {
                        for char in column {
                            let point = CGPoint(x: char.x, y: char.y)
                            var text = Text(char.character)
                                .font(.system(size: 16, design: .monospaced))
                                .foregroundColor(char.isLead ? .jarvisGreen : Color.jarvisGreen.opacity(0.5))

                            context.draw(text, at: point)
                        }
                    }
                }
                .opacity(opacity)
            }
        }
        .onAppear {
            initializeMatrix()
            startMatrixAnimation()

            withAnimation(.easeOut(duration: 1.0)) {
                opacity = 1.0
            }

            // Fade out
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                withAnimation(.easeOut(duration: 0.5)) {
                    opacity = 0
                }
            }
        }
    }

    private func initializeMatrix() {
        let columnCount = 40
        let characters = "JARVIS01アイウエオカキクケコ"

        for i in 0..<columnCount {
            var column: [MatrixCharacter] = []
            for j in 0..<50 {
                let char = MatrixCharacter(
                    character: String(characters.randomElement()!),
                    x: CGFloat(i * 30),
                    y: CGFloat(j * 20),
                    isLead: j == 0
                )
                column.append(char)
            }
            columns.append(column)
        }
    }

    private func startMatrixAnimation() {
        Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { timer in
            for i in 0..<columns.count {
                for j in 0..<columns[i].count {
                    columns[i][j].y += 2
                    if columns[i][j].y > 1000 {
                        columns[i][j].y = -20
                    }
                }
            }
        }
    }
}

struct MatrixCharacter {
    var character: String
    var x: CGFloat
    var y: CGFloat
    var isLead: Bool
}

#Preview {
    LoadingHUDView {
        print("Loading complete!")
    }
}
