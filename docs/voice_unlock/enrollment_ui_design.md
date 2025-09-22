# Voice Unlock Enrollment UI Design Specification

## Overview
SwiftUI-based enrollment interface with real-time audio visualization, dynamic feedback, and progressive enhancement. Zero hardcoding - all text, colors, and behaviors configurable.

## Design Principles

### 1. **Accessibility First**
- Full VoiceOver support
- Keyboard navigation
- High contrast mode
- Dynamic type support
- Reduced motion options

### 2. **Progressive Disclosure**
- Simple initial view
- Advanced options hidden
- Contextual help
- Step-by-step guidance

### 3. **Real-time Feedback**
- Live audio waveform
- Quality indicators
- Progress visualization
- Clear error states

## Screen Flows

### 1. Welcome Screen
```
+----------------------------------+
|                                  |
|      🎙️ JARVIS Voice Unlock      |
|                                  |
|   "Secure your Mac with your    |
|    unique voice biometrics"      |
|                                  |
|  [Get Started]  [Learn More]     |
|                                  |
+----------------------------------+
```

**Dynamic Elements:**
- Icon: Configurable SF Symbol
- Title: From `config.enrollment.app_title`
- Subtitle: From `config.enrollment.welcome_message`
- Buttons: Localized strings

### 2. Permissions Screen
```
+----------------------------------+
|  Microphone Access Required      |
|                                  |
|  🎤 JARVIS needs access to your  |
|     microphone to recognize      |
|     your voice                   |
|                                  |
|  • Audio is processed locally    |
|  • Never sent to the cloud       |
|  • Deleted after processing      |
|                                  |
|  [Grant Access]  [Not Now]       |
+----------------------------------+
```

**Dynamic Elements:**
- Privacy points from config
- Button actions configurable
- Skip option if already granted

### 3. Noise Calibration
```
+----------------------------------+
|  Setting Up...                   |
|                                  |
|  🔇 Calibrating ambient noise    |
|                                  |
|  ░░░░░░░░░░░░░░░░░░░░ 45%       |
|                                  |
|  Please remain quiet...          |
|                                  |
|  Noise Level: ▁▂▁▃▁▂▁           |
|                                  |
+----------------------------------+
```

**Real-time Updates:**
- Progress bar animation
- Live noise visualization
- Dynamic status text
- Auto-advance when complete

### 4. Voice Enrollment (Main Screen)
```
+----------------------------------+
|  Sample 1 of 3                🔊 |
|                                  |
|  Please say:                     |
|  "Hello JARVIS, this is Alex"    |
|                                  |
|  ╭─────────────────────────╮    |
|  │ ▁▃▅▇▅▃▁ ▂▄▆▄▂ ▁▃▅▇▅▃ │    |
|  │   Live Audio Waveform    │    |
|  ╰─────────────────────────╯    |
|                                  |
|  Quality: ●●●●○ Good            |
|  Volume:  ████████░░ 80%        |
|  Clarity: ●●●●● Excellent       |
|                                  |
|  [Retry] [Skip] [Continue]       |
|                                  |
|  ○ ● ○  Progress               |
+----------------------------------+
```

**Dynamic Components:**
- Phrase: From enrollment service
- Waveform: Real-time visualization
- Quality indicators: Live updates
- Progress dots: Current/total samples
- Buttons: Context-sensitive

### 5. Audio Visualization Components

#### Waveform Visualizer
```swift
struct WaveformView: View {
    @ObservedObject var audioData: AudioVisualizerData
    
    var body: some View {
        GeometryReader { geometry in
            Path { path in
                // Dynamic waveform based on audio buffer
                // Smooth animations with .spring()
                // Configurable colors and stroke
            }
        }
    }
}
```

#### Quality Indicators
```swift
struct QualityIndicator: View {
    let quality: Float // 0.0 - 1.0
    let label: String
    
    var body: some View {
        HStack {
            // Dynamic dot indicators
            ForEach(0..<5) { index in
                Circle()
                    .fill(dotColor(for: index))
                    .frame(width: 8, height: 8)
            }
            Text(qualityLabel)
                .font(.caption)
        }
    }
    
    // Quality thresholds from config
    var qualityLabel: String {
        switch quality {
        case 0.85...: return "Excellent"
        case 0.7...: return "Good"
        case 0.5...: return "Fair"
        default: return "Poor"
        }
    }
}
```

### 6. Sample Review Screen
```
+----------------------------------+
|  Review Your Samples            |
|                                  |
|  Sample 1: ●●●●● Excellent      |
|  Sample 2: ●●●●○ Good           |
|  Sample 3: ●●●○○ Fair - Retry?  |
|                                  |
|  Overall Quality: Good          |
|  Consistency: 92%               |
|                                  |
|  ⚠️ Sample 3 has background     |
|     noise. Consider retrying.    |
|                                  |
|  [Add More] [Retry Low] [Done]  |
+----------------------------------+
```

**Intelligent Feedback:**
- Quality assessment per sample
- Consistency scoring
- Actionable recommendations
- One-click improvements

### 7. Completion Screen
```
+----------------------------------+
|       ✅ Setup Complete!         |
|                                  |
|  Your voice profile has been     |
|  securely saved to Keychain      |
|                                  |
|  🔐 Security Score: 94/100       |
|                                  |
|  You can now unlock your Mac     |
|  by saying your unlock phrase    |
|                                  |
|  [Test Unlock] [Configure] [Done]|
|                                  |
+----------------------------------+
```

## UI Components Library

### 1. **JAudioLevelMeter**
```swift
@available(iOS 14.0, macOS 11.0, *)
struct JAudioLevelMeter: View {
    @Binding var level: Float
    var config: AudioMeterConfig = .default
    
    var body: some View {
        // Smooth, real-time level visualization
        // Configurable colors, size, orientation
        // Accessibility support
    }
}
```

### 2. **JVoiceWaveform**
```swift
struct JVoiceWaveform: View {
    @ObservedObject var audioBuffer: AudioBuffer
    var style: WaveformStyle = .smooth
    
    // Real-time waveform with multiple styles:
    // - .smooth: Curved lines
    // - .bars: Vertical bars
    // - .dots: Particle effect
}
```

### 3. **JQualityRing**
```swift
struct JQualityRing: View {
    let score: Float
    var showLabel: Bool = true
    
    // Circular progress indicator
    // Gradient colors based on score
    // Animated transitions
}
```

### 4. **JPhraseDisplay**
```swift
struct JPhraseDisplay: View {
    let phrase: String
    @State private var highlightedWord: Int = -1
    
    // Word-by-word highlighting
    // Karaoke-style guidance
    // Accessibility annotations
}
```

## Animations & Transitions

### 1. **Enrollment Progress**
```swift
.transition(.asymmetric(
    insertion: .move(edge: .trailing).combined(with: .opacity),
    removal: .move(edge: .leading).combined(with: .opacity)
))
.animation(.spring(response: 0.6, dampingFraction: 0.8))
```

### 2. **Success Feedback**
```swift
// Haptic feedback
HapticFeedback.notification(type: .success)

// Visual celebration
ConfettiView()
    .opacity(showSuccess ? 1 : 0)
    .animation(.easeOut(duration: 2))
```

### 3. **Error States**
```swift
.shake(times: 3)
.foregroundColor(.red)
.hapticFeedback(.error)
```

## Adaptive Layouts

### Compact View (iPhone/Small Window)
- Single column
- Stacked elements
- Collapsible sections
- Swipe navigation

### Regular View (iPad/Mac)
- Two-column layout
- Side panel for help
- Larger visualizations
- Keyboard shortcuts

## Color Schemes

### Light Mode
```swift
extension Color {
    static let jPrimary = Color("JARVISPrimary") // Dynamic
    static let jSecondary = Color("JARVISSecondary")
    static let jSuccess = Color.green
    static let jWarning = Color.orange
    static let jError = Color.red
}
```

### Dark Mode
- Automatic adjustments
- Increased contrast
- Subtle gradients
- Neon accents for JARVIS theme

## Accessibility Features

### 1. **VoiceOver**
```swift
.accessibilityLabel("Say the phrase: \(currentPhrase)")
.accessibilityHint("Double tap to start recording")
.accessibilityValue("Sample \(currentSample) of \(totalSamples)")
```

### 2. **Keyboard Navigation**
- Tab order configuration
- Arrow key support
- Space/Enter activation
- Escape to cancel

### 3. **Dynamic Type**
```swift
.font(.system(.body, design: .rounded))
.dynamicTypeSize(...DynamicTypeSize.xxxLarge)
```

## Error Handling UI

### Common Errors
1. **No Microphone**
   - Clear explanation
   - System settings button
   - Alternative options

2. **Too Noisy**
   - Visual noise indicator
   - Tips for quiet environment
   - Retry with coaching

3. **Poor Quality**
   - Specific feedback
   - Visual guides
   - Progressive hints

## Configuration UI

### Settings Panel
```
+----------------------------------+
|  Voice Unlock Settings          |
|                                  |
|  Phrases                        |
|  ○ Default phrases              |
|  ● Custom phrases               |
|  [Edit Phrases...]              |
|                                  |
|  Security                       |
|  ☑️ Require liveness detection   |
|  ☑️ Anti-spoofing protection    |
|  ○ Basic ● Enhanced ○ Maximum  |
|                                  |
|  Advanced                       |
|  Samples required: [3] ▼        |
|  Quality threshold: ████░ 70%   |
|                                  |
+----------------------------------+
```

## SwiftUI Code Structure

### Main Enrollment View
```swift
struct VoiceEnrollmentView: View {
    @StateObject private var viewModel = EnrollmentViewModel()
    @Environment(\.colorScheme) var colorScheme
    
    var body: some View {
        NavigationView {
            ZStack {
                // Background gradient
                LinearGradient(
                    gradient: backgroundGradient,
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
                .ignoresSafeArea()
                
                // Content
                VStack {
                    switch viewModel.state {
                    case .welcome:
                        WelcomeView()
                    case .permissions:
                        PermissionsView()
                    case .calibrating:
                        CalibrationView()
                    case .enrolling:
                        EnrollmentView()
                    case .reviewing:
                        ReviewView()
                    case .complete:
                        CompletionView()
                    }
                }
            }
        }
        .environmentObject(viewModel)
    }
}
```

### View Model
```swift
class EnrollmentViewModel: ObservableObject {
    @Published var state: EnrollmentState = .welcome
    @Published var samples: [VoiceSample] = []
    @Published var currentPhrase: String = ""
    @Published var audioLevel: Float = 0
    @Published var quality: QualityMetrics = .zero
    
    private let enrollmentService: EnrollmentService
    private let audioCapture: AudioCaptureService
    private let config: VoiceUnlockConfig
    
    // All business logic here
    // No hardcoded values
    // Reactive updates
}
```

## Testing Considerations

### 1. **Audio Simulation**
- Mock audio input for testing
- Quality level simulation
- Error state testing

### 2. **Accessibility Testing**
- VoiceOver navigation
- Keyboard-only usage
- High contrast mode
- Reduced motion

### 3. **Localization**
- All strings from config/resources
- RTL language support
- Dynamic phrase generation

## Performance Optimization

### 1. **Audio Processing**
- Background queue for analysis
- Debounced UI updates
- Memory-efficient buffers

### 2. **Animations**
- GPU-accelerated where possible
- Reduced motion alternatives
- Battery-efficient updates

### 3. **State Management**
- Minimal re-renders
- Computed properties
- Lazy loading of resources

This design provides a professional, accessible, and highly configurable enrollment experience that aligns with the JARVIS brand while maintaining security and usability.