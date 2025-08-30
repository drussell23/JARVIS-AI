// swift-tools-version: 5.7
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "JARVISCommandClassifier",
    platforms: [
        .macOS(.v10_15)
    ],
    products: [
        // Main library
        .library(
            name: "CommandClassifier",
            targets: ["CommandClassifier"]),
        
        // System Control library
        .library(
            name: "SystemControl",
            targets: ["SystemControl"]),
        
        // Performance Core library
        .library(
            name: "PerformanceCore",
            type: .dynamic,
            targets: ["PerformanceCore"]),
        
        // Advanced Command Classifier
        .library(
            name: "AdvancedCommandClassifier",
            targets: ["AdvancedCommandClassifier"]),
        
        // Executable for command-line testing
        .executable(
            name: "jarvis-classifier",
            targets: ["CommandClassifierCLI"]),
        
        // System Control executable
        .executable(
            name: "jarvis-system-control",
            targets: ["SystemControlCLI"]),
        
        // Dynamic library for Python integration
        .library(
            name: "CommandClassifierDynamic",
            type: .dynamic,
            targets: ["CommandClassifier", "SystemControl", "AdvancedCommandClassifier"])
    ],
    dependencies: [
        // No external dependencies - using only Apple frameworks
    ],
    targets: [
        // Main classifier target
        .target(
            name: "CommandClassifier",
            dependencies: [],
            path: "Sources/CommandClassifier"),
        
        // System Control target
        .target(
            name: "SystemControl",
            dependencies: [],
            path: "Sources/SystemControl"),
        
        // Advanced Command Classifier with ML
        .target(
            name: "AdvancedCommandClassifier",
            dependencies: [],
            path: "Sources/AdvancedCommandClassifier"),
        
        // Performance Core - Audio, Vision, System monitoring
        .target(
            name: "PerformanceCore",
            dependencies: [],
            path: "Sources/PerformanceCore",
            linkerSettings: [
                .linkedFramework("Accelerate"),
                .linkedFramework("CoreML"),
                .linkedFramework("Vision"),
                .linkedFramework("Metal"),
                .linkedFramework("MetalPerformanceShaders"),
                .linkedFramework("AVFoundation"),
                .linkedFramework("CoreImage"),
                .linkedFramework("IOKit")
            ]),
        
        // CLI tool for testing
        .executableTarget(
            name: "CommandClassifierCLI",
            dependencies: ["CommandClassifier", "SystemControl"],
            path: "Sources/CommandClassifierCLI"),
        
        // System Control CLI
        .executableTarget(
            name: "SystemControlCLI",
            dependencies: ["SystemControl", "CommandClassifier"],
            path: "Sources/SystemControlCLI"),
        
        // Tests
        .testTarget(
            name: "CommandClassifierTests",
            dependencies: ["CommandClassifier"],
            path: "Tests/CommandClassifierTests"),
    ]
)