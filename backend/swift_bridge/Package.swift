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
        
        // Executable for command-line testing
        .executable(
            name: "jarvis-classifier",
            targets: ["CommandClassifierCLI"]),
        
        // Dynamic library for Python integration
        .library(
            name: "CommandClassifierDynamic",
            type: .dynamic,
            targets: ["CommandClassifier"])
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
        
        // CLI tool for testing
        .executableTarget(
            name: "CommandClassifierCLI",
            dependencies: ["CommandClassifier"],
            path: "Sources/CommandClassifierCLI"),
        
        // Tests
        .testTarget(
            name: "CommandClassifierTests",
            dependencies: ["CommandClassifier"],
            path: "Tests/CommandClassifierTests"),
    ]
)