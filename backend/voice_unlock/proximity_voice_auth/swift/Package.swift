// swift-tools-version: 5.7
import PackageDescription

let package = Package(
    name: "JARVISProximityAuth",
    platforms: [
        .macOS(.v11)
    ],
    products: [
        .library(
            name: "JARVISProximityAuth",
            targets: ["JARVISProximityAuth"]),
        .executable(
            name: "ProximityService",
            targets: ["ProximityService"])
    ],
    dependencies: [
        // No external dependencies for now
    ],
    targets: [
        .target(
            name: "JARVISProximityAuth",
            dependencies: [],
            path: "Sources/JARVISProximityAuth"),
        .executableTarget(
            name: "ProximityService",
            dependencies: ["JARVISProximityAuth"],
            path: "Sources/ProximityService")
    ]
)