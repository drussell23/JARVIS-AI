import Foundation
import SystemControl
import CommandClassifier

/// JARVIS System Control CLI
/// Provides command-line interface for system control operations
@main
struct SystemControlCLI {
    static func main() async {
        let arguments = CommandLine.arguments
        
        // Check for JSON mode
        if arguments.contains("--json") {
            await handleJSONMode()
        } else {
            await handleInteractiveMode()
        }
    }
    
    static func handleJSONMode() async {
        guard let jsonIndex = CommandLine.arguments.firstIndex(of: "--json"),
              jsonIndex + 1 < CommandLine.arguments.count else {
            printError("Missing JSON data after --json flag")
            exit(1)
        }
        
        let jsonString = CommandLine.arguments[jsonIndex + 1]
        
        guard let jsonData = jsonString.data(using: .utf8),
              let command = try? JSONDecoder().decode(SystemCommand.self, from: jsonData) else {
            printError("Invalid JSON command")
            exit(1)
        }
        
        // Execute command
        let result = await executeCommand(command)
        
        // Output result as JSON
        if let resultData = try? JSONEncoder().encode(result),
           let resultString = String(data: resultData, encoding: .utf8) {
            print(resultString)
        } else {
            printError("Failed to encode result")
            exit(1)
        }
    }
    
    static func handleInteractiveMode() async {
        print("JARVIS System Control v1.0")
        print("Type 'help' for available commands or 'exit' to quit")
        print("")
        
        while true {
            print("> ", terminator: "")
            fflush(stdout)
            
            guard let input = readLine()?.trimmingCharacters(in: .whitespacesAndNewlines) else {
                break
            }
            
            if input.lowercased() == "exit" {
                break
            }
            
            if input.lowercased() == "help" {
                printHelp()
                continue
            }
            
            // Use command classifier to understand the command
            let classifier = CommandClassifier()
            let analysis = classifier.analyzeCommand(input)
            
            // Convert to system command
            if let command = convertToSystemCommand(analysis) {
                let result = await executeCommand(command)
                printResult(result)
            } else {
                print("Command not understood: \(input)")
                print("Classification: \(analysis.type.rawValue) - \(analysis.intent.rawValue)")
            }
        }
    }
    
    static func executeCommand(_ command: SystemCommand) async -> SystemCommandResult {
        let systemControl = SystemControl()
        let startTime = Date()
        
        do {
            let operation: SystemOperation
            
            switch command.type {
            case "app_lifecycle":
                operation = try createAppOperation(command)
            case "system_preference":
                operation = try createPreferenceOperation(command)
            case "file_system":
                operation = try createFileOperation(command)
            case "clipboard":
                operation = try createClipboardOperation(command)
            default:
                throw SystemControlError.invalidParameter("Unknown operation type: \(command.type)")
            }
            
            let result = try await systemControl.execute(operation)
            
            return SystemCommandResult(
                success: result.success,
                data: serializeResult(result.result),
                error: nil,
                executionTime: Date().timeIntervalSince(startTime)
            )
            
        } catch {
            return SystemCommandResult(
                success: false,
                data: nil,
                error: error.localizedDescription,
                executionTime: Date().timeIntervalSince(startTime)
            )
        }
    }
    
    static func createAppOperation(_ command: SystemCommand) throws -> SystemOperation {
        guard let operation = command.operation else {
            throw SystemControlError.invalidParameter("Missing operation")
        }
        
        switch operation {
        case "launch":
            guard let bundleId = command.parameters["bundle_identifier"] as? String else {
                throw SystemControlError.invalidParameter("Missing bundle_identifier")
            }
            return .appLifecycle(.launch(bundleIdentifier: bundleId))
            
        case "close":
            guard let bundleId = command.parameters["bundle_identifier"] as? String else {
                throw SystemControlError.invalidParameter("Missing bundle_identifier")
            }
            let force = command.parameters["force"] as? Bool ?? false
            return .appLifecycle(.close(bundleIdentifier: bundleId, force: force))
            
        case "minimize":
            guard let bundleId = command.parameters["bundle_identifier"] as? String else {
                throw SystemControlError.invalidParameter("Missing bundle_identifier")
            }
            return .appLifecycle(.minimize(bundleIdentifier: bundleId))
            
        case "maximize":
            guard let bundleId = command.parameters["bundle_identifier"] as? String else {
                throw SystemControlError.invalidParameter("Missing bundle_identifier")
            }
            return .appLifecycle(.maximize(bundleIdentifier: bundleId))
            
        case "hide":
            guard let bundleId = command.parameters["bundle_identifier"] as? String else {
                throw SystemControlError.invalidParameter("Missing bundle_identifier")
            }
            return .appLifecycle(.hide(bundleIdentifier: bundleId))
            
        case "switch_to":
            guard let bundleId = command.parameters["bundle_identifier"] as? String else {
                throw SystemControlError.invalidParameter("Missing bundle_identifier")
            }
            return .appLifecycle(.switchTo(bundleIdentifier: bundleId))
            
        case "list_running":
            return .appLifecycle(.listRunning)
            
        case "get_info":
            guard let bundleId = command.parameters["bundle_identifier"] as? String else {
                throw SystemControlError.invalidParameter("Missing bundle_identifier")
            }
            return .appLifecycle(.getInfo(bundleIdentifier: bundleId))
            
        default:
            throw SystemControlError.invalidParameter("Unknown app operation: \(operation)")
        }
    }
    
    static func createPreferenceOperation(_ command: SystemCommand) throws -> SystemOperation {
        guard let operation = command.operation else {
            throw SystemControlError.invalidParameter("Missing operation")
        }
        
        switch operation {
        case "set_volume":
            guard let level = command.parameters["level"] as? Double else {
                throw SystemControlError.invalidParameter("Missing level parameter")
            }
            return .systemPreference(.setVolume(Float(level)))
            
        case "set_brightness":
            guard let level = command.parameters["level"] as? Double else {
                throw SystemControlError.invalidParameter("Missing level parameter")
            }
            return .systemPreference(.setBrightness(Float(level)))
            
        case "toggle_wifi":
            guard let enabled = command.parameters["enabled"] as? Bool else {
                throw SystemControlError.invalidParameter("Missing enabled parameter")
            }
            return .systemPreference(.toggleWiFi(enabled))
            
        case "toggle_bluetooth":
            guard let enabled = command.parameters["enabled"] as? Bool else {
                throw SystemControlError.invalidParameter("Missing enabled parameter")
            }
            return .systemPreference(.toggleBluetooth(enabled))
            
        case "set_do_not_disturb":
            guard let enabled = command.parameters["enabled"] as? Bool else {
                throw SystemControlError.invalidParameter("Missing enabled parameter")
            }
            return .systemPreference(.setDoNotDisturb(enabled))
            
        case "get_dark_mode":
            return .systemPreference(.getDarkMode)
            
        case "set_dark_mode":
            guard let enabled = command.parameters["enabled"] as? Bool else {
                throw SystemControlError.invalidParameter("Missing enabled parameter")
            }
            return .systemPreference(.setDarkMode(enabled))
            
        case "get_system_info":
            return .systemPreference(.getSystemInfo)
            
        default:
            throw SystemControlError.invalidParameter("Unknown preference operation: \(operation)")
        }
    }
    
    static func createFileOperation(_ command: SystemCommand) throws -> SystemOperation {
        guard let operation = command.operation else {
            throw SystemControlError.invalidParameter("Missing operation")
        }
        
        switch operation {
        case "copy":
            guard let source = command.parameters["source"] as? String,
                  let destination = command.parameters["destination"] as? String else {
                throw SystemControlError.invalidParameter("Missing source or destination")
            }
            return .fileSystem(.copy(from: URL(fileURLWithPath: source), 
                                   to: URL(fileURLWithPath: destination)))
            
        case "move":
            guard let source = command.parameters["source"] as? String,
                  let destination = command.parameters["destination"] as? String else {
                throw SystemControlError.invalidParameter("Missing source or destination")
            }
            return .fileSystem(.move(from: URL(fileURLWithPath: source), 
                                   to: URL(fileURLWithPath: destination)))
            
        case "delete":
            guard let path = command.parameters["path"] as? String else {
                throw SystemControlError.invalidParameter("Missing path")
            }
            return .fileSystem(.delete(URL(fileURLWithPath: path)))
            
        case "create_directory":
            guard let path = command.parameters["path"] as? String else {
                throw SystemControlError.invalidParameter("Missing path")
            }
            return .fileSystem(.createDirectory(URL(fileURLWithPath: path)))
            
        case "search":
            guard let query = command.parameters["query"] as? String,
                  let directory = command.parameters["directory"] as? String else {
                throw SystemControlError.invalidParameter("Missing query or directory")
            }
            return .fileSystem(.search(query: query, in: URL(fileURLWithPath: directory)))
            
        case "get_info":
            guard let path = command.parameters["path"] as? String else {
                throw SystemControlError.invalidParameter("Missing path")
            }
            return .fileSystem(.getInfo(URL(fileURLWithPath: path)))
            
        case "set_permissions":
            guard let path = command.parameters["path"] as? String,
                  let permissions = command.parameters["permissions"] as? Int else {
                throw SystemControlError.invalidParameter("Missing path or permissions")
            }
            return .fileSystem(.setPermissions(URL(fileURLWithPath: path), permissions: permissions))
            
        default:
            throw SystemControlError.invalidParameter("Unknown file operation: \(operation)")
        }
    }
    
    static func createClipboardOperation(_ command: SystemCommand) throws -> SystemOperation {
        guard let operation = command.operation else {
            throw SystemControlError.invalidParameter("Missing operation")
        }
        
        switch operation {
        case "read":
            return .clipboard(.read)
            
        case "write":
            guard let text = command.parameters["text"] as? String else {
                throw SystemControlError.invalidParameter("Missing text parameter")
            }
            return .clipboard(.write(text))
            
        case "clear":
            return .clipboard(.clear)
            
        case "get_history":
            return .clipboard(.getHistory)
            
        default:
            throw SystemControlError.invalidParameter("Unknown clipboard operation: \(operation)")
        }
    }
    
    static func convertToSystemCommand(_ analysis: CommandAnalysis) -> SystemCommand? {
        var type: String?
        var operation: String?
        var parameters: [String: Any] = [:]
        
        // Map command type and intent to system operations
        switch (analysis.type, analysis.intent) {
        case (.system, .openApp):
            type = "app_lifecycle"
            operation = "launch"
            if let app = analysis.entities["app"] {
                parameters["bundle_identifier"] = bundleIdentifierForApp(app)
            }
            
        case (.system, .closeApp):
            type = "app_lifecycle"
            operation = "close"
            if let app = analysis.entities["app"] {
                parameters["bundle_identifier"] = bundleIdentifierForApp(app)
            }
            
        case (.system, .switchApp):
            type = "app_lifecycle"
            operation = "switch_to"
            if let app = analysis.entities["app"] {
                parameters["bundle_identifier"] = bundleIdentifierForApp(app)
            }
            
        case (.vision, .analyzeScreen):
            // This would be handled by vision system
            return nil
            
        default:
            return nil
        }
        
        guard let cmdType = type, let cmdOp = operation else {
            return nil
        }
        
        return SystemCommand(
            type: cmdType,
            operation: cmdOp,
            parameters: parameters
        )
    }
    
    static func bundleIdentifierForApp(_ appName: String) -> String {
        // Common app mappings
        let mappings: [String: String] = [
            "safari": "com.apple.Safari",
            "mail": "com.apple.mail",
            "messages": "com.apple.MobileSMS",
            "slack": "com.tinyspeck.slackmacgap",
            "chrome": "com.google.Chrome",
            "firefox": "org.mozilla.firefox",
            "vscode": "com.microsoft.VSCode",
            "terminal": "com.apple.Terminal",
            "finder": "com.apple.finder",
            "xcode": "com.apple.dt.Xcode"
        ]
        
        return mappings[appName.lowercased()] ?? "com.apple.\(appName)"
    }
    
    static func serializeResult(_ result: Any?) -> [String: Any]? {
        guard let result = result else { return nil }
        
        // Convert result to dictionary
        if let dict = result as? [String: Any] {
            return dict
        } else if let array = result as? [[String: Any]] {
            return ["items": array]
        } else {
            return ["value": String(describing: result)]
        }
    }
    
    static func printResult(_ result: SystemCommandResult) {
        if result.success {
            print("✅ Success")
            if let data = result.data {
                print("Result: \(data)")
            }
        } else {
            print("❌ Failed")
            if let error = result.error {
                print("Error: \(error)")
            }
        }
        print("Execution time: \(String(format: "%.3f", result.executionTime))s")
    }
    
    static func printError(_ message: String) {
        FileHandle.standardError.write(Data("Error: \(message)\n".utf8))
    }
    
    static func printHelp() {
        print("""
        
        JARVIS System Control Commands:
        
        App Control:
          open <app>         - Launch an application
          close <app>        - Close an application
          switch to <app>    - Switch to an application
          list apps          - List running applications
        
        System Preferences:
          set volume <0-100> - Set system volume
          set brightness <0-100> - Set display brightness
          toggle wifi on/off - Toggle WiFi
          dark mode on/off   - Toggle dark mode
        
        File Operations:
          copy <src> <dst>   - Copy file or directory
          move <src> <dst>   - Move file or directory
          delete <path>      - Delete file or directory
          search <query>     - Search for files
        
        Clipboard:
          read clipboard     - Read clipboard content
          write clipboard    - Write to clipboard
          clear clipboard    - Clear clipboard
        
        General:
          help              - Show this help
          exit              - Exit the program
        
        """)
    }
}

// MARK: - Data Structures

struct SystemCommand: Codable {
    let type: String
    let operation: String?
    let parameters: [String: Any]
    
    enum CodingKeys: String, CodingKey {
        case type, operation, parameters
    }
    
    init(type: String, operation: String?, parameters: [String: Any]) {
        self.type = type
        self.operation = operation
        self.parameters = parameters
    }
    
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        type = try container.decode(String.self, forKey: .type)
        operation = try container.decodeIfPresent(String.self, forKey: .operation)
        
        // Decode parameters as generic dictionary
        if let params = try? container.decode([String: String].self, forKey: .parameters) {
            parameters = params
        } else if let params = try? container.decode([String: Int].self, forKey: .parameters) {
            parameters = params
        } else if let params = try? container.decode([String: Double].self, forKey: .parameters) {
            parameters = params
        } else if let params = try? container.decode([String: Bool].self, forKey: .parameters) {
            parameters = params
        } else {
            parameters = [:]
        }
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(type, forKey: .type)
        try container.encodeIfPresent(operation, forKey: .operation)
        // Parameters encoding would need custom implementation
    }
}

struct SystemCommandResult: Codable {
    let success: Bool
    let data: [String: Any]?
    let error: String?
    let executionTime: Double
}