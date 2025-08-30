import Foundation
import AppKit
import UniformTypeIdentifiers

/// Advanced clipboard management with history, type detection, and text manipulation
public class ClipboardManager {
    
    // MARK: - Types
    
    public struct ClipboardItem {
        public let content: Any
        public let type: ClipboardContentType
        public let timestamp: Date
        public let sourceApp: String?
        public let metadata: [String: Any]
        
        public var textRepresentation: String? {
            switch type {
            case .text(let string):
                return string
            case .rtf:
                if let data = content as? Data,
                   let attributed = NSAttributedString(rtf: data, documentAttributes: nil) {
                    return attributed.string
                }
            case .html:
                if let data = content as? Data,
                   let html = String(data: data, encoding: .utf8) {
                    return html.replacingOccurrences(of: "<[^>]+>", with: "", options: .regularExpression)
                }
            case .url(let url):
                return url.absoluteString
            case .filePaths(let paths):
                return paths.joined(separator: "\n")
            default:
                return nil
            }
        }
    }
    
    public enum ClipboardContentType: Equatable {
        case text(String)
        case rtf
        case html
        case image
        case pdf
        case url(URL)
        case filePaths([String])
        case custom(UTType)
    }
    
    public struct TextManipulation {
        public enum Operation {
            case uppercase
            case lowercase
            case capitalize
            case reverse
            case removeWhitespace
            case trim
            case base64Encode
            case base64Decode
            case urlEncode
            case urlDecode
            case jsonFormat
            case removeHtmlTags
            case extractUrls
            case extractEmails
            case wordCount
            case characterCount
        }
    }
    
    // MARK: - Properties
    
    private static let shared = ClipboardManager()
    private var history: [ClipboardItem] = []
    private let maxHistorySize = 100
    private var changeObserver: Any?
    private let historyQueue = DispatchQueue(label: "com.jarvis.clipboard.history", attributes: .concurrent)
    
    // MARK: - Initialization
    
    private init() {
        startMonitoring()
    }
    
    // MARK: - Basic Operations
    
    public static func read() -> ClipboardItem? {
        let pasteboard = NSPasteboard.general
        
        // Determine content type
        let types = pasteboard.types ?? []
        let timestamp = Date()
        let sourceApp = NSWorkspace.shared.frontmostApplication?.localizedName
        
        // Text content
        if types.contains(.string), let string = pasteboard.string(forType: .string) {
            return ClipboardItem(
                content: string,
                type: .text(string),
                timestamp: timestamp,
                sourceApp: sourceApp,
                metadata: [:]
            )
        }
        
        // RTF content
        if types.contains(.rtf), let data = pasteboard.data(forType: .rtf) {
            return ClipboardItem(
                content: data,
                type: .rtf,
                timestamp: timestamp,
                sourceApp: sourceApp,
                metadata: ["size": data.count]
            )
        }
        
        // HTML content
        if types.contains(.html), let data = pasteboard.data(forType: .html) {
            return ClipboardItem(
                content: data,
                type: .html,
                timestamp: timestamp,
                sourceApp: sourceApp,
                metadata: ["size": data.count]
            )
        }
        
        // URL
        if types.contains(.URL), let urlString = pasteboard.string(forType: .URL),
           let url = URL(string: urlString) {
            return ClipboardItem(
                content: url,
                type: .url(url),
                timestamp: timestamp,
                sourceApp: sourceApp,
                metadata: ["scheme": url.scheme ?? ""]
            )
        }
        
        // File paths
        if types.contains(.fileURL) {
            if let urls = pasteboard.readObjects(forClasses: [NSURL.self], options: nil) as? [URL] {
                let paths = urls.map { $0.path }
                return ClipboardItem(
                    content: paths,
                    type: .filePaths(paths),
                    timestamp: timestamp,
                    sourceApp: sourceApp,
                    metadata: ["count": paths.count]
                )
            }
        }
        
        // Image
        if types.contains(.tiff) || types.contains(.png), 
           let image = NSImage(pasteboard: pasteboard) {
            return ClipboardItem(
                content: image,
                type: .image,
                timestamp: timestamp,
                sourceApp: sourceApp,
                metadata: [
                    "width": image.size.width,
                    "height": image.size.height
                ]
            )
        }
        
        // PDF
        if types.contains(.pdf), let data = pasteboard.data(forType: .pdf) {
            return ClipboardItem(
                content: data,
                type: .pdf,
                timestamp: timestamp,
                sourceApp: sourceApp,
                metadata: ["size": data.count]
            )
        }
        
        return nil
    }
    
    public static func write(_ text: String) -> Bool {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        return pasteboard.setString(text, forType: .string)
    }
    
    public static func writeRichText(_ attributedString: NSAttributedString) -> Bool {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        return pasteboard.writeObjects([attributedString])
    }
    
    public static func writeImage(_ image: NSImage) -> Bool {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        return pasteboard.writeObjects([image])
    }
    
    public static func writeFiles(_ urls: [URL]) -> Bool {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        return pasteboard.writeObjects(urls as [NSURL])
    }
    
    public static func clear() -> Bool {
        NSPasteboard.general.clearContents()
        return true
    }
    
    // MARK: - History Management
    
    public static func getHistory(limit: Int = 50) -> [ClipboardItem] {
        return shared.historyQueue.sync {
            Array(shared.history.prefix(limit))
        }
    }
    
    public static func clearHistory() {
        shared.historyQueue.async(flags: .barrier) {
            shared.history.removeAll()
        }
    }
    
    public static func restoreFromHistory(at index: Int) -> Bool {
        guard let item = shared.historyQueue.sync(execute: {
            index < shared.history.count ? shared.history[index] : nil
        }) else {
            return false
        }
        
        switch item.type {
        case .text(let string):
            return write(string)
        case .rtf:
            if let data = item.content as? Data,
               let attributed = NSAttributedString(rtf: data, documentAttributes: nil) {
                return writeRichText(attributed)
            }
        case .image:
            if let image = item.content as? NSImage {
                return writeImage(image)
            }
        case .url(let url):
            return write(url.absoluteString)
        case .filePaths(let paths):
            let urls = paths.compactMap { URL(fileURLWithPath: $0) }
            return writeFiles(urls)
        default:
            break
        }
        
        return false
    }
    
    // MARK: - Text Manipulation
    
    public static func manipulateText(_ text: String, operation: TextManipulation.Operation) -> String {
        switch operation {
        case .uppercase:
            return text.uppercased()
            
        case .lowercase:
            return text.lowercased()
            
        case .capitalize:
            return text.capitalized
            
        case .reverse:
            return String(text.reversed())
            
        case .removeWhitespace:
            return text.replacingOccurrences(of: "\\s+", with: "", options: .regularExpression)
            
        case .trim:
            return text.trimmingCharacters(in: .whitespacesAndNewlines)
            
        case .base64Encode:
            return Data(text.utf8).base64EncodedString()
            
        case .base64Decode:
            guard let data = Data(base64Encoded: text),
                  let decoded = String(data: data, encoding: .utf8) else {
                return text
            }
            return decoded
            
        case .urlEncode:
            return text.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? text
            
        case .urlDecode:
            return text.removingPercentEncoding ?? text
            
        case .jsonFormat:
            guard let data = text.data(using: .utf8),
                  let json = try? JSONSerialization.jsonObject(with: data),
                  let formatted = try? JSONSerialization.data(withJSONObject: json, options: .prettyPrinted),
                  let result = String(data: formatted, encoding: .utf8) else {
                return text
            }
            return result
            
        case .removeHtmlTags:
            return text.replacingOccurrences(of: "<[^>]+>", with: "", options: .regularExpression)
            
        case .extractUrls:
            let detector = try? NSDataDetector(types: NSTextCheckingResult.CheckingType.link.rawValue)
            let matches = detector?.matches(in: text, range: NSRange(text.startIndex..., in: text)) ?? []
            return matches.compactMap { match in
                guard let range = Range(match.range, in: text) else { return nil }
                return String(text[range])
            }.joined(separator: "\n")
            
        case .extractEmails:
            let emailRegex = "[A-Z0-9a-z._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,64}"
            let regex = try? NSRegularExpression(pattern: emailRegex)
            let matches = regex?.matches(in: text, range: NSRange(text.startIndex..., in: text)) ?? []
            return matches.compactMap { match in
                guard let range = Range(match.range, in: text) else { return nil }
                return String(text[range])
            }.joined(separator: "\n")
            
        case .wordCount:
            let words = text.components(separatedBy: .whitespacesAndNewlines)
                .filter { !$0.isEmpty }
            return "\(words.count) words"
            
        case .characterCount:
            return "\(text.count) characters"
        }
    }
    
    public static func manipulateClipboard(operation: TextManipulation.Operation) -> Bool {
        guard let current = read(),
              let text = current.textRepresentation else {
            return false
        }
        
        let manipulated = manipulateText(text, operation: operation)
        return write(manipulated)
    }
    
    // MARK: - Smart Features
    
    public static func smartPaste() -> String? {
        guard let item = read() else { return nil }
        
        // Detect context and format accordingly
        if let text = item.textRepresentation {
            // Clean up common formatting issues
            var cleaned = text
            
            // Remove multiple spaces
            cleaned = cleaned.replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
            
            // Fix smart quotes
            cleaned = cleaned.replacingOccurrences(of: """, with: "\"")
            cleaned = cleaned.replacingOccurrences(of: """, with: "\"")
            cleaned = cleaned.replacingOccurrences(of: "'", with: "'")
            cleaned = cleaned.replacingOccurrences(of: "'", with: "'")
            
            // Remove trailing whitespace
            cleaned = cleaned.trimmingCharacters(in: .whitespacesAndNewlines)
            
            return cleaned
        }
        
        return nil
    }
    
    public static func mergeClipboardHistory(separator: String = "\n") -> Bool {
        let items = getHistory(limit: 10)
        let texts = items.compactMap { $0.textRepresentation }
        
        guard !texts.isEmpty else { return false }
        
        let merged = texts.joined(separator: separator)
        return write(merged)
    }
    
    // MARK: - Monitoring
    
    private func startMonitoring() {
        changeObserver = NotificationCenter.default.addObserver(
            forName: NSPasteboard.didChangeNotification,
            object: NSPasteboard.general,
            queue: .main
        ) { [weak self] _ in
            self?.handleClipboardChange()
        }
    }
    
    private func handleClipboardChange() {
        guard let item = Self.read() else { return }
        
        historyQueue.async(flags: .barrier) { [weak self] in
            guard let self = self else { return }
            
            // Don't add duplicates
            if let last = self.history.first,
               self.areItemsEqual(last, item) {
                return
            }
            
            // Add to history
            self.history.insert(item, at: 0)
            
            // Trim history
            if self.history.count > self.maxHistorySize {
                self.history = Array(self.history.prefix(self.maxHistorySize))
            }
        }
    }
    
    private func areItemsEqual(_ item1: ClipboardItem, _ item2: ClipboardItem) -> Bool {
        guard item1.type == item2.type else { return false }
        
        switch (item1.type, item2.type) {
        case (.text(let text1), .text(let text2)):
            return text1 == text2
        case (.url(let url1), .url(let url2)):
            return url1 == url2
        case (.filePaths(let paths1), .filePaths(let paths2)):
            return paths1 == paths2
        default:
            return false
        }
    }
    
    deinit {
        if let observer = changeObserver {
            NotificationCenter.default.removeObserver(observer)
        }
    }
}

// MARK: - Clipboard Utilities

public extension ClipboardManager {
    
    static func copyToClipboardWithNotification(_ text: String, notify: Bool = true) -> Bool {
        let success = write(text)
        
        if success && notify {
            // Post notification for UI feedback
            NotificationCenter.default.post(
                name: NSNotification.Name("JARVISClipboardCopied"),
                object: nil,
                userInfo: ["text": text]
            )
        }
        
        return success
    }
    
    static func getClipboardStats() -> [String: Any] {
        let history = getHistory()
        var stats: [String: Any] = [:]
        
        stats["historyCount"] = history.count
        stats["currentType"] = read()?.type.description ?? "empty"
        
        // Type distribution
        var typeCount: [String: Int] = [:]
        for item in history {
            let typeKey = item.type.description
            typeCount[typeKey, default: 0] += 1
        }
        stats["typeDistribution"] = typeCount
        
        // Source apps
        let sources = history.compactMap { $0.sourceApp }
        let uniqueSources = Set(sources)
        stats["sourceApps"] = Array(uniqueSources)
        
        return stats
    }
}

extension ClipboardManager.ClipboardContentType: CustomStringConvertible {
    public var description: String {
        switch self {
        case .text: return "text"
        case .rtf: return "rtf"
        case .html: return "html"
        case .image: return "image"
        case .pdf: return "pdf"
        case .url: return "url"
        case .filePaths: return "files"
        case .custom(let type): return type.identifier
        }
    }
}