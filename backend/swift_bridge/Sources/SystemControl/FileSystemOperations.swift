import Foundation
import UniformTypeIdentifiers

/// Enhanced file system operations with advanced search and metadata handling
public class FileSystemOperations {
    
    // MARK: - Types
    
    public struct FileSearchOptions {
        public let includeHidden: Bool
        public let recursive: Bool
        public let fileTypes: [UTType]?
        public let sizeRange: ClosedRange<Int>?
        public let dateRange: ClosedRange<Date>?
        public let contentSearch: String?
        public let maxResults: Int
        
        public init(
            includeHidden: Bool = false,
            recursive: Bool = true,
            fileTypes: [UTType]? = nil,
            sizeRange: ClosedRange<Int>? = nil,
            dateRange: ClosedRange<Date>? = nil,
            contentSearch: String? = nil,
            maxResults: Int = 100
        ) {
            self.includeHidden = includeHidden
            self.recursive = recursive
            self.fileTypes = fileTypes
            self.sizeRange = sizeRange
            self.dateRange = dateRange
            self.contentSearch = contentSearch
            self.maxResults = maxResults
        }
    }
    
    public struct FileInfo {
        public let url: URL
        public let name: String
        public let size: Int64
        public let creationDate: Date
        public let modificationDate: Date
        public let isDirectory: Bool
        public let isHidden: Bool
        public let permissions: String
        public let owner: String
        public let group: String
        public let type: UTType?
        public let attributes: [FileAttributeKey: Any]
    }
    
    public struct SearchResult {
        public let file: FileInfo
        public let relevanceScore: Double
        public let matchedContent: String?
    }
    
    // MARK: - File Operations
    
    public static func copyFile(from source: URL, to destination: URL, overwrite: Bool = false) throws {
        let fileManager = FileManager.default
        
        // Check if source exists
        guard fileManager.fileExists(atPath: source.path) else {
            throw SystemControlError.notFound("Source file not found: \(source.path)")
        }
        
        // Check if destination exists
        if fileManager.fileExists(atPath: destination.path) {
            if overwrite {
                try fileManager.removeItem(at: destination)
            } else {
                throw SystemControlError.operationFailed("Destination already exists: \(destination.path)")
            }
        }
        
        // Create destination directory if needed
        let destDir = destination.deletingLastPathComponent()
        if !fileManager.fileExists(atPath: destDir.path) {
            try fileManager.createDirectory(at: destDir, withIntermediateDirectories: true)
        }
        
        // Perform copy
        try fileManager.copyItem(at: source, to: destination)
    }
    
    public static func moveFile(from source: URL, to destination: URL, overwrite: Bool = false) throws {
        let fileManager = FileManager.default
        
        // Check if source exists
        guard fileManager.fileExists(atPath: source.path) else {
            throw SystemControlError.notFound("Source file not found: \(source.path)")
        }
        
        // Check if destination exists
        if fileManager.fileExists(atPath: destination.path) {
            if overwrite {
                try fileManager.removeItem(at: destination)
            } else {
                throw SystemControlError.operationFailed("Destination already exists: \(destination.path)")
            }
        }
        
        // Create destination directory if needed
        let destDir = destination.deletingLastPathComponent()
        if !fileManager.fileExists(atPath: destDir.path) {
            try fileManager.createDirectory(at: destDir, withIntermediateDirectories: true)
        }
        
        // Perform move
        try fileManager.moveItem(at: source, to: destination)
    }
    
    public static func deleteFile(_ url: URL, permanently: Bool = false) throws {
        let fileManager = FileManager.default
        
        guard fileManager.fileExists(atPath: url.path) else {
            throw SystemControlError.notFound("File not found: \(url.path)")
        }
        
        if permanently {
            try fileManager.removeItem(at: url)
        } else {
            // Move to trash
            try fileManager.trashItem(at: url, resultingItemURL: nil)
        }
    }
    
    public static func createDirectory(_ url: URL, withIntermediateDirectories: Bool = true) throws {
        try FileManager.default.createDirectory(
            at: url,
            withIntermediateDirectories: withIntermediateDirectories,
            attributes: nil
        )
    }
    
    // MARK: - Advanced Search
    
    public static func searchFiles(in directory: URL, options: FileSearchOptions) throws -> [SearchResult] {
        var results: [SearchResult] = []
        let fileManager = FileManager.default
        
        // Create enumerator options
        var enumeratorOptions: FileManager.DirectoryEnumerationOptions = []
        if !options.includeHidden {
            enumeratorOptions.insert(.skipsHiddenFiles)
        }
        if !options.recursive {
            enumeratorOptions.insert(.skipsSubdirectoryDescendants)
        }
        
        // Properties to fetch
        let properties: [URLResourceKey] = [
            .nameKey,
            .fileSizeKey,
            .creationDateKey,
            .contentModificationDateKey,
            .isDirectoryKey,
            .isHiddenKey,
            .fileSecurityKey,
            .contentTypeKey
        ]
        
        guard let enumerator = fileManager.enumerator(
            at: directory,
            includingPropertiesForKeys: properties,
            options: enumeratorOptions
        ) else {
            throw SystemControlError.operationFailed("Failed to create file enumerator")
        }
        
        for case let fileURL as URL in enumerator {
            // Skip if we've hit max results
            if results.count >= options.maxResults {
                break
            }
            
            do {
                let fileInfo = try getFileInfo(fileURL)
                
                // Apply filters
                if !matchesFilters(fileInfo, options: options) {
                    continue
                }
                
                // Calculate relevance score
                var relevanceScore = 1.0
                var matchedContent: String?
                
                // Content search if requested
                if let contentSearch = options.contentSearch, !fileInfo.isDirectory {
                    if let (score, content) = try searchFileContent(fileURL, for: contentSearch) {
                        relevanceScore = score
                        matchedContent = content
                    } else {
                        continue // Skip files that don't match content search
                    }
                }
                
                results.append(SearchResult(
                    file: fileInfo,
                    relevanceScore: relevanceScore,
                    matchedContent: matchedContent
                ))
            } catch {
                // Skip files we can't read
                continue
            }
        }
        
        // Sort by relevance
        results.sort { $0.relevanceScore > $1.relevanceScore }
        
        return results
    }
    
    private static func matchesFilters(_ fileInfo: FileInfo, options: FileSearchOptions) -> Bool {
        // File type filter
        if let fileTypes = options.fileTypes {
            guard let fileType = fileInfo.type else { return false }
            let matches = fileTypes.contains { requestedType in
                fileType.conforms(to: requestedType)
            }
            if !matches { return false }
        }
        
        // Size filter
        if let sizeRange = options.sizeRange {
            if !sizeRange.contains(Int(fileInfo.size)) {
                return false
            }
        }
        
        // Date filter
        if let dateRange = options.dateRange {
            if !dateRange.contains(fileInfo.modificationDate) {
                return false
            }
        }
        
        return true
    }
    
    private static func searchFileContent(_ url: URL, for searchTerm: String) throws -> (Double, String)? {
        // Only search text files
        guard let type = try? url.resourceValues(forKeys: [.contentTypeKey]).contentType,
              type.conforms(to: .text) else {
            return nil
        }
        
        // Read file content
        let content = try String(contentsOf: url, encoding: .utf8)
        
        // Search for term
        let lowercasedContent = content.lowercased()
        let lowercasedSearch = searchTerm.lowercased()
        
        guard lowercasedContent.contains(lowercasedSearch) else {
            return nil
        }
        
        // Calculate relevance score based on frequency
        let occurrences = lowercasedContent.components(separatedBy: lowercasedSearch).count - 1
        let relevance = min(Double(occurrences) / 10.0, 1.0)
        
        // Extract context around first match
        if let range = lowercasedContent.range(of: lowercasedSearch) {
            let startIndex = content.index(range.lowerBound, offsetBy: -50, limitedBy: content.startIndex) ?? content.startIndex
            let endIndex = content.index(range.upperBound, offsetBy: 50, limitedBy: content.endIndex) ?? content.endIndex
            let context = String(content[startIndex..<endIndex])
            return (relevance, context)
        }
        
        return (relevance, "")
    }
    
    // MARK: - File Information
    
    public static func getFileInfo(_ url: URL) throws -> FileInfo {
        let fileManager = FileManager.default
        let attributes = try fileManager.attributesOfItem(atPath: url.path)
        
        let resourceValues = try url.resourceValues(forKeys: [
            .nameKey,
            .fileSizeKey,
            .creationDateKey,
            .contentModificationDateKey,
            .isDirectoryKey,
            .isHiddenKey,
            .contentTypeKey
        ])
        
        // Get permissions
        let permissions = (attributes[.posixPermissions] as? NSNumber)?.intValue ?? 0
        let permissionString = String(format: "%o", permissions)
        
        // Get owner and group
        let owner = (attributes[.ownerAccountName] as? String) ?? "unknown"
        let group = (attributes[.groupOwnerAccountName] as? String) ?? "unknown"
        
        return FileInfo(
            url: url,
            name: resourceValues.name ?? url.lastPathComponent,
            size: Int64(resourceValues.fileSize ?? 0),
            creationDate: resourceValues.creationDate ?? Date(),
            modificationDate: resourceValues.contentModificationDate ?? Date(),
            isDirectory: resourceValues.isDirectory ?? false,
            isHidden: resourceValues.isHidden ?? false,
            permissions: permissionString,
            owner: owner,
            group: group,
            type: resourceValues.contentType,
            attributes: attributes
        )
    }
    
    // MARK: - Permissions
    
    public static func setFilePermissions(_ url: URL, permissions: Int) throws {
        let attributes = [FileAttributeKey.posixPermissions: permissions]
        try FileManager.default.setAttributes(attributes, ofItemAtPath: url.path)
    }
    
    public static func setFileOwner(_ url: URL, owner: String?, group: String?) throws {
        var attributes: [FileAttributeKey: Any] = [:]
        
        if let owner = owner {
            attributes[.ownerAccountName] = owner
        }
        
        if let group = group {
            attributes[.groupOwnerAccountName] = group
        }
        
        try FileManager.default.setAttributes(attributes, ofItemAtPath: url.path)
    }
    
    // MARK: - Batch Operations
    
    public static func batchCopy(files: [(source: URL, destination: URL)], 
                                stopOnError: Bool = false) throws -> [URL: Error] {
        var errors: [URL: Error] = [:]
        
        for (source, destination) in files {
            do {
                try copyFile(from: source, to: destination)
            } catch {
                errors[source] = error
                if stopOnError {
                    throw SystemControlError.operationFailed("Batch copy failed at \(source.path): \(error)")
                }
            }
        }
        
        return errors
    }
    
    public static func batchDelete(files: [URL], permanently: Bool = false) throws -> [URL: Error] {
        var errors: [URL: Error] = [:]
        
        for file in files {
            do {
                try deleteFile(file, permanently: permanently)
            } catch {
                errors[file] = error
            }
        }
        
        return errors
    }
    
    // MARK: - Monitoring
    
    public static func monitorDirectory(_ url: URL, 
                                      callback: @escaping (URL, FileSystemEvent) -> Void) -> FileSystemMonitor {
        return FileSystemMonitor(url: url, callback: callback)
    }
}

// MARK: - File System Monitoring

public enum FileSystemEvent {
    case created
    case modified
    case deleted
    case renamed(to: URL)
    case attributesChanged
}

public class FileSystemMonitor {
    private let url: URL
    private let callback: (URL, FileSystemEvent) -> Void
    private var streamRef: FSEventStreamRef?
    private let queue = DispatchQueue(label: "com.jarvis.fsmonitor")
    
    init(url: URL, callback: @escaping (URL, FileSystemEvent) -> Void) {
        self.url = url
        self.callback = callback
        start()
    }
    
    deinit {
        stop()
    }
    
    private func start() {
        let pathsToWatch = [url.path] as CFArray
        
        var context = FSEventStreamContext()
        context.info = Unmanaged.passUnretained(self).toOpaque()
        
        let flags = UInt32(kFSEventStreamCreateFlagUseCFTypes | kFSEventStreamCreateFlagFileEvents)
        
        streamRef = FSEventStreamCreate(
            kCFAllocatorDefault,
            fsEventsCallback,
            &context,
            pathsToWatch,
            FSEventStreamEventId(kFSEventStreamEventIdSinceNow),
            1.0,
            flags
        )
        
        if let stream = streamRef {
            FSEventStreamSetDispatchQueue(stream, queue)
            FSEventStreamStart(stream)
        }
    }
    
    func stop() {
        if let stream = streamRef {
            FSEventStreamStop(stream)
            FSEventStreamInvalidate(stream)
            FSEventStreamRelease(stream)
            streamRef = nil
        }
    }
}

private func fsEventsCallback(
    streamRef: ConstFSEventStreamRef,
    clientCallBackInfo: UnsafeMutableRawPointer?,
    numEvents: Int,
    eventPaths: UnsafeMutableRawPointer,
    eventFlags: UnsafePointer<FSEventStreamEventFlags>,
    eventIds: UnsafePointer<FSEventStreamEventId>
) {
    guard let info = clientCallBackInfo else { return }
    
    let monitor = Unmanaged<FileSystemMonitor>.fromOpaque(info).takeUnretainedValue()
    let paths = unsafeBitCast(eventPaths, to: NSArray.self) as! [String]
    
    for i in 0..<numEvents {
        let path = paths[i]
        let flags = eventFlags[i]
        let url = URL(fileURLWithPath: path)
        
        var event: FileSystemEvent?
        
        if flags & UInt32(kFSEventStreamEventFlagItemCreated) != 0 {
            event = .created
        } else if flags & UInt32(kFSEventStreamEventFlagItemModified) != 0 {
            event = .modified
        } else if flags & UInt32(kFSEventStreamEventFlagItemRemoved) != 0 {
            event = .deleted
        } else if flags & UInt32(kFSEventStreamEventFlagItemRenamed) != 0 {
            // For rename, we'd need to track the rename pairs
            event = .renamed(to: url)
        } else if flags & UInt32(kFSEventStreamEventFlagItemXattrMod) != 0 {
            event = .attributesChanged
        }
        
        if let event = event {
            monitor.callback(url, event)
        }
    }
}