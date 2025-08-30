import Foundation
import IOKit
import IOKit.ps
import SystemConfiguration

/// High-performance system monitor for resource tracking
/// Uses IOKit for direct hardware access with minimal overhead
@available(macOS 10.15, *)
public class SystemMonitor {
    
    // MARK: - Properties
    private let updateInterval: TimeInterval
    private var timer: Timer?
    private let queue = DispatchQueue(label: "system.monitor", qos: .utility)
    
    // Cached values to reduce system calls
    private var cachedMetrics = SystemMetrics()
    private let cacheDuration: TimeInterval = 1.0
    private var lastUpdateTime: TimeInterval = 0
    
    // IOKit references
    private var cpuInfo: processor_info_array_t?
    private var prevCpuInfo: processor_info_array_t?
    private var numCpuInfo: mach_msg_type_number_t = 0
    private var numPrevCpuInfo: mach_msg_type_number_t = 0
    
    // MARK: - Types
    public struct SystemMetrics: Codable {
        var cpuUsagePercent: Double = 0
        var memoryUsedMB: Int = 0
        var memoryAvailableMB: Int = 0
        var memoryTotalMB: Int = 0
        var memoryPressure: MemoryPressure = .normal
        var diskReadBytesPerSec: Int64 = 0
        var diskWriteBytesPerSec: Int64 = 0
        var networkInBytesPerSec: Int64 = 0
        var networkOutBytesPerSec: Int64 = 0
        var thermalState: ThermalState = .nominal
        var timestamp: Double = Date().timeIntervalSince1970
    }
    
    public enum MemoryPressure: String, Codable {
        case normal
        case warning
        case urgent
        case critical
    }
    
    public enum ThermalState: String, Codable {
        case nominal
        case fair
        case serious
        case critical
    }
    
    // MARK: - Initialization
    public init(updateInterval: TimeInterval = 5.0) {
        self.updateInterval = updateInterval
    }
    
    deinit {
        stopMonitoring()
        deallocateCPUInfo()
    }
    
    // MARK: - Public Methods
    
    /// Get current system metrics with caching
    public func getCurrentMetrics() -> SystemMetrics {
        let now = Date().timeIntervalSince1970
        
        // Return cached value if still fresh
        if now - lastUpdateTime < cacheDuration {
            return cachedMetrics
        }
        
        // Update metrics
        queue.sync {
            updateMetrics()
        }
        
        return cachedMetrics
    }
    
    /// Start continuous monitoring
    public func startMonitoring(callback: @escaping (SystemMetrics) -> Void) {
        stopMonitoring()
        
        timer = Timer.scheduledTimer(withTimeInterval: updateInterval, repeats: true) { [weak self] _ in
            guard let self = self else { return }
            
            self.queue.async {
                self.updateMetrics()
                callback(self.cachedMetrics)
            }
        }
    }
    
    /// Stop monitoring
    public func stopMonitoring() {
        timer?.invalidate()
        timer = nil
    }
    
    /// Get detailed memory info
    public func getMemoryInfo() -> (used: Int, available: Int, total: Int, pressure: MemoryPressure) {
        let info = getVMStatistics()
        
        let pageSize = vm_kernel_page_size
        let total = Int(ProcessInfo.processInfo.physicalMemory / 1024 / 1024)
        let free = Int(info.free_count) * Int(pageSize) / 1024 / 1024
        let inactive = Int(info.inactive_count) * Int(pageSize) / 1024 / 1024
        let available = free + inactive
        let used = total - available
        
        // Calculate memory pressure
        let pressure: MemoryPressure
        let usagePercent = Double(used) / Double(total) * 100
        
        switch usagePercent {
        case 0..<60:
            pressure = .normal
        case 60..<75:
            pressure = .warning
        case 75..<85:
            pressure = .urgent
        default:
            pressure = .critical
        }
        
        return (used: used, available: available, total: total, pressure: pressure)
    }
    
    /// Get CPU usage with minimal overhead
    public func getCPUUsage() -> Double {
        var cpuUsagePercent: Double = 0
        var newCpuInfo: processor_info_array_t?
        var newNumCpuInfo: mach_msg_type_number_t = 0
        
        // Get CPU info
        var cpuInfoCount: mach_msg_type_number_t = 0
        let result = host_processor_info(
            mach_host_self(),
            PROCESSOR_CPU_LOAD_INFO,
            &cpuInfoCount,
            &newCpuInfo,
            &newNumCpuInfo
        )
        
        guard result == KERN_SUCCESS, let currentCpuInfo = newCpuInfo else {
            return 0
        }
        
        if let prevCpuInfo = prevCpuInfo {
            // Calculate CPU usage
            let cpuCount = Int(newNumCpuInfo) / Int(CPU_STATE_MAX)
            var totalUsage: Double = 0
            
            for i in 0..<cpuCount {
                let offset = Int(CPU_STATE_MAX) * i
                
                let userDiff = Double(currentCpuInfo[offset + Int(CPU_STATE_USER)] - prevCpuInfo[offset + Int(CPU_STATE_USER)])
                let systemDiff = Double(currentCpuInfo[offset + Int(CPU_STATE_SYSTEM)] - prevCpuInfo[offset + Int(CPU_STATE_SYSTEM)])
                let idleDiff = Double(currentCpuInfo[offset + Int(CPU_STATE_IDLE)] - prevCpuInfo[offset + Int(CPU_STATE_IDLE)])
                let niceDiff = Double(currentCpuInfo[offset + Int(CPU_STATE_NICE)] - prevCpuInfo[offset + Int(CPU_STATE_NICE)])
                
                let total = userDiff + systemDiff + idleDiff + niceDiff
                if total > 0 {
                    let usage = (userDiff + systemDiff + niceDiff) / total * 100
                    totalUsage += usage
                }
            }
            
            cpuUsagePercent = totalUsage / Double(cpuCount)
        }
        
        // Swap and deallocate old info
        deallocatePrevCPUInfo()
        self.prevCpuInfo = newCpuInfo
        self.numPrevCpuInfo = newNumCpuInfo
        self.cpuInfo = newCpuInfo
        self.numCpuInfo = newNumCpuInfo
        
        return cpuUsagePercent
    }
    
    /// Check thermal state
    public func getThermalState() -> ThermalState {
        let state = ProcessInfo.processInfo.thermalState
        
        switch state {
        case .nominal:
            return .nominal
        case .fair:
            return .fair
        case .serious:
            return .serious
        case .critical:
            return .critical
        @unknown default:
            return .nominal
        }
    }
    
    // MARK: - Private Methods
    
    private func updateMetrics() {
        let now = Date().timeIntervalSince1970
        
        // CPU usage
        cachedMetrics.cpuUsagePercent = getCPUUsage()
        
        // Memory info
        let memory = getMemoryInfo()
        cachedMetrics.memoryUsedMB = memory.used
        cachedMetrics.memoryAvailableMB = memory.available
        cachedMetrics.memoryTotalMB = memory.total
        cachedMetrics.memoryPressure = memory.pressure
        
        // Thermal state
        cachedMetrics.thermalState = getThermalState()
        
        // Update timestamp
        cachedMetrics.timestamp = now
        lastUpdateTime = now
    }
    
    private func getVMStatistics() -> vm_statistics64 {
        var info = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size) / 4
        
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(
                    mach_host_self(),
                    HOST_VM_INFO64,
                    $0,
                    &count
                )
            }
        }
        
        guard result == KERN_SUCCESS else {
            return vm_statistics64()
        }
        
        return info
    }
    
    private func deallocateCPUInfo() {
        if let cpuInfo = cpuInfo {
            vm_deallocate(
                mach_task_self_,
                vm_address_t(bitPattern: cpuInfo),
                vm_size_t(numCpuInfo)
            )
            self.cpuInfo = nil
        }
    }
    
    private func deallocatePrevCPUInfo() {
        if let prevCpuInfo = prevCpuInfo {
            vm_deallocate(
                mach_task_self_,
                vm_address_t(bitPattern: prevCpuInfo),
                vm_size_t(numPrevCpuInfo)
            )
            self.prevCpuInfo = nil
        }
    }
}

// MARK: - C Interface for Python
private var globalMonitor: SystemMonitor?

@_cdecl("system_monitor_create")
public func system_monitor_create(updateInterval: Double) -> Bool {
    globalMonitor = SystemMonitor(updateInterval: updateInterval)
    return true
}

@_cdecl("system_monitor_get_metrics")
public func system_monitor_get_metrics(
    cpuUsage: UnsafeMutablePointer<Double>,
    memoryUsed: UnsafeMutablePointer<Int32>,
    memoryAvailable: UnsafeMutablePointer<Int32>,
    memoryTotal: UnsafeMutablePointer<Int32>
) -> Bool {
    guard let monitor = globalMonitor else { return false }
    
    let metrics = monitor.getCurrentMetrics()
    
    cpuUsage.pointee = metrics.cpuUsagePercent
    memoryUsed.pointee = Int32(metrics.memoryUsedMB)
    memoryAvailable.pointee = Int32(metrics.memoryAvailableMB)
    memoryTotal.pointee = Int32(metrics.memoryTotalMB)
    
    return true
}

@_cdecl("system_monitor_get_memory_pressure")
public func system_monitor_get_memory_pressure() -> UnsafePointer<CChar>? {
    guard let monitor = globalMonitor else { return nil }
    
    let metrics = monitor.getCurrentMetrics()
    let pressure = metrics.memoryPressure.rawValue
    
    return (pressure as NSString).utf8String
}

@_cdecl("system_monitor_destroy")
public func system_monitor_destroy() {
    globalMonitor?.stopMonitoring()
    globalMonitor = nil
}