import Foundation
import IOKit
import IOKit.pwr_mgt
import CoreAudio
import SystemConfiguration
import CoreBluetooth

/// Enhanced System Preferences controller with native API implementations
public class SystemPreferences {
    
    // MARK: - Audio Control
    
    public static func setSystemVolume(_ level: Float) throws {
        guard level >= 0 && level <= 1 else {
            throw SystemControlError.invalidParameter("Volume must be between 0 and 1")
        }
        
        var defaultOutputDeviceID = AudioDeviceID(0)
        var defaultOutputDeviceIDSize = UInt32(MemoryLayout.size(ofValue: defaultOutputDeviceID))
        
        var getDefaultOutputDevicePropertyAddress = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultOutputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        
        let status = AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject),
            &getDefaultOutputDevicePropertyAddress,
            0,
            nil,
            &defaultOutputDeviceIDSize,
            &defaultOutputDeviceID
        )
        
        guard status == noErr else {
            throw SystemControlError.operationFailed("Failed to get default output device")
        }
        
        var volume = Float32(level)
        let volumeSize = UInt32(MemoryLayout.size(ofValue: volume))
        
        var volumePropertyAddress = AudioObjectPropertyAddress(
            mSelector: kAudioHardwareServiceDeviceProperty_VirtualMainVolume,
            mScope: kAudioDevicePropertyScopeOutput,
            mElement: kAudioObjectPropertyElementMain
        )
        
        let volumeStatus = AudioObjectSetPropertyData(
            defaultOutputDeviceID,
            &volumePropertyAddress,
            0,
            nil,
            volumeSize,
            &volume
        )
        
        guard volumeStatus == noErr else {
            throw SystemControlError.operationFailed("Failed to set volume")
        }
    }
    
    public static func getSystemVolume() throws -> Float {
        var defaultOutputDeviceID = AudioDeviceID(0)
        var defaultOutputDeviceIDSize = UInt32(MemoryLayout.size(ofValue: defaultOutputDeviceID))
        
        var getDefaultOutputDevicePropertyAddress = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultOutputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        
        let status = AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject),
            &getDefaultOutputDevicePropertyAddress,
            0,
            nil,
            &defaultOutputDeviceIDSize,
            &defaultOutputDeviceID
        )
        
        guard status == noErr else {
            throw SystemControlError.operationFailed("Failed to get default output device")
        }
        
        var volume = Float32(0)
        var volumeSize = UInt32(MemoryLayout.size(ofValue: volume))
        
        var volumePropertyAddress = AudioObjectPropertyAddress(
            mSelector: kAudioHardwareServiceDeviceProperty_VirtualMainVolume,
            mScope: kAudioDevicePropertyScopeOutput,
            mElement: kAudioObjectPropertyElementMain
        )
        
        let volumeStatus = AudioObjectGetPropertyData(
            defaultOutputDeviceID,
            &volumePropertyAddress,
            0,
            nil,
            &volumeSize,
            &volume
        )
        
        guard volumeStatus == noErr else {
            throw SystemControlError.operationFailed("Failed to get volume")
        }
        
        return Float(volume)
    }
    
    // MARK: - Display Brightness
    
    public static func setDisplayBrightness(_ level: Float) throws {
        guard level >= 0 && level <= 1 else {
            throw SystemControlError.invalidParameter("Brightness must be between 0 and 1")
        }
        
        let service = IOServiceGetMatchingService(kIOMasterPortDefault, IOServiceMatching("IODisplayConnect"))
        guard service != 0 else {
            throw SystemControlError.operationFailed("Failed to find display service")
        }
        
        defer { IOObjectRelease(service) }
        
        let brightness = level
        let result = IODisplaySetFloatParameter(service, 0, kIODisplayBrightnessKey as CFString, brightness)
        
        guard result == kIOReturnSuccess else {
            throw SystemControlError.operationFailed("Failed to set brightness")
        }
    }
    
    public static func getDisplayBrightness() throws -> Float {
        let service = IOServiceGetMatchingService(kIOMasterPortDefault, IOServiceMatching("IODisplayConnect"))
        guard service != 0 else {
            throw SystemControlError.operationFailed("Failed to find display service")
        }
        
        defer { IOObjectRelease(service) }
        
        var brightness: Float = 0
        let result = IODisplayGetFloatParameter(service, 0, kIODisplayBrightnessKey as CFString, &brightness)
        
        guard result == kIOReturnSuccess else {
            throw SystemControlError.operationFailed("Failed to get brightness")
        }
        
        return brightness
    }
    
    // MARK: - Network Control
    
    public static func getWiFiStatus() -> Bool {
        let client = CWWiFiClient.shared()
        return client.interface()?.powerOn() ?? false
    }
    
    public static func setWiFiEnabled(_ enabled: Bool) throws {
        let client = CWWiFiClient.shared()
        guard let interface = client.interface() else {
            throw SystemControlError.operationFailed("No WiFi interface found")
        }
        
        do {
            try interface.setPower(enabled)
        } catch {
            throw SystemControlError.operationFailed("Failed to set WiFi power: \(error)")
        }
    }
    
    public static func getCurrentWiFiNetwork() -> String? {
        let client = CWWiFiClient.shared()
        return client.interface()?.ssid()
    }
    
    // MARK: - Bluetooth Control
    
    private static var bluetoothManager: CBCentralManager?
    
    public static func getBluetoothStatus(completion: @escaping (Bool) -> Void) {
        class BluetoothDelegate: NSObject, CBCentralManagerDelegate {
            let completion: (Bool) -> Void
            
            init(completion: @escaping (Bool) -> Void) {
                self.completion = completion
            }
            
            func centralManagerDidUpdateState(_ central: CBCentralManager) {
                completion(central.state == .poweredOn)
            }
        }
        
        let delegate = BluetoothDelegate(completion: completion)
        bluetoothManager = CBCentralManager(delegate: delegate, queue: nil)
    }
    
    // MARK: - Do Not Disturb
    
    public static func setDoNotDisturb(_ enabled: Bool) throws {
        // This requires using private APIs or scripting the Notification Center
        // For now, we'll use AppleScript as a workaround
        let script = """
        tell application "System Preferences"
            reveal anchor "dnd" of pane id "com.apple.preference.notifications"
        end tell
        
        tell application "System Events"
            tell application process "System Preferences"
                if \(enabled ? "not" : "") (exists checkbox 1 of group 1 of window "Notifications") then
                    click checkbox 1 of group 1 of window "Notifications"
                end if
            end tell
        end tell
        
        quit application "System Preferences"
        """
        
        let appleScript = NSAppleScript(source: script)
        var error: NSDictionary?
        appleScript?.executeAndReturnError(&error)
        
        if error != nil {
            throw SystemControlError.operationFailed("Failed to set Do Not Disturb")
        }
    }
    
    // MARK: - Power Management
    
    public static func preventSleep() throws -> IOPMAssertionID {
        var assertionID: IOPMAssertionID = 0
        let reason = "JARVIS System Control Operation" as CFString
        
        let result = IOPMAssertionCreateWithName(
            kIOPMAssertionTypeNoDisplaySleep as CFString,
            IOPMAssertionLevel(kIOPMAssertionLevelOn),
            reason,
            &assertionID
        )
        
        guard result == kIOReturnSuccess else {
            throw SystemControlError.operationFailed("Failed to prevent sleep")
        }
        
        return assertionID
    }
    
    public static func allowSleep(_ assertionID: IOPMAssertionID) {
        IOPMAssertionRelease(assertionID)
    }
    
    // MARK: - System Information
    
    public static func getBatteryInfo() -> [String: Any]? {
        let snapshot = IOPSCopyPowerSourcesInfo()?.takeRetainedValue()
        let sources = IOPSCopyPowerSourcesList(snapshot)?.takeRetainedValue() as? [Any]
        
        guard let source = sources?.first else { return nil }
        
        let info = IOPSGetPowerSourceDescription(snapshot, source as CFTypeRef)?.takeUnretainedValue() as? [String: Any]
        
        return [
            "isCharging": info?[kIOPSIsChargingKey] as? Bool ?? false,
            "currentCapacity": info?[kIOPSCurrentCapacityKey] as? Int ?? 0,
            "maxCapacity": info?[kIOPSMaxCapacityKey] as? Int ?? 100,
            "timeRemaining": info?[kIOPSTimeToEmptyKey] as? Int ?? -1
        ]
    }
    
    public static func getThermalState() -> ProcessInfo.ThermalState {
        return ProcessInfo.processInfo.thermalState
    }
    
    // MARK: - Accessibility
    
    public static func checkAccessibilityPermission() -> Bool {
        let options = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true] as CFDictionary
        return AXIsProcessTrustedWithOptions(options)
    }
    
    public static func requestAccessibilityPermission() {
        let options = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true] as CFDictionary
        AXIsProcessTrustedWithOptions(options)
    }
}

// MARK: - CoreWLAN imports (for WiFi control)

import CoreWLAN

extension CWWiFiClient {
    static func shared() -> CWWiFiClient {
        return CWWiFiClient()
    }
}