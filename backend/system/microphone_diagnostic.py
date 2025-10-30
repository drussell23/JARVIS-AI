#!/usr/bin/env python3
"""
Comprehensive Microphone Diagnostic and Auto-Fix System

This module provides automated detection and resolution of microphone issues for JARVIS.
It performs system-level diagnostics, identifies blocking applications, tests microphone
access, and applies automatic fixes where possible.

The diagnostic system supports multiple platforms with enhanced functionality on macOS,
including Core Audio service management, TCC permission checking, and browser compatibility
testing for Web Speech API integration.

Example:
    >>> diagnostic = MicrophoneDiagnostic()
    >>> results = diagnostic.run_diagnostic()
    >>> print(diagnostic.generate_report(results))
"""

import subprocess
import platform
import json
import time
import os
import sys
import psutil
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class MicrophoneStatus(Enum):
    """Enumeration of possible microphone status states.
    
    Attributes:
        AVAILABLE: Microphone is accessible and ready for use
        BUSY: Microphone is currently in use by another application
        PERMISSION_DENIED: Application lacks necessary permissions
        NOT_FOUND: No microphone devices detected
        ERROR: General error or unknown status
    """
    AVAILABLE = "available"
    BUSY = "busy"
    PERMISSION_DENIED = "permission_denied"
    NOT_FOUND = "not_found"
    ERROR = "error"

class BrowserType(Enum):
    """Enumeration of supported web browsers for Web Speech API.
    
    Attributes:
        CHROME: Google Chrome browser
        SAFARI: Apple Safari browser
        FIREFOX: Mozilla Firefox browser
        EDGE: Microsoft Edge browser
        UNKNOWN: Unidentified or unsupported browser
    """
    CHROME = "chrome"
    SAFARI = "safari"
    FIREFOX = "firefox"
    EDGE = "edge"
    UNKNOWN = "unknown"

@dataclass
class DiagnosticResult:
    """Result of a single diagnostic check.
    
    Attributes:
        check_name: Human-readable name of the diagnostic check
        status: Whether the check passed (True) or failed (False)
        message: Descriptive message about the check result
        fix_available: Whether an automatic fix is available
        fix_command: Optional command to execute for fixing the issue
    """
    check_name: str
    status: bool
    message: str
    fix_available: bool = False
    fix_command: Optional[str] = None

@dataclass
class MicrophoneDevice:
    """Information about a detected microphone device.
    
    Attributes:
        name: Display name of the microphone device
        device_id: System identifier for the device
        is_default: Whether this is the default input device
        is_available: Whether the device is currently available for use
    """
    name: str
    device_id: str
    is_default: bool
    is_available: bool

class MicrophoneDiagnostic:
    """
    Comprehensive microphone diagnostic and auto-fix system.
    
    This class provides a complete suite of diagnostic tools for detecting and
    resolving microphone-related issues. It can identify blocking applications,
    test device access, check permissions, and apply automatic fixes.
    
    Attributes:
        platform: Current operating system platform
        diagnostic_results: List of completed diagnostic check results
        blocking_apps: List of applications currently using the microphone
        available_devices: List of detected microphone devices
        audio_apps: Known applications that commonly use microphone access
    
    Example:
        >>> diagnostic = MicrophoneDiagnostic()
        >>> results = diagnostic.run_diagnostic()
        >>> if results['status'] == MicrophoneStatus.AVAILABLE:
        ...     print("Microphone is ready!")
    """
    
    def __init__(self):
        """Initialize the diagnostic system with platform detection and app lists."""
        self.platform = platform.system()
        self.diagnostic_results: List[DiagnosticResult] = []
        self.blocking_apps: List[str] = []
        self.available_devices: List[MicrophoneDevice] = []
        
        # Common apps that use microphone
        self.audio_apps = [
            "zoom.us", "Teams", "Discord", "Slack", "Skype",
            "FaceTime", "WhatsApp", "Telegram", "Signal",
            "OBS", "QuickTime Player", "Voice Memos", "GarageBand",
            "Audacity", "ScreenFloat", "CleanMyMac", "Loom",
            "Chrome", "Safari", "Firefox", "Edge"
        ]
        
    def run_diagnostic(self) -> Dict[str, Any]:
        """Run the complete diagnostic suite and return comprehensive results.
        
        Performs all available diagnostic checks including platform compatibility,
        blocking application detection, device enumeration, browser compatibility,
        microphone access testing, and automatic fix application.
        
        Returns:
            Dict containing diagnostic results with the following keys:
                - timestamp: When the diagnostic was run
                - platform: Operating system platform
                - checks: List of diagnostic check results
                - devices: List of detected microphone devices
                - blocking_apps: List of applications using microphone
                - fixes_applied: List of automatic fixes that were applied
                - status: Overall microphone status
                - recommendations: List of user recommendations
        
        Example:
            >>> diagnostic = MicrophoneDiagnostic()
            >>> results = diagnostic.run_diagnostic()
            >>> print(f"Status: {results['status']}")
            >>> print(f"Found {len(results['devices'])} devices")
        """
        print("\n🔍 JARVIS Microphone Diagnostic System")
        print("=" * 50)
        
        results: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "platform": self.platform,
            "checks": [],
            "devices": [],
            "blocking_apps": [],
            "fixes_applied": [],
            "status": MicrophoneStatus.ERROR
        }
        
        # 1. Check platform compatibility
        if self.platform == "Darwin":
            self._check_macos_compatibility()
        elif self.platform == "Linux":
            print("ℹ️  Linux platform detected - limited diagnostic available")
        elif self.platform == "Windows":
            print("ℹ️  Windows platform detected - limited diagnostic available")
        else:
            print(f"⚠️  Unsupported platform: {self.platform}")
            return results
        
        # 2. Check for blocking applications
        print("\n📱 Checking for apps using microphone...")
        self.blocking_apps = self._find_blocking_apps()
        results["blocking_apps"] = self.blocking_apps
        
        # 3. List available microphone devices
        print("\n🎤 Detecting microphone devices...")
        self.available_devices = self._list_microphone_devices()
        results["devices"] = [
            {
                "name": d.name,
                "device_id": d.device_id,
                "is_default": d.is_default,
                "is_available": d.is_available
            }
            for d in self.available_devices
        ]
        
        # 4. Check browser compatibility
        print("\n🌐 Checking browser compatibility...")
        self._check_browser_compatibility()
        
        # 5. Test microphone access
        print("\n🔊 Testing microphone access...")
        mic_status = self._test_microphone_access()
        
        # 6. Apply automatic fixes if needed
        if mic_status != MicrophoneStatus.AVAILABLE:
            print("\n🔧 Applying automatic fixes...")
            fixes = self._apply_automatic_fixes()
            results["fixes_applied"] = fixes
            
            # Re-test after fixes
            mic_status = self._test_microphone_access()
        
        # 7. Generate recommendations
        recommendations = self._generate_recommendations()
        
        # Compile results
        results["checks"] = [
            {
                "name": r.check_name,
                "passed": r.status,
                "message": r.message,
                "fix_available": r.fix_available
            }
            for r in self.diagnostic_results
        ]
        results["status"] = mic_status
        results["recommendations"] = recommendations
        
        return results
    
    def _check_macos_compatibility(self) -> None:
        """Check macOS-specific requirements and system compatibility.
        
        Verifies macOS version compatibility, microphone permissions, and
        Core Audio service status. Adds results to diagnostic_results list.
        """
        # Check macOS version
        try:
            version = subprocess.run(
                ["sw_vers", "-productVersion"],
                capture_output=True,
                text=True
            ).stdout.strip()
            
            major_version = int(version.split('.')[0])
            if major_version >= 10:
                self.diagnostic_results.append(
                    DiagnosticResult(
                        "macOS Version",
                        True,
                        f"macOS {version} is compatible"
                    )
                )
            else:
                self.diagnostic_results.append(
                    DiagnosticResult(
                        "macOS Version",
                        False,
                        f"macOS {version} may have compatibility issues"
                    )
                )
        except Exception as e:
            logger.warning(f"Could not check macOS version: {e}")
        
        # Check if Terminal/IDE has microphone permission
        self._check_macos_microphone_permission()
        
        # Check Core Audio
        self._check_core_audio()
    
    def _check_macos_microphone_permission(self) -> None:
        """Check if the current process has microphone permission on macOS.
        
        Uses tccutil to verify microphone access permissions in the TCC database.
        Adds permission status to diagnostic results with fix recommendations.
        """
        try:
            # Check TCC database for microphone permissions
            result = subprocess.run(
                ["tccutil", "check", "Microphone"],
                capture_output=True
            )
            
            has_permission = result.returncode == 0
            
            self.diagnostic_results.append(
                DiagnosticResult(
                    "Microphone Permission",
                    has_permission,
                    "App has microphone permission" if has_permission else "App needs microphone permission",
                    fix_available=not has_permission,
                    fix_command="tccutil reset Microphone"
                )
            )
        except Exception as e:
            logger.warning(f"Could not check microphone permission: {e}")
            # Fallback check
            self.diagnostic_results.append(
                DiagnosticResult(
                    "Microphone Permission",
                    False,
                    "Could not verify microphone permission",
                    fix_available=True
                )
            )
    
    def _check_core_audio(self) -> None:
        """Check the status of the Core Audio service on macOS.
        
        Verifies that the coreaudiod daemon is running, which is essential
        for audio device functionality. Provides restart command if needed.
        """
        try:
            result = subprocess.run(
                ["pgrep", "coreaudiod"],
                capture_output=True
            )
            
            is_running = result.returncode == 0
            
            self.diagnostic_results.append(
                DiagnosticResult(
                    "Core Audio Service",
                    is_running,
                    "Core Audio is running" if is_running else "Core Audio is not running",
                    fix_available=not is_running,
                    fix_command="sudo killall coreaudiod"
                )
            )
        except Exception as e:
            logger.warning(f"Could not check Core Audio status: {e}")
    
    def _find_blocking_apps(self) -> List[str]:
        """Find applications that might be using the microphone.
        
        Scans running processes for known audio applications and checks
        system resources for audio device access using lsof.
        
        Returns:
            List of application names that may be blocking microphone access.
            
        Example:
            >>> diagnostic = MicrophoneDiagnostic()
            >>> blocking = diagnostic._find_blocking_apps()
            >>> print(f"Found {len(blocking)} blocking apps")
        """
        blocking_apps = []
        
        if self.platform == "Darwin":
            # Check for running audio apps
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    proc_name = proc.info['name']
                    for app in self.audio_apps:
                        if app.lower() in proc_name.lower():
                            blocking_apps.append(proc_name)
                            print(f"  ⚠️  Found: {proc_name} (PID: {proc.info['pid']})")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Check lsof for audio device access
            try:
                result = subprocess.run(
                    ["lsof", "+D", "/dev"],
                    capture_output=True,
                    text=True
                )
                
                if "audio" in result.stdout.lower():
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if "audio" in line.lower():
                            parts = line.split()
                            if parts:
                                app_name = parts[0]
                                if app_name not in blocking_apps:
                                    blocking_apps.append(app_name)
            except Exception as e:
                logger.warning(f"Could not check lsof for audio devices: {e}")
        
        if blocking_apps:
            self.diagnostic_results.append(
                DiagnosticResult(
                    "Blocking Applications",
                    False,
                    f"Found {len(blocking_apps)} apps that may be using microphone",
                    fix_available=True
                )
            )
        else:
            self.diagnostic_results.append(
                DiagnosticResult(
                    "Blocking Applications",
                    True,
                    "No blocking applications detected"
                )
            )
        
        return blocking_apps
    
    def _list_microphone_devices(self) -> List[MicrophoneDevice]:
        """Enumerate available microphone devices on the system.
        
        Uses system_profiler on macOS to detect audio input devices and
        their configuration status.
        
        Returns:
            List of MicrophoneDevice objects representing detected devices.
            
        Example:
            >>> diagnostic = MicrophoneDiagnostic()
            >>> devices = diagnostic._list_microphone_devices()
            >>> for device in devices:
            ...     print(f"{device.name}: {'Default' if device.is_default else 'Available'}")
        """
        devices = []
        
        if self.platform == "Darwin":
            try:
                # Use system_profiler to get audio devices
                result = subprocess.run(
                    ["system_profiler", "SPAudioDataType", "-json"],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    audio_devices = data.get("SPAudioDataType", [])
                    
                    for device in audio_devices:
                        if device.get("_name", "").lower() in ["microphone", "input"]:
                            devices.append(
                                MicrophoneDevice(
                                    name=device.get("_name", "Unknown"),
                                    device_id=device.get("coreaudio_device_id", ""),
                                    is_default=device.get("coreaudio_default_audio_input_device", "") == "Yes",
                                    is_available=True
                                )
                            )
            except Exception as e:
                logger.warning(f"Could not enumerate audio devices: {e}")
        
        if devices:
            self.diagnostic_results.append(
                DiagnosticResult(
                    "Microphone Devices",
                    True,
                    f"Found {len(devices)} microphone device(s)"
                )
            )
        else:
            self.diagnostic_results.append(
                DiagnosticResult(
                    "Microphone Devices",
                    False,
                    "No microphone devices found",
                    fix_available=True
                )
            )
        
        return devices
    
    def _check_browser_compatibility(self) -> None:
        """Check browser compatibility for Web Speech API support.
        
        Identifies running browsers and evaluates their Web Speech API support
        level. Chrome and Edge have full support, Safari has partial support,
        and Firefox has no support.
        """
        browsers = {
            BrowserType.CHROME: self._is_browser_running("Google Chrome"),
            BrowserType.SAFARI: self._is_browser_running("Safari"),
            BrowserType.FIREFOX: self._is_browser_running("Firefox"),
            BrowserType.EDGE: self._is_browser_running("Microsoft Edge")
        }
        
        # Check which browsers support Web Speech API
        compatible_browsers = [BrowserType.CHROME, BrowserType.EDGE]
        partial_support = [BrowserType.SAFARI]
        no_support = [BrowserType.FIREFOX]
        
        running_compatible = False
        for browser, is_running in browsers.items():
            if is_running and browser in compatible_browsers:
                running_compatible = True
                print(f"  ✅ {browser.value.title()} is running (Full support)")
            elif is_running and browser in partial_support:
                print(f"  ⚠️  {browser.value.title()} is running (Partial support)")
            elif is_running and browser in no_support:
                print(f"  ❌ {browser.value.title()} is running (No Web Speech API support)")
        
        self.diagnostic_results.append(
            DiagnosticResult(
                "Browser Compatibility",
                running_compatible,
                "Compatible browser detected" if running_compatible else "No fully compatible browser running",
                fix_available=not running_compatible
            )
        )
    
    def _is_browser_running(self, browser_name: str) -> bool:
        """Check if a specific browser process is currently running.
        
        Args:
            browser_name: Name of the browser to check for (e.g., "Google Chrome")
            
        Returns:
            True if the browser process is found, False otherwise.
            
        Example:
            >>> diagnostic = MicrophoneDiagnostic()
            >>> if diagnostic._is_browser_running("Google Chrome"):
            ...     print("Chrome is running")
        """
        for proc in psutil.process_iter(['name']):
            try:
                if browser_name.lower() in proc.info['name'].lower():
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False
    
    def _test_microphone_access(self) -> MicrophoneStatus:
        """Test actual microphone access by attempting to record audio.
        
        Attempts to record a brief audio sample using sox to verify that
        the microphone is accessible and not blocked by permissions or
        other applications.
        
        Returns:
            MicrophoneStatus indicating the current state of microphone access.
            
        Raises:
            No exceptions are raised; all errors are caught and converted
            to appropriate MicrophoneStatus values.
            
        Example:
            >>> diagnostic = MicrophoneDiagnostic()
            >>> status = diagnostic._test_microphone_access()
            >>> if status == MicrophoneStatus.AVAILABLE:
            ...     print("Microphone is ready!")
        """
        if self.platform == "Darwin":
            try:
                # Try to record a short audio sample
                test_file = "/tmp/jarvis_mic_test.wav"
                result = subprocess.run(
                    ["sox", "-d", test_file, "trim", "0", "0.1"],
                    capture_output=True,
                    stderr=subprocess.PIPE,
                    timeout=2
                )
                
                if result.returncode == 0:
                    # Clean up test file
                    if os.path.exists(test_file):
                        os.remove(test_file)
                    
                    self.diagnostic_results.append(
                        DiagnosticResult(
                            "Microphone Access Test",
                            True,
                            "Microphone is accessible"
                        )
                    )
                    return MicrophoneStatus.AVAILABLE
                else:
                    error_msg = result.stderr.decode() if result.stderr else "Unknown error"
                    if "Permission denied" in error_msg:
                        return MicrophoneStatus.PERMISSION_DENIED
                    elif "Device not configured" in error_msg:
                        return MicrophoneStatus.NOT_FOUND
                    else:
                        return MicrophoneStatus.BUSY
            except subprocess.TimeoutExpired:
                return MicrophoneStatus.BUSY
            except FileNotFoundError:
                # sox not installed, try alternative method
                print("  ℹ️  sox not found, skipping audio test")
                return MicrophoneStatus.ERROR
        
        return MicrophoneStatus.ERROR
    
    def _apply_automatic_fixes(self) -> List[str]:
        """Apply automatic fixes for common microphone issues.
        
        Attempts to resolve detected issues by restarting services,
        identifying blocking applications, and providing guidance for
        permission issues.
        
        Returns:
            List of strings describing the fixes that were applied or attempted.
            
        Example:
            >>> diagnostic = MicrophoneDiagnostic()
            >>> fixes = diagnostic._apply_automatic_fixes()
            >>> for fix in fixes:
            ...     print(f"Applied: {fix}")
        """
        fixes_applied = []
        
        # 1. Restart Core Audio if needed
        core_audio_check = next((r for r in self.diagnostic_results if r.check_name == "Core Audio Service"), None)
        if core_audio_check and not core_audio_check.status:
            print("  🔧 Restarting Core Audio...")
            try:
                subprocess.run(["sudo", "killall", "coreaudiod"], capture_output=True)
                time.sleep(2)
                fixes_applied.append("Restarted Core Audio service")
            except Exception as e:
                logger.warning(f"Could not restart Core Audio: {e}")
                print("  ⚠️  Could not restart Core Audio (may need sudo)")
        
        # 2. Kill blocking applications (with user confirmation)
        if self.blocking_apps:
            print(f"\n  ⚠️  Found {len(self.blocking_apps)} apps using microphone:")
            for app in self.blocking_apps:
                print(f"     - {app}")
            
            # In automated mode, we'll just log them
            fixes_applied.append(f"Identified {len(self.blocking_apps)} blocking apps")
        
        # 3. Reset microphone permissions if needed
        permission_check = next((r for r in self.diagnostic_results if r.check_name == "Microphone Permission"), None)
        if permission_check and not permission_check.status:
            print("  🔧 Microphone permission needs to be granted")
            print("     Please grant permission when prompted by your browser")
            fixes_applied.append("Microphone permission reset required")
        
        return fixes_applied
    
    def _generate_recommendations(self) -> List[str]:
        """Generate specific recommendations based on diagnostic results.
        
        Analyzes the diagnostic results to provide actionable recommendations
        for resolving microphone issues and optimizing JARVIS performance.
        
        Returns:
            List of recommendation strings for the user to follow.
            
        Example:
            >>> diagnostic = MicrophoneDiagnostic()
            >>> recommendations = diagnostic._generate_recommendations()
            >>> for rec in recommendations:
            ...     print(f"Recommendation: {rec}")
        """
        recommendations = []
        
        # Browser recommendations
        if not any(r.status for r in self.diagnostic_results if r.check_name == "Browser Compatibility"):
            recommendations.append("Use Chrome or Edge for best JARVIS voice compatibility")
        
        # Blocking apps recommendations
        if self.blocking_apps:
            recommendations.append(f"Close these apps before using JARVIS: {', '.join(self.blocking_apps[:3])}")
        
        # Device recommendations
        if not self.available_devices:
            recommendations.append("Connect a microphone or check System Preferences → Sound → Input")
        
        # Permission recommendations
        permission_check = next((r for r in self.diagnostic_results if r.check_name == "Microphone Permission"), None)
        if permission_check and not permission_check.status:
            recommendations.append("Grant microphone permission in System Preferences → Security & Privacy → Privacy → Microphone")
        
        return recommendations
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable diagnostic report from results.
        
        Creates a formatted text report summarizing all diagnostic findings,
        device information, blocking applications, applied fixes, and
        recommendations.
        
        Args:
            results: Dictionary containing diagnostic results from run_diagnostic()
            
        Returns:
            Formatted string report suitable for display or logging.
            
        Example:
            >>> diagnostic = MicrophoneDiagnostic()
            >>> results = diagnostic.run_diagnostic()
            >>> report = diagnostic.generate_report(results)
            >>> print(report)
        """
        report = []
        report.append("\n" + "=" * 60)
        report.append("📊 JARVIS MICROPHONE DIAGNOSTIC REPORT")
        report.append("=" * 60)
        report.append(f"Time: {results['timestamp']}")
        report.append(f"Platform: {results['platform']}")
        report.append(f"Status: {results['status'].value.upper()}")
        report.append("")
        
        # Diagnostic checks
        report.append("🔍 DIAGNOSTIC CHECKS:")
        report.append("-" * 40)
        for check in results['checks']:
            status = "✅" if check['passed'] else "❌"
            report.append(f"{status} {check['name']}: {check['message']}")
        
        # Devices
        if results['devices']:
            report.append("\n🎤 MICROPHONE DEVICES:")
            report.append("-" * 40)
            for device in results['devices']:
                default = " (Default)" if device['is_default'] else ""
                available = "Available" if device['is_available'] else "Unavailable"
                report.append(f"• {device['name']}{default} - {available}")
        
        # Blocking apps
        if results['blocking_apps']:
            report.append("\n⚠️  APPS USING MICROPHONE:")
            report.append("-" * 40)
            for app in results['blocking_apps']:
                report.append(f"• {app}")
        
        # Fixes applied
        if results['fixes_applied']:
            report.append("\n🔧 FIXES APPLIED:")
            report.append("-" * 40)
            for fix in results['fixes_applied']:
                report.append(f"• {fix}")
        
        # Recommendations
        if results.get('recommendations'):
            report.append("\n💡 RECOMMENDATIONS:")
            report.append("-" * 40)
            for i, rec in enumerate(results['recommendations'], 1):
                report.append(f"{i}. {rec}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def save_diagnostic_log(self, results: Dict[str, Any], filepath: str = "microphone_diagnostic.log") -> None:
        """Save diagnostic results to a log file.
        
        Writes both the human-readable report and raw diagnostic data
        to a specified log file for later analysis or troubleshooting.
        
        Args:
            results: Dictionary containing diagnostic results from run_diagnostic()
            filepath: Path where the log file should be saved
            
        Example:
            >>> diagnostic = MicrophoneDiagnostic()
            >>> results = diagnostic.run_diagnostic()
            >>> diagnostic.save_diagnostic_log(results, "my_diagnostic.log")
        """
        report = self.generate_report(results)
        
        with open(filepath, 'w') as f:
            f.write(report)
            f.write("\n\nRAW DIAGNOSTIC DATA:\n")
            f.write(json.dumps(results, indent=2, default=str))
        
        print(f"\n📄 Diagnostic log saved to: {filepath}")

def run_diagnostic_sync() -> Dict[str, Any]:
    """Run diagnostic synchronously and save results to log.
    
    Convenience function that creates a MicrophoneDiagnostic instance,
    runs the complete diagnostic suite, displays the report, and saves
    the results to a log file.
    
    Returns:
        Dictionary containing complete diagnostic results.
        
    Example:
        >>> results = run_diagnostic_sync()
        >>> if results['status'] == MicrophoneStatus.AVAILABLE:
        ...     print("Microphone is working!")
    """
    diagnostic = MicrophoneDiagnostic()
    results = diagnostic.run_diagnostic()
    report = diagnostic.generate_report(results)
    print(report)
    
    # Save log
    log_path = os.path.join(os.path.dirname(__file__), "../../logs/microphone_diagnostic.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    diagnostic.save_diagnostic_log(results, log_path)
    
    return results

def main() -> int:
    """Run diagnostic from command line interface.
    
    Main entry point for command-line execution. Displays platform
    compatibility warnings, runs the diagnostic suite, generates
    a report, and saves results to a log file.
    
    Returns:
        Exit code: 0 if microphone is available, 1 if issues detected.
        
    Example:
        $ python microphone_diagnostic.py
        # Runs complete diagnostic and displays results
    """
    if platform.system() != "Darwin":
        print("⚠️  This diagnostic tool is currently optimized for macOS")
        print("   Limited functionality on other platforms")
    
    diagnostic = MicrophoneDiagnostic()
    results = diagnostic.run_diagnostic()
    report = diagnostic.generate_report(results)
    print(report)
    
    # Save log
    diagnostic.save_diagnostic_log(results)
    
    # Return status code based on microphone availability
    if results['status'] == MicrophoneStatus.AVAILABLE:
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())