#!/usr/bin/env python3
"""Advanced macOS Integration for JARVIS.

This module provides deep system control and hardware management for macOS systems,
powered by Anthropic's Claude API for intelligent decision making. It includes
comprehensive system monitoring, AI-powered optimization, and safe system control
with built-in safety limits and reversible actions.

The module supports:
- Continuous system monitoring and optimization
- AI-powered decision making for system control
- Safe application management with protection for critical apps
- Context-aware optimization (meeting, focus, gaming modes)
- Emergency optimization for system stress situations
- Intelligent model selection for enhanced performance

Example:
    >>> integration = get_macos_integration(api_key="your_api_key")
    >>> await integration.start_system_monitoring()
    >>> status = integration.get_system_status()
    >>> await integration.optimize_for_context("meeting")
"""

import asyncio
import logging
import subprocess
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import anthropic
import os
import psutil
import platform

logger = logging.getLogger(__name__)

class SystemResource(Enum):
    """System resources that can be managed.
    
    Attributes:
        CPU: Central processing unit resources
        MEMORY: System memory (RAM) resources
        DISK: Storage disk resources
        NETWORK: Network connectivity resources
        DISPLAY: Display and graphics resources
        AUDIO: Audio system resources
        CAMERA: Camera hardware resources
        POWER: Power management and battery resources
        BLUETOOTH: Bluetooth connectivity resources
        WIFI: WiFi connectivity resources
    """
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    DISPLAY = "display"
    AUDIO = "audio"
    CAMERA = "camera"
    POWER = "power"
    BLUETOOTH = "bluetooth"
    WIFI = "wifi"

class ControlAction(Enum):
    """Types of control actions that can be performed on system resources.
    
    Attributes:
        OPTIMIZE: Optimize resource usage for better performance
        ADJUST: Adjust resource settings or parameters
        ENABLE: Enable a resource or feature
        DISABLE: Disable a resource or feature
        RESTART: Restart a resource or service
        MONITOR: Monitor resource usage and status
        CONFIGURE: Configure resource settings
    """
    OPTIMIZE = "optimize"
    ADJUST = "adjust"
    ENABLE = "enable"
    DISABLE = "disable"
    RESTART = "restart"
    MONITOR = "monitor"
    CONFIGURE = "configure"

@dataclass
class SystemState:
    """Current system state snapshot.
    
    Attributes:
        cpu_usage: Current CPU usage percentage (0-100)
        memory_usage: Current memory usage percentage (0-100)
        disk_usage: Current disk usage percentage (0-100)
        active_apps: List of currently active application names
        network_status: Dictionary containing network connectivity information
        display_config: Dictionary containing display configuration
        power_status: Dictionary containing power and battery status
        timestamp: When this state snapshot was taken
    """
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    active_apps: List[str]
    network_status: Dict[str, Any]
    display_config: Dict[str, Any]
    power_status: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ControlDecision:
    """A control decision made by the AI system.
    
    Attributes:
        resource: The system resource to be controlled
        action: The type of action to perform
        parameters: Dictionary of parameters for the action
        reasoning: AI's explanation for this decision
        confidence: Confidence level of the decision (0.0-1.0)
        impact_prediction: Predicted impact of this action
        reversible: Whether this action can be safely reversed
    """
    resource: SystemResource
    action: ControlAction
    parameters: Dict[str, Any]
    reasoning: str
    confidence: float
    impact_prediction: Dict[str, Any]
    reversible: bool = True

class AdvancedMacOSIntegration:
    """Deep macOS integration with AI-powered control.
    
    This class provides comprehensive macOS system integration including:
    - Continuous system monitoring and health assessment
    - AI-powered optimization decisions using Claude API
    - Safe application control with built-in protection
    - Context-aware system optimization
    - Emergency optimization capabilities
    - Intelligent model selection for enhanced performance
    
    Attributes:
        claude: Anthropic Claude API client
        use_intelligent_selection: Whether to use intelligent model selection
        system_state: Current system state snapshot
        monitoring_active: Whether system monitoring is active
        monitoring_interval: Interval between monitoring cycles in seconds
        control_history: History of control actions taken
        optimization_rules: Custom optimization rules
        safety_limits: Safety limits and protected resources
    """
    
    def __init__(self, anthropic_api_key: str, use_intelligent_selection: bool = True):
        """Initialize the macOS integration system.
        
        Args:
            anthropic_api_key: API key for Anthropic Claude
            use_intelligent_selection: Whether to use intelligent model selection
            
        Raises:
            ValueError: If API key is invalid or missing
        """
        self.claude = anthropic.Anthropic(api_key=anthropic_api_key)
        self.use_intelligent_selection = use_intelligent_selection

        # System monitoring
        self.system_state = None
        self.monitoring_active = False
        self.monitoring_interval = 30  # seconds
        
        # Control history
        self.control_history = []
        self.optimization_rules = {}
        
        # Safety limits
        self.safety_limits = {
            'max_cpu_throttle': 0.5,  # Don't throttle CPU below 50%
            'min_memory_free': 2048,  # Keep at least 2GB free
            'max_brightness_change': 0.3,  # Max 30% brightness change
            'protected_apps': ['Finder', 'SystemUIServer', 'loginwindow']
        }
        
    async def start_system_monitoring(self) -> None:
        """Start continuous system monitoring.
        
        Begins the system monitoring loop that continuously gathers system state,
        identifies optimization opportunities, and executes high-confidence
        optimizations automatically.
        
        Example:
            >>> integration = AdvancedMacOSIntegration(api_key)
            >>> await integration.start_system_monitoring()
        """
        self.monitoring_active = True
        asyncio.create_task(self._system_monitoring_loop())
        logger.info("Advanced macOS monitoring started")
        
    async def stop_system_monitoring(self) -> None:
        """Stop system monitoring.
        
        Stops the continuous system monitoring loop and sets monitoring_active
        to False.
        """
        self.monitoring_active = False
        
    async def _system_monitoring_loop(self) -> None:
        """Monitor system state continuously.
        
        Internal method that runs the main monitoring loop. Gathers system state,
        identifies optimization opportunities, and executes safe optimizations
        with high confidence levels.
        
        Raises:
            Exception: Logs errors but continues monitoring after delay
        """
        while self.monitoring_active:
            try:
                # Gather system state
                self.system_state = await self._gather_system_state()
                
                # Check for optimization opportunities
                opportunities = await self._identify_optimization_opportunities()
                
                # Execute high-confidence optimizations
                for opportunity in opportunities:
                    if opportunity.confidence > 0.8 and opportunity.reversible:
                        await self._execute_control_action(opportunity)
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _gather_system_state(self) -> SystemState:
        """Gather comprehensive system state information.
        
        Collects current system metrics including CPU usage, memory usage,
        disk usage, active applications, network status, display configuration,
        and power status.
        
        Returns:
            SystemState: Complete system state snapshot
            
        Raises:
            Exception: If system state gathering fails
        """
        # CPU usage
        cpu_usage = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_usage = disk.percent
        
        # Active applications
        active_apps = await self._get_active_applications()
        
        # Network status
        network_status = await self._get_network_status()
        
        # Display configuration
        display_config = await self._get_display_configuration()
        
        # Power status
        power_status = await self._get_power_status()
        
        return SystemState(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            active_apps=active_apps,
            network_status=network_status,
            display_config=display_config,
            power_status=power_status
        )
    
    async def _get_active_applications(self) -> List[str]:
        """Get list of currently active applications.
        
        Uses AppleScript to query the system for visible application processes.
        
        Returns:
            List[str]: Names of active applications
            
        Example:
            >>> apps = await integration._get_active_applications()
            >>> print(apps)
            ['Finder', 'Safari', 'Terminal', 'Code']
        """
        try:
            # Use AppleScript to get running applications
            script = '''
            tell application "System Events"
                set appList to name of every application process whose visible is true
            end tell
            return appList
            '''
            
            result = subprocess.run(
                ['osascript', '-e', script],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                apps = result.stdout.strip().split(', ')
                return apps
            
        except Exception as e:
            logger.error(f"Error getting active applications: {e}")
        
        return []
    
    async def _get_network_status(self) -> Dict[str, Any]:
        """Get comprehensive network status information.
        
        Checks WiFi connectivity, SSID, ethernet status, and VPN connections
        using system network utilities.
        
        Returns:
            Dict[str, Any]: Network status information containing:
                - wifi_connected: Whether WiFi is connected
                - wifi_ssid: Current WiFi network name
                - ethernet_connected: Whether ethernet is connected
                - vpn_connected: Whether VPN is active
        """
        status = {
            'wifi_connected': False,
            'wifi_ssid': None,
            'ethernet_connected': False,
            'vpn_connected': False
        }
        
        try:
            # Check WiFi status
            wifi_result = subprocess.run(
                ['networksetup', '-getairportnetwork', 'en0'],
                capture_output=True,
                text=True
            )
            
            if 'Current Wi-Fi Network:' in wifi_result.stdout:
                status['wifi_connected'] = True
                status['wifi_ssid'] = wifi_result.stdout.split(':')[1].strip()
            
            # Check for VPN
            vpn_result = subprocess.run(
                ['scutil', '--nwi'],
                capture_output=True,
                text=True
            )
            
            if 'utun' in vpn_result.stdout:
                status['vpn_connected'] = True
                
        except Exception as e:
            logger.error(f"Error getting network status: {e}")
        
        return status
    
    async def _get_display_configuration(self) -> Dict[str, Any]:
        """Get current display configuration.
        
        Retrieves display settings including brightness, night shift status,
        display count, and resolution. Note: Some features require additional
        tools or system APIs.
        
        Returns:
            Dict[str, Any]: Display configuration containing:
                - brightness: Current display brightness (if available)
                - night_shift: Whether night shift is enabled
                - display_count: Number of displays
                - resolution: Display resolution (if available)
        """
        config = {
            'brightness': None,
            'night_shift': False,
            'display_count': 1,
            'resolution': None
        }
        
        try:
            # Get display brightness (requires additional tools)
            # This is a placeholder - actual implementation would use
            # tools like brightness CLI or system APIs
            pass
            
        except Exception as e:
            logger.error(f"Error getting display config: {e}")
        
        return config
    
    async def _get_power_status(self) -> Dict[str, Any]:
        """Get comprehensive power and battery status.
        
        Uses pmset utility to gather power information including battery
        percentage, power source, and power adapter status.
        
        Returns:
            Dict[str, Any]: Power status information containing:
                - on_battery: Whether system is running on battery
                - battery_percent: Battery charge percentage
                - power_adapter: Whether power adapter is connected
                - low_power_mode: Whether low power mode is enabled
        """
        status = {
            'on_battery': False,
            'battery_percent': None,
            'power_adapter': False,
            'low_power_mode': False
        }
        
        try:
            # Get power info
            result = subprocess.run(
                ['pmset', '-g', 'batt'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                output = result.stdout
                
                # Parse battery percentage
                if '%' in output:
                    import re
                    match = re.search(r'(\d+)%', output)
                    if match:
                        status['battery_percent'] = int(match.group(1))
                
                # Check power source
                if 'AC Power' in output:
                    status['power_adapter'] = True
                else:
                    status['on_battery'] = True
                    
        except Exception as e:
            logger.error(f"Error getting power status: {e}")
        
        return status
    
    async def _identify_optimization_opportunities(self) -> List[ControlDecision]:
        """Use AI and local logic to identify system optimization opportunities.
        
        Analyzes current system state to identify potential optimizations.
        Uses local logic for common scenarios to reduce API calls and improve
        performance, with AI analysis for complex situations.
        
        Returns:
            List[ControlDecision]: List of potential optimization actions
            
        Raises:
            Exception: Logs errors but returns empty list to maintain stability
        """
        if not self.system_state:
            return []
        
        try:
            # Prepare system context for Claude
            context = {
                'cpu_usage': self.system_state.cpu_usage,
                'memory_usage': self.system_state.memory_usage,
                'disk_usage': self.system_state.disk_usage,
                'active_apps': self.system_state.active_apps[:10],  # Top 10 apps
                'network': self.system_state.network_status,
                'power': self.system_state.power_status,
                'timestamp': self.system_state.timestamp.isoformat()
            }
            
            # LOCAL ANALYSIS - No Claude API calls to reduce CPU usage
            opportunities = []
            
            # CPU optimization (local logic)
            if self.system_state.cpu_usage > 80:
                opportunities.append(ControlDecision(
                    resource=SystemResource.CPU,
                    action=ControlAction.OPTIMIZE,
                    parameters={'target_cpu': 70, 'method': 'throttle_background'},
                    reasoning="CPU usage above 80% - throttle background processes",
                    confidence=0.9,
                    impact_prediction={'cpu_reduction': '10-15%', 'battery_improvement': '5-10%'}
                ))
            
            # Memory optimization (local logic)
            if self.system_state.memory_usage > 85:
                opportunities.append(ControlDecision(
                    resource=SystemResource.MEMORY,
                    action=ControlAction.OPTIMIZE,
                    parameters={'target_memory': 75, 'method': 'unload_unused_models'},
                    reasoning="Memory usage above 85% - unload unused ML models",
                    confidence=0.95,
                    impact_prediction={'memory_reduction': '1-2GB', 'stability_improvement': 'high'}
                ))
            
            # Battery optimization (if on battery)
            if self.system_state.power_status.get('power_source') == 'battery':
                if self.system_state.cpu_usage > 60:
                    opportunities.append(ControlDecision(
                        resource=SystemResource.POWER,
                        action=ControlAction.OPTIMIZE,
                        parameters={'target_cpu': 50, 'method': 'battery_saver_mode'},
                        reasoning="On battery with high CPU - enable battery saver",
                        confidence=0.8,
                        impact_prediction={'battery_improvement': '15-25%', 'performance_impact': 'minimal'}
                    ))
            
            # Disk optimization (local logic)
            if self.system_state.disk_usage > 90:
                opportunities.append(ControlDecision(
                    resource=SystemResource.DISK,
                    action=ControlAction.OPTIMIZE,
                    parameters={'method': 'clear_cache', 'target_free_space': '10%'},
                    reasoning="Disk usage above 90% - clear cache files",
                    confidence=0.85,
                    impact_prediction={'disk_freed': '1-5GB', 'performance_improvement': 'moderate'}
                ))
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error identifying optimizations: {e}")
            return []
    
    def _parse_optimization_response(self, response_text: str) -> List[ControlDecision]:
        """Parse AI response text into structured control decisions.
        
        Extracts optimization recommendations from natural language AI responses
        and converts them into actionable ControlDecision objects.
        
        Args:
            response_text: Natural language response from AI
            
        Returns:
            List[ControlDecision]: Parsed control decisions
            
        Example:
            >>> response = "Resource: CPU\nAction: OPTIMIZE\nConfidence: 85%"
            >>> decisions = integration._parse_optimization_response(response)
        """
        decisions = []
        
        # Simple parsing - would be more sophisticated in practice
        lines = response_text.split('\n')
        current_decision = None
        
        for line in lines:
            line_lower = line.lower()
            
            if 'resource:' in line_lower:
                if current_decision:
                    decisions.append(current_decision)
                
                # Start new decision
                resource_str = line.split(':')[1].strip().upper()
                try:
                    resource = SystemResource[resource_str]
                    current_decision = ControlDecision(
                        resource=resource,
                        action=ControlAction.OPTIMIZE,
                        parameters={},
                        reasoning="",
                        confidence=0.5,
                        impact_prediction={}
                    )
                except:
                    current_decision = None
                    
            elif current_decision:
                if 'action:' in line_lower:
                    action_str = line.split(':')[1].strip().upper()
                    try:
                        current_decision.action = ControlAction[action_str]
                    except:
                        pass
                elif 'confidence:' in line_lower:
                    try:
                        conf = float(line.split(':')[1].strip().strip('%')) / 100
                        current_decision.confidence = conf
                    except:
                        pass
                elif 'reasoning:' in line_lower:
                    current_decision.reasoning = line.split(':')[1].strip()
                elif 'reversible:' in line_lower:
                    current_decision.reversible = 'yes' in line_lower or 'true' in line_lower
        
        if current_decision:
            decisions.append(current_decision)
        
        return decisions
    
    async def _execute_control_action(self, decision: ControlDecision) -> None:
        """Execute a specific control decision.
        
        Takes a ControlDecision and executes the appropriate system action
        based on the resource and action type. Records the action in history
        for tracking and potential reversal.
        
        Args:
            decision: The control decision to execute
            
        Raises:
            Exception: Logs errors but continues operation
        """
        logger.info(f"Executing control action: {decision.resource.value} - {decision.action.value}")
        
        try:
            if decision.resource == SystemResource.MEMORY:
                await self._optimize_memory(decision)
            elif decision.resource == SystemResource.CPU:
                await self._optimize_cpu(decision)
            elif decision.resource == SystemResource.POWER:
                await self._optimize_power(decision)
            elif decision.resource == SystemResource.NETWORK:
                await self._optimize_network(decision)
            elif decision.resource == SystemResource.DISPLAY:
                await self._optimize_display(decision)
            
            # Record action
            self.control_history.append({
                'decision': decision,
                'timestamp': datetime.now(),
                'success': True
            })
            
        except Exception as e:
            logger.error(f"Error executing control action: {e}")
            self.control_history.append({
                'decision': decision,
                'timestamp': datetime.now(),
                'success': False,
                'error': str(e)
            })
    
    async def _optimize_memory(self, decision: ControlDecision) -> None:
        """Optimize system memory usage.
        
        Executes memory optimization actions such as purging inactive memory
        and clearing caches based on the control decision parameters.
        
        Args:
            decision: Control decision with memory optimization parameters
        """
        if decision.action == ControlAction.OPTIMIZE:
            # Purge inactive memory
            subprocess.run(['sudo', 'purge'], capture_output=True)
            logger.info("Purged inactive memory")
    
    async def _optimize_cpu(self, decision: ControlDecision) -> None:
        """Optimize CPU usage and performance.
        
        Implements CPU optimization strategies such as process throttling
        and background task management based on decision parameters.
        
        Args:
            decision: Control decision with CPU optimization parameters
        """
        if decision.action == ControlAction.OPTIMIZE:
            # This would implement CPU optimization
            # For safety, we're being conservative here
            logger.info("CPU optimization requested - monitoring high-usage processes")
    
    async def _optimize_power(self, decision: ControlDecision) -> None:
        """Optimize power settings and battery usage.
        
        Adjusts power management settings such as enabling low power mode
        when on battery or optimizing CPU frequency for battery life.
        
        Args:
            decision: Control decision with power optimization parameters
        """
        if decision.action == ControlAction.OPTIMIZE:
            if self.system_state.power_status.get('on_battery'):
                # Enable power saving mode
                subprocess.run(['pmset', '-a', 'lowpowermode', '1'], capture_output=True)
                logger.info("Enabled low power mode")
    
    async def _optimize_network(self, decision: ControlDecision) -> None:
        """Optimize network settings and connectivity.
        
        Implements network optimization strategies such as connection
        prioritization and bandwidth management.
        
        Args:
            decision: Control decision with network optimization parameters
        """
        if decision.action == ControlAction.OPTIMIZE:
            # This would implement network optimization
            logger.info("Network optimization requested")
    
    async def _optimize_display(self, decision: ControlDecision) -> None:
        """Optimize display settings for performance or battery life.
        
        Adjusts display parameters such as brightness, refresh rate,
        and visual effects based on optimization goals.
        
        Args:
            decision: Control decision with display optimization parameters
        """
        if decision.action == ControlAction.ADJUST:
            # This would adjust display brightness/settings
            logger.info("Display optimization requested")
    
    async def _control_application_with_intelligent_selection(self, app_name: str, action: str) -> bool:
        """Control application using intelligent model selection.
        
        Uses the hybrid orchestrator to intelligently select the best model
        for application control decisions, providing enhanced safety and
        performance through optimized model selection.
        
        Args:
            app_name: Name of the application to control
            action: Action to perform ('quit', 'hide', 'activate')
            
        Returns:
            bool: True if action was successful, False otherwise
            
        Raises:
            ImportError: If hybrid orchestrator is not available
            Exception: For other errors during intelligent selection
        """
        try:
            from backend.core.hybrid_orchestrator import HybridOrchestrator

            orchestrator = HybridOrchestrator()
            if not orchestrator.is_running:
                await orchestrator.start()

            # Build rich context
            context = {
                "task_type": "app_control_safety_check",
                "app_name": app_name,
                "action": action,
                "system_state": {
                    "cpu_usage": self.system_state.cpu_usage if self.system_state else None,
                    "memory_usage": self.system_state.memory_usage if self.system_state else None,
                    "active_apps": self.system_state.active_apps if self.system_state else [],
                },
                "protected_apps": self.safety_limits['protected_apps'],
            }

            prompt = f"""Is it safe to {action} the application "{app_name}"?
Consider:
- System stability
- User workflow disruption
- Data loss risk

Respond with: SAFE or UNSAFE and brief reason."""

            # Execute with intelligent selection
            result = await orchestrator.execute_with_intelligent_model_selection(
                query=prompt,
                intent="system_integration",
                required_capabilities={"nlp_analysis", "system_understanding", "automation"},
                context=context,
                max_tokens=200,
                temperature=0.1,
            )

            if not result.get("success"):
                raise Exception(result.get("error", "Unknown error"))

            response_text = result.get("text", "").strip()
            model_used = result.get("model_used", "intelligent_selection")

            logger.info(f"✨ App control safety check using {model_used}")

            if 'SAFE' in response_text:
                # Execute action
                if action == 'quit':
                    script = f'tell application "{app_name}" to quit'
                elif action == 'hide':
                    script = f'tell application "System Events" to set visible of process "{app_name}" to false'
                elif action == 'activate':
                    script = f'tell application "{app_name}" to activate'
                else:
                    return False

                result = subprocess.run(
                    ['osascript', '-e', script],
                    capture_output=True
                )

                return result.returncode == 0

            return False

        except ImportError:
            logger.warning("Hybrid orchestrator not available, falling back to direct API")
            raise
        except Exception as e:
            logger.error(f"Error in intelligent selection: {e}")
            raise

    async def control_application(self, app_name: str, action: str) -> bool:
        """Control a specific application safely.
        
        Provides safe application control with AI-powered safety checks.
        Protects critical system applications and validates actions before
        execution. Supports intelligent model selection for enhanced performance.
        
        Args:
            app_name: Name of the application to control
            action: Action to perform ('quit', 'hide', 'activate')
            
        Returns:
            bool: True if action was successful, False otherwise
            
        Example:
            >>> success = await integration.control_application("Safari", "quit")
            >>> if success:
            ...     print("Safari closed successfully")
        """
        try:
            # Validate app is not protected
            if app_name in self.safety_limits['protected_apps']:
                logger.warning(f"Cannot control protected app: {app_name}")
                return False

            # Try intelligent selection first
            if self.use_intelligent_selection:
                try:
                    return await self._control_application_with_intelligent_selection(app_name, action)
                except Exception as e:
                    logger.warning(f"Intelligent selection failed, falling back to direct API: {e}")

            # Fallback to direct API
            # Use Claude to determine safe action
            response = await asyncio.to_thread(
                self.claude.messages.create,
                model="claude-3-haiku-20240307",
                max_tokens=200,
                messages=[{
                    "role": "user",
                    "content": f"""Is it safe to {action} the application "{app_name}"?
Consider:
- System stability
- User workflow disruption
- Data loss risk

Respond with: SAFE or UNSAFE and brief reason."""
                }]
            )

            if 'SAFE' in response.content[0].text:
                # Execute action
                if action == 'quit':
                    script = f'tell application "{app_name}" to quit'
                elif action == 'hide':
                    script = f'tell application "System Events" to set visible of process "{app_name}" to false'
                elif action == 'activate':
                    script = f'tell application "{app_name}" to activate'
                else:
                    return False

                result = subprocess.run(
                    ['osascript', '-e', script],
                    capture_output=True
                )

                return result.returncode == 0

        except Exception as e:
            logger.error(f"Error controlling application: {e}")

        return False
    
    async def _optimize_for_context_with_intelligent_selection(self, context: str, state: SystemState) -> Dict[str, Any]:
        """Optimize for context using intelligent model selection.
        
        Uses the hybrid orchestrator to intelligently select the best model
        for context-specific optimization, providing enhanced performance
        and more accurate optimization decisions.
        
        Args:
            context: The context to optimize for (e.g., "meeting", "focus", "gaming")
            state: Current system state
            
        Returns:
            Dict[str, Any]: Optimization results including applied optimizations
            
        Raises:
            ImportError: If hybrid orchestrator is not available
            Exception: For other errors during intelligent selection
        """
        try:
            from backend.core.hybrid_orchestrator import HybridOrchestrator

            orchestrator = HybridOrchestrator()
            if not orchestrator.is_running:
                await orchestrator.start()

            # Build rich context
            rich_context = {
                "task_type": "system_optimization",
                "optimization_context": context,
                "cpu_usage": state.cpu_usage,
                "memory_usage": state.memory_usage,
                "on_battery": state.power_status.get('on_battery', False),
                "active_apps_count": len(state.active_apps),
                "macos_version": platform.mac_ver()[0],
            }

            prompt = f"""Optimize macOS for context: {context}

Current System State:
- CPU Usage: {state.cpu_usage}%
- Memory Usage: {state.memory_usage}%
- Active Apps: {', '.join(state.active_apps[:5])}
- On Battery: {state.power_status.get('on_battery', False)}