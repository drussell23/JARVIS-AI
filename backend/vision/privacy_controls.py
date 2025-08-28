#!/usr/bin/env python3
"""
Privacy Control System for JARVIS Multi-Window Intelligence
Manages window visibility permissions and sensitive content protection
"""

import json
import logging
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
import re
import hashlib

from .window_detector import WindowDetector, WindowInfo

logger = logging.getLogger(__name__)

@dataclass
class PrivacyRule:
    """Privacy rule for window filtering"""
    rule_id: str
    rule_type: str  # 'blacklist', 'whitelist', 'pattern', 'temporary'
    target: str  # app name, window title pattern, or domain
    action: str  # 'hide', 'blur', 'exclude'
    expires_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True

@dataclass
class PrivacyMode:
    """Privacy mode configuration"""
    mode_name: str  # 'normal', 'meeting', 'focused', 'private'
    description: str
    rules: List[PrivacyRule]
    auto_hide_sensitive: bool = True
    blur_content: bool = False
    exclude_from_analysis: bool = False

@dataclass
class SensitiveContent:
    """Detected sensitive content in a window"""
    window_id: int
    content_type: str  # 'password', 'financial', 'personal', etc.
    confidence: float
    detected_patterns: List[str]
    recommended_action: str  # 'hide', 'blur', 'warn'

class PrivacyControlSystem:
    """Manages privacy settings and sensitive content protection"""
    
    def __init__(self, config_dir: str = "~/.jarvis/privacy"):
        self.window_detector = WindowDetector()
        
        # Configuration storage
        self.config_dir = Path(config_dir).expanduser()
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.rules_file = self.config_dir / "privacy_rules.json"
        self.modes_file = self.config_dir / "privacy_modes.json"
        self.history_file = self.config_dir / "privacy_history.json"
        
        # Load configurations
        self.rules: List[PrivacyRule] = self._load_rules()
        self.modes: Dict[str, PrivacyMode] = self._load_modes()
        self.current_mode: str = "normal"
        
        # Default sensitive patterns
        self.sensitive_patterns = {
            'password': [
                r'password', r'passwd', r'pwd', r'passcode',
                r'api[_\s]?key', r'secret[_\s]?key', r'auth[_\s]?token'
            ],
            'financial': [
                r'bank', r'account\s+number', r'routing\s+number',
                r'credit\s+card', r'ssn', r'social\s+security',
                r'salary', r'compensation', r'invoice'
            ],
            'personal': [
                r'medical', r'health\s+record', r'prescription',
                r'diagnosis', r'private', r'confidential',
                r'personal\s+information'
            ],
            'legal': [
                r'contract', r'agreement', r'legal\s+document',
                r'nda', r'non[_\s]?disclosure', r'lawsuit'
            ]
        }
        
        # Apps that are always sensitive
        self.always_sensitive_apps = {
            '1Password', 'LastPass', 'Bitwarden', 'KeePass', 'Dashlane',
            'Keychain Access', 'Banking', 'TurboTax', 'Health'
        }
        
        # Initialize default modes if none exist
        if not self.modes:
            self._initialize_default_modes()
    
    def filter_windows(self, windows: List[WindowInfo], 
                      respect_mode: bool = True) -> Tuple[List[WindowInfo], List[WindowInfo]]:
        """Filter windows based on privacy rules and current mode"""
        allowed_windows = []
        blocked_windows = []
        
        # Get current mode rules
        mode_rules = []
        if respect_mode and self.current_mode in self.modes:
            mode_rules = self.modes[self.current_mode].rules
        
        # Combine with global rules
        all_rules = self.rules + mode_rules
        
        for window in windows:
            if self._should_block_window(window, all_rules):
                blocked_windows.append(window)
            else:
                allowed_windows.append(window)
        
        return allowed_windows, blocked_windows
    
    def detect_sensitive_content(self, windows: List[WindowInfo]) -> List[SensitiveContent]:
        """Detect sensitive content in windows"""
        sensitive_findings = []
        
        for window in windows:
            # Check if app is always sensitive
            if any(app in window.app_name for app in self.always_sensitive_apps):
                sensitive_findings.append(SensitiveContent(
                    window_id=window.window_id,
                    content_type='sensitive_app',
                    confidence=1.0,
                    detected_patterns=[window.app_name],
                    recommended_action='hide'
                ))
                continue
            
            # Check window title for sensitive patterns
            if window.window_title:
                title_lower = window.window_title.lower()
                
                for content_type, patterns in self.sensitive_patterns.items():
                    matches = []
                    for pattern in patterns:
                        if re.search(pattern, title_lower, re.IGNORECASE):
                            matches.append(pattern)
                    
                    if matches:
                        confidence = min(len(matches) * 0.3, 1.0)
                        sensitive_findings.append(SensitiveContent(
                            window_id=window.window_id,
                            content_type=content_type,
                            confidence=confidence,
                            detected_patterns=matches,
                            recommended_action='blur' if confidence < 0.7 else 'hide'
                        ))
                        break  # Only report highest priority finding
        
        return sensitive_findings
    
    def set_privacy_mode(self, mode_name: str) -> bool:
        """Set current privacy mode"""
        if mode_name not in self.modes:
            logger.error(f"Unknown privacy mode: {mode_name}")
            return False
        
        self.current_mode = mode_name
        logger.info(f"Privacy mode set to: {mode_name}")
        
        # Log mode change
        self._log_privacy_event("mode_change", {"from": self.current_mode, "to": mode_name})
        
        return True
    
    def add_privacy_rule(self, rule_type: str, target: str, 
                        action: str = "hide", duration_minutes: Optional[int] = None) -> PrivacyRule:
        """Add a new privacy rule"""
        rule = PrivacyRule(
            rule_id=f"rule_{datetime.now().timestamp()}",
            rule_type=rule_type,
            target=target,
            action=action,
            expires_at=datetime.now() + timedelta(minutes=duration_minutes) if duration_minutes else None
        )
        
        self.rules.append(rule)
        self._save_rules()
        
        # Log rule addition
        self._log_privacy_event("rule_added", asdict(rule))
        
        return rule
    
    def create_temporary_privacy_session(self, duration_minutes: int = 30) -> str:
        """Create a temporary privacy session that hides all windows"""
        session_id = f"temp_session_{datetime.now().timestamp()}"
        
        # Add temporary rule to hide everything
        rule = self.add_privacy_rule(
            rule_type="temporary",
            target="*",
            action="exclude",
            duration_minutes=duration_minutes
        )
        
        logger.info(f"Created temporary privacy session: {session_id} for {duration_minutes} minutes")
        
        return session_id
    
    def get_window_privacy_status(self, window: WindowInfo) -> Dict[str, any]:
        """Get privacy status for a specific window"""
        status = {
            "window_id": window.window_id,
            "app_name": window.app_name,
            "is_sensitive": False,
            "is_blocked": False,
            "reasons": [],
            "recommended_action": "allow"
        }
        
        # Check if blocked by rules
        all_rules = self.rules
        if self.current_mode in self.modes:
            all_rules += self.modes[self.current_mode].rules
        
        if self._should_block_window(window, all_rules):
            status["is_blocked"] = True
            status["reasons"].append("Blocked by privacy rule")
            status["recommended_action"] = "hide"
        
        # Check for sensitive content
        sensitive = self.detect_sensitive_content([window])
        if sensitive:
            status["is_sensitive"] = True
            status["reasons"].extend([f"{s.content_type} detected" for s in sensitive])
            status["recommended_action"] = sensitive[0].recommended_action
        
        return status
    
    def generate_privacy_report(self) -> Dict[str, any]:
        """Generate a privacy report of current settings and detections"""
        windows = self.window_detector.get_all_windows()
        allowed, blocked = self.filter_windows(windows)
        sensitive = self.detect_sensitive_content(allowed)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "current_mode": self.current_mode,
            "statistics": {
                "total_windows": len(windows),
                "allowed_windows": len(allowed),
                "blocked_windows": len(blocked),
                "sensitive_windows": len(sensitive)
            },
            "active_rules": len([r for r in self.rules if r.is_active]),
            "blocked_apps": list(set(w.app_name for w in blocked)),
            "sensitive_findings": [
                {
                    "app": next((w.app_name for w in windows if w.window_id == s.window_id), "Unknown"),
                    "type": s.content_type,
                    "confidence": s.confidence
                }
                for s in sensitive
            ],
            "recommendations": self._generate_privacy_recommendations(windows, sensitive)
        }
        
        return report
    
    def _should_block_window(self, window: WindowInfo, rules: List[PrivacyRule]) -> bool:
        """Check if window should be blocked by rules"""
        # Clean expired rules
        active_rules = [r for r in rules 
                       if r.is_active and (not r.expires_at or r.expires_at > datetime.now())]
        
        for rule in active_rules:
            if rule.action not in ['hide', 'exclude']:
                continue
            
            # Check rule match
            if rule.rule_type == 'blacklist':
                if rule.target in window.app_name:
                    return True
                if window.window_title and rule.target in window.window_title:
                    return True
            
            elif rule.rule_type == 'pattern':
                if window.window_title and re.search(rule.target, window.window_title, re.IGNORECASE):
                    return True
            
            elif rule.rule_type == 'temporary' and rule.target == '*':
                return True
        
        return False
    
    def _initialize_default_modes(self) -> None:
        """Initialize default privacy modes"""
        self.modes = {
            'normal': PrivacyMode(
                mode_name='normal',
                description='Standard privacy protection',
                rules=[],
                auto_hide_sensitive=True,
                blur_content=False,
                exclude_from_analysis=False
            ),
            'meeting': PrivacyMode(
                mode_name='meeting',
                description='Enhanced privacy for screen sharing',
                rules=[
                    PrivacyRule(
                        rule_id='meeting_messages',
                        rule_type='blacklist',
                        target='Messages',
                        action='hide'
                    ),
                    PrivacyRule(
                        rule_id='meeting_slack',
                        rule_type='blacklist',
                        target='Slack',
                        action='hide'
                    ),
                    PrivacyRule(
                        rule_id='meeting_passwords',
                        rule_type='blacklist',
                        target='1Password',
                        action='hide'
                    )
                ],
                auto_hide_sensitive=True,
                blur_content=True,
                exclude_from_analysis=False
            ),
            'focused': PrivacyMode(
                mode_name='focused',
                description='Hide distracting windows',
                rules=[
                    PrivacyRule(
                        rule_id='focus_social',
                        rule_type='pattern',
                        target='twitter|facebook|instagram|reddit',
                        action='hide'
                    )
                ],
                auto_hide_sensitive=False,
                blur_content=False,
                exclude_from_analysis=False
            ),
            'private': PrivacyMode(
                mode_name='private',
                description='Maximum privacy - exclude all from analysis',
                rules=[
                    PrivacyRule(
                        rule_id='private_all',
                        rule_type='temporary',
                        target='*',
                        action='exclude'
                    )
                ],
                auto_hide_sensitive=True,
                blur_content=True,
                exclude_from_analysis=True
            )
        }
        
        self._save_modes()
    
    def _generate_privacy_recommendations(self, windows: List[WindowInfo], 
                                        sensitive: List[SensitiveContent]) -> List[str]:
        """Generate privacy recommendations"""
        recommendations = []
        
        # Check for multiple sensitive windows
        if len(sensitive) > 3:
            recommendations.append(
                "Multiple sensitive windows detected. Consider using 'meeting' privacy mode."
            )
        
        # Check for password managers
        password_managers = [w for w in windows 
                           if any(pm in w.app_name for pm in ['1Password', 'LastPass', 'Bitwarden'])]
        if password_managers:
            recommendations.append(
                "Password manager detected. Hide before screen sharing."
            )
        
        # Check for communication apps during focus time
        if self.current_mode == 'focused':
            comm_apps = [w for w in windows 
                        if any(app in w.app_name for app in ['Discord', 'Slack', 'Messages'])]
            if comm_apps:
                recommendations.append(
                    "Communication apps open during focus mode. Consider closing for better concentration."
                )
        
        return recommendations
    
    def _log_privacy_event(self, event_type: str, details: Dict) -> None:
        """Log privacy-related events"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "details": details
        }
        
        # Load existing history
        history = []
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    history = json.load(f)
            except:
                pass
        
        # Add new event
        history.append(event)
        
        # Keep last 1000 events
        history = history[-1000:]
        
        # Save
        try:
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving privacy history: {e}")
    
    def _load_rules(self) -> List[PrivacyRule]:
        """Load privacy rules from file"""
        if self.rules_file.exists():
            try:
                with open(self.rules_file, 'r') as f:
                    data = json.load(f)
                    rules = []
                    for r_data in data:
                        rule = PrivacyRule(
                            rule_id=r_data['rule_id'],
                            rule_type=r_data['rule_type'],
                            target=r_data['target'],
                            action=r_data['action'],
                            expires_at=datetime.fromisoformat(r_data['expires_at']) if r_data.get('expires_at') else None,
                            created_at=datetime.fromisoformat(r_data['created_at']),
                            is_active=r_data.get('is_active', True)
                        )
                        rules.append(rule)
                    return rules
            except Exception as e:
                logger.error(f"Error loading rules: {e}")
        return []
    
    def _save_rules(self) -> None:
        """Save privacy rules to file"""
        try:
            data = []
            for rule in self.rules:
                r_data = asdict(rule)
                r_data['expires_at'] = rule.expires_at.isoformat() if rule.expires_at else None
                r_data['created_at'] = rule.created_at.isoformat()
                data.append(r_data)
            
            with open(self.rules_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving rules: {e}")
    
    def _load_modes(self) -> Dict[str, PrivacyMode]:
        """Load privacy modes from file"""
        if self.modes_file.exists():
            try:
                with open(self.modes_file, 'r') as f:
                    data = json.load(f)
                    modes = {}
                    for mode_name, m_data in data.items():
                        # Load rules for mode
                        rules = []
                        for r_data in m_data.get('rules', []):
                            rule = PrivacyRule(
                                rule_id=r_data['rule_id'],
                                rule_type=r_data['rule_type'],
                                target=r_data['target'],
                                action=r_data['action']
                            )
                            rules.append(rule)
                        
                        mode = PrivacyMode(
                            mode_name=mode_name,
                            description=m_data['description'],
                            rules=rules,
                            auto_hide_sensitive=m_data.get('auto_hide_sensitive', True),
                            blur_content=m_data.get('blur_content', False),
                            exclude_from_analysis=m_data.get('exclude_from_analysis', False)
                        )
                        modes[mode_name] = mode
                    return modes
            except Exception as e:
                logger.error(f"Error loading modes: {e}")
        return {}
    
    def _save_modes(self) -> None:
        """Save privacy modes to file"""
        try:
            data = {}
            for mode_name, mode in self.modes.items():
                # Convert rules to dict and handle datetime
                rules_data = []
                for rule in mode.rules:
                    r_dict = asdict(rule)
                    # Convert datetime to isoformat
                    if 'created_at' in r_dict and hasattr(rule.created_at, 'isoformat'):
                        r_dict['created_at'] = rule.created_at.isoformat()
                    if 'expires_at' in r_dict and rule.expires_at:
                        r_dict['expires_at'] = rule.expires_at.isoformat()
                    rules_data.append(r_dict)
                
                m_data = {
                    'description': mode.description,
                    'rules': rules_data,
                    'auto_hide_sensitive': mode.auto_hide_sensitive,
                    'blur_content': mode.blur_content,
                    'exclude_from_analysis': mode.exclude_from_analysis
                }
                data[mode_name] = m_data
            
            with open(self.modes_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving modes: {e}")

def test_privacy_controls():
    """Test privacy control system"""
    print("🔒 Testing Privacy Control System")
    print("=" * 50)
    
    privacy_system = PrivacyControlSystem()
    
    # Get current windows
    windows = privacy_system.window_detector.get_all_windows()
    print(f"\n📊 Found {len(windows)} windows")
    
    # Test sensitive content detection
    print("\n🔍 Detecting sensitive content...")
    sensitive = privacy_system.detect_sensitive_content(windows)
    
    if sensitive:
        print(f"\n⚠️  Found {len(sensitive)} sensitive windows:")
        for s in sensitive[:5]:
            window = next((w for w in windows if w.window_id == s.window_id), None)
            if window:
                print(f"   • {window.app_name}: {s.content_type} (confidence: {s.confidence:.0%})")
                print(f"     Action: {s.recommended_action}")
    else:
        print("   No sensitive content detected")
    
    # Test privacy filtering
    print(f"\n🛡️  Testing privacy filtering (mode: {privacy_system.current_mode})...")
    allowed, blocked = privacy_system.filter_windows(windows)
    
    print(f"   Allowed: {len(allowed)} windows")
    print(f"   Blocked: {len(blocked)} windows")
    
    if blocked:
        print(f"\n   Blocked windows:")
        for window in blocked[:3]:
            print(f"     • {window.app_name} - {window.window_title or 'Untitled'}")
    
    # Test privacy modes
    print("\n🔐 Testing privacy modes...")
    for mode_name in ['meeting', 'focused', 'private']:
        privacy_system.set_privacy_mode(mode_name)
        allowed, blocked = privacy_system.filter_windows(windows)
        print(f"   {mode_name}: {len(blocked)} blocked, {len(allowed)} allowed")
    
    # Reset to normal
    privacy_system.set_privacy_mode('normal')
    
    # Generate privacy report
    print("\n📄 Generating privacy report...")
    report = privacy_system.generate_privacy_report()
    
    print(f"   Current mode: {report['current_mode']}")
    print(f"   Statistics: {report['statistics']}")
    
    if report['recommendations']:
        print(f"\n   Recommendations:")
        for rec in report['recommendations']:
            print(f"     • {rec}")
    
    print("\n✅ Privacy control test complete!")

if __name__ == "__main__":
    test_privacy_controls()