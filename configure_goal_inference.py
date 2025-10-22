#!/usr/bin/env python3
"""
Configuration Manager for Goal Inference + Autonomous Decision Engine
Easily adjust settings for optimal performance
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import argparse


class GoalInferenceConfigurator:
    """Manage Goal Inference and Autonomous Engine configuration"""

    def __init__(self):
        self.config_path = Path("backend/config/integration_config.json")
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Load current configuration"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        else:
            print(f"❌ Config file not found at {self.config_path}")
            return {}

    def save_config(self):
        """Save configuration to file"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"✅ Configuration saved to {self.config_path}")

    def display_current_config(self):
        """Display current configuration"""
        print("\n" + "=" * 60)
        print("📊 CURRENT GOAL INFERENCE CONFIGURATION")
        print("=" * 60)

        for section, settings in self.config.items():
            print(f"\n🔧 {section.upper()}:")
            for key, value in settings.items():
                if isinstance(value, dict):
                    print(f"   {key}:")
                    for k, v in value.items():
                        print(f"      {k}: {v}")
                else:
                    print(f"   {key}: {value}")

    def set_preset(self, preset: str):
        """Apply a preset configuration"""
        presets = {
            "aggressive": {
                "description": "Highly proactive, learns quickly, suggests often",
                "changes": {
                    "goal_inference.min_goal_confidence": 0.65,
                    "autonomous_decisions.min_decision_confidence": 0.60,
                    "integration.proactive_suggestion_threshold": 0.75,
                    "integration.enable_automation": True,
                    "learning.pattern_confidence_boost": 0.10
                }
            },
            "balanced": {
                "description": "Default balanced settings",
                "changes": {
                    "goal_inference.min_goal_confidence": 0.75,
                    "autonomous_decisions.min_decision_confidence": 0.70,
                    "integration.proactive_suggestion_threshold": 0.85,
                    "integration.enable_automation": False,
                    "learning.pattern_confidence_boost": 0.05
                }
            },
            "conservative": {
                "description": "Cautious, requires high confidence",
                "changes": {
                    "goal_inference.min_goal_confidence": 0.85,
                    "autonomous_decisions.min_decision_confidence": 0.80,
                    "integration.proactive_suggestion_threshold": 0.90,
                    "integration.enable_automation": False,
                    "learning.pattern_confidence_boost": 0.02
                }
            },
            "learning": {
                "description": "Optimized for learning your patterns",
                "changes": {
                    "learning.enabled": True,
                    "learning.min_samples_for_pattern": 2,
                    "learning.pattern_confidence_boost": 0.10,
                    "learning.feedback_weight": 0.15,
                    "autonomous_decisions.exploration_rate": 0.2
                }
            },
            "performance": {
                "description": "Maximum speed, aggressive caching",
                "changes": {
                    "performance.max_prediction_cache_size": 200,
                    "performance.cache_ttl_seconds": 600,
                    "performance.parallel_processing": True,
                    "display_optimization.preload_resources": True
                }
            }
        }

        if preset not in presets:
            print(f"❌ Unknown preset: {preset}")
            print(f"   Available presets: {', '.join(presets.keys())}")
            return

        print(f"\n🎯 Applying preset: {preset}")
        print(f"   Description: {presets[preset]['description']}")

        for path, value in presets[preset]['changes'].items():
            self.set_value(path, value, silent=True)

        self.save_config()
        print("✅ Preset applied successfully!")

    def set_value(self, path: str, value: Any, silent: bool = False):
        """Set a configuration value using dot notation"""
        keys = path.split('.')
        current = self.config

        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Set the value
        old_value = current.get(keys[-1], "not set")
        current[keys[-1]] = value

        if not silent:
            print(f"✅ Set {path}:")
            print(f"   Old value: {old_value}")
            print(f"   New value: {value}")

    def get_value(self, path: str) -> Any:
        """Get a configuration value using dot notation"""
        keys = path.split('.')
        current = self.config

        for key in keys:
            if key in current:
                current = current[key]
            else:
                return None

        return current

    def interactive_mode(self):
        """Interactive configuration mode"""
        print("\n" + "=" * 60)
        print("🎮 INTERACTIVE CONFIGURATION MODE")
        print("=" * 60)

        while True:
            print("\nOptions:")
            print("1. View current configuration")
            print("2. Apply preset (aggressive/balanced/conservative/learning/performance)")
            print("3. Change specific setting")
            print("4. Toggle automation")
            print("5. Adjust confidence thresholds")
            print("6. Configure display preferences")
            print("7. Save and exit")
            print("8. Exit without saving")

            choice = input("\nEnter choice (1-8): ").strip()

            if choice == '1':
                self.display_current_config()

            elif choice == '2':
                preset = input("Enter preset name: ").strip().lower()
                self.set_preset(preset)

            elif choice == '3':
                path = input("Enter setting path (e.g., learning.enabled): ").strip()
                value = input("Enter new value: ").strip()

                # Try to parse value as appropriate type
                try:
                    if value.lower() in ['true', 'false']:
                        value = value.lower() == 'true'
                    elif '.' in value:
                        value = float(value)
                    elif value.isdigit():
                        value = int(value)
                except:
                    pass

                self.set_value(path, value)

            elif choice == '4':
                current = self.get_value('integration.enable_automation')
                new_value = not current
                self.set_value('integration.enable_automation', new_value)
                print(f"✅ Automation {'enabled' if new_value else 'disabled'}")

            elif choice == '5':
                print("\nConfidence Thresholds:")
                print("1. Goal confidence (current: {})".format(
                    self.get_value('goal_inference.min_goal_confidence')))
                print("2. Decision confidence (current: {})".format(
                    self.get_value('autonomous_decisions.min_decision_confidence')))
                print("3. Proactive suggestion (current: {})".format(
                    self.get_value('integration.proactive_suggestion_threshold')))

                sub_choice = input("Which to adjust (1-3): ").strip()
                new_value = float(input("Enter new value (0.0-1.0): ").strip())

                if sub_choice == '1':
                    self.set_value('goal_inference.min_goal_confidence', new_value)
                elif sub_choice == '2':
                    self.set_value('autonomous_decisions.min_decision_confidence', new_value)
                elif sub_choice == '3':
                    self.set_value('integration.proactive_suggestion_threshold', new_value)

            elif choice == '6':
                print("\nDisplay Preferences:")
                print("1. Enable predictive connection:",
                      self.get_value('display_optimization.enable_predictive_connection'))
                print("2. Default display:",
                      self.get_value('display_optimization.default_display'))

                sub_choice = input("Which to change (1-2): ").strip()

                if sub_choice == '1':
                    current = self.get_value('display_optimization.enable_predictive_connection')
                    self.set_value('display_optimization.enable_predictive_connection', not current)
                elif sub_choice == '2':
                    display = input("Enter default display name: ").strip()
                    self.set_value('display_optimization.default_display', display)

            elif choice == '7':
                self.save_config()
                print("✅ Configuration saved. Exiting...")
                break

            elif choice == '8':
                print("Exiting without saving...")
                break


def main():
    """Main configuration interface"""
    parser = argparse.ArgumentParser(description="Configure Goal Inference + Autonomous Engine")

    parser.add_argument('--preset', choices=['aggressive', 'balanced', 'conservative', 'learning', 'performance'],
                       help='Apply a preset configuration')
    parser.add_argument('--set', nargs=2, metavar=('PATH', 'VALUE'),
                       help='Set a specific configuration value')
    parser.add_argument('--get', metavar='PATH',
                       help='Get a specific configuration value')
    parser.add_argument('--show', action='store_true',
                       help='Show current configuration')
    parser.add_argument('--interactive', action='store_true',
                       help='Enter interactive configuration mode')
    parser.add_argument('--enable-automation', action='store_true',
                       help='Enable automatic actions')
    parser.add_argument('--disable-automation', action='store_true',
                       help='Disable automatic actions')

    args = parser.parse_args()

    configurator = GoalInferenceConfigurator()

    if args.preset:
        configurator.set_preset(args.preset)

    elif args.set:
        path, value = args.set
        # Parse value
        if value.lower() in ['true', 'false']:
            value = value.lower() == 'true'
        elif '.' in value:
            value = float(value)
        elif value.isdigit():
            value = int(value)

        configurator.set_value(path, value)
        configurator.save_config()

    elif args.get:
        value = configurator.get_value(args.get)
        print(f"{args.get}: {value}")

    elif args.show:
        configurator.display_current_config()

    elif args.interactive:
        configurator.interactive_mode()

    elif args.enable_automation:
        configurator.set_value('integration.enable_automation', True)
        configurator.save_config()
        print("✅ Automation enabled")

    elif args.disable_automation:
        configurator.set_value('integration.enable_automation', False)
        configurator.save_config()
        print("✅ Automation disabled")

    else:
        print("Goal Inference Configuration Manager")
        print("=====================================\n")
        print("Usage examples:")
        print("  python configure_goal_inference.py --show")
        print("  python configure_goal_inference.py --preset aggressive")
        print("  python configure_goal_inference.py --set learning.enabled true")
        print("  python configure_goal_inference.py --interactive")
        print("  python configure_goal_inference.py --enable-automation")
        print("\nUse --help for all options")


if __name__ == "__main__":
    main()