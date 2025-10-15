#!/usr/bin/env python3
"""
Start JARVIS Display Monitoring
================================

Advanced display monitoring system with:
- Multi-method detection (AppleScript, CoreGraphics, Yabai)
- Voice integration
- Smart caching
- Event-driven callbacks
- Configuration management

Usage:
    python3 start_tv_monitoring.py [--config CONFIG_PATH] [--simple]

Options:
    --config PATH    Path to custom configuration file
    --simple         Use simple legacy monitor instead
    --test-voice     Test voice output before starting
    --add-display    Add a new display to monitoring

Author: Derek Russell
Date: 2025-10-15
Version: 2.0
"""

import asyncio
import sys
import signal
import argparse
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


async def main():
    """Main function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="JARVIS Display Monitoring")
    parser.add_argument("--config", help="Path to configuration file", default=None)
    parser.add_argument("--simple", action="store_true", help="Use simple legacy monitor")
    parser.add_argument("--test-voice", action="store_true", help="Test voice output")
    parser.add_argument("--add-display", action="store_true", help="Add a new display")
    parser.add_argument("--list-displays", action="store_true", help="List monitored displays")
    parser.add_argument("--status", action="store_true", help="Show monitor status")

    args = parser.parse_args()

    print("=" * 80)
    print("🖥️  JARVIS Advanced Display Monitor")
    print("=" * 80)
    print()

    # Handle special commands
    if args.add_display:
        await add_display_interactive()
        return

    if args.list_displays:
        list_displays(args.config)
        return

    # Use simple monitor if requested
    if args.simple:
        await run_simple_monitor()
        return

    # Use advanced monitor
    await run_advanced_monitor(args)


async def run_advanced_monitor(args):
    """Run the advanced display monitor"""
    from display.advanced_display_monitor import get_display_monitor
    from display.display_voice_handler import create_voice_handler
    from display.display_config_manager import get_config_manager

    # Load configuration
    config_manager = get_config_manager(args.config)

    # Create voice handler
    voice_handler = create_voice_handler()

    # Test voice if requested
    if args.test_voice:
        print("🎤 Testing voice output...")
        await voice_handler.test_voice()
        print("✅ Voice test complete")
        return

    # Create monitor
    monitor = get_display_monitor(
        config_path=args.config,
        voice_handler=voice_handler
    )

    # Show configuration summary
    summary = config_manager.get_summary()
    print("📋 Configuration:")
    print(f"   • Monitoring: {'✅ Enabled' if summary['monitoring_enabled'] else '❌ Disabled'}")
    print(f"   • Displays: {summary['monitored_displays_count']} monitored")
    print(f"   • Voice: {'✅ Enabled' if summary['voice_enabled'] else '❌ Disabled'}")
    print(f"   • Detection: {', '.join(summary['detection_methods'])}")
    print(f"   • Caching: {'✅ Enabled' if summary['caching_enabled'] else '❌ Disabled'}")
    print()

    # List monitored displays
    print("📺 Monitored Displays:")
    for display in config_manager.get_monitored_displays():
        status = "✅" if display.get('enabled', True) else "❌"
        auto = "🤖 Auto" if display.get('auto_connect', False) else "📢 Prompt"
        print(f"   {status} {display['name']} - {auto}")
    print()

    # Show status if requested
    if args.status:
        status = monitor.get_status()
        print("🔍 Monitor Status:")
        print(f"   • Running: {status['is_monitoring']}")
        print(f"   • Available: {status['available_displays']}")
        print(f"   • Connected: {status['connected_displays']}")
        return

    # Register event callbacks
    async def on_display_detected(display, detected_name):
        print(f"\n✨ Detected: {display.name} ({detected_name})")

    async def on_display_connected(display):
        print(f"\n✅ Connected: {display.name}")

    async def on_display_lost(display):
        print(f"\n❌ Lost: {display.name}")

    async def on_error(error):
        print(f"\n⚠️  Error: {error}")

    monitor.register_callback('display_detected', on_display_detected)
    monitor.register_callback('display_connected', on_display_connected)
    monitor.register_callback('display_lost', on_display_lost)
    monitor.register_callback('error', on_error)

    # Start monitoring
    print("🚀 Starting display monitoring...")
    print("   Press Ctrl+C to stop")
    print()
    print("=" * 80)
    print()

    await monitor.start()

    # Set up signal handler
    def signal_handler(sig, frame):
        print("\n\n🛑 Shutting down display monitor...")
        asyncio.create_task(monitor.stop())
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Keep running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping display monitor...")
        await monitor.stop()


async def run_simple_monitor():
    """Run the simple legacy monitor"""
    from display.simple_tv_monitor import get_tv_monitor

    print("📺 Using simple legacy monitor")
    print("   Monitoring for: Living Room TV")
    print("   ⏰ Check interval: Every 10 seconds")
    print()
    print("   When your TV becomes available, JARVIS will prompt you to connect.")
    print("   Press Ctrl+C to stop monitoring.")
    print()
    print("=" * 80)
    print()

    monitor = get_tv_monitor("Living Room TV")
    await monitor.start()

    def signal_handler(sig, frame):
        print("\n\n🛑 Stopping TV monitor...")
        asyncio.create_task(monitor.stop())
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping TV monitor...")
        await monitor.stop()


async def add_display_interactive():
    """Interactive display addition"""
    from display.display_config_manager import get_config_manager

    print("➕ Add New Display to Monitoring")
    print("=" * 80)
    print()

    config_manager = get_config_manager()

    # Get display information
    display_id = input("Display ID (e.g., 'living_room_tv'): ").strip()
    display_name = input("Display Name (e.g., 'Living Room TV'): ").strip()
    display_type = input("Display Type (airplay/hdmi/thunderbolt/usb_c/wireless) [airplay]: ").strip() or "airplay"

    # Aliases
    aliases_input = input("Aliases (comma-separated, e.g., 'Living Room,LG TV') [none]: ").strip()
    aliases = [a.strip() for a in aliases_input.split(',')] if aliases_input else []

    # Auto-connect
    auto_connect_input = input("Auto-connect when detected? (yes/no) [no]: ").strip().lower()
    auto_connect = auto_connect_input in ['yes', 'y']

    # Auto-prompt
    auto_prompt_input = input("Auto-prompt when detected? (yes/no) [yes]: ").strip().lower()
    auto_prompt = auto_prompt_input not in ['no', 'n']

    # Connection mode
    connection_mode = input("Connection mode (extend/mirror) [extend]: ").strip() or "extend"

    # Priority
    priority_input = input("Priority (1-10) [1]: ").strip()
    priority = int(priority_input) if priority_input.isdigit() else 1

    # Create display config
    display_config = {
        'id': display_id,
        'name': display_name,
        'display_type': display_type,
        'aliases': aliases,
        'auto_connect': auto_connect,
        'auto_prompt': auto_prompt,
        'connection_mode': connection_mode,
        'priority': priority,
        'enabled': True
    }

    # Confirm
    print()
    print("Display Configuration:")
    print(f"  ID: {display_id}")
    print(f"  Name: {display_name}")
    print(f"  Type: {display_type}")
    print(f"  Aliases: {aliases}")
    print(f"  Auto-connect: {auto_connect}")
    print(f"  Auto-prompt: {auto_prompt}")
    print(f"  Mode: {connection_mode}")
    print(f"  Priority: {priority}")
    print()

    confirm = input("Add this display? (yes/no): ").strip().lower()

    if confirm in ['yes', 'y']:
        if config_manager.add_display(display_config):
            print(f"✅ Added {display_name} to monitoring")
        else:
            print(f"❌ Failed to add display")
    else:
        print("❌ Cancelled")


def list_displays(config_path=None):
    """List all monitored displays"""
    from display.display_config_manager import get_config_manager

    config_manager = get_config_manager(config_path)

    print("📺 Monitored Displays:")
    print("=" * 80)
    print()

    displays = config_manager.get_monitored_displays()

    if not displays:
        print("   No displays configured for monitoring")
        print()
        print("   Add a display with: python3 start_tv_monitoring.py --add-display")
    else:
        for i, display in enumerate(displays, 1):
            status = "✅ Enabled" if display.get('enabled', True) else "❌ Disabled"
            auto = "🤖 Auto-connect" if display.get('auto_connect', False) else "📢 Prompt"

            print(f"{i}. {display['name']}")
            print(f"   ID: {display['id']}")
            print(f"   Type: {display['display_type']}")
            print(f"   Status: {status}")
            print(f"   Action: {auto}")
            print(f"   Mode: {display.get('connection_mode', 'extend')}")
            if display.get('aliases'):
                print(f"   Aliases: {', '.join(display['aliases'])}")
            print()

    print("=" * 80)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n✅ Display monitoring stopped")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
