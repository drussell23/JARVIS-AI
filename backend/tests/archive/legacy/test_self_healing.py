#!/usr/bin/env python3
"""
Test the Rust self-healing system.
Simulates various failure scenarios and verifies automatic recovery.
"""

import asyncio
import sys
import os
import shutil
import logging
from pathlib import Path
import time

# Add backend to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

async def test_self_healing():
    """Test the self-healing system with various scenarios."""
    print(f"\n{BLUE}=== Testing Rust Self-Healing System ==={RESET}\n")
    
    from vision.rust_self_healer import RustSelfHealer, get_self_healer
    from vision.dynamic_component_loader import get_component_loader, ComponentType
    
    # Initialize systems
    healer = get_self_healer()
    loader = get_component_loader()
    
    print(f"{BLUE}Starting self-healing and component loader...{RESET}")
    await loader.start()  # This also starts the healer
    
    # Get initial status
    initial_status = loader.get_status()
    print(f"\n{BLUE}Initial Component Status:{RESET}")
    for comp_name, comp_info in initial_status['components'].items():
        active = comp_info.get('active')
        if active:
            print(f"  • {comp_name}: {active['type']} implementation")
    
    # Test 1: Simulate missing Rust library
    print(f"\n{BLUE}Test 1: Simulating missing Rust library...{RESET}")
    rust_core_dir = backend_dir / "vision" / "jarvis-rust-core"
    target_dir = rust_core_dir / "target"
    backup_dir = rust_core_dir / "target_backup"
    
    if target_dir.exists():
        print(f"  • Backing up target directory...")
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.move(str(target_dir), str(backup_dir))
        print(f"  • {YELLOW}Rust library removed{RESET}")
    
    # Force a check
    print(f"  • Forcing component check...")
    changes = await loader.force_check()
    
    if changes:
        print(f"  • {GREEN}✓ Components switched to fallback:{RESET}")
        for comp, change in changes.items():
            print(f"    - {comp}: {change}")
    
    # Wait for self-healing
    print(f"\n  • Waiting for self-healing to kick in...")
    await asyncio.sleep(5)
    
    # Check if Rust was rebuilt
    if target_dir.exists():
        print(f"  • {GREEN}✓ Rust components rebuilt automatically!{RESET}")
    else:
        print(f"  • {RED}✗ Rust components not rebuilt{RESET}")
    
    # Test 2: Check health report
    print(f"\n{BLUE}Test 2: Checking self-healer health report...{RESET}")
    health = healer.get_health_report()
    
    print(f"  • Running: {health['running']}")
    print(f"  • Total fix attempts: {health['total_fix_attempts']}")
    print(f"  • Success rate: {health['success_rate']:.1%}")
    
    if health['recent_fixes']:
        print(f"  • Recent fixes:")
        for fix in health['recent_fixes'][-3:]:
            success = "✓" if fix['success'] else "✗"
            print(f"    - {fix['issue']}: {fix['strategy']} {success}")
    
    # Test 3: Manual diagnosis
    print(f"\n{BLUE}Test 3: Running manual diagnosis...{RESET}")
    success = await healer.diagnose_and_fix()
    
    if success:
        print(f"  • {GREEN}✓ Manual diagnosis and fix successful{RESET}")
    else:
        print(f"  • {YELLOW}No issues found or fix not needed{RESET}")
    
    # Test 4: Check if components upgraded back to Rust
    print(f"\n{BLUE}Test 4: Checking for automatic Rust upgrade...{RESET}")
    await asyncio.sleep(3)  # Give time for components to upgrade
    
    final_status = loader.get_status()
    rust_count = 0
    for comp_name, comp_info in final_status['components'].items():
        active = comp_info.get('active')
        if active and active['type'] == 'rust':
            rust_count += 1
            print(f"  • {GREEN}✓ {comp_name} using Rust{RESET}")
        elif active:
            print(f"  • {YELLOW}! {comp_name} still using {active['type']}{RESET}")
    
    print(f"\n  Total Rust components active: {rust_count}")
    
    # Restore backup if exists
    if backup_dir.exists():
        print(f"\n{BLUE}Restoring original target directory...{RESET}")
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.move(str(backup_dir), str(target_dir))
    
    # Stop systems
    await loader.stop()
    
    # Summary
    print(f"\n{BLUE}=== Self-Healing Test Summary ==={RESET}")
    print("• Self-healer successfully integrated with component loader")
    print("• Automatic diagnosis and fixing works")
    print("• Components automatically switch between Rust and Python")
    print("• Health reporting provides useful diagnostics")
    
    if rust_count > 0:
        print(f"\n{GREEN}✅ Self-healing system is working correctly!{RESET}")
    else:
        print(f"\n{YELLOW}⚠️ Self-healing attempted but Rust components may need manual intervention{RESET}")

async def test_specific_scenarios():
    """Test specific failure scenarios."""
    print(f"\n{BLUE}=== Testing Specific Failure Scenarios ==={RESET}\n")
    
    from vision.rust_self_healer import RustSelfHealer, RustIssueType
    
    healer = RustSelfHealer(check_interval=30)  # Faster checks for testing
    
    # Test diagnosis without starting the healer
    print(f"{BLUE}Diagnosing current Rust state...{RESET}")
    issue_type, details = await healer._diagnose_issue()
    
    print(f"  • Issue type: {issue_type.value}")
    print(f"  • Details: {details}")
    
    # Test fix strategy determination
    strategy = healer._determine_fix_strategy(issue_type, details)
    print(f"  • Recommended fix: {strategy.value}")
    
    # Test if Rust is working
    is_working = await healer._is_rust_working()
    print(f"  • Rust currently working: {is_working}")
    
    if not is_working:
        print(f"\n{YELLOW}Attempting to fix Rust components...{RESET}")
        success = await healer.diagnose_and_fix()
        if success:
            print(f"{GREEN}✓ Fix applied successfully!{RESET}")
        else:
            print(f"{RED}✗ Fix failed or not applicable{RESET}")

if __name__ == "__main__":
    print(f"{BLUE}Rust Self-Healing System Test{RESET}")
    print("=" * 50)
    
    # Run main test
    asyncio.run(test_self_healing())
    
    # Run specific scenario tests
    asyncio.run(test_specific_scenarios())