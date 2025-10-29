#!/usr/bin/env python3
"""
GCP VM Status and Management CLI
=================================

Standalone tool to check GCP VM status, create/terminate VMs, and view costs.
Can be called from start_system.py or used independently.

Usage:
    python gcp_vm_status.py              # Show status
    python gcp_vm_status.py --create     # Create VM manually
    python gcp_vm_status.py --terminate  # Terminate all VMs
    python gcp_vm_status.py --costs      # Show cost summary
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.cost_tracker import get_cost_tracker
from core.gcp_vm_manager import get_gcp_vm_manager
from core.platform_memory_monitor import get_memory_monitor


class Colors:
    """ANSI color codes"""

    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


async def show_vm_status(verbose: bool = False) -> dict:
    """
    Show current GCP VM status

    Returns:
        dict with status information
    """
    print(f"\n{Colors.HEADER}{'='*70}")
    print(f"{Colors.BOLD}☁️  GCP VM Auto-Creation Status{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}\n")

    enabled = os.getenv("GCP_VM_ENABLED", "true").lower() == "true"

    if not enabled:
        print(f"{Colors.YELLOW}⚠️  GCP VM auto-creation is DISABLED{Colors.ENDC}")
        print(f"   Set GCP_VM_ENABLED=true to enable")
        return {"enabled": False, "vms": []}

    print(f"{Colors.GREEN}✅ GCP VM auto-creation is ENABLED{Colors.ENDC}\n")

    try:
        # Get VM manager
        manager = await get_gcp_vm_manager()

        # Show configuration
        print(f"{Colors.BOLD}Configuration:{Colors.ENDC}")
        print(f"  Project ID:      {manager.config.project_id}")
        print(f"  Region:          {manager.config.region}")
        print(f"  Zone:            {manager.config.zone}")
        print(f"  Machine Type:    {manager.config.machine_type} (4 vCPU, 32GB RAM)")
        print(
            f"  Spot VMs:        {'Yes' if manager.config.use_spot else 'No'} (${manager.config.spot_max_price:.3f}/hour max)"
        )
        print(f"  Daily Budget:    ${manager.config.daily_budget_usd:.2f}")
        print(f"  Max Concurrent:  {manager.config.max_concurrent_vms} VMs")
        print(f"  Max Lifetime:    {manager.config.max_vm_lifetime_hours:.1f} hours")

        # Show statistics
        stats = manager.get_stats()
        print(f"\n{Colors.BOLD}Statistics:{Colors.ENDC}")
        print(f"  Total Created:   {stats['total_created']}")
        print(f"  Total Failed:    {stats['total_failed']}")
        print(f"  Total Terminated: {stats['total_terminated']}")
        print(f"  Currently Active: {stats['current_active']}")
        print(f"  Total Cost:      ${stats['total_cost']:.4f}")

        # Show active VMs
        if manager.managed_vms:
            print(f"\n{Colors.BOLD}Active VMs:{Colors.ENDC}")
            for vm_name, vm in manager.managed_vms.items():
                vm.update_cost()  # Update cost before displaying

                status_color = Colors.GREEN if vm.is_healthy else Colors.YELLOW
                print(f"\n  {status_color}● {vm_name}{Colors.ENDC}")
                print(f"    State:      {vm.state.value}")
                print(f"    IP:         {vm.ip_address or 'N/A'}")
                print(f"    Internal:   {vm.internal_ip or 'N/A'}")
                print(f"    Uptime:     {vm.uptime_hours:.2f} hours")
                print(f"    Cost:       ${vm.total_cost:.4f}")
                print(f"    Components: {', '.join(vm.components)}")
                print(f"    Trigger:    {vm.trigger_reason}")
                print(f"    Health:     {vm.health_status}")

                if verbose:
                    print(f"    Instance ID: {vm.instance_id}")
                    print(f"    Zone:        {vm.zone}")
                    print(f"    Created:     {vm.created_at}")
        else:
            print(f"\n{Colors.CYAN}ℹ️  No active VMs currently running{Colors.ENDC}")

        # Show memory status
        print(f"\n{Colors.BOLD}Local Memory Status:{Colors.ENDC}")
        memory_monitor = get_memory_monitor()
        snapshot = memory_monitor.capture_snapshot()

        memory_pct = (snapshot.used_gb / snapshot.total_gb * 100) if snapshot.total_gb > 0 else 0
        memory_color = (
            Colors.GREEN if memory_pct < 70 else Colors.YELLOW if memory_pct < 85 else Colors.RED
        )

        print(f"  Total RAM:   {snapshot.total_gb:.1f} GB")
        print(
            f"  Used RAM:    {memory_color}{snapshot.used_gb:.1f} GB ({memory_pct:.1f}%){Colors.ENDC}"
        )
        print(f"  Available:   {snapshot.available_gb:.1f} GB")
        print(f"  Pressure:    {snapshot.pressure_level}")

        if snapshot.gcp_shift_recommended:
            print(f"  {Colors.YELLOW}⚠️  GCP shift recommended: {snapshot.reasoning}{Colors.ENDC}")

        return {
            "enabled": True,
            "vms": list(manager.managed_vms.keys()),
            "stats": stats,
            "memory_pressure": snapshot.pressure_level,
        }

    except Exception as e:
        print(f"\n{Colors.RED}❌ Error retrieving VM status: {e}{Colors.ENDC}")
        return {"enabled": True, "error": str(e)}


async def create_vm_interactive():
    """Interactive VM creation"""
    print(f"\n{Colors.CYAN}🚀 Manual VM Creation{Colors.ENDC}\n")

    # Get memory snapshot
    memory_monitor = get_memory_monitor()
    snapshot = memory_monitor.capture_snapshot()

    # Get VM manager
    manager = await get_gcp_vm_manager()

    # Ask for components
    print("Select components to run on VM:")
    print("  1. VISION + CHATBOTS (recommended for most tasks)")
    print("  2. VISION + CHATBOTS + ML_MODELS (heavy ML tasks)")
    print("  3. All heavy components (VISION, CHATBOTS, ML_MODELS, LOCAL_LLM)")
    print("  4. Custom selection")

    choice = input(f"\n{Colors.BOLD}Choice [1-4]: {Colors.ENDC}").strip()

    if choice == "1":
        components = ["VISION", "CHATBOTS"]
    elif choice == "2":
        components = ["VISION", "CHATBOTS", "ML_MODELS"]
    elif choice == "3":
        components = ["VISION", "CHATBOTS", "ML_MODELS", "LOCAL_LLM"]
    elif choice == "4":
        comp_input = input("Enter components (comma-separated): ").strip()
        components = [c.strip() for c in comp_input.split(",")]
    else:
        components = ["VISION", "CHATBOTS"]

    print(f"\n{Colors.CYAN}Creating VM with components: {', '.join(components)}{Colors.ENDC}")

    # Check if creation is allowed
    should_create, reason, confidence = await manager.should_create_vm(
        snapshot, trigger_reason="Manual creation via CLI"
    )

    if not should_create:
        print(f"\n{Colors.YELLOW}⚠️  VM creation not recommended: {reason}{Colors.ENDC}")
        proceed = input(f"{Colors.BOLD}Create anyway? [y/N]: {Colors.ENDC}").strip().lower()
        if proceed != "y":
            print("Cancelled.")
            return

    # Create VM
    print(f"\n{Colors.CYAN}Creating VM...{Colors.ENDC}")
    vm = await manager.create_vm(
        components=components, trigger_reason="Manual creation via CLI", metadata={"manual": True}
    )

    if vm:
        print(f"\n{Colors.GREEN}✅ VM created successfully!{Colors.ENDC}")
        print(f"  Name:       {vm.name}")
        print(f"  IP:         {vm.ip_address}")
        print(f"  Internal:   {vm.internal_ip}")
        print(f"  Components: {', '.join(vm.components)}")
        print(f"  Cost:       ${vm.cost_per_hour:.3f}/hour")
        print(
            f"\n{Colors.CYAN}Backend will be available at: http://{vm.ip_address}:8010{Colors.ENDC}"
        )
    else:
        print(f"\n{Colors.RED}❌ Failed to create VM{Colors.ENDC}")


async def terminate_vms():
    """Terminate all active VMs"""
    print(f"\n{Colors.YELLOW}🛑 VM Termination{Colors.ENDC}\n")

    manager = await get_gcp_vm_manager()

    if not manager.managed_vms:
        print(f"{Colors.CYAN}ℹ️  No active VMs to terminate{Colors.ENDC}")
        return

    print(f"Active VMs:")
    for vm_name, vm in manager.managed_vms.items():
        print(f"  • {vm_name} (uptime: {vm.uptime_hours:.2f}h, cost: ${vm.total_cost:.4f})")

    confirm = input(f"\n{Colors.BOLD}Terminate all VMs? [y/N]: {Colors.ENDC}").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return

    print(f"\n{Colors.CYAN}Terminating VMs...{Colors.ENDC}")
    await manager.cleanup_all_vms(reason="Manual termination via CLI")

    print(f"\n{Colors.GREEN}✅ All VMs terminated{Colors.ENDC}")


async def show_costs():
    """Show cost summary"""
    print(f"\n{Colors.HEADER}{'='*70}")
    print(f"{Colors.BOLD}💰 GCP VM Cost Summary{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}\n")

    try:
        cost_tracker = await get_cost_tracker()

        # Daily cost
        daily_cost = await cost_tracker.get_daily_cost()
        budget = float(os.getenv("GCP_VM_DAILY_BUDGET", "5.0"))
        budget_pct = (daily_cost / budget * 100) if budget > 0 else 0

        budget_color = (
            Colors.GREEN if budget_pct < 70 else Colors.YELLOW if budget_pct < 90 else Colors.RED
        )

        print(f"{Colors.BOLD}Today:{Colors.ENDC}")
        print(
            f"  Cost:   {budget_color}${daily_cost:.4f}{Colors.ENDC} / ${budget:.2f} ({budget_pct:.1f}%)"
        )

        # Show active sessions
        active_sessions = cost_tracker.get_active_sessions()
        if active_sessions:
            print(f"\n{Colors.BOLD}Active Sessions:{Colors.ENDC}")
            for session in active_sessions:
                runtime_hours = session.get("runtime_hours", 0)
                session_cost = session.get("cost", 0)
                print(f"  • {session.get('instance_id', 'unknown')}")
                print(f"    Runtime: {runtime_hours:.2f}h")
                print(f"    Cost:    ${session_cost:.4f}")
                print(f"    Started: {session.get('created_at', 'unknown')}")

        # Show historical costs (if available)
        # This would require additional methods in cost_tracker

        print()

    except Exception as e:
        print(f"\n{Colors.RED}❌ Error retrieving costs: {e}{Colors.ENDC}")


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="GCP VM Status and Management")
    parser.add_argument("--create", action="store_true", help="Create a VM interactively")
    parser.add_argument("--terminate", action="store_true", help="Terminate all active VMs")
    parser.add_argument("--costs", action="store_true", help="Show cost summary")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    try:
        if args.create:
            await create_vm_interactive()
        elif args.terminate:
            await terminate_vms()
        elif args.costs:
            await show_costs()
        else:
            # Default: show status
            await show_vm_status(verbose=args.verbose)

    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Interrupted{Colors.ENDC}")
    except Exception as e:
        print(f"\n{Colors.RED}Error: {e}{Colors.ENDC}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
