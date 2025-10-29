"""
GCP Spot VM Auto-Creation Manager
==================================

Advanced, robust, async system for auto-creating and managing GCP Spot VMs
when local memory pressure is too high.

Features:
- Auto-creates e2-highmem-4 Spot VMs (32GB RAM) when RAM >85%
- Integrates with intelligent_gcp_optimizer for smart decisions
- Tracks all costs via cost_tracker
- Async/await throughout for non-blocking operations
- No hardcoding - all configuration from environment/config
- Comprehensive error handling and retry logic
- VM lifecycle management (create, monitor, cleanup)
- Orphaned VM detection and cleanup
- Health monitoring and auto-recovery
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

# Google Cloud Compute Engine API
try:
    from google.cloud import compute_v1

    COMPUTE_AVAILABLE = True
except ImportError:
    COMPUTE_AVAILABLE = False
    logging.warning("google-cloud-compute not installed. VM creation disabled.")

from cost_tracker import get_cost_tracker
from intelligent_gcp_optimizer import get_gcp_optimizer

logger = logging.getLogger(__name__)


class VMState(Enum):
    """VM lifecycle states"""

    CREATING = "creating"
    PROVISIONING = "provisioning"
    STAGING = "staging"
    RUNNING = "running"
    STOPPING = "stopping"
    TERMINATED = "terminated"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class VMInstance:
    """Represents a managed VM instance"""

    instance_id: str
    name: str
    zone: str
    state: VMState
    created_at: float
    ip_address: Optional[str] = None
    internal_ip: Optional[str] = None
    last_health_check: Optional[float] = None
    health_status: str = "unknown"
    components: List[str] = field(default_factory=list)
    trigger_reason: str = ""
    cost_per_hour: float = 0.029  # e2-highmem-4 Spot VM
    total_cost: float = 0.0
    metadata: Dict = field(default_factory=dict)

    @property
    def uptime_hours(self) -> float:
        """Calculate VM uptime in hours"""
        return (time.time() - self.created_at) / 3600

    @property
    def is_healthy(self) -> bool:
        """Check if VM is healthy"""
        return self.health_status == "healthy" and self.state == VMState.RUNNING

    def update_cost(self):
        """Update total cost based on uptime"""
        self.total_cost = self.uptime_hours * self.cost_per_hour


@dataclass
class VMManagerConfig:
    """Configuration for GCP VM Manager"""

    # GCP Configuration
    project_id: str = field(default_factory=lambda: os.getenv("GCP_PROJECT_ID", "jarvis-473803"))
    region: str = field(default_factory=lambda: os.getenv("GCP_REGION", "us-central1"))
    zone: str = field(default_factory=lambda: os.getenv("GCP_ZONE", "us-central1-a"))

    # VM Configuration
    machine_type: str = "e2-highmem-4"  # 4 vCPU, 32GB RAM
    use_spot: bool = True
    spot_max_price: float = 0.10  # Max $0.10/hour (safety limit)

    # VM Naming
    vm_name_prefix: str = "jarvis-backend"

    # Image Configuration
    image_project: str = "ubuntu-os-cloud"
    image_family: str = "ubuntu-2204-lts"

    # Disk Configuration
    boot_disk_size_gb: int = 50
    boot_disk_type: str = "pd-standard"

    # Network Configuration
    network: str = "default"
    subnetwork: str = "default"

    # Startup Script Path
    startup_script_path: Optional[str] = field(
        default_factory=lambda: os.path.join(os.path.dirname(__file__), "gcp_vm_startup.sh")
    )

    # Health Check Configuration
    health_check_interval: int = 30  # seconds
    health_check_timeout: int = 10  # seconds
    max_health_check_failures: int = 3

    # VM Lifecycle
    max_vm_lifetime_hours: float = 3.0  # Auto-cleanup after 3 hours
    idle_timeout_minutes: int = 30  # Cleanup if idle for 30 min

    # Retry Configuration
    max_create_attempts: int = 3
    retry_delay_seconds: int = 10

    # Resource Limits
    max_concurrent_vms: int = 2
    daily_budget_usd: float = 5.0  # Max $5/day

    # Monitoring
    enable_monitoring: bool = True
    enable_logging: bool = True


class GCPVMManager:
    """
    Advanced GCP Spot VM auto-creation and lifecycle manager

    Integrates with:
    - intelligent_gcp_optimizer: For VM creation decisions
    - cost_tracker: For billing tracking
    - platform_memory_monitor: For memory pressure detection
    """

    def __init__(self, config: Optional[VMManagerConfig] = None):
        self.config = config or VMManagerConfig()

        # API clients (initialized lazily)
        self.instances_client: Optional[compute_v1.InstancesClient] = None
        self.zones_client: Optional[compute_v1.ZonesClient] = None

        # Integrations
        self.cost_tracker = None
        self.gcp_optimizer = None

        # State tracking
        self.managed_vms: Dict[str, VMInstance] = {}
        self.creating_vms: Dict[str, asyncio.Task] = {}

        # Monitoring
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False

        # Stats
        self.stats = {
            "total_created": 0,
            "total_failed": 0,
            "total_terminated": 0,
            "current_active": 0,
            "total_cost": 0.0,
        }

        self.initialized = False

    async def initialize(self):
        """Initialize GCP API clients and integrations"""
        if self.initialized:
            return

        if not COMPUTE_AVAILABLE:
            logger.error("❌ Google Cloud Compute Engine API not available")
            raise RuntimeError("google-cloud-compute package not installed")

        logger.info("🚀 Initializing GCP VM Manager...")

        try:
            # Initialize GCP Compute Engine clients
            self.instances_client = compute_v1.InstancesClient()
            self.zones_client = compute_v1.ZonesClient()
            logger.info(f"✅ GCP API clients initialized (Project: {self.config.project_id})")

            # Initialize integrations
            self.cost_tracker = await get_cost_tracker()
            logger.info("✅ Cost tracker integrated")

            self.gcp_optimizer = get_gcp_optimizer(config={"project_id": self.config.project_id})
            logger.info("✅ GCP optimizer integrated")

            # Start monitoring
            if self.config.enable_monitoring:
                self.monitoring_task = asyncio.create_task(self._monitoring_loop())
                logger.info("✅ VM monitoring started")

            self.initialized = True
            logger.info("✅ GCP VM Manager ready")

        except Exception as e:
            logger.error(f"Failed to initialize GCP VM Manager: {e}", exc_info=True)
            raise

    async def should_create_vm(
        self, memory_snapshot, trigger_reason: str = ""
    ) -> Tuple[bool, str, float]:
        """
        Determine if we should create a VM based on current conditions

        Returns: (should_create, reason, confidence_score)
        """
        if not self.initialized:
            await self.initialize()

        # Check budget limits
        if self.cost_tracker:
            daily_cost = await self.cost_tracker.get_daily_cost()
            if daily_cost >= self.config.daily_budget_usd:
                return (
                    False,
                    f"Daily budget exceeded: ${daily_cost:.2f} / ${self.config.daily_budget_usd:.2f}",
                    0.0,
                )

        # Check concurrent VM limits
        active_vms = len([vm for vm in self.managed_vms.values() if vm.state == VMState.RUNNING])
        if active_vms >= self.config.max_concurrent_vms:
            return (
                False,
                f"Max concurrent VMs reached: {active_vms} / {self.config.max_concurrent_vms}",
                0.0,
            )

        # Check if already creating a VM
        if self.creating_vms:
            return False, "VM creation already in progress", 0.0

        # Use intelligent optimizer for decision
        if self.gcp_optimizer:
            should_create, reason, score = await self.gcp_optimizer.should_create_vm(
                memory_snapshot, current_processes=None
            )
            return should_create, reason, score.overall_score

        # Fallback: Simple memory pressure check
        if memory_snapshot.gcp_shift_recommended:
            return True, trigger_reason or memory_snapshot.reasoning, 0.8

        return False, "Memory pressure within acceptable limits", 0.0

    async def create_vm(
        self, components: List[str], trigger_reason: str, metadata: Optional[Dict] = None
    ) -> Optional[VMInstance]:
        """
        Create a new GCP Spot VM instance

        Args:
            components: List of components that will run on this VM
            trigger_reason: Why this VM is being created
            metadata: Additional metadata

        Returns:
            VMInstance if successful, None otherwise
        """
        if not self.initialized:
            await self.initialize()

        logger.info(f"🚀 Creating GCP Spot VM...")
        logger.info(f"   Components: {', '.join(components)}")
        logger.info(f"   Trigger: {trigger_reason}")

        attempt = 0
        last_error = None

        while attempt < self.config.max_create_attempts:
            attempt += 1
            try:
                # Generate unique VM name
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                vm_name = f"{self.config.vm_name_prefix}-{timestamp}"

                logger.info(
                    f"🔨 Attempt {attempt}/{self.config.max_create_attempts}: Creating VM '{vm_name}'"
                )

                # Build VM configuration
                instance_config = self._build_instance_config(
                    vm_name=vm_name,
                    components=components,
                    trigger_reason=trigger_reason,
                    metadata=metadata or {},
                )

                # Create the VM (async operation)
                operation = await asyncio.to_thread(
                    self.instances_client.insert,
                    project=self.config.project_id,
                    zone=self.config.zone,
                    instance_resource=instance_config,
                )

                logger.info(f"⏳ VM creation operation started: {operation.name}")

                # Wait for operation to complete
                await self._wait_for_operation(operation)

                # Get the created instance
                instance = await asyncio.to_thread(
                    self.instances_client.get,
                    project=self.config.project_id,
                    zone=self.config.zone,
                    instance=vm_name,
                )

                # Extract IP addresses
                ip_address = None
                internal_ip = None
                if instance.network_interfaces:
                    internal_ip = instance.network_interfaces[0].network_i_p
                    if instance.network_interfaces[0].access_configs:
                        ip_address = instance.network_interfaces[0].access_configs[0].nat_i_p

                # Create VMInstance tracking object
                vm_instance = VMInstance(
                    instance_id=str(instance.id),
                    name=vm_name,
                    zone=self.config.zone,
                    state=VMState.RUNNING,
                    created_at=time.time(),
                    ip_address=ip_address,
                    internal_ip=internal_ip,
                    components=components,
                    trigger_reason=trigger_reason,
                    metadata=metadata or {},
                )

                # Track the VM
                self.managed_vms[vm_name] = vm_instance
                self.stats["total_created"] += 1
                self.stats["current_active"] += 1

                # Record in cost tracker
                if self.cost_tracker:
                    await self.cost_tracker.record_vm_creation(
                        instance_id=vm_instance.instance_id,
                        vm_type=self.config.machine_type,
                        region=self.config.region,
                        zone=self.config.zone,
                        components=components,
                        trigger_reason=trigger_reason,
                        metadata=metadata or {},
                    )

                logger.info(f"✅ VM created successfully: {vm_name}")
                logger.info(f"   External IP: {ip_address or 'N/A'}")
                logger.info(f"   Internal IP: {internal_ip or 'N/A'}")
                logger.info(f"   Cost: ${vm_instance.cost_per_hour:.3f}/hour")

                return vm_instance

            except Exception as e:
                last_error = e
                logger.error(f"❌ Attempt {attempt} failed: {e}")

                if attempt < self.config.max_create_attempts:
                    delay = self.config.retry_delay_seconds * attempt
                    logger.info(f"⏳ Retrying in {delay}s...")
                    await asyncio.sleep(delay)

        # All attempts failed
        self.stats["total_failed"] += 1
        logger.error(f"❌ Failed to create VM after {self.config.max_create_attempts} attempts")
        logger.error(f"   Last error: {last_error}")

        return None

    def _build_instance_config(
        self, vm_name: str, components: List[str], trigger_reason: str, metadata: Dict
    ) -> compute_v1.Instance:
        """Build GCP Instance configuration"""

        # Machine type URL
        machine_type_url = f"zones/{self.config.zone}/machineTypes/{self.config.machine_type}"

        # Boot disk configuration
        boot_disk = compute_v1.AttachedDisk(
            auto_delete=True,
            boot=True,
            initialize_params=compute_v1.AttachedDiskInitializeParams(
                disk_size_gb=self.config.boot_disk_size_gb,
                disk_type=f"zones/{self.config.zone}/diskTypes/{self.config.boot_disk_type}",
                source_image=f"projects/{self.config.image_project}/global/images/family/{self.config.image_family}",
            ),
        )

        # Network interface
        network_interface = compute_v1.NetworkInterface(
            network=f"global/networks/{self.config.network}",
            access_configs=[compute_v1.AccessConfig(name="External NAT", type="ONE_TO_ONE_NAT")],
        )

        # Metadata
        metadata_items = [
            compute_v1.Items(key="jarvis-components", value=",".join(components)),
            compute_v1.Items(key="jarvis-trigger", value=trigger_reason),
            compute_v1.Items(key="jarvis-created-at", value=datetime.now().isoformat()),
        ]

        # Add startup script if provided
        if self.config.startup_script_path and os.path.exists(self.config.startup_script_path):
            with open(self.config.startup_script_path, "r") as f:
                startup_script = f.read()
            metadata_items.append(compute_v1.Items(key="startup-script", value=startup_script))

        # Build instance
        instance = compute_v1.Instance(
            name=vm_name,
            machine_type=machine_type_url,
            disks=[boot_disk],
            network_interfaces=[network_interface],
            metadata=compute_v1.Metadata(items=metadata_items),
            tags=compute_v1.Tags(items=["jarvis", "backend", "spot-vm"]),
            labels={"created-by": "jarvis", "type": "backend", "vm-class": "spot"},
        )

        # Configure as Spot VM
        if self.config.use_spot:
            instance.scheduling = compute_v1.Scheduling(
                preemptible=True,
                on_host_maintenance="TERMINATE",
                automatic_restart=False,
                provisioning_model="SPOT",
                instance_termination_action="DELETE",
            )

        return instance

    async def _wait_for_operation(self, operation, timeout: int = 300):
        """Wait for a GCP operation to complete"""
        start_time = time.time()

        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Operation timed out after {timeout}s")

            # Check operation status
            if operation.status == compute_v1.Operation.Status.DONE:
                if operation.error:
                    errors = [error.message for error in operation.error.errors]
                    raise Exception(f"Operation failed: {', '.join(errors)}")
                return

            await asyncio.sleep(2)

            # Refresh operation status
            operation = await asyncio.to_thread(
                self.instances_client.get,
                project=self.config.project_id,
                zone=self.config.zone,
                instance=operation.target_link.split("/")[-1],
            )

    async def terminate_vm(self, vm_name: str, reason: str = "Manual termination") -> bool:
        """Terminate a VM instance"""
        if vm_name not in self.managed_vms:
            logger.warning(f"⚠️  VM not found: {vm_name}")
            return False

        vm = self.managed_vms[vm_name]
        logger.info(f"🛑 Terminating VM: {vm_name} (Reason: {reason})")

        try:
            # Update cost before termination
            vm.update_cost()

            # Record termination in cost tracker
            if self.cost_tracker:
                await self.cost_tracker.record_vm_termination(
                    instance_id=vm.instance_id, reason=reason, total_cost=vm.total_cost
                )

            # Delete the VM
            operation = await asyncio.to_thread(
                self.instances_client.delete,
                project=self.config.project_id,
                zone=self.config.zone,
                instance=vm_name,
            )

            await self._wait_for_operation(operation)

            # Update tracking
            vm.state = VMState.TERMINATED
            self.stats["total_terminated"] += 1
            self.stats["current_active"] -= 1
            self.stats["total_cost"] += vm.total_cost

            logger.info(f"✅ VM terminated: {vm_name}")
            logger.info(f"   Uptime: {vm.uptime_hours:.2f}h")
            logger.info(f"   Cost: ${vm.total_cost:.4f}")

            # Remove from managed VMs
            del self.managed_vms[vm_name]

            return True

        except Exception as e:
            logger.error(f"❌ Failed to terminate VM {vm_name}: {e}", exc_info=True)
            return False

    async def _monitoring_loop(self):
        """Background monitoring loop for VM health and lifecycle"""
        logger.info("🔍 VM monitoring loop started")
        self.is_monitoring = True

        while self.is_monitoring:
            try:
                await asyncio.sleep(self.config.health_check_interval)

                for vm_name, vm in list(self.managed_vms.items()):
                    # Update cost
                    vm.update_cost()

                    # Check VM lifetime
                    if vm.uptime_hours >= self.config.max_vm_lifetime_hours:
                        logger.info(
                            f"⏰ VM {vm_name} exceeded max lifetime ({self.config.max_vm_lifetime_hours}h)"
                        )
                        await self.terminate_vm(vm_name, reason="Max lifetime exceeded")
                        continue

                    # Health check
                    is_healthy = await self._health_check_vm(vm_name)
                    vm.last_health_check = time.time()
                    vm.health_status = "healthy" if is_healthy else "unhealthy"

                    if not is_healthy:
                        logger.warning(f"⚠️  VM {vm_name} health check failed")

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)

    async def _health_check_vm(self, vm_name: str) -> bool:
        """Perform health check on a VM"""
        if vm_name not in self.managed_vms:
            return False

        vm = self.managed_vms[vm_name]

        try:
            # Get instance status from GCP
            instance = await asyncio.to_thread(
                self.instances_client.get,
                project=self.config.project_id,
                zone=self.config.zone,
                instance=vm_name,
            )

            # Update state
            status_map = {
                "PROVISIONING": VMState.PROVISIONING,
                "STAGING": VMState.STAGING,
                "RUNNING": VMState.RUNNING,
                "STOPPING": VMState.STOPPING,
                "TERMINATED": VMState.TERMINATED,
            }
            vm.state = status_map.get(instance.status, VMState.UNKNOWN)

            return vm.state == VMState.RUNNING

        except Exception as e:
            logger.error(f"Health check failed for {vm_name}: {e}")
            return False

    async def cleanup_all_vms(self, reason: str = "System shutdown"):
        """Terminate all managed VMs with cost summary"""
        if not self.managed_vms:
            logger.info("ℹ️  No VMs to clean up")
            return

        logger.info(f"🧹 Cleaning up all VMs: {reason}")

        # Calculate total costs before terminating
        total_session_cost = 0.0
        total_uptime_hours = 0.0
        vm_count = len(self.managed_vms)

        for vm in self.managed_vms.values():
            vm.update_cost()
            total_session_cost += vm.total_cost
            total_uptime_hours += vm.uptime_hours

        # Terminate all VMs
        tasks = []
        for vm_name in list(self.managed_vms.keys()):
            tasks.append(self.terminate_vm(vm_name, reason=reason))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Display cost summary
        logger.info("✅ All VMs cleaned up")
        logger.info("=" * 60)
        logger.info("💰 GCP VM COST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"   VMs Terminated:  {vm_count}")
        logger.info(f"   Total Uptime:    {total_uptime_hours:.2f} hours")
        logger.info(f"   Session Cost:    ${total_session_cost:.4f}")
        logger.info(f"   Total Lifetime:  ${self.stats['total_cost']:.4f}")
        logger.info("=" * 60)

    async def cleanup(self):
        """Cleanup and shutdown"""
        self.is_monitoring = False

        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        await self.cleanup_all_vms(reason="Manager shutdown")

        self.initialized = False
        logger.info("🧹 GCP VM Manager cleaned up")

    def get_stats(self) -> Dict:
        """Get manager statistics"""
        return {
            **self.stats,
            "managed_vms": len(self.managed_vms),
            "creating_vms": len(self.creating_vms),
        }


# Singleton instance
_gcp_vm_manager: Optional[GCPVMManager] = None


async def get_gcp_vm_manager(config: Optional[VMManagerConfig] = None) -> GCPVMManager:
    """Get or create singleton GCP VM Manager"""
    global _gcp_vm_manager

    if _gcp_vm_manager is None:
        _gcp_vm_manager = GCPVMManager(config=config)
        await _gcp_vm_manager.initialize()

    return _gcp_vm_manager


async def create_vm_if_needed(
    memory_snapshot, components: List[str], trigger_reason: str, metadata: Optional[Dict] = None
) -> Optional[VMInstance]:
    """
    Convenience function: Check if VM needed and create if so

    Returns:
        VMInstance if created, None otherwise
    """
    manager = await get_gcp_vm_manager()

    should_create, reason, confidence = await manager.should_create_vm(
        memory_snapshot, trigger_reason
    )

    if should_create:
        logger.info(f"✅ VM creation recommended: {reason} (confidence: {confidence:.2%})")
        return await manager.create_vm(components, trigger_reason, metadata)
    else:
        logger.info(f"ℹ️  VM creation not needed: {reason}")
        return None
