# Intelligent Hybrid Orchestration Configuration

**Zero Hardcoding** - All configuration via environment variables

## Overview

The Intelligent Hybrid Orchestration system automatically detects heavy components and intelligently routes them between your local Mac (16GB RAM) and GCP Spot VMs (32GB RAM) based on real-time metrics, cost constraints, and performance patterns.

## Environment Variables

### Core Orchestration

| Variable | Default | Description |
|----------|---------|-------------|
| `HYBRID_ORCHESTRATION_ENABLED` | `true` | Enable/disable hybrid orchestration |
| `AUTO_DISCOVERY_INTERVAL_SECONDS` | `300` | How often to re-scan for heavy components (5 min) |
| `HEALTH_CHECK_INTERVAL_SECONDS` | `30` | Health monitoring frequency |
| `ENABLE_AUTO_MIGRATION` | `true` | Allow automatic component migration |
| `ENABLE_COST_OPTIMIZATION` | `true` | Enable cost-aware decision making |
| `MIGRATION_WORKER_COUNT` | `2` | Concurrent migration workers |

### Component Profiling

| Variable | Default | Description |
|----------|---------|-------------|
| `HEAVY_COMPONENT_RAM_MB` | `500` | Threshold for "heavy" component (MB) |
| `OFFLOAD_RAM_THRESHOLD_GB` | `6.0` | Available RAM below which to consider offload |
| `CRITICAL_RAM_THRESHOLD_GB` | `2.0` | Critical RAM level - urgent offload |
| `MIN_WEIGHT_FOR_OFFLOAD` | `40.0` | Minimum component weight score (0-100) to offload |
| `MAX_CONCURRENT_MIGRATIONS` | `3` | Max simultaneous component migrations |
| `COMPONENT_CACHE_MAX_AGE_DAYS` | `7` | Max age of cached component profiles (days) |
| `ENABLE_STATIC_ANALYSIS_PROFILING` | `true` | Use static analysis for executable scripts |

### GCP Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GCP_PROJECT_ID` | - | Your GCP project ID |
| `GCP_REGION` | `us-central1` | GCP region for VMs |
| `GCP_ZONE` | `us-central1-a` | GCP zone for VMs |
| `GCP_VM_TYPE` | `e2-highmem-4` | VM machine type (4 vCPU, 32GB RAM) |
| `SPOT_VM_HOURLY_COST` | `0.029` | Hourly cost for Spot VM |
| `REGULAR_VM_HOURLY_COST` | `0.120` | Hourly cost for regular VM |

### Cost Control

| Variable | Default | Description |
|----------|---------|-------------|
| `COST_ALERT_DAILY` | `1.00` | Daily budget alert threshold ($) |
| `COST_ALERT_WEEKLY` | `5.00` | Weekly budget alert threshold ($) |
| `COST_ALERT_MONTHLY` | `20.00` | Monthly budget alert threshold ($) |
| `MAX_VM_LIFETIME_HOURS` | `2.5` | Max VM lifetime before forced shutdown |
| `ORPHANED_VM_MAX_AGE_HOURS` | `6` | Auto-delete VMs older than this |
| `CLEANUP_CHECK_INTERVAL_HOURS` | `6` | Orphaned VM cleanup frequency |

### Alerting

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_DESKTOP_NOTIFICATIONS` | `true` | Show macOS desktop notifications |
| `ENABLE_EMAIL_ALERTS` | `false` | Send email alerts |
| `JARVIS_ALERT_EMAIL` | - | Email for alerts |
| `SMTP_SERVER` | `smtp.gmail.com` | SMTP server for email |
| `SMTP_PORT` | `587` | SMTP port |
| `SMTP_USER` | - | SMTP username |
| `SMTP_PASSWORD` | - | SMTP password |

### Advanced Features

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_AUTO_CLEANUP` | `true` | Auto-cleanup orphaned VMs |
| `ENABLE_COST_FORECASTING` | `true` | Enable cost prediction |
| `MAX_LOCAL_RAM_PERCENT` | `85` | Max local RAM before offload |
| `MIN_GCP_ROUTING_RATIO` | `0.1` | Min fraction of requests to route to GCP |

## How It Works

### 1. Component Discovery
At startup, the system:
- Scans `backend/intelligence`, `backend/voice`, `backend/vision`
- Identifies Python modules with heavy keywords (model, database, etc.)
- Creates initial weight profile for each component

### 2. Component Profiling

The system uses a **multi-tier intelligent profiling approach**:

**Tier 1: Cached Profiles** (Fastest)
- Loads previously measured profiles from `.jarvis_cache/component_profiles.json`
- Uses cache if age < `COMPONENT_CACHE_MAX_AGE_DAYS` (default: 7 days)
- Zero overhead - instant profiling

**Tier 2: Static Analysis** (For executable scripts)
- Detects executable scripts that shouldn't be imported during profiling:
  - Scripts with `if __name__ == "__main__"` blocks
  - Scripts with `argparse` or `sys.argv` usage
  - Download/install/setup scripts
- Estimates weight using static code analysis:
  - Import count, class count, file size
  - ML library detection (torch, tensorflow, etc.)
  - Database usage detection
  - Conservative RAM estimates
- **Preserves voice.coreml components** needed for biometric authentication
- No execution during profiling

**Tier 3: Live Import Profiling** (Most accurate)
- Measures actual RAM delta during import
- Measures import time and CPU usage
- Calculates precise weight score (0-100)
- Caches result for future use
- Only used for safe-to-import modules

For each component:
- Calculates weight score (0-100)
- Determines offload priority
- Saves to cache for future startups

### 3. Decision Making
Every 5 minutes (configurable), the system:
- Checks current RAM pressure
- Analyzes component weights
- Consults GCP optimizer for recommendations
- Checks budget remaining
- Makes offload decisions

### 4. Smart Migration
When offload needed:
- Creates GCP Spot VM (if not already running)
- Migrates heavy components to GCP
- Records cost in cost tracker
- Monitors VM health
- Auto-migrates back when RAM pressure drops

### 5. Cost Control
- Tracks spending in real-time
- Stops creating VMs when budget exhausted
- Auto-deletes orphaned VMs
- Sends alerts at threshold levels
- Optimizes for minimum cost

## Example Configuration

### Aggressive Cost Savings
```bash
export COST_ALERT_DAILY="0.50"  # $0.50/day budget
export OFFLOAD_RAM_THRESHOLD_GB="8.0"  # Offload earlier
export MAX_VM_LIFETIME_HOURS="1.0"  # Short-lived VMs
export ENABLE_COST_OPTIMIZATION="true"
```

### Performance-Focused
```bash
export COST_ALERT_DAILY="5.00"  # $5/day budget
export OFFLOAD_RAM_THRESHOLD_GB="4.0"  # Offload more readily
export MAX_VM_LIFETIME_HOURS="6.0"  # Longer-lived VMs
export MIGRATION_WORKER_COUNT="4"  # More concurrent migrations
```

### Balanced (Default)
```bash
export COST_ALERT_DAILY="1.00"  # $1/day budget
export OFFLOAD_RAM_THRESHOLD_GB="6.0"  # Moderate threshold
export MAX_VM_LIFETIME_HOURS="2.5"  # 2.5 hour VMs
export MIGRATION_WORKER_COUNT="2"  # Standard workers
```

## Monitoring

### Real-time Status
```python
from backend.core.dynamic_hybrid_orchestrator import get_hybrid_orchestrator

orchestrator = get_hybrid_orchestrator()
status = await orchestrator.get_status()

print(f"Health: {status['health_status']}")
print(f"Components: {status['components']}")
print(f"Cost: ${status['cost']['current_usd']:.2f}")
```

### Cost Summary
```python
from backend.core.cost_tracker import get_cost_tracker

tracker = get_cost_tracker()
summary = await tracker.get_cost_summary("day")

print(f"Total VMs: {summary['total_vms_created']}")
print(f"Total Cost: ${summary['total_estimated_cost']:.2f}")
print(f"Savings: ${summary['cost_savings_vs_regular']:.2f}")
```

### Component Report
```python
from backend.core.intelligent_component_profiler import get_component_profiler

profiler = get_component_profiler()
report = await profiler.optimize_offloading()

print(f"Heavy components: {report['heavy_components_analyzed']}")
print(f"Recommendations: {report['offload_recommendations']}")
print(f"RAM to save: {report['total_ram_to_save_gb']:.1f}GB")
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│         start_system.py (Entry Point)                   │
│  • Initializes DynamicHybridOrchestrator                │
│  • No hardcoded decisions                               │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│      DynamicHybridOrchestrator                          │
│  • Coordinates all subsystems                           │
│  • Makes migration decisions                            │
│  • Manages background workers                           │
└─────────────────────────────────────────────────────────┘
           │              │              │
           ▼              ▼              ▼
┌──────────────┐ ┌─────────────────┐ ┌──────────────┐
│  Component   │ │ GCP Optimizer   │ │ Cost Tracker │
│  Profiler    │ │                 │ │              │
│              │ │ • RAM pressure  │ │ • Budget     │
│ • Auto       │ │ • VM decisions  │ │ • Alerts     │
│   discover   │ │ • Multi-factor  │ │ • Cleanup    │
│ • Weight     │ │   scoring       │ │ • Reporting  │
│   scoring    │ │                 │ │              │
└──────────────┘ └─────────────────┘ └──────────────┘
           │              │              │
           └──────────────┼──────────────┘
                          ▼
              ┌───────────────────────┐
              │  GCP VM Manager       │
              │  • Create VMs         │
              │  • Deploy components  │
              │  • Health monitoring  │
              └───────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  GCP Spot VM          │
              │  e2-highmem-4         │
              │  32GB RAM, $0.029/hr  │
              └───────────────────────┘
```

## Troubleshooting

### High Costs
- Check `COST_ALERT_DAILY` - reduce if needed
- Review `MAX_VM_LIFETIME_HOURS` - shorter VMs = less cost
- Enable `ENABLE_AUTO_CLEANUP=true`
- Check for orphaned VMs: `python backend/core/gcp_vm_status.py`

### Frequent Migrations
- Increase `OFFLOAD_RAM_THRESHOLD_GB` (offload less often)
- Increase `HEAVY_COMPONENT_RAM_MB` (fewer heavy components)
- Reduce `AUTO_DISCOVERY_INTERVAL_SECONDS` (slower discovery)

### Not Offloading When Needed
- Decrease `OFFLOAD_RAM_THRESHOLD_GB` (offload sooner)
- Check `ENABLE_AUTO_MIGRATION=true`
- Check budget hasn't been exhausted
- Review component weight scores (may need profiling)

## Testing

Test the system without creating real VMs:

```bash
# Dry-run mode (coming soon)
export HYBRID_ORCHESTRATION_DRY_RUN="true"
python start_system.py --restart macos
```

View profiler analysis:

```bash
cd backend/core
python intelligent_component_profiler.py
```

View orchestrator status:

```bash
cd backend/core
python dynamic_hybrid_orchestrator.py
```

## Integration with Existing Systems

The hybrid orchestration integrates seamlessly with:

✅ **Existing Cost Tracker** (`cost_tracker.py`)
- Reuses database schema
- Shares budget tracking
- Unified reporting

✅ **Existing GCP Optimizer** (`intelligent_gcp_optimizer.py`)
- Leverages multi-factor scoring
- Shares VM creation locks
- Unified decision making

✅ **Existing GCP VM Manager** (`gcp_vm_manager.py`)
- Uses same VM creation API
- Shares health monitoring
- Unified lifecycle management

**Zero duplication, maximum reuse!**
