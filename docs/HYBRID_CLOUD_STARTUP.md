# Hybrid Cloud Startup Flow

## Overview

JARVIS uses a **hybrid cloud intelligence architecture** to prevent memory exhaustion on local macOS (16GB RAM) by automatically offloading heavy components to GCP Spot VMs (32GB RAM) when memory pressure is detected.

This ensures JARVIS can run in **full mode** with all features, while preventing local memory thrashing and OOM conditions.

---

## Architecture

### Local macOS (16GB RAM)
- **CRITICAL priority components** (authentication, screen lock detection)
- Minimal overhead, fast startup
- Handles core security and UI interactions

### GCP Spot VM (32GB RAM)
- **MEDIUM/LOW/DEFERRED priority components** (ML models, heavy processing)
- Auto-created when RAM ≥ 80%
- Cost-optimized: $0.029/hour with daily budget tracking
- Instance type: `e2-highmem-4` (4 vCPU, 32GB RAM)

---

## Startup Flow

### 1. Component Warmup Initialization

```
[START] Component warmup begins
   ↓
[CHECK] Current memory usage: X%
```

### 2. Memory Pressure Detection

**If RAM < 80%:**
```
[NORMAL] Load all registered components locally
   ↓
[COMPLETE] Warmup finished
```

**If RAM ≥ 80%:**
```
[ALERT] ⚠️  High memory pressure detected!
   ↓
[ACTIVATE] Hybrid cloud intelligence
```

### 3. Hybrid Cloud Activation (RAM ≥ 80%)

```
[ANALYZE] Get detailed memory snapshot
   │
   ├─ macOS: pressure level (normal/warn/critical)
   ├─ macOS: page outs (swapping activity)
   ├─ Linux: PSI metrics (process stalls)
   └─ Platform: available memory, reclaimable cache
   ↓
[SPLIT] Separate components by priority
   │
   ├─ CRITICAL components → Load locally
   └─ MEDIUM/LOW/DEFERRED → Offload candidates
   ↓
[DECIDE] Consult intelligent optimizer
   │
   ├─ Check daily budget remaining
   ├─ Check VM creation quotas
   ├─ Analyze workload patterns
   ├─ Calculate pressure score (0-100)
   └─ Predict if VM will be useful
   ↓
```

**If optimizer recommends VM:**
```
[CREATE] Spin up GCP Spot VM
   │
   ├─ Instance: e2-highmem-4 (32GB RAM)
   ├─ Cost: $0.029/hour
   ├─ Zone: us-central1-a
   └─ Timeout: 300s for creation
   ↓
[SUCCESS] VM created at IP: X.X.X.X
   │
   ├─ Record in cost tracker
   ├─ Track uptime and costs
   └─ Enable health monitoring
   ↓
[LOAD] Critical components locally
[OFFLOAD] Heavy components to VM (future: via API)
   ↓
[COMPLETE] Hybrid mode active
```

**If optimizer denies VM:**
```
[DENY] VM creation not recommended
   │
   ├─ Reason: Budget exhausted / quota limit / workload too short
   └─ Fallback to critical-only loading
   ↓
[FALLBACK] Load only CRITICAL components locally
   ↓
[COMPLETE] Minimal mode (graceful degradation)
```

---

## Component Priority Levels

| Priority | When Loaded | Examples |
|----------|-------------|----------|
| **CRITICAL** | Always local | Screen lock detection, voice authentication |
| **HIGH** | Local if RAM < 80% | Context handlers, NLP processors |
| **MEDIUM** | Offload if RAM ≥ 80% | Vision systems, learning databases |
| **LOW** | Offload if RAM ≥ 80% | Analytics, telemetry |
| **DEFERRED** | Never auto-loaded | Heavy ML models, experimental features |

---

## Intelligent Decision-Making

The `IntelligentGCPOptimizer` uses multi-factor analysis to prevent unnecessary VM creation:

### Pressure Scoring (0-100)

**Factors:**
- Memory pressure score (35% weight)
- Swap activity score (25% weight)
- Trend analysis (15% weight)
- Predicted pressure in 60s (15% weight)
- Time of day patterns (5% weight)
- Historical stability (5% weight)

**Thresholds:**
- 60-79: Warning (consider GCP)
- 80-94: Critical (recommend GCP)
- 95-100: Emergency (urgent GCP)

### Cost Protection

- Daily budget limit: $1.00 (configurable)
- Max VM creations per day: 10
- Prevents VM churn: 5-minute cooldown after destruction
- Tracks actual vs estimated costs
- Learns from historical VM sessions

### Workload Detection

- **Coding spike**: Short-term RAM increase (no VM needed)
- **ML training**: Sustained high usage (VM recommended)
- **Browser heavy**: Cache pressure (local can handle)
- **Sustained load**: Persistent high RAM (VM recommended)

---

## Integration Points

### 1. Component Warmup System
**File:** `backend/core/component_warmup.py`

```python
async def warmup_all(self) -> Dict[str, Any]:
    # Check memory
    mem_percent = psutil.virtual_memory().percent

    if mem_percent >= 80:
        # Activate hybrid cloud
        memory_snapshot = await memory_monitor.get_memory_pressure()
        vm_instance = await create_vm_if_needed(
            memory_snapshot=memory_snapshot,
            components=offloadable_components,
            trigger_reason="High memory pressure during warmup"
        )
```

### 2. Platform Memory Monitor
**File:** `backend/core/platform_memory_monitor.py`

- macOS: `memory_pressure` command + `vm_stat` page outs
- Linux: PSI (Pressure Stall Information) + `/proc/meminfo`
- Distinguishes cache vs actual pressure

### 3. GCP VM Manager
**File:** `backend/core/gcp_vm_manager.py`

- Creates/monitors/terminates Spot VMs
- Integrates with Google Cloud Compute Engine API
- Health checks every 30 seconds
- Auto-cleanup after 3 hours max lifetime

### 4. Intelligent Optimizer
**File:** `backend/core/intelligent_gcp_optimizer.py`

- Multi-factor pressure analysis
- Budget tracking and enforcement
- Workload pattern detection
- VM creation locking (prevents duplicates)

---

## Example Startup Scenarios

### Scenario 1: Normal Startup (75% RAM)
```
[WARMUP] 💾 Memory check: 75.0% used, 4.0GB available
[WARMUP] 🚀 Starting component warmup (18 components registered)
[WARMUP] Loading 5 CRITICAL priority components...
[WARMUP] ✅ Critical components ready in 2.34s
[WARMUP] Loading 8 HIGH priority components...
[WARMUP] ✅ High priority ready in 4.12s
[WARMUP] 🎉 Warmup complete in 8.45s (18/18 components ready)
```

### Scenario 2: High Memory Pressure (82% RAM)
```
[WARMUP] 💾 Memory check: 82.3% used, 2.8GB available
[WARMUP] ⚠️  High memory pressure (82.3%) detected! Activating hybrid cloud intelligence...
[WARMUP] 📊 Memory pressure analysis: elevated (macOS pressure=warn, 2.8GB available)
[WARMUP] 📦 Component split: 5 critical (local), 13 offloadable (GCP candidate)
[WARMUP] 🔍 GCP Recommended (score: 72.4/100)
[WARMUP]    ⚠️  CRITICAL: Score 72.4/100; Workload: coding; Budget remaining: $0.85
[WARMUP] 🚀 Creating GCP Spot VM...
[WARMUP]    Components: learning_database, yabai_detector, multi_space_window_detector, ...
[WARMUP] 🔨 Attempt 1/3: Creating VM 'jarvis-backend-20250112-004532'
[WARMUP] ✅ VM created successfully: jarvis-backend-20250112-004532
[WARMUP]    External IP: 34.123.45.67
[WARMUP]    Internal IP: 10.128.0.2
[WARMUP]    Cost: $0.029/hour
[WARMUP] 🚀 VM has 32GB RAM (vs local 16GB) for heavy components
[WARMUP] 🎯 Loading 5 CRITICAL components locally
[WARMUP] ✅ Critical components ready in 2.18s
[WARMUP] ☁️  Heavy components can be offloaded to VM at 34.123.45.67
[WARMUP] 🎉 Warmup complete in 45.23s (5/18 components ready locally, 13 offloadable to GCP)
```

### Scenario 3: Budget Exhausted (85% RAM)
```
[WARMUP] 💾 Memory check: 85.1% used, 2.4GB available
[WARMUP] ⚠️  High memory pressure (85.1%) detected! Activating hybrid cloud intelligence...
[WARMUP] 📊 Memory pressure analysis: high (macOS pressure=warn, 2.4GB available)
[WARMUP] 📦 Component split: 5 critical (local), 13 offloadable (GCP candidate)
[WARMUP] ℹ️  VM creation not needed: ❌ Daily budget exhausted ($1.00 / $1.00)
[WARMUP] ⚠️  Could not create GCP VM (budget/quota/optimizer decision)
[WARMUP] 🎯 Falling back: Loading only CRITICAL components locally
[WARMUP] ✅ Critical components ready in 2.05s
[WARMUP] 🎉 Warmup complete in 3.42s (5/18 components ready)
```

---

## Cost Tracking

### Daily Budget Report
```json
{
  "date": "2025-01-12",
  "budget_limit": 1.00,
  "current_spend": 0.15,
  "remaining_budget": 0.85,
  "vm_sessions_today": 2,
  "vm_creation_count": 2,
  "total_decisions": 15
}
```

### VM Session Tracking
```json
{
  "vm_id": "jarvis-backend-20250112-004532",
  "created_at": "2025-01-12T00:45:32Z",
  "trigger_reason": "High memory pressure (82.3%) during warmup",
  "initial_cost_estimate": 0.058,
  "actual_runtime_seconds": 5420,
  "actual_cost": 0.043,
  "components": ["learning_database", "yabai_detector", ...],
  "was_useful": true
}
```

---

## Configuration

### Environment Variables

```bash
# Component warmup mode
WARMUP_MODE=manual              # manual (lazy), hybrid (eager), dynamic (auto-discover)
WARMUP_LEARNING=false           # Enable performance learning

# GCP Configuration
GCP_PROJECT_ID=jarvis-473803
GCP_REGION=us-central1
GCP_ZONE=us-central1-a

# Budget & Limits
GCP_DAILY_BUDGET=1.00           # Max $1/day
GCP_MAX_VMS_PER_DAY=10          # Max 10 VM creations/day
GCP_VM_MAX_LIFETIME_HOURS=3.0   # Auto-cleanup after 3 hours
```

### Thresholds (in code)

```python
# Memory pressure thresholds
MEMORY_PRESSURE_THRESHOLD = 80.0     # % RAM usage to trigger hybrid cloud

# Optimizer thresholds
PRESSURE_SCORE_WARNING = 60.0        # Start considering GCP
PRESSURE_SCORE_CRITICAL = 80.0       # Strong recommendation
PRESSURE_SCORE_EMERGENCY = 95.0      # Urgent creation

# Cost protection
MIN_VM_RUNTIME_SECONDS = 300         # Don't create VM for <5min workloads
VM_WARMDOWN_SECONDS = 600            # Keep VM alive 10min after pressure drops
```

---

## Monitoring & Observability

### Startup Logs
- Memory pressure detection
- Component priority split
- Optimizer decision reasoning
- VM creation status
- Cost tracking

### Metrics Tracked
- Total components registered
- Components loaded locally vs offloaded
- VM creation success/failure rate
- Average startup time (local vs hybrid)
- Daily cost accumulation
- Memory pressure over time

### Health Checks
- VM instance health (every 30s)
- Component initialization status
- Connection to VM (for remote components)
- Budget remaining alerts

---

## Future Enhancements

### Phase 1 (Current)
- ✅ Automatic VM creation on high memory pressure
- ✅ Intelligent optimizer with multi-factor analysis
- ✅ Cost tracking and budget enforcement
- ✅ Component priority-based splitting

### Phase 2 (Planned)
- [ ] Remote component loading via gRPC/REST API
- [ ] Seamless component migration to/from VM
- [ ] Real-time load balancing based on usage
- [ ] Multi-VM orchestration for parallel processing

### Phase 3 (Future)
- [ ] Kubernetes-based auto-scaling
- [ ] Regional failover and high availability
- [ ] Machine learning for predictive VM creation
- [ ] Cost optimization via reserved instances

---

## Troubleshooting

### VM Creation Fails
**Symptom:** "Could not create GCP VM"

**Possible Causes:**
1. Daily budget exhausted → Wait until next day or increase budget
2. GCP quota limits → Request quota increase in GCP Console
3. API not enabled → Enable Compute Engine API
4. Auth issues → Check GCP credentials in environment

### Components Stuck in PENDING
**Symptom:** Components never reach READY state

**Possible Causes:**
1. Timeout too short → Increase component timeout
2. Dependency not met → Check dependency graph
3. Import error → Verify component module exists
4. Memory pressure → VM creation may have failed

### Unexpectedly High Costs
**Symptom:** GCP costs higher than expected

**Possible Causes:**
1. VM not auto-terminated → Check max lifetime setting
2. Multiple VMs created → Check VM creation locking
3. Long-running workload → Adjust workload detection
4. Budget not enforced → Verify budget file exists

---

## Key Files

| File | Purpose |
|------|---------|
| `backend/core/component_warmup.py` | Main warmup system with hybrid cloud integration |
| `backend/core/platform_memory_monitor.py` | Cross-platform memory pressure detection |
| `backend/core/gcp_vm_manager.py` | GCP Spot VM lifecycle management |
| `backend/core/intelligent_gcp_optimizer.py` | Multi-factor VM creation decision engine |
| `backend/api/component_warmup_config.py` | Component registration and discovery |
| `start_system.py` | System startup orchestration |

---

## Summary

JARVIS's hybrid cloud startup flow prevents local memory exhaustion by:

1. **Detecting** memory pressure ≥80% during component warmup
2. **Analyzing** pressure with platform-specific metrics (macOS/Linux)
3. **Deciding** whether to create GCP VM using intelligent optimizer
4. **Creating** cost-optimized Spot VM (32GB, $0.029/hour) if recommended
5. **Splitting** components: CRITICAL local, MEDIUM/LOW/DEFERRED offloadable
6. **Tracking** costs and usage with daily budget enforcement
7. **Monitoring** VM health and auto-cleanup to prevent runaway costs

This ensures **full mode capabilities** while protecting the local macOS from memory thrashing and OOM conditions.
