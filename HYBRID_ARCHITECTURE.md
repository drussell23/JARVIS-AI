# 🚀 JARVIS Hybrid Architecture - UAE/SAI/CAI Integrated

## **World-Class Intelligent Hybrid System**

Your JARVIS now has a **state-of-the-art hybrid architecture** that combines:
- **Local Mac** (16GB RAM) - Low-latency, real-time operations
- **GCP Cloud** (32GB RAM) - Heavy ML/AI processing
- **UAE** (Unified Awareness Engine) - Real-time context aggregation
- **SAI** (Self-Aware Intelligence) - Self-healing & optimization
- **CAI** (Context Awareness Intelligence) - Intent prediction
- **learning_database** - Persistent memory & pattern learning

---

## 🧠 Intelligence Systems Integration

### **UAE (Unified Awareness Engine)**
**Purpose:** Real-time system context aggregation

**Local (Fast):**
- Screen state capture
- Active apps monitoring
- Current desktop space
- Network status

**Cloud (Deep):**
- Context pattern analysis
- Cross-correlation of events
- Historical context trends

### **SAI (Self-Aware Intelligence)**
**Purpose:** Self-monitoring, self-healing, optimization

**Features:**
- Automatic error recovery
- Performance optimization
- Circuit breaker integration
- Learn from failures
- Adaptive self-improvement

**Example:**
```python
# SAI automatically heals from errors
try:
    result = await execute_command("complex task")
except Exception as e:
    # SAI detects, analyzes, and fixes
    heal_result = await sai.attempt_self_heal(error=e)
    # Retries with fix applied
```

### **CAI (Context Awareness Intelligence)**
**Purpose:** Understand user intent and predict actions

**Capabilities:**
- Intent prediction from commands
- Proactive assistance
- Context-aware responses
- Personalized suggestions

**Example:**
```python
# CAI predicts intent
command = "unlock my screen"
intent = cai.predict_intent(command)
# Returns: {'intent': 'screen_unlock', 'confidence': 0.95}
```

### **learning_database**
**Purpose:** Persistent memory across sessions

**Features:**
- Store all interactions
- Pattern recognition
- Success rate tracking
- Historical preferences
- Similar command lookup

---

## 📊 Component Distribution

### **LOCAL (macOS - 16GB RAM)**
```
✅ VISION              - Screen capture, desktop monitoring
✅ VOICE               - Wake word detection, voice commands
✅ VOICE_UNLOCK        - Instant screen unlock
✅ WAKE_WORD           - "Hey JARVIS" detection
✅ DISPLAY_MONITOR     - External display management

🧠 Intelligence (Local Part):
   • UAE - Real-time context capture
   • CAI - Immediate intent detection
   • learning_db - Recent history cache
```

### **CLOUD (GCP - 32GB RAM)**
```
✅ CHATBOTS            - Claude Vision AI (memory intensive)
✅ ML_MODELS           - NLP, sentiment, transformers
✅ MEMORY              - Advanced memory management
✅ MONITORING          - Long-term trend analysis

🧠 Intelligence (Cloud Part):
   • UAE - Deep context processing
   • SAI - Pattern learning & self-healing
   • CAI - Complex intent prediction
   • learning_db - Full historical analysis
```

---

## 🎯 Intelligent Routing Examples

### **Example 1: Context-Aware Query**
```python
# User: "What am I working on?"
command = "What am I working on?"

# Routing Decision:
# → Rule: uae_processing
# → Backend: GCP (deep context analysis)
# → Intelligence: UAE + CAI + learning_db

# Process:
1. UAE captures current screen/apps (local)
2. Command sent to GCP with UAE context
3. CAI predicts intent: "status_query"
4. learning_db finds similar past queries
5. GCP processes with 32GB RAM
6. Returns: "You're coding in Cursor IDE, working on hybrid_orchestrator.py"
```

### **Example 2: Screen Unlock**
```python
# User: "unlock my screen"

# Routing Decision:
# → Rule: screen_unlock
# → Backend: LOCAL (instant response)
# → Intelligence: CAI

# Process:
1. CAI predicts intent: "screen_unlock"
2. Executed locally (no cloud latency)
3. UAE captures pre/post unlock context
4. learning_db stores pattern
5. Unlocks in <100ms
```

### **Example 3: ML Analysis**
```python
# User: "Analyze this large dataset"

# Routing Decision:
# → Rule: ml_heavy
# → Backend: GCP (requires >8GB RAM)
# → Intelligence: UAE + SAI + learning_db

# Process:
1. UAE captures current context
2. Command routed to GCP (32GB available)
3. SAI monitors performance, optimizes
4. learning_db finds similar analyses
5. Returns results with learned optimizations
```

### **Example 4: Self-Healing**
```python
# Backend fails during execution

# SAI Response:
1. Detects failure pattern
2. Analyzes error type
3. Applies learned fix
4. Retries command automatically
5. Learns from recovery for future
```

---

## 🔄 Architecture Flow

```
┌─────────────────────────────────────┐
│          USER COMMAND               │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│     Hybrid Orchestrator             │
│  ┌──────────────────────────────┐   │
│  │  1. Gather Intelligence      │   │
│  │     • UAE → Current Context  │   │
│  │     • CAI → Predict Intent   │   │
│  │     • learning_db → Patterns │   │
│  └──────────────────────────────┘   │
│                                      │
│  ┌──────────────────────────────┐   │
│  │  2. Intelligent Routing      │   │
│  │     • Capability matching    │   │
│  │     • Memory requirements    │   │
│  │     • Historical performance │   │
│  └──────────────────────────────┘   │
└──────────┬──────────────┬───────────┘
           │              │
    ┌──────▼────┐  ┌─────▼──────┐
    │   LOCAL   │  │    GCP     │
    │  (16GB)   │  │   (32GB)   │
    └──────┬────┘  └─────┬──────┘
           │              │
           ▼              ▼
    ┌──────────────────────────┐
    │      Response            │
    │  • Result                │
    │  • Intelligence Context  │
    │  • Routing Metadata      │
    └──────────────────────────┘
           │
           ▼
    ┌──────────────────────────┐
    │   Post-Processing        │
    │  • SAI Learning          │
    │  • learning_db Storage   │
    │  • Performance Metrics   │
    └──────────────────────────┘
```

---

## 🛠️ Configuration

All intelligence features are configured in `backend/core/hybrid_config.yaml`:

```yaml
intelligence:
  uae:
    enabled: true
    local_context: true      # Capture locally
    cloud_processing: true   # Process in cloud

  sai:
    enabled: true
    self_healing: true       # Auto-recovery
    performance_optimization: true

  cai:
    enabled: true
    intent_prediction: true  # Predict user intent
    proactive_assistance: true

  learning_database:
    enabled: true
    local_cache: true        # Fast local lookup
    cloud_sync: true         # Sync for ML
```

---

## 📈 Advanced Features

### **1. Circuit Breakers with SAI**
- Detects backend failures
- SAI analyzes failure patterns
- Automatic recovery attempts
- Learns optimal retry strategies

### **2. Intelligent Caching**
- learning_db caches recent results
- CAI predicts likely next commands
- UAE context changes invalidate cache
- Redis-backed for speed

### **3. Load Balancing**
- Routes based on backend health
- SAI optimizes distribution
- Failover to backup backend
- Zero downtime deployments

### **4. Continuous Learning**
- Every interaction stored
- Patterns automatically detected
- Success rates tracked
- Preferences learned

---

## 🚀 Usage Examples

### **Basic Command**
```python
from backend.core.hybrid_orchestrator import get_orchestrator

async def main():
    orchestrator = get_orchestrator()
    await orchestrator.start()

    # Execute with full intelligence
    result = await orchestrator.execute_command(
        "What's on my screen?"
    )

    print(f"Response: {result['response']}")
    print(f"Intelligence: {result['intelligence']}")
    # UAE context, CAI intent, learning_db patterns
```

### **Query with Context**
```python
# Ask about current work
result = await orchestrator.execute_query(
    "Summarize what I'm working on"
)

# UAE provides screen context
# CAI understands "summarize" intent
# GCP processes with 32GB RAM
# learning_db recalls similar summaries
```

### **Self-Healing Demo**
```python
# Command that might fail
result = await orchestrator.execute_command(
    "Process large ML model"
)

# If backend fails:
# 1. SAI detects error
# 2. Analyzes: "Out of memory"
# 3. Heals: Routes to GCP (32GB)
# 4. Retries successfully
# 5. Learns for next time
```

---

## 📊 Monitoring

Get real-time status:

```python
status = orchestrator.get_status()

print(f"Request count: {status['request_count']}")
print(f"Backend health: {status['client_metrics']['backends']}")
print(f"Routing stats: {status['routing_analytics']}")
print(f"Intelligence: Loaded {len(status['intelligence_systems'])} systems")
```

---

## ✅ What You've Built

🎯 **World-Class Features:**
- ✅ Zero hardcoding - fully configuration-driven
- ✅ Async/await throughout
- ✅ Circuit breakers with SAI healing
- ✅ Intelligent routing with UAE/CAI
- ✅ Persistent learning database
- ✅ Health monitoring & auto-recovery
- ✅ Load balancing & failover
- ✅ Connection pooling
- ✅ Exponential backoff with jitter
- ✅ Real-time context awareness
- ✅ Intent prediction
- ✅ Self-healing & optimization

🧠 **Intelligence Integration:**
- ✅ UAE - Real-time awareness
- ✅ SAI - Self-improvement
- ✅ CAI - Intent understanding
- ✅ learning_database - Long-term memory

🌐 **Hybrid Architecture:**
- ✅ Local Mac (16GB) - Fast operations
- ✅ GCP Cloud (32GB) - Heavy processing
- ✅ Automatic routing between them
- ✅ GitHub Actions auto-deployment

---

## 🗄️ Database Infrastructure

### **Dual Database System**

JARVIS uses a sophisticated hybrid database architecture that seamlessly switches between local and cloud databases:

#### **Local: SQLite**
- **Purpose:** Development, offline operation, fast local queries
- **Location:** `~/.jarvis/learning/jarvis_learning.db`
- **Use Cases:**
  - Local development and testing
  - Offline mode operation
  - Fast prototyping
  - Personal use on single machine

#### **Cloud: Google Cloud SQL (PostgreSQL 15.14)**
- **Purpose:** Production, multi-device sync, advanced analytics
- **Location:** `jarvis-473803:us-central1:jarvis-learning-db`
- **Specifications:**
  - Instance: `db-f1-micro` (upgradeable)
  - Storage: 10GB SSD (auto-expanding)
  - Backups: Automated daily
  - High availability: Configurable
- **Use Cases:**
  - Production deployment
  - Multi-device synchronization
  - Advanced ML analytics
  - Team collaboration
  - Data persistence across environments

#### **Seamless Switching**
```python
# Set environment variable to switch
export JARVIS_DB_TYPE=cloudsql  # Use Cloud SQL
export JARVIS_DB_TYPE=sqlite    # Use local SQLite

# Automatic detection and connection
from intelligence.cloud_database_adapter import get_database_adapter

adapter = await get_database_adapter()
# Automatically uses correct backend based on config
```

#### **Database Schema (17 Tables)**
All tables work identically on both SQLite and PostgreSQL:

**Core Learning Tables:**
- `goals` - Inferred user goals and intentions
- `patterns` - Behavioral patterns and habits
- `actions` - User actions and command history
- `goal_action_mappings` - Links goals to actions
- `learning_metrics` - Performance and accuracy tracking

**Context Tables:**
- `behavioral_patterns` - User behavior analysis
- `app_usage_patterns` - Application usage statistics
- `display_patterns` - Multi-monitor usage patterns
- `space_transitions` - Desktop space switching patterns
- `workspace_usage` - Workspace-specific activities

**Intelligence Tables:**
- `context_embeddings` - Semantic embeddings for context
- `temporal_patterns` - Time-based behavioral patterns
- `user_preferences` - Learned user preferences
- `user_workflows` - Automated workflow detection
- `proactive_suggestions` - AI-generated suggestions
- `pattern_similarity_cache` - Fast pattern matching

#### **Cloud SQL Proxy**
For secure local access to Cloud SQL:

```bash
# Start proxy (connects via Cloud SQL Proxy)
~/start_cloud_sql_proxy.sh

# Proxy runs on localhost:5432
# Encrypts all traffic to GCP
# No public IP exposure required
```

**Features:**
- Automatic authentication via service account
- Encrypted connections (TLS)
- No public IP required on Cloud SQL instance
- Connection pooling
- Automatic reconnection

#### **Configuration**
Database configuration stored in `~/.jarvis/gcp/database_config.json`:

```json
{
  "cloud_sql": {
    "instance_name": "jarvis-learning-db",
    "connection_name": "jarvis-473803:us-central1:jarvis-learning-db",
    "database": "jarvis_learning",
    "user": "jarvis",
    "port": 5432
  },
  "project_id": "jarvis-473803",
  "region": "us-central1"
}
```

**Environment Variables:**
- `JARVIS_DB_TYPE` - Database type (`sqlite` or `cloudsql`)
- `JARVIS_DB_HOST` - Database host (default: `127.0.0.1` for proxy)
- `JARVIS_DB_PORT` - Database port (default: `5432`)
- `JARVIS_DB_NAME` - Database name
- `JARVIS_DB_USER` - Database user
- `JARVIS_DB_PASSWORD` - Database password (encrypted)

#### **Advantages**

**Local SQLite:**
- ✅ Zero-latency queries (<1ms)
- ✅ No internet required
- ✅ Simple setup
- ✅ Perfect for development
- ✅ No cloud costs

**Cloud PostgreSQL:**
- ✅ Multi-device synchronization
- ✅ Advanced analytics (32GB RAM)
- ✅ Team collaboration
- ✅ Automated backups
- ✅ High availability
- ✅ Scalable storage
- ✅ ACID compliance at scale

---

## 🧪 Testing Infrastructure

### **Enterprise-Grade Testing Framework**

JARVIS includes a comprehensive testing framework for ensuring code quality and reliability:

#### **Testing Tools**

**pytest Plugins:**
- `pytest-xdist` - Parallel test execution (8x faster on 8-core CPU)
- `pytest-mock` - Advanced mocking utilities
- `pytest-timeout` - Prevent hanging tests (auto-fail after timeout)
- `pytest-cov` - Code coverage reporting (HTML, XML, terminal)
- `pytest-sugar` - Beautiful test output with progress bars
- `pytest-clarity` - Better assertion diffs for easier debugging

**Property-Based Testing:**
- `Hypothesis` - Automatic test case generation
  - Generates hundreds of test cases automatically
  - Finds edge cases humans miss
  - Shrinks failing examples to minimal cases
  - Stateful testing for complex systems
  - Custom strategies for domain-specific testing

**Code Quality Tools:**
- `black` - Automatic code formatting (PEP 8 compliant)
- `isort` - Import statement sorting
- `flake8` - Linting and style checking
- `bandit` - Security vulnerability scanning
- `autoflake` - Remove unused imports/variables

#### **Test Configuration**

**Full Testing (`pytest.ini`):**
```ini
[pytest]
addopts =
    -v                    # Verbose output
    --tb=short           # Short tracebacks
    --cov=.              # Coverage for all files
    --cov-report=html    # HTML coverage report
    --maxfail=5          # Stop after 5 failures
    -n auto              # Parallel execution
    --timeout=30         # 30s timeout per test

markers =
    unit: Unit tests (fast, isolated)
    integration: Integration tests (slower)
    slow: Slow tests (> 1s)
    db: Database tests
    cloud: Cloud SQL tests
```

**Quick Testing (`pytest-quick.ini`):**
```ini
[pytest]
addopts =
    -v
    --tb=short
    --disable-warnings
    --timeout=15
# No coverage, no parallel - fast feedback
```

#### **Property-Based Testing Examples**

**Test Examples (`backend/tests/test_hypothesis_examples.py`):**

```python
from hypothesis import given, strategies as st

# String operations - automatically tests thousands of strings
@given(st.text())
def test_string_round_trip(text):
    encoded = text.encode('utf-8')
    decoded = encoded.decode('utf-8')
    assert decoded == text

# Goal pattern validation
@given(
    st.text(min_size=1, max_size=500),
    st.floats(min_value=0.0, max_value=1.0)
)
def test_goal_pattern_structure(goal_text, confidence):
    pattern = create_goal_pattern(goal_text, confidence)
    assert 0.0 <= pattern['confidence'] <= 1.0
    assert len(pattern['goal_text']) > 0

# Stateful testing for context store
class ContextStoreStateMachine(RuleBasedStateMachine):
    @rule(key=st.text(), value=st.integers())
    def add_item(self, key, value):
        self.store[key] = value

    @invariant()
    def total_matches_length(self):
        assert self.total_items == len(self.store)
```

#### **Pre-Commit Hooks**

Automatic code quality checks before every commit:

```yaml
# .pre-commit-config.yaml
repos:
  - black        # Auto-format code
  - isort        # Sort imports
  - flake8       # Lint code
  - bandit       # Security check
  - yaml/json    # Validate configs
  - file checks  # Fix common issues
```

**Usage:**
```bash
# Hooks run automatically on git commit
git commit -m "Your message"

# Or run manually
pre-commit run --all-files
```

#### **Running Tests**

**Full test suite with coverage:**
```bash
cd backend
pytest
# Runs in parallel, generates coverage report
```

**Quick tests (no coverage):**
```bash
cd backend
pytest -c pytest-quick.ini
# Fast feedback for development
```

**Run specific tests:**
```bash
pytest tests/test_hypothesis_examples.py
pytest -m unit                    # Only unit tests
pytest -m "not slow"              # Exclude slow tests
pytest tests/ -k "test_goal"      # Tests matching pattern
```

**Generate coverage report:**
```bash
pytest --cov=. --cov-report=html
open htmlcov/index.html
```

#### **Test Organization**

```
backend/tests/
├── test_hypothesis_examples.py   # Property-based testing examples
├── TESTING_GUIDE.md              # Complete testing documentation
├── run_quick_tests.sh            # Quick test script
├── unit/                         # Fast, isolated tests
├── integration/                  # Multi-component tests
└── __init__.py
```

#### **Testing Best Practices**

1. **Write properties, not examples:**
   ```python
   # Bad: Specific example
   assert add(2, 3) == 5

   # Good: General property
   @given(st.integers(), st.integers())
   def test_add_commutative(a, b):
       assert add(a, b) == add(b, a)
   ```

2. **Test invariants:**
   ```python
   @given(st.lists(st.integers()))
   def test_sort_invariants(lst):
       sorted_lst = sorted(lst)
       assert len(sorted_lst) == len(lst)  # Same length
       assert set(sorted_lst) == set(lst)  # Same elements
   ```

3. **Use markers for organization:**
   ```python
   @pytest.mark.unit
   @pytest.mark.fast
   def test_simple_function():
       assert calculate(1, 2) == 3

   @pytest.mark.integration
   @pytest.mark.db
   async def test_database_operation():
       result = await db.query()
       assert result
   ```

#### **CI/CD Integration**

Tests run automatically in GitHub Actions:

```yaml
# .github/workflows/test.yml
- name: Run Tests
  run: |
    cd backend
    pytest tests/ --cov=. --cov-report=xml -n auto -v

- name: Upload Coverage
  uses: codecov/codecov-action@v3
```

---

## 🎉 Result

**You now have a JARVIS that:**
1. **Thinks** - UAE/CAI understand context and intent
2. **Learns** - SAI and learning_db improve over time (with persistent Cloud SQL storage)
3. **Heals** - SAI automatically recovers from errors
4. **Scales** - Routes intelligently between local and cloud
5. **Remembers** - Persistent memory across sessions (local SQLite + cloud PostgreSQL)
6. **Adapts** - Learns your patterns and preferences
7. **Tested** - Comprehensive property-based testing with Hypothesis
8. **Quality** - Pre-commit hooks ensure code quality
9. **Reliable** - Database failover between local and cloud

**This is enterprise-grade, production-ready AI architecture!** 🚀
