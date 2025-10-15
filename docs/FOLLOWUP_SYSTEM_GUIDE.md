# Context-Aware Follow-Up Handling System

**Version:** 1.0
**Author:** Derek J. Russell
**Date:** October 8th, 2025

---

## Overview

The Context-Aware Follow-Up Handling System transforms JARVIS from a one-shot command executor into an intelligent conversational agent that can maintain context across multi-turn dialogues. This system enables natural interactions like:

```
JARVIS: "I can see your Terminal. Would you like me to describe what's displayed?"
You: "yes"
JARVIS: "You're running a Python script. The output shows 'ModuleNotFoundError: No module named requests'. Would you like me to help fix this?"
```

## Key Features

### 🎯 Dynamic Intent Classification
- **ML-Ready Architecture**: Supports both lexical and semantic classifiers
- **Pluggable Classifiers**: Add/remove classifiers at runtime
- **Ensemble Strategies**: Combine multiple signals with weighted voting
- **No Hardcoding**: All patterns loaded from configuration files

### 💾 Flexible Context Storage
- **Multiple Backends**: In-memory, Redis, or custom implementations
- **LRU Eviction**: Automatic memory management
- **TTL & Decay**: Time-based relevance scoring
- **Query DSL**: Fluent query interface for context retrieval

### 🔀 Adaptive Routing
- **Plugin Architecture**: Register handlers as plugins
- **Middleware Support**: Pre/post-processing hooks
- **Context-Aware**: Routes based on intent + active context
- **Fallback Handling**: Graceful degradation

### 🔍 Semantic Matching
- **Embedding Support**: OpenAI, SentenceTransformers, or custom
- **Hybrid Scoring**: Combines semantic + keyword + recency
- **Caching**: Built-in embedding cache for performance

### 📊 Observability
- **Structured Events**: All operations emit telemetry events
- **Multiple Sinks**: Logging, Prometheus, OpenTelemetry
- **Latency Tracking**: Context manager for operation timing
- **Error Tracking**: Automatic error event emission

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Input                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Intent Classification                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Lexical    │  │   Semantic   │  │   Context    │          │
│  │  Classifier  │  │  Classifier  │  │    Aware     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                          │                                       │
│                          ▼                                       │
│                  ┌───────────────┐                              │
│                  │   Ensemble    │                              │
│                  │   Strategy    │                              │
│                  └───────────────┘                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Context Retrieval                            │
│  ┌──────────────────────────────────────────────────┐          │
│  │           Context Store (Memory/Redis)           │          │
│  │  • Query by category, tags, relevance            │          │
│  │  • Semantic matching                             │          │
│  │  • TTL & expiry management                       │          │
│  └──────────────────────────────────────────────────┘          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Adaptive Router                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Middleware  │  │     Route    │  │   Handler    │          │
│  │   Pipeline   │→ │   Matching   │→ │  Execution   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Handler Plugins                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Vision     │  │   Command    │  │    Custom    │          │
│  │  Follow-Up   │  │   Executor   │  │   Handlers   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                        Response
```

---

## Quick Start

### 1. Installation

```bash
# Core dependencies
pip install asyncio dataclasses

# Optional: Redis support
pip install redis

# Optional: Semantic embeddings
pip install sentence-transformers
# OR
pip install openai

# Optional: Observability
pip install prometheus-client opentelemetry-api
```

### 2. Basic Setup

```python
from backend.examples.bootstrap_followup_system import FollowUpSystem

# Initialize system
system = FollowUpSystem()
await system.start()

# Track a pending question (after JARVIS asks user)
context_id = await system.track_pending_question(
    question_text="Would you like me to describe what's in the Terminal?",
    window_type="terminal",
    window_id="term_1",
    space_id="space_1",
    snapshot_id="snap_123",
    summary="Terminal with Python error",
    ocr_text="ModuleNotFoundError: No module named 'requests'",
)

# Process user response
response = await system.process_user_input("yes")
print(response)

# Clean up
await system.stop()
```

### 3. Run Demo

```bash
python -m backend.examples.bootstrap_followup_system
```

---

## Component Guide

### Intent Classification

#### Lexical Classifier (Pattern-Based)

```python
from backend.core.intent.adaptive_classifier import LexicalClassifier

patterns = {
    "follow_up": ["yes", "no", "tell me more", "show me"],
    "vision": ["terminal", "browser", "code", "screen"],
}

classifier = LexicalClassifier(
    name="lexical",
    patterns=patterns,
    case_sensitive=False,
    word_boundary=True,  # Match whole words only
)

signals = classifier.classify("yes, show me", {})
# Returns: [IntentSignal(label="follow_up", confidence=0.85, ...)]
```

#### Semantic Classifier (Embedding-Based)

```python
from backend.core.intent.adaptive_classifier import SemanticEmbeddingClassifier

# Using OpenAI embeddings
async def embed_fn(text):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key="sk-...")
    resp = await client.embeddings.create(model="text-embedding-3-small", input=text)
    return resp.data[0].embedding

# Pre-compute reference embeddings for intents
intent_embeddings = {
    "follow_up": [
        await embed_fn("yes"),
        await embed_fn("tell me more"),
        await embed_fn("show me"),
    ],
}

classifier = SemanticEmbeddingClassifier(
    name="semantic",
    embedding_fn=embed_fn,
    intent_embeddings=intent_embeddings,
    threshold=0.75,
)

signals = await classifier.classify_async("absolutely, go ahead", {})
```

#### Intent Engine (Ensemble)

```python
from backend.core.intent.adaptive_classifier import AdaptiveIntentEngine, WeightedVotingStrategy

engine = AdaptiveIntentEngine(
    classifiers=[lexical_classifier, semantic_classifier],
    strategy=WeightedVotingStrategy(
        source_weights={"lexical": 1.0, "semantic": 1.5},
        min_confidence=0.6,
    ),
)

result = await engine.classify("yes, please do")
print(f"Intent: {result.primary_intent}, Confidence: {result.confidence}")
```

### Context Storage

#### In-Memory Store

```python
from backend.core.context.memory_store import InMemoryContextStore
from backend.core.models.context_envelope import ContextEnvelope, VisionContextPayload

store = InMemoryContextStore(max_size=1000)

# Create context
envelope = ContextEnvelope(
    metadata=ContextMetadata(category=ContextCategory.VISION),
    payload=VisionContextPayload(
        window_type="terminal",
        window_id="w1",
        space_id="s1",
        snapshot_id="snap1",
        summary="Terminal detected",
    ),
    ttl_seconds=120,
)

# Add to store
context_id = await store.add(envelope)

# Retrieve
retrieved = await store.get(context_id)

# Query
from backend.core.context.store_interface import ContextQuery

query = (
    ContextQuery()
    .with_category("VISION")
    .with_tag("terminal")
    .sort_by_relevance()
    .limit(5)
)
results = await store.query(query)
```

#### Redis Store

```python
from backend.core.context.redis_store import RedisContextStore

store = RedisContextStore(
    redis_url="redis://localhost:6379",
    key_prefix="jarvis:context:",
)

await store.connect()

# Same API as InMemoryContextStore
context_id = await store.add(envelope)

# Optimized Redis queries
results = await store.get_by_category("VISION")
results = await store.get_by_tags("terminal", "error")
top_relevant = await store.get_top_relevant(limit=10)

await store.disconnect()
```

### Routing & Handlers

#### Custom Handler

```python
from backend.core.routing.adaptive_router import BaseRouteHandler, RoutingResult

class MyCustomHandler(BaseRouteHandler):
    def __init__(self):
        super().__init__(name="custom_handler", priority=80)

    async def handle(self, user_input, intent, context, extras):
        if not context:
            return RoutingResult(
                success=False,
                response="No context available",
            )

        # Your logic here
        response_text = f"Handling {intent.primary_intent} for {context.metadata.category.name}"

        return RoutingResult(
            success=True,
            response=response_text,
            metadata={"handler": "custom"},
        )
```

#### Register Handler

```python
from backend.core.routing.adaptive_router import RouteConfig, AdaptiveRouter, RouteMatcher

matcher = RouteMatcher()
router = AdaptiveRouter(matcher=matcher)

# Add route
route = RouteConfig(
    intent_label="follow_up",
    handler=MyCustomHandler(),
    requires_context=True,
    context_categories=("VISION", "COMMAND"),
    min_confidence=0.7,
)

router.add_route(route)

# Route request
result = await router.route(
    user_input="yes",
    intent=intent_result,
    context=context_envelope,
)
```

#### Plugin Architecture

```python
from backend.core.routing.adaptive_router import HandlerPlugin

class MyPlugin(HandlerPlugin):
    @property
    def routes(self):
        return [
            RouteConfig(
                intent_label="custom_intent",
                handler=MyCustomHandler(),
                min_confidence=0.6,
            )
        ]

    async def on_load(self):
        print("Plugin loaded")

    async def on_unload(self):
        print("Plugin unloaded")

# Register
registry = PluginRegistry(router)
await registry.register_plugin("my_plugin", MyPlugin())
```

### Telemetry

```python
from backend.core.telemetry.events import get_telemetry, LatencyTracker

telemetry = get_telemetry()

# Track events
await telemetry.track_context_created(
    context_id="ctx123",
    category="VISION",
    priority="HIGH",
    ttl_seconds=120,
)

await telemetry.track_intent_detected(
    intent_label="follow_up",
    confidence=0.92,
    classifiers=["lexical"],
)

# Track latency
async with LatencyTracker("process_followup", telemetry):
    result = await process_followup(user_input)

# Track errors
await telemetry.track_error(
    operation="context_retrieval",
    error_type="TimeoutError",
    error_message="Context store timeout",
)
```

---

## Configuration

### Intent Patterns (JSON)

Create `backend/config/followup_intents.json`:

```json
{
  "intents": [
    {
      "label": "follow_up",
      "patterns": ["yes", "no", "tell me more", "show me"],
      "examples": ["yes", "tell me more about that"],
      "metadata": {
        "category": "interaction",
        "priority": 100
      }
    }
  ]
}
```

Load dynamically:

```python
from backend.core.intent.intent_registry import IntentRegistry

registry = IntentRegistry(config_path=Path("backend/config/followup_intents.json"))
patterns = registry.get_all_patterns()
```

### Environment Variables

```bash
# Redis configuration
JARVIS_REDIS_URL=redis://localhost:6379
JARVIS_CONTEXT_STORE_BACKEND=redis  # or "memory"

# Telemetry
JARVIS_TELEMETRY_ENABLED=true
JARVIS_PROMETHEUS_PORT=9090

# Context settings
JARVIS_CONTEXT_TTL_SECONDS=120
JARVIS_CONTEXT_MAX_SIZE=1000
```

---

## Testing

### Run Tests

```bash
# All tests
pytest backend/tests/

# Specific test file
pytest backend/tests/test_adaptive_classifier.py

# With coverage
pytest --cov=backend/core --cov-report=html
```

### Test Structure

```
backend/tests/
├── test_context_envelope.py       # Context models
├── test_adaptive_classifier.py    # Intent classification
├── test_context_store.py          # Storage implementations
└── test_integration_followup.py   # End-to-end flows
```

---

## Performance Optimization

### Embedding Caching

```python
from backend.core.matching.semantic_matcher import CachedEmbeddingProvider

provider = CachedEmbeddingProvider(
    provider=base_provider,
    max_cache_size=10000,
)
```

### Redis for Distributed Systems

- Use Redis for multi-instance deployments
- Automatic TTL expiry reduces cleanup overhead
- Sorted sets for fast relevance queries

### Context Store Auto-Cleanup

```python
store = InMemoryContextStore(auto_cleanup_interval=60)
await store.start_auto_cleanup()
```

---

## Best Practices

### 1. Context Lifecycle

```python
# Create context when JARVIS asks a question
context_id = await track_pending_question(...)

# User responds → retrieve context
context = await store.get(context_id)

# Process response
result = await router.route(user_input, intent, context)

# Mark as consumed (prevents reuse)
if result.success:
    await store.mark_consumed(context_id)
```

### 2. Intent Confidence Thresholds

- **Follow-up**: ≥ 0.75 (high confidence needed)
- **Vision**: ≥ 0.60
- **Command**: ≥ 0.70

### 3. Context Relevance

```python
# Use decay rate for time-sensitive contexts
envelope = ContextEnvelope(
    ...,
    ttl_seconds=120,
    decay_rate=0.02,  # 2% per second
)

# High priority for user-facing questions
metadata = ContextMetadata(priority=ContextPriority.CRITICAL)
```

### 4. Error Handling

```python
try:
    result = await router.route(...)
except Exception as e:
    await telemetry.track_error(
        operation="routing",
        error_type=type(e).__name__,
        error_message=str(e),
    )
    # Fallback response
    result = RoutingResult(
        success=False,
        response="I encountered an error processing your request.",
    )
```

---

## Troubleshooting

### Issue: Context not retrieved

**Symptoms**: User says "yes" but system responds "No pending context"

**Solutions**:
1. Check context TTL (may have expired)
2. Verify context was added to store
3. Check query filters (category/tag mismatch)

```python
# Debug: List all contexts
all_contexts = await store.get_all()
print(f"Active contexts: {len(all_contexts)}")
for ctx in all_contexts:
    print(f"  {ctx.metadata.id}: {ctx.metadata.category.name}, valid={ctx.is_valid()}")
```

### Issue: Intent misclassification

**Symptoms**: "yes" classified as "unknown" instead of "follow_up"

**Solutions**:
1. Check pattern definitions
2. Lower confidence threshold
3. Add more pattern variations

```python
# Debug: See all signals
result = await intent_engine.classify("yes")
print(f"Primary: {result.primary_intent}, Confidence: {result.confidence}")
for signal in result.all_signals:
    print(f"  {signal.label}: {signal.confidence} ({signal.source})")
```

### Issue: High memory usage

**Solutions**:
1. Reduce `max_size` in InMemoryContextStore
2. Lower TTL values
3. Enable auto-cleanup
4. Switch to Redis backend

---

## Migration Guide

### From Hardcoded to Configuration

**Before:**
```python
FOLLOW_UP_PATTERNS = ["yes", "no", "tell me more"]
```

**After:**
```python
registry = IntentRegistry(config_path=Path("config/intents.json"))
patterns = registry.get_all_patterns()
```

### Adding Semantic Classification

```python
# 1. Install dependencies
# pip install sentence-transformers

# 2. Create provider
from backend.core.matching.semantic_matcher import SentenceTransformerProvider

provider = SentenceTransformerProvider(model_name="all-MiniLM-L6-v2")

# 3. Add to intent engine
semantic_classifier = SemanticEmbeddingClassifier(
    name="semantic",
    embedding_fn=provider.embed,
    intent_embeddings=registry.get_all_embeddings(),
)

engine.add_classifier(semantic_classifier)
```

---

## Future Enhancements

- [ ] Multi-modal context (visual + text)
- [ ] Cross-space context persistence
- [ ] Proactive reasoning ("You said yes earlier...")
- [ ] Few-shot learning for custom intents
- [ ] Graph-based context relationships
- [ ] Natural language query interface

---

## Support

For issues, questions, or contributions:
- GitHub: [JARVIS-AI-Agent](https://github.com/yourusername/JARVIS-AI-Agent)
- Documentation: See `docs/` directory
- Examples: See `backend/examples/`

---

**Built with ❤️ for natural, context-aware AI interactions**
