# Resolver Systems Comparison & Integration Strategy

## System Comparison

### Contextual Query Resolver (Section 1.2 - Just Implemented)
**Location:** `backend/context_intelligence/resolvers/contextual_query_resolver.py`

**Focus:** Space and monitor resolution for vision queries

**Features:**
- ✅ 6 resolution strategies (USE_ACTIVE_SPACE, USE_LAST_QUERIED, etc.)
- ✅ Yabai integration for active space detection
- ✅ 10-turn conversation history (basic)
- ✅ Space/monitor pronoun resolution ("it", "that", "them")
- ✅ 5-second caching for active space
- ✅ Multi-monitor aware

**Strengths:**
- Laser-focused on space/monitor resolution
- Directly integrated with Yabai
- Fast and lightweight (~750 lines)
- Already integrated with Unified Command Processor

**Limitations:**
- Only tracks spaces/monitors, not entities
- Simple conversation tracking (just space numbers)
- No visual attention tracking
- No intent classification
- No OCR integration

---

### Implicit Reference Resolver (Pre-Existing)
**Location:** `backend/core/nlp/implicit_reference_resolver.py`

**Focus:** Multi-modal context understanding and entity resolution

**Features:**
- ✅ **Visual Attention Tracking** (50 events history)
  - Records what user was looking at via OCR
  - Tracks content_type (error, code, documentation, terminal_output)
  - Significance levels (critical, high, normal, low)
  - OCR text hashing for deduplication

- ✅ **Rich Conversational Context** (10 turns)
  - Extracts entities (errors, files, commands, functions)
  - Tracks which entities were discussed
  - Current topic tracking

- ✅ **Query Intent Classification** (11 intents)
  - EXPLAIN, DESCRIBE, LOCATE, STATUS
  - DIAGNOSE, FIX, PREVENT
  - RECALL, COMPARE, SUMMARIZE
  - CLARIFY, UNKNOWN

- ✅ **Advanced Reference Types**
  - PRONOUN: "it", "that", "this"
  - DEMONSTRATIVE: "that error", "this file"
  - POSSESSIVE: "my code", "your suggestion"
  - IMPLICIT: "the error" (which error?)
  - EXPLICIT: "the error in terminal" (specific)

- ✅ **Multi-Strategy Resolution**
  1. Conversational context (what was discussed)
  2. Visual attention (what was seen on screen)
  3. Workspace context graph (recent errors/events)
  4. Temporal relevance (decay over 5 minutes)

- ✅ **Intent-Specific Responses**
  - Generates different responses based on intent
  - Integrated with MultiSpaceContextGraph
  - Rich error diagnosis and fix suggestions

**Strengths:**
- Comprehensive multi-modal understanding
- OCR-based visual attention tracking
- Rich entity extraction and tracking
- Intent-aware response generation
- Temporal relevance scoring

**Limitations:**
- NOT integrated with Unified Command Processor
- Does NOT resolve space/monitor numbers
- Requires MultiSpaceContextGraph (dependency)
- More complex (~900 lines)

---

## Integration Strategy

### Recommended Approach: **Layered Integration**

Both systems serve different but complementary purposes. They should work together in layers:

```
User Query: "What does it say?"
    ↓
Layer 1: Implicit Reference Resolver
    ↓
Resolves: "it" = error in Terminal (from visual attention)
Intent: DESCRIBE
Entity Type: error
    ↓
Layer 2: Contextual Query Resolver
    ↓
Resolves: Terminal is on Space 3
Space: 3 (from last queried space)
    ↓
Layer 3: Unified Command Processor
    ↓
Executes: Analyze Space 3, focus on Terminal, describe error
    ↓
Result: "The error in Terminal (Space 3) is: FileNotFoundError..."
```

### Integration Points

#### 1. Unified Command Processor Integration

**Current State:**
- ✅ Contextual Query Resolver integrated (lines 199-214)
- ❌ Implicit Reference Resolver NOT integrated

**Proposed Enhancement:**

```python
# In unified_command_processor.py

def __init__(self, ...):
    # Existing
    self.contextual_resolver = None  # Space/monitor resolution

    # NEW: Add implicit reference resolver
    self.implicit_resolver = None    # Entity/intent resolution

    self._initialize_resolvers()

def _initialize_resolvers(self):
    """Initialize both resolver systems"""
    # Contextual query resolver (space/monitor)
    from context_intelligence.resolvers import get_contextual_resolver
    self.contextual_resolver = get_contextual_resolver()

    # Implicit reference resolver (entity/intent)
    from core.nlp.implicit_reference_resolver import get_implicit_resolver
    self.implicit_resolver = get_implicit_resolver()

async def _resolve_vision_query(self, query: str) -> Dict[str, Any]:
    """
    Two-stage resolution:
    1. Implicit resolver: Resolve entity ("it" = error in Terminal)
    2. Contextual resolver: Resolve space (Terminal is on Space 3)
    """
    resolution = {
        "original_query": query,
        "entity_resolution": None,
        "space_resolution": None,
        "intent": None
    }

    # Stage 1: Implicit reference resolution
    if self.implicit_resolver:
        implicit_result = await self.implicit_resolver.resolve_query(query)
        resolution["entity_resolution"] = {
            "intent": implicit_result.get("intent"),
            "referent": implicit_result.get("referent"),
            "confidence": implicit_result.get("confidence")
        }
        resolution["intent"] = implicit_result.get("intent")

        # If implicit resolver found a specific space, use it
        if implicit_result.get("context", {}).get("space_id"):
            resolution["space_resolution"] = {
                "spaces": [implicit_result["context"]["space_id"]],
                "strategy": "implicit_reference",
                "confidence": 1.0
            }
            return resolution

    # Stage 2: Contextual space resolution (if needed)
    if self.contextual_resolver:
        space_result = await self.contextual_resolver.resolve_query(query)
        resolution["space_resolution"] = {
            "spaces": space_result.resolved_spaces,
            "monitors": space_result.resolved_monitors,
            "strategy": space_result.strategy_used.value,
            "confidence": space_result.confidence
        }

    return resolution
```

#### 2. Shared Conversation Context

Both systems track conversation history independently. They should share:

```python
class UnifiedConversationContext:
    """Shared conversation context for both resolvers"""

    def __init__(self):
        self.turns = deque(maxlen=10)
        self.entities = {}        # From implicit resolver
        self.spaces = {}          # From contextual resolver
        self.visual_attention = deque(maxlen=50)  # From implicit resolver

    def add_turn(self, user_query, response, context):
        """Record turn for both resolvers"""
        turn = {
            "timestamp": datetime.now(),
            "query": user_query,
            "response": response,
            "entities": self._extract_entities(context),
            "spaces": self._extract_spaces(context),
            "visual_attention": context.get("visual_attention")
        }
        self.turns.append(turn)
```

#### 3. Visual Attention + Space Mapping

Combine visual attention tracking with space resolution:

```python
# When visual attention is recorded
implicit_resolver.record_visual_attention(
    space_id=space_id,      # From contextual resolver
    app_name=app_name,
    ocr_text=ocr_text,
    content_type="error",
    significance="critical"
)

# Later, when query "what does it say?" comes in:
# 1. Implicit resolver finds: "it" = error in Terminal (visual attention)
# 2. Implicit resolver knows: error was on space_id=3
# 3. No need for contextual resolver - already have space!
```

#### 4. Intent-Aware Query Routing

Use implicit resolver's intent classification to route queries:

```python
# In unified command processor

async def process_command(self, command_text: str) -> Dict[str, Any]:
    # Get intent from implicit resolver
    if self.implicit_resolver:
        parsed = self.implicit_resolver.query_analyzer.analyze(command_text)
        intent = parsed.intent

        # Route based on intent
        if intent in [QueryIntent.EXPLAIN, QueryIntent.DESCRIBE, QueryIntent.DIAGNOSE]:
            # These need full implicit resolution
            result = await self._handle_with_implicit_resolver(command_text)
        elif intent == QueryIntent.STATUS:
            # These need space resolution
            result = await self._handle_with_contextual_resolver(command_text)
        else:
            # Default routing
            result = await self._execute_command(...)
```

---

## Recommended Implementation Plan

### Phase 1: Verify Implicit Resolver Status ✅ (Do Now)

**Goal:** Understand current integration state

**Tasks:**
1. Check if MultiSpaceContextGraph is initialized
2. Check if implicit resolver is being used anywhere
3. Verify dependencies are available

**Commands:**
```bash
# Check for MultiSpaceContextGraph usage
grep -r "MultiSpaceContextGraph" backend/ --include="*.py"

# Check for implicit resolver usage
grep -r "ImplicitReferenceResolver\|get_implicit_resolver" backend/ --include="*.py"

# Check if it's initialized
grep -r "initialize_implicit_resolver" backend/ --include="*.py"
```

---

### Phase 2: Initialize Implicit Resolver (Quick Win)

**Goal:** Get implicit resolver running alongside contextual resolver

**Changes:**
1. Add implicit resolver to `unified_command_processor.py` init
2. Initialize both resolvers
3. Add fallback if not available

**Estimated Time:** 15 minutes
**Risk:** Low (graceful fallback)

---

### Phase 3: Two-Stage Resolution (Enhancement)

**Goal:** Combine both resolvers for comprehensive understanding

**Changes:**
1. Modify `_resolve_vision_query()` to use both systems
2. Implicit resolver first (entity resolution)
3. Contextual resolver second (space resolution)
4. Combine results

**Estimated Time:** 30 minutes
**Risk:** Medium (need to test interaction)

---

### Phase 4: Shared Context (Advanced)

**Goal:** Unify conversation tracking

**Changes:**
1. Create `UnifiedConversationContext`
2. Both resolvers use shared context
3. Visual attention + space mapping

**Estimated Time:** 1-2 hours
**Risk:** High (major refactoring)

---

## Benefits of Integration

### For Users

1. **Richer Understanding**
   - Before: "What's happening?" → Just shows active space
   - After: "What's happening?" → Knows you mean the error you just saw in Terminal on Space 3

2. **Intent-Aware Responses**
   - Before: Generic vision analysis
   - After: Different responses for "explain that" vs "fix that" vs "what is that?"

3. **Visual Memory**
   - Before: No memory of what was on screen
   - After: "What did it say 2 minutes ago?" → Recalls OCR text from visual attention

4. **Multi-Modal Context**
   - Combines: conversation + visual attention + workspace state + space/monitor info

### For System

1. **Comprehensive Resolution**
   - Entity resolution (implicit resolver)
   - Space resolution (contextual resolver)
   - Both working together

2. **Better Confidence**
   - Cross-verify between systems
   - Higher confidence when both agree

3. **Fallback Support**
   - If one fails, other still works
   - Graceful degradation

---

## Conclusion

**Yes, the implicit_reference_resolver.py can GREATLY enhance the system!**

**Recommended Next Step:**
Run Phase 1 (Verification) to check current state, then decide on Phase 2 (Quick Win) or Phase 3 (Full Integration).

The two systems are complementary:
- **Contextual Query Resolver**: Lightweight, space-focused, already integrated ✅
- **Implicit Reference Resolver**: Comprehensive, entity-focused, NOT integrated ❌

Integrating both would give JARVIS the best of both worlds:
- "What's happening?" → Active space detection (contextual)
- "What does it say?" → OCR text from visual attention (implicit)
- "How do I fix it?" → Error entity + space + intent-aware response (both!)
