# Predictive/Analytical Queries Integration

## Overview

The predictive/analytical query system provides JARVIS with advanced capabilities to analyze progress, detect bugs, recommend next steps, and provide semantic code understanding using Claude Vision.

## Architecture

```
User Query → Intent Analyzer → Context-Aware Handler → Predictive Query Handler → Response
                                          ↓                        ↓
                                  Predictive Analyzer        Claude Vision
                                          ↓                        ↓
                                    Git Metrics              Code Analysis
                                    Error Patterns           Screenshots
                                    Workflow Analysis
                                    Recommendations
```

## Components

### 1. Intent Analyzer (`analyzers/intent_analyzer.py`)
- Detects predictive query intent type
- Pattern matching for query classification
- Determines if screen access is needed

### 2. Predictive Analyzer (`analyzers/predictive_analyzer.py`)
- **GitMetricsCollector**: Tracks commits, velocity, code changes
- **ErrorPatternCollector**: Detects recurring bugs
- **WorkflowAnalyzer**: Measures efficiency and focus time
- **RecommendationEngine**: Generates actionable recommendations

### 3. Predictive Query Handler (`handlers/predictive_query_handler.py`)
- **ClaudeVisionAnalyzer**: Semantic code analysis using Claude 3.5 Sonnet
- **Screenshot Integration**: Captures and analyzes visible code
- **Context-Aware**: Integrates with workspace context

### 4. Context-Aware Handler (`handlers/context_aware_handler.py`)
- Routes predictive queries to specialized handler
- Manages screen lock state
- Formats responses

### 5. JARVIS Integration (`integrations/jarvis_integration.py`)
- Routes predictive queries through voice command pipeline
- Manages WebSocket updates
- Integrates with command queue

## Supported Query Types

### 1. Progress Checks
```
"Am I making progress?"
"How much progress have I made?"
"What's my progress?"
```

**Returns:**
- Commits today/this week
- Lines added/removed
- Files modified
- Velocity trend
- Efficiency score

### 2. Next Steps
```
"What should I work on next?"
"What's next?"
"What to do next?"
```

**Returns:**
- Prioritized recommendations
- Bug fixes to address
- Task suggestions based on state

### 3. Bug Detection
```
"Are there any bugs?"
"Any potential issues?"
"What's wrong?"
```

**Returns:**
- Recurring error patterns
- Bug severity levels
- Suggested fixes
- Affected locations

### 4. Code Explanation (with Vision)
```
"Explain this code"
"What does this code do?"
"How does this work?"
```

**Returns:**
- Semantic code analysis
- Function descriptions
- Pattern explanations
- Visual analysis (if screen available)

### 5. Workflow Optimization
```
"How can I improve my workflow?"
"Optimize my workflow"
"Work more efficiently"
```

**Returns:**
- Efficiency score
- Context switch analysis
- Focus time metrics
- Optimization recommendations

### 6. Quality Assessment
```
"How's my code quality?"
"Code quality assessment"
```

**Returns:**
- Test coverage analysis
- Code quality metrics
- Improvement suggestions

## Usage

### Via Voice Command
```python
# Automatically routed through JARVIS integration
"JARVIS, am I making progress?"
"JARVIS, what should I work on next?"
"JARVIS, are there any bugs?"
```

### Programmatic API
```python
from backend.context_intelligence.handlers import handle_predictive_query

# Simple usage
result = await handle_predictive_query("Am I making progress?")
print(result.response_text)

# With options
from backend.context_intelligence.handlers import (
    PredictiveQueryHandler,
    PredictiveQueryRequest
)

handler = PredictiveQueryHandler()

request = PredictiveQueryRequest(
    query="Explain this code",
    use_vision=True,
    capture_screen=True,
    repo_path="/path/to/repo"
)

response = await handler.handle_query(request)
print(response.response_text)
```

### Direct Analyzer Usage
```python
from backend.context_intelligence.analyzers import analyze_query

result = await analyze_query(
    "Am I making progress?",
    context={"repo_path": "."}
)

print(result.response_text)

# Access detailed metrics
if result.metrics:
    print(f"Commits today: {result.metrics.commits_today}")
    print(f"Velocity: {result.metrics.velocity_trend}")

# Access recommendations
for rec in result.recommendations:
    print(f"{rec.priority.upper()}: {rec.title}")
    print(f"  {rec.description}")
```

## Configuration

### Enable/Disable Features
```python
from backend.context_intelligence.analyzers import initialize_predictive_analyzer

analyzer = initialize_predictive_analyzer(
    enable_git_metrics=True,
    enable_error_tracking=True,
    enable_workflow_analysis=True
)
```

### Claude Vision Setup
```python
from backend.context_intelligence.handlers import initialize_predictive_handler

# With Claude API key
handler = initialize_predictive_handler(
    claude_api_key="your-api-key",
    enable_vision=True
)

# Without Vision (text-only)
handler = initialize_predictive_handler(
    enable_vision=False
)
```

## Integration Points

### 1. Intent Analysis
`backend/context_intelligence/analyzers/intent_analyzer.py:17`
- Added `PREDICTIVE_QUERY` to `IntentType` enum
- Pattern matching for all query types

### 2. Context-Aware Handler
`backend/context_intelligence/handlers/context_aware_handler.py:60-63`
- Routes predictive queries to specialized handler
- Manages screen access requirements

### 3. JARVIS Integration
`backend/context_intelligence/integrations/jarvis_integration.py:107-121`
- Detects predictive queries early in pipeline
- Routes to appropriate handler

## Response Format

```python
{
    "success": true,
    "command": "Am I making progress?",
    "messages": ["📊 Insights\n- ✅ 3 commit(s) today..."],
    "context": {
        "query_type": "progress_check",
        "confidence": 0.85,
        "insights": [...]
    },
    "result": {
        "analytics": {
            "metrics": {
                "commits_today": 3,
                "commits_this_week": 15,
                "lines_added": 234,
                "velocity_trend": "increasing"
            },
            "bug_patterns": [],
            "recommendations": [...]
        },
        "vision_analysis": null
    },
    "timestamp": "2025-10-19T00:45:00"
}
```

## Error Handling

The system gracefully handles:
- Missing git repository
- Unavailable Claude Vision API
- No metrics available
- Screen lock during visual queries

Example error response:
```python
{
    "success": false,
    "command": "Explain this code",
    "messages": ["Error: Claude Vision not available"],
    "summary": "Error: Install anthropic library"
}
```

## Testing

### Run Demo
```bash
# All scenarios
python backend/context_intelligence/demo_predictive_queries.py

# Interactive mode
python backend/context_intelligence/demo_predictive_queries.py --scenario interactive

# Specific scenario
python backend/context_intelligence/demo_predictive_queries.py --scenario progress
python backend/context_intelligence/demo_predictive_queries.py --scenario bugs
python backend/context_intelligence/demo_predictive_queries.py --scenario next_steps
```

### Manual Testing
```python
import asyncio
from backend.context_intelligence.handlers import handle_predictive_query

async def test():
    result = await handle_predictive_query("Am I making progress?")
    print(result.response_text)

asyncio.run(test())
```

## Performance

- **Git Metrics**: Cached for 5 minutes
- **Error Patterns**: Analyzed from last 24 hours
- **Workflow Analysis**: Last 24 hours of activity
- **Claude Vision**: ~2-5 seconds per request

## Dependencies

Required:
- Python 3.8+
- Git (for metrics collection)
- asyncio

Optional:
- `anthropic` library (for Claude Vision)
  ```bash
  pip install anthropic
  ```

## Future Enhancements

- [ ] Machine learning for better bug prediction
- [ ] Integration with CI/CD pipelines
- [ ] Multi-repository analysis
- [ ] Historical trend visualization
- [ ] Custom metric collectors
- [ ] Plugin system for additional analyzers

## Troubleshooting

### "No metrics available"
- Ensure you're in a git repository
- Check git is installed: `git --version`

### "Claude Vision not available"
- Install: `pip install anthropic`
- Set API key in environment or config

### "Screen capture failed"
- Verify Yabai is running (macOS)
- Check screen permissions

## Contributing

To add new query types:

1. Add pattern to `IntentType.PREDICTIVE_QUERY` in `intent_analyzer.py`
2. Add query type to `PredictiveQueryType` enum in `predictive_analyzer.py`
3. Implement analyzer logic in `PredictiveAnalyzer.analyze()`
4. Add recommendation logic in `RecommendationEngine`

## License

Part of JARVIS AI Assistant project.
