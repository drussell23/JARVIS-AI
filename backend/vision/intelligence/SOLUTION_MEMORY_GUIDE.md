# Solution Memory Bank Guide

## Overview

The Solution Memory Bank is an intelligent system that captures, stores, and applies solutions to recurring problems. It learns from past problem resolutions to provide automated assistance for similar issues in the future.

## Architecture

### Memory Allocation (100MB Total)

1. **Solution Database (60MB)**
   - Stores complete solution entries
   - Problem signatures with visual patterns
   - Action sequences and parameters
   - Success metrics and refinements

2. **Index Structures (20MB)**
   - Hash-based problem lookup
   - Vector similarity index (FAISS/LSH)
   - Type-based categorization
   - Keyword search indices

3. **Application Engine (20MB)**
   - Solution execution logic
   - Adaptation algorithms
   - Verification systems
   - Learning buffers

### Multi-Language Components

1. **Python (Core Logic)**
   - Solution capture and storage
   - ML-based similarity matching
   - Learning and refinement system
   - API integration

2. **Rust (High Performance)**
   - SIMD-accelerated vector similarity
   - Locality-Sensitive Hashing (LSH)
   - Pattern mining algorithms
   - Parallel search operations

3. **Swift (Native macOS)**
   - Automated solution execution
   - UI element interaction
   - Keyboard/mouse automation
   - Application control

## Solution Structure

### Problem Signature
Identifies unique problems through:
- **Visual Patterns**: Screenshot features, UI elements
- **Error Messages**: Exact error text and codes
- **Context State**: Application version, OS, environment
- **Symptoms**: Observable behaviors and indicators
- **Problem Type**: Error, Performance, Configuration, etc.

### Solution Details
Complete resolution information:
- **Action Sequence**: Ordered steps to resolve
- **Success Rate**: Historical effectiveness
- **Average Time**: Typical execution duration
- **Side Effects**: Known consequences
- **Requirements**: Prerequisites for execution
- **Confidence**: Solution reliability score

### Application Context
When and how to apply:
- **Applicable Conditions**: When solution works
- **Prerequisites**: Required state/setup
- **Variations**: Alternative approaches
- **Limitations**: Known constraints
- **Compatibility**: App/OS versions

### Learning Metadata
Continuous improvement tracking:
- **Usage Count**: Times applied
- **Success/Failure Count**: Outcome tracking
- **User Ratings**: Feedback scores
- **Refinement History**: Changes over time
- **Feedback Notes**: User comments

## Usage Examples

### 1. Capturing a Solution

```python
from backend.vision.intelligence.solution_memory_bank import (
    get_solution_memory_bank, ProblemSignature, ProblemType
)

memory_bank = get_solution_memory_bank()

# Define the problem
problem = ProblemSignature(
    visual_pattern={
        'app': 'vscode',
        'dialog_type': 'error',
        'color_scheme': 'dark'
    },
    error_messages=[
        "Cannot find module 'numpy'",
        "Import error: No module named numpy"
    ],
    context_state={
        'app_version': '1.75.0',
        'python_version': '3.9.0',
        'os': 'macOS 14.0'
    },
    symptoms=[
        'red_squiggly_lines',
        'import_error',
        'module_not_found'
    ],
    problem_type=ProblemType.ERROR
)

# Define solution steps
steps = [
    {
        'action': 'open_terminal',
        'target': 'integrated_terminal',
        'parameters': {'shortcut': 'cmd+`'},
        'wait_condition': 'terminal_ready'
    },
    {
        'action': 'type',
        'parameters': {'text': 'pip install numpy'},
        'verification': 'no_errors'
    },
    {
        'action': 'key',
        'target': 'return',
        'parameters': {},
        'wait_condition': 'installation_complete',
        'timeout': 60.0
    }
]

# Capture the solution
solution = await memory_bank.capture_solution(
    problem=problem,
    solution_steps=steps,
    success=True,
    execution_time=45.2,
    context={
        'python_env': 'venv',
        'package_manager': 'pip'
    }
)
```

### 2. Finding Similar Solutions

```python
# New similar problem
new_problem = ProblemSignature(
    error_messages=["ModuleNotFoundError: No module named 'pandas'"],
    symptoms=['import_error', 'module_missing'],
    problem_type=ProblemType.ERROR
)

# Find similar solutions
similar = await memory_bank.find_similar_solutions(
    new_problem,
    threshold=0.7  # 70% similarity minimum
)

for solution_id, similarity in similar[:3]:
    print(f"Solution {solution_id}: {similarity:.2%} match")
```

### 3. Getting Recommendations

```python
# Get intelligent recommendations
recommendations = await memory_bank.get_solution_recommendations(
    problem=new_problem,
    context={
        'app': 'jupyter',
        'python_version': '3.9',
        'urgency': 'high'
    }
)

for rec in recommendations:
    print(f"Score: {rec['score']:.2f}")
    print(f"Auto-applicable: {rec['auto_applicable']}")
    print(f"Success rate: {rec['success_rate']:.2%}")
    print(f"Estimated time: {rec['estimated_time']:.1f}s")
```

### 4. Applying Solutions

```python
# Apply the best solution
best_solution_id = recommendations[0]['solution_id']

# With automation callback
async def execute_step(step, context):
    print(f"Executing: {step.action}")
    # Actual automation would happen here
    return {'success': True, 'message': 'Step completed'}

result = await memory_bank.apply_solution(
    solution_id=best_solution_id,
    current_context={'app': 'jupyter', 'cell': 1},
    execute_callback=execute_step
)

if result['success']:
    print(f"Problem resolved in {result['execution_time']:.1f}s")
```

### 5. Refining Solutions

```python
# User provides feedback
await memory_bank.refine_solution(
    solution_id=best_solution_id,
    refinements={
        'steps': [{
            'index': 1,
            'parameters': {
                'text': 'pip install pandas numpy matplotlib'
            }
        }],
        'add_steps': [{
            'action': 'restart_kernel',
            'target': 'jupyter',
            'parameters': {'menu': 'Kernel > Restart'}
        }]
    },
    user_feedback="Installing multiple packages together is faster",
    rating=0.9
)
```

## Integration with Vision Analyzer

The Solution Memory Bank integrates seamlessly with the Claude Vision Analyzer:

```python
from backend.vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer

analyzer = ClaudeVisionAnalyzer()

# Automatic solution capture during analysis
result, metrics = await analyzer.analyze_screenshot(
    screenshot,
    "Analyze this error and suggest a solution",
    capture_solution=True  # Enable solution capture
)

# If a solution was found and applied
if result.get('solution_applied'):
    print(f"Applied solution: {result['solution_id']}")
    print(f"Success: {result['solution_success']}")
```

## Advanced Features

### 1. Pattern Mining

The Rust-powered pattern analyzer identifies common action sequences:

```python
# Analyze patterns in solutions
patterns = memory_bank.analyze_solution_patterns(
    min_support=0.3  # Patterns in 30%+ of solutions
)

for pattern in patterns:
    print(f"Pattern: {' → '.join(pattern.action_sequence)}")
    print(f"Occurrence: {pattern.occurrence_count}")
    print(f"Success rate: {pattern.success_rate:.2%}")
```

### 2. Similarity Search

High-performance vector similarity using SIMD:

```python
# Custom similarity search
results = memory_bank.search_similar_vectors(
    query_vector,
    top_k=10,
    use_rust=True  # Enable Rust acceleration
)
```

### 3. Automated Execution

Native macOS automation with Swift:

```python
# Execute solution with native automation
from backend.vision.solution_automation import execute_solution_native

await execute_solution_native(
    solution.solution_details.action_sequence,
    target_app="com.microsoft.VSCode",
    capture_screenshots=True
)
```

## Configuration

### Environment Variables

```bash
# Storage
export SOLUTION_STORAGE_PATH="./jarvis_solutions.db"

# Matching
export SOLUTION_MIN_CONFIDENCE=0.6
export SOLUTION_MAX_RESULTS=5
export SOLUTION_AUTO_APPLY_THRESHOLD=0.9

# ML Features
export SOLUTION_USE_ML=true
export SOLUTION_USE_RUST=true

# Automation
export SOLUTION_AUTOMATION_ENABLED=true
export SOLUTION_CAPTURE_SCREENSHOTS=true
```

## Best Practices

### 1. Problem Definition
- Be specific with error messages
- Include visual context when available
- Specify exact symptoms observed
- Record complete context state

### 2. Solution Design
- Keep steps atomic and verifiable
- Include wait conditions for async operations
- Add verification steps for critical actions
- Document side effects and requirements

### 3. Continuous Learning
- Regularly refine solutions based on outcomes
- Provide user feedback and ratings
- Monitor effectiveness metrics
- Update for new app versions

### 4. Performance
- Use Rust acceleration for large databases
- Enable ML features for better matching
- Batch similar problem searches
- Prune old/ineffective solutions

## Troubleshooting

### Low Match Scores
- Ensure problem signatures are detailed
- Check similarity threshold settings
- Verify ML features are enabled
- Consider expanding symptom descriptions

### Automation Failures
- Verify accessibility permissions (macOS)
- Check target application is running
- Ensure UI elements haven't changed
- Add longer wait conditions

### Memory Issues
- Monitor solution database size
- Enable solution pruning
- Adjust memory allocations
- Archive old solutions

## Metrics and Monitoring

```python
# Get system statistics
stats = memory_bank.get_statistics()
print(f"Total solutions: {stats['total_solutions']}")
print(f"Average effectiveness: {stats['average_effectiveness']:.2%}")
print(f"Success rate: {stats['successes']/stats['applications']:.2%}")

# Memory usage
memory = memory_bank.get_memory_usage()
print(f"Database: {memory['solution_database']/1024/1024:.1f} MB")
print(f"Indices: {memory['index_structures']/1024/1024:.1f} MB")
```

## Future Enhancements

1. **Cloud Sync**: Share solutions across devices
2. **Community Solutions**: Crowdsourced problem fixes
3. **AI Refinement**: GPT-powered solution optimization
4. **Visual Verification**: Computer vision for validation
5. **Cross-Platform**: Windows and Linux support