# Goal Inference System Guide

## Overview

The Goal Inference System is part of the Intelligent Understanding System (IUS) for JARVIS Vision. It understands user objectives beyond immediate actions by analyzing context, patterns, and temporal information to infer what the user is trying to accomplish.

## Goal Hierarchy

The system recognizes three levels of goals:

### 1. High-Level Goals (Long-term objectives)
- **Project Completion** - Working towards completing a project
- **Problem Solving** - Solving technical or conceptual problems
- **Information Gathering** - Researching and collecting information
- **Communication** - Engaging in messaging, email, or meetings
- **Learning/Research** - Studying new concepts or technologies

### 2. Intermediate Goals (Session/task goals)
- **Feature Implementation** - Building new functionality
- **Bug Fixing** - Resolving errors and issues
- **Document Preparation** - Creating or editing documents
- **Meeting Preparation** - Getting ready for meetings
- **Response Composition** - Writing emails or messages

### 3. Immediate Goals (Current action goals)
- **Find Information** - Searching for specific data
- **Fix Error** - Resolving an immediate error
- **Complete Form** - Filling out forms or inputs
- **Send Message** - Sending communication
- **Review Content** - Reviewing documents or code

## Architecture

### Evidence Collection
The system collects evidence from multiple sources:
- **Recent Actions** - User's recent interactions
- **Open Applications** - Currently active apps
- **Content Being Viewed** - What's on screen
- **Time Context** - Time of day, day of week
- **Calendar Context** - Meetings and appointments (via Swift integration)
- **Historical Patterns** - Past behavior

### Goal Hypothesis
1. Generate possible goals based on evidence
2. Score each hypothesis using pattern matching
3. Validate against learned patterns
4. Select high-confidence goals

### Goal Tracking
- Monitor progress towards goals
- Detect goal changes or abandonment
- Identify goal completion
- Learn success patterns for future inference

## Memory Allocation

The Goal Inference System uses 80MB of memory:
- **Goal Templates**: 20MB - Pattern definitions
- **Inference Engine**: 30MB - Hypothesis generation
- **Tracking Data**: 30MB - Progress monitoring

## Multi-Language Components

### Python (Main orchestration)
- Goal hierarchy management
- Evidence collection
- Integration with VSMS and Activity Recognition

### Rust (High-performance matching)
- Pattern matching engine (`goal_patterns.rs`)
- Real-time inference optimization
- Memory-efficient caching

### Swift (Native macOS integration)
- Calendar context provider (`calendar_context_provider.swift`)
- Native event access
- Time-based evidence collection

## Configuration

Environment variables to control the Goal Inference System:

```bash
# Enable/disable goal inference
export GOAL_INFERENCE_ENABLED=true

# Set confidence thresholds
export GOAL_CONFIDENCE_THRESHOLD=0.7

# Enable calendar integration (macOS only)
export CALENDAR_INTEGRATION_ENABLED=true
```

## API Usage

### Get Inferred Goals
```python
# Get all currently inferred goals
goals = await analyzer.get_inferred_goals()

# Response includes:
# - total_active: Number of active goals
# - by_level: Goals organized by hierarchy level
# - high_confidence_goals: Most confident goals
# - recently_updated: Recently changed goals
# - near_completion: Goals close to completion
```

### Get Goal Insights
```python
# Get detailed insights for a specific goal
insights = await analyzer.get_goal_insights(goal_id)

# Insights include:
# - goal type and level
# - description and confidence
# - progress (0-1)
# - duration since creation
# - parent/child relationships
# - predicted completion time
```

### Track Goal Progress
```python
# Update goal progress
result = await analyzer.track_goal_progress(goal_id, progress_delta=0.25)

# Returns:
# - success: Whether update succeeded
# - current_progress: Updated progress value
```

## Integration with Other Systems

### Activity Recognition
- Goals are linked to recognized activities
- Activities provide evidence for goal inference
- Progress in activities updates goal progress

### VSMS (Visual State Management)
- State transitions inform goal changes
- Application states provide goal context
- Stuck states indicate goal blockers

### Temporal Context Engine
- Historical patterns improve goal inference
- Event sequences reveal goal patterns
- Time-based predictions for goal completion

## Evidence Types and Weights

1. **Application Evidence** (weight: 0.9)
   - Which apps are open indicates goal type
   - App combinations reveal complex goals

2. **Content Evidence** (weight: 0.8)
   - Document types suggest goals
   - Error messages indicate problem-solving

3. **Action Evidence** (weight: 1.0)
   - User interactions directly indicate intent
   - Action sequences reveal workflows

4. **Time Evidence** (weight: 0.7)
   - Working hours suggest professional goals
   - Calendar events provide goal deadlines

5. **Communication Evidence** (weight: 0.8)
   - Email/chat apps indicate communication goals
   - Message drafts show response composition

6. **Historical Evidence** (weight: 0.6)
   - Past patterns predict current goals
   - Success patterns improve accuracy

## Testing

Run the test scripts to verify integration:

```bash
# Test goal inference
python backend/vision/intelligence/test_goal_inference.py

# Test calendar integration (macOS)
swift backend/vision/calendar_context_provider.swift test
```

## Common Use Cases

### 1. Development Work
- Detects: Feature implementation, bug fixing
- Evidence: IDE open, error messages, code files
- Tracks: Progress through compilation, testing

### 2. Communication Tasks
- Detects: Email composition, meeting prep
- Evidence: Email/chat apps, calendar events
- Tracks: Message drafts, response completion

### 3. Research Activities
- Detects: Information gathering, learning
- Evidence: Browser tabs, note-taking apps
- Tracks: Topics explored, notes created

### 4. Document Work
- Detects: Document preparation, review
- Evidence: Office apps, document content
- Tracks: Editing progress, review completion

### 5. Problem Solving
- Detects: Debugging, troubleshooting
- Evidence: Error messages, search queries
- Tracks: Error resolution, solution finding

## Best Practices

1. **Evidence Quality**
   - More evidence types improve accuracy
   - Recent evidence weights higher
   - Contradictory evidence reduces confidence

2. **Goal Relationships**
   - High-level goals have intermediate children
   - Intermediate goals have immediate children
   - Progress bubbles up the hierarchy

3. **Learning Integration**
   - System learns from completed goals
   - Success patterns improve future inference
   - User-specific patterns personalize inference

## Future Enhancements

- Natural language goal specification
- Multi-user goal coordination
- Goal recommendation engine
- Integration with task management tools
- Predictive goal scheduling