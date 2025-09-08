# Activity Recognition Engine Guide

## Overview

The Activity Recognition Engine is part of the Intelligent Understanding System (IUS) for JARVIS Vision. It identifies what tasks users are performing and provides insights into their workflow, progress, and potential blockers.

## Key Components

### 1. Task Identification System
- Recognizes primary activities: Coding, Communication, Research, Document Creation, Data Analysis
- Maps visual states to task templates
- Learns user-specific patterns over time

### 2. Task Inference Engine  
- Infers user intent from context
- Predicts task goals and next steps
- Detects when users are stuck or need help

### 3. Progress Monitoring
- Tracks task completion percentage
- Estimates time remaining
- Identifies blockers and suggests solutions

## Configuration

Set these environment variables to control the Activity Recognition Engine:

```bash
# Enable/disable the engine
export ACTIVITY_RECOGNITION_ENABLED=true

# Enable task inference
export TASK_INFERENCE=true  

# Enable progress monitoring
export PROGRESS_MONITORING=true
```

## API Usage

### Get Current Activities
```python
# Get all currently active tasks
activities = await analyzer.get_current_activities()

# Each activity contains:
# - task_id: Unique identifier
# - task_name: Human-readable name
# - primary_activity: Type of activity (coding, communication, etc.)
# - completion_percentage: Progress (0-100)
# - is_stuck: Whether user appears blocked
# - applications: Apps involved in the task
```

### Get Activity Summary
```python
# Get overall productivity insights
summary = await analyzer.get_activity_summary()

# Summary includes:
# - total_tasks: Number of tasks tracked
# - active_tasks: Currently active
# - productivity_score: Overall productivity (0-1)
# - time_distribution: How time is spent
# - workflow_patterns: Common patterns
# - recommendations: Productivity tips
```

### Get Task Insights
```python
# Get detailed insights for a specific task
insights = await analyzer.get_activity_insights(task_id)

# Insights include:
# - detailed progress breakdown
# - identified blockers
# - suggested next steps
# - estimated completion time
```

## Integration with VSMS

The Activity Recognition Engine is fully integrated with:
- **Visual State Management System (VSMS)** - For state tracking
- **Semantic Scene Graph** - For understanding relationships
- **Temporal Context Engine** - For workflow patterns

When analyzing a screenshot, activity data is automatically included:

```python
result, metrics = await analyzer.analyze_screenshot(screenshot, prompt)

# Activity data is in:
activity = result['vsms_core']['activity']
```

## Memory Allocation

The Activity Recognition Engine uses 100MB of memory:
- Task templates and patterns: 30MB
- Active task tracking: 40MB  
- Progress monitoring: 30MB

## Multi-Language Support

Core engine is in Python with performance-critical components in:
- **Rust** - Pattern matching and real-time inference
- **Swift** - macOS native app integration

## Testing

Run the test script to verify integration:

```bash
python backend/vision/intelligence/test_activity_recognition.py
```

## Common Use Cases

1. **Development Workflow Tracking**
   - Identifies coding sessions
   - Tracks debugging progress
   - Detects when stuck on errors

2. **Communication Management**
   - Tracks email/chat sessions
   - Identifies important conversations
   - Suggests response priorities

3. **Research & Learning**
   - Tracks research progress
   - Identifies information gathering patterns
   - Suggests relevant resources

4. **Document Creation**
   - Monitors writing progress
   - Detects writer's block
   - Suggests content organization

5. **Data Analysis**
   - Tracks analysis workflow
   - Identifies data exploration patterns
   - Suggests visualizations

## Future Enhancements

- Goal setting and tracking
- Team collaboration insights
- Cross-application workflow optimization
- Personalized productivity coaching
- Integration with calendar and task managers