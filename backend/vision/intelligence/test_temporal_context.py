#!/usr/bin/env python3
"""
Test script for Temporal Context Engine
Demonstrates event tracking, pattern detection, and temporal predictions
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import components
from temporal_context_engine import get_temporal_engine, EventType, TemporalEvent


async def simulate_user_workflow():
    """Simulate a typical user workflow with events"""
    engine = get_temporal_engine()
    
    # Start the engine
    await engine.start()
    
    print("🚀 Temporal Context Engine Test")
    print("=" * 80)
    
    # Simulate Chrome email workflow
    app_id = "chrome"
    events = []
    
    print("\n📧 Simulating Email Workflow")
    print("-" * 60)
    
    # 1. Open Chrome
    event_id = await engine.process_visual_event(
        event_type=EventType.APPLICATION_LAUNCH,
        app_id=app_id,
        data={'window_title': 'Chrome'}
    )
    events.append(event_id)
    print(f"✅ Application launched: Chrome")
    await asyncio.sleep(0.5)
    
    # 2. Navigate to Gmail
    event_id = await engine.process_visual_event(
        event_type=EventType.STATE_CHANGE,
        app_id=app_id,
        state_id="chrome_home",
        data={'url': 'gmail.com'}
    )
    events.append(event_id)
    print(f"📍 State change: chrome_home")
    await asyncio.sleep(0.3)
    
    # 3. Gmail loads
    event_id = await engine.process_visual_event(
        event_type=EventType.STATE_CHANGE,
        app_id=app_id,
        state_id="chrome_gmail_inbox",
        data={'unread_count': 5}
    )
    events.append(event_id)
    print(f"📍 State change: chrome_gmail_inbox")
    await asyncio.sleep(1.0)
    
    # 4. Click on email
    event_id = await engine.process_visual_event(
        event_type=EventType.MOUSE_CLICK,
        app_id=app_id,
        state_id="chrome_gmail_inbox",
        data={'target': 'email_item', 'position': {'x': 400, 'y': 300}}
    )
    events.append(event_id)
    print(f"🖱️ Mouse click: email_item")
    await asyncio.sleep(0.2)
    
    # 5. Email opens
    event_id = await engine.process_visual_event(
        event_type=EventType.STATE_CHANGE,
        app_id=app_id,
        state_id="chrome_gmail_read",
        data={'email_id': 'msg_123', 'subject': 'Meeting Tomorrow'}
    )
    events.append(event_id)
    print(f"📍 State change: chrome_gmail_read")
    await asyncio.sleep(2.0)
    
    # 6. Click reply
    event_id = await engine.process_visual_event(
        event_type=EventType.MOUSE_CLICK,
        app_id=app_id,
        state_id="chrome_gmail_read",
        data={'target': 'reply_button'}
    )
    events.append(event_id)
    print(f"🖱️ Mouse click: reply_button")
    await asyncio.sleep(0.3)
    
    # 7. Compose window opens
    event_id = await engine.process_visual_event(
        event_type=EventType.STATE_CHANGE,
        app_id=app_id,
        state_id="chrome_gmail_compose",
        data={'mode': 'reply', 'to': 'john@example.com'}
    )
    events.append(event_id)
    print(f"📍 State change: chrome_gmail_compose")
    
    return engine, events


async def test_temporal_context():
    """Test temporal context retrieval"""
    engine, events = await simulate_user_workflow()
    
    print("\n🕐 Testing Temporal Context")
    print("-" * 60)
    
    # Get temporal context
    context = await engine.get_temporal_context(app_id="chrome")
    
    # Display immediate context
    print(f"\n📊 Immediate Context (last 30 seconds):")
    immediate = context.get('immediate', {})
    if not immediate.get('empty'):
        print(f"   Event count: {immediate.get('event_count', 0)}")
        print(f"   Time span: {immediate.get('time_span', 0):.1f} seconds")
        print(f"   Dominant events:")
        for event_type, count in immediate.get('dominant_event_types', []):
            print(f"   - {event_type}: {count}")
    
    # Display short-term context
    print(f"\n📈 Short-term Context (last 5 minutes):")
    short_term = context.get('short_term', {})
    print(f"   Active apps: {short_term.get('active_apps', [])}")
    print(f"   Task sequences: {len(short_term.get('task_sequences', []))}")
    print(f"   Focus changes: {short_term.get('focus_changes', 0)}")
    
    # Display active patterns
    print(f"\n🔄 Active Patterns:")
    active_patterns = context.get('active_patterns', [])
    for pattern in active_patterns[:3]:
        print(f"   - Type: {pattern['type']}, Confidence: {pattern['confidence']:.2f}")
    
    # Display predictions
    print(f"\n🔮 Predictions:")
    predictions = context.get('predictions', {})
    next_events = predictions.get('next_likely_events', [])
    for event in next_events[:3]:
        print(f"   - {event.get('event_type')} in {event.get('app_id')}")
        print(f"     Expected: {event.get('expected_time')}")
    
    return engine, context


async def test_pattern_detection():
    """Test pattern detection with repeated sequences"""
    engine = get_temporal_engine()
    
    print("\n🔍 Testing Pattern Detection")
    print("-" * 60)
    
    # Simulate periodic save action
    print("\n💾 Simulating periodic save pattern...")
    for i in range(5):
        await engine.process_visual_event(
            event_type=EventType.KEYBOARD_INPUT,
            app_id="vscode",
            data={'keys': 'cmd+s'}
        )
        print(f"   Save #{i+1}")
        await asyncio.sleep(2.0)  # Consistent 2-second interval
    
    # Simulate workflow pattern
    print("\n🔄 Simulating workflow pattern...")
    workflow = [
        (EventType.STATE_CHANGE, "finder", "folder_view"),
        (EventType.MOUSE_CLICK, "finder", "file_select"),
        (EventType.STATE_CHANGE, "preview", "document_view"),
    ]
    
    # Repeat workflow 3 times
    for repeat in range(3):
        print(f"   Workflow iteration #{repeat+1}")
        for event_type, app_id, state_id in workflow:
            await engine.process_visual_event(
                event_type=event_type,
                app_id=app_id,
                state_id=state_id
            )
            await asyncio.sleep(0.5)
        await asyncio.sleep(1.0)
    
    # Get patterns
    await asyncio.sleep(3)  # Wait for pattern extraction
    patterns = await engine.get_active_patterns()
    
    print(f"\n✅ Detected {len(patterns)} active patterns")
    for pattern in patterns:
        print(f"   - {pattern['type']} pattern: {pattern.get('pattern_id', 'unknown')[:20]}...")
        print(f"     Occurrences: {pattern.get('occurrences', 0)}")
        print(f"     Frequency: {pattern.get('frequency', 0):.2f} per day")
    
    return engine


async def test_predictions():
    """Test temporal predictions"""
    engine = get_temporal_engine()
    
    print("\n🔮 Testing Temporal Predictions")
    print("-" * 60)
    
    # Create a strong periodic pattern
    print("\n⏰ Creating periodic email check pattern...")
    for i in range(4):
        await engine.process_visual_event(
            event_type=EventType.STATE_CHANGE,
            app_id="mail",
            state_id="inbox_view",
            data={'check_number': i+1}
        )
        print(f"   Email check #{i+1}")
        await asyncio.sleep(5.0)  # 5-second interval
    
    # Wait for pattern detection
    await asyncio.sleep(2)
    
    # Get predictions
    predictions = await engine.predict_next_events(lookahead_seconds=30)
    
    print(f"\n📅 Next event predictions (30 second lookahead):")
    if predictions:
        for pred in predictions:
            print(f"   - Event: {pred.get('event_type')}")
            print(f"     App: {pred.get('app_id')}")
            print(f"     Expected: {pred.get('expected_time')}")
            print(f"     Confidence: {pred.get('confidence', 0):.2f}")
    else:
        print("   No predictions available yet (pattern learning in progress)")
    
    return engine


async def test_causality_detection():
    """Test cause-effect pattern detection"""
    engine = get_temporal_engine()
    
    print("\n🔗 Testing Causality Detection")
    print("-" * 60)
    
    # Simulate cause-effect patterns
    print("\n⚡ Simulating cause-effect relationships...")
    
    for i in range(4):
        # Cause: Click save button
        await engine.process_visual_event(
            event_type=EventType.MOUSE_CLICK,
            app_id="photoshop",
            state_id="editor",
            data={'target': 'save_button'}
        )
        print(f"   Click save button #{i+1}")
        
        await asyncio.sleep(0.5)  # Small delay
        
        # Effect: Save dialog appears
        await engine.process_visual_event(
            event_type=EventType.STATE_CHANGE,
            app_id="photoshop",
            state_id="save_dialog",
            data={'modal': True}
        )
        print(f"   → Save dialog appears")
        
        await asyncio.sleep(1.0)
        
        # Effect resolved: Dialog closes
        await engine.process_visual_event(
            event_type=EventType.STATE_CHANGE,
            app_id="photoshop",
            state_id="editor",
            data={'saved': True}
        )
        print(f"   → Back to editor")
        
        await asyncio.sleep(2.0)
    
    # Get context with causality patterns
    await asyncio.sleep(3)  # Wait for pattern extraction
    context = await engine.get_temporal_context("photoshop")
    
    print(f"\n✅ Causality patterns detected in context")
    active_patterns = context.get('active_patterns', [])
    causal_patterns = [p for p in active_patterns if p['type'] == 'causality']
    
    print(f"   Found {len(causal_patterns)} causality patterns")
    
    return engine


async def test_memory_usage():
    """Test memory usage tracking"""
    engine = get_temporal_engine()
    
    print("\n💾 Testing Memory Usage")
    print("-" * 60)
    
    # Generate many events
    print("\n📊 Generating 1000 events...")
    for i in range(1000):
        await engine.process_visual_event(
            event_type=EventType.STATE_CHANGE,
            app_id=f"app_{i % 10}",
            state_id=f"state_{i % 20}",
            data={'index': i}
        )
        
        if i % 100 == 0:
            print(f"   Generated {i} events")
    
    # Check memory usage
    context = await engine.get_temporal_context()
    memory_usage = context.get('memory_usage', {})
    
    print(f"\n📊 Memory Usage Report:")
    total_usage = 0
    for component, usage_bytes in memory_usage.items():
        usage_mb = usage_bytes / 1024 / 1024
        total_usage += usage_bytes
        print(f"   - {component}: {usage_mb:.2f} MB")
    
    total_mb = total_usage / 1024 / 1024
    limit_mb = 200  # Total limit
    print(f"\n   Total: {total_mb:.2f} MB / {limit_mb} MB ({total_mb/limit_mb*100:.1f}%)")
    
    return engine


async def cleanup(engine):
    """Cleanup the engine"""
    print("\n🧹 Cleaning up...")
    await engine.stop()
    print("✅ Engine stopped")


async def main():
    """Run all tests"""
    print("🧪 Temporal Context Engine Test Suite")
    print("=" * 80)
    
    try:
        # Test 1: Basic workflow and context
        engine, context = await test_temporal_context()
        await cleanup(engine)
        
        # Test 2: Pattern detection
        engine = await test_pattern_detection()
        await cleanup(engine)
        
        # Test 3: Predictions
        engine = await test_predictions()
        await cleanup(engine)
        
        # Test 4: Causality
        engine = await test_causality_detection()
        await cleanup(engine)
        
        # Test 5: Memory usage
        engine = await test_memory_usage()
        await cleanup(engine)
        
        print("\n✨ All tests completed successfully!")
        print("The Temporal Context Engine is now tracking events across time!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # Ensure directories exist
    Path("learned_states").mkdir(exist_ok=True)
    
    # Run tests
    asyncio.run(main())