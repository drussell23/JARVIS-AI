# Vision Folder Cleanup Plan for 16GB macOS System

## ✅ FILES TO KEEP (Feasible & Enhance claude_vision_analyzer_main.py)

### Core Files
1. **claude_vision_analyzer_main.py** - Main vision system (fully optimized for 16GB)
2. **vision_system_claude_only.py** - Lightweight Claude-only backup system
3. **memory_efficient_vision_analyzer.py** - Memory-optimized analyzer
4. **optimized_claude_vision.py** - Speed-optimized lightweight analyzer

### Support Systems
5. **ml_intent_classifier_claude.py** - Claude-based intent classification (no local ML)
6. **lazy_vision_engine.py** - Lazy loading for memory efficiency
7. **dynamic_vision_engine.py** - Dynamic capability wrapper

### Rust Integration (Zero-Copy & Performance)
8. **rust_integration.py** - Main Rust bridge
9. **rust_bridge.py** - Low-level Rust interface
10. **jarvis-rust-core/** - Entire Rust core (memory-efficient)

### Utilities & Examples
11. **jarvis_sliding_window_example.py** - Sliding window integration example
12. **test_sliding_window_integration.py** - Integration tests
13. **screen_capture_fallback.py** - Fallback capture mechanism
14. **ocr_processor.py** - OCR utilities (if lightweight)
15. **window_detector.py** - Window detection utilities

### Documentation
16. **VISION_INTEGRATION_GUIDE.md** - Integration documentation
17. **VISION_SYSTEM_ENHANCEMENT_ROADMAP.md** - Enhancement roadmap

## ❌ FILES TO REMOVE (Not Feasible for 16GB RAM)

### Heavy ML Systems
1. **vision_system_v2.py** - Contains PyTorch, transformers (4-8GB+ RAM)
2. **phase2_integrated_system.py** - Heavy parallel processing & quantization
3. **phase3_production_system.py** - Builds on Phase 2, adds overhead
4. **intelligent_vision_integration_v2.py** - Overly complex event architecture

### Redundant/Obsolete Files
5. **continuous_vision_monitor.py** - Memory accumulation issues
6. **rust_accelerated_learning.py** - ML training (not feasible)
7. **optimized_continuous_learning.py** - Continuous learning (heavy)
8. **parallel_processing_pipeline.py** - Heavy parallel processing
9. **rust_vision_processor.py** - Redundant with rust_integration.py

### Complex Systems (Nice-to-have but not essential)
10. **multi_window_capture.py** - Complex multi-window handling
11. **dynamic_multi_window_engine.py** - Redundant with sliding window
12. **semantic_understanding_engine.py** - Heavy semantic processing
13. **proactive_vision_assistant.py** - Complex proactive system
14. **workflow_learning.py** - ML-based workflow learning

### Monitoring & Analytics (Overhead)
15. **enhanced_monitoring.py** - Monitoring overhead
16. **monitoring_observability.py** - Additional monitoring
17. **performance_optimizer.py** - Runtime optimization overhead
18. **realtime_vision_monitor.py** - Continuous monitoring

### Other Complex Systems
19. **personalization_engine.py** - ML personalization
20. **smart_caching_system.py** - Complex caching (main already has cache)
21. **smart_query_router.py** - Query routing complexity
22. **vision_plugin_system.py** - Plugin overhead
23. **vision_query_optimizer.py** - Query optimization overhead

### Workspace-Specific (Keep only if needed)
24. **workspace_analyzer.py** - Complex workspace analysis
25. **workspace_optimizer.py** - Workspace optimization
26. **jarvis_workspace_integration.py** - Workspace integration
27. **meeting_preparation.py** - Specific use case

### Safety/Security (Keep only if required)
28. **circuit_breaker.py** - Circuit breaker pattern
29. **fault_tolerance_system.py** - Fault tolerance
30. **gradual_rollout_system.py** - Gradual rollout
31. **safety_verification_framework.py** - Safety framework
32. **privacy_controls.py** - Privacy controls

## 🔧 FILES TO MODIFY

1. **continuous_screen_analyzer.py** - Add better memory management
2. **window_analysis.py** - Ensure no hardcoding
3. **window_relationship_detector.py** - Make configurable
4. **swift_vision_integration.py** - Check Swift integration feasibility

## Summary

### Keep: 17 core files + Rust core
### Remove: 32 files (heavy ML, redundant, complex systems)
### Modify: 4 files

This cleanup will:
- Free up significant disk space
- Reduce complexity
- Ensure all remaining files are optimized for 16GB RAM
- Maintain all essential vision capabilities via Claude API
- Keep performance optimizations via Rust integration

## Removal Commands

```bash
# Remove heavy ML systems
rm vision_system_v2.py
rm phase2_integrated_system.py
rm phase3_production_system.py
rm intelligent_vision_integration_v2.py

# Remove redundant files
rm continuous_vision_monitor.py
rm rust_accelerated_learning.py
rm optimized_continuous_learning.py
rm parallel_processing_pipeline.py
rm rust_vision_processor.py

# Remove complex systems
rm multi_window_capture.py
rm dynamic_multi_window_engine.py
rm semantic_understanding_engine.py
rm proactive_vision_assistant.py
rm workflow_learning.py

# Remove monitoring overhead
rm enhanced_monitoring.py
rm monitoring_observability.py
rm performance_optimizer.py
rm realtime_vision_monitor.py

# Remove other complex systems
rm personalization_engine.py
rm smart_caching_system.py
rm smart_query_router.py
rm vision_plugin_system.py
rm vision_query_optimizer.py

# Remove workspace-specific (unless needed)
rm workspace_analyzer.py
rm workspace_optimizer.py
rm jarvis_workspace_integration.py
rm meeting_preparation.py

# Remove safety/security (unless required)
rm circuit_breaker.py
rm fault_tolerance_system.py
rm gradual_rollout_system.py
rm safety_verification_framework.py
rm privacy_controls.py
```