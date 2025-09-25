#!/usr/bin/env python3
"""Debug learned apps in pattern learner"""

import sys
sys.path.append('/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')

from api.unified_command_processor import UnifiedCommandProcessor

def debug_learned_apps():
    """Check what apps are learned"""
    processor = UnifiedCommandProcessor()
    
    print("Debugging Learned Apps")
    print("=" * 50)
    
    # Check pattern learner
    if hasattr(processor.pattern_learner, 'learned_apps'):
        print(f"\nLearned apps: {processor.pattern_learner.learned_apps}")
    
    # Check if safari is learned
    test_apps = ['safari', 'Safari', 'music', 'Music', 'weather', 'Weather']
    for app in test_apps:
        is_learned = processor.pattern_learner.is_learned_app(app)
        print(f"\nIs '{app}' learned? {is_learned}")
    
    # Check all app verbs
    print(f"\nApp verbs: {processor.pattern_learner.app_verbs}")
    
    # Try to get dynamic controller apps
    try:
        from system_control.dynamic_app_controller import get_dynamic_app_controller
        controller = get_dynamic_app_controller()
        
        print(f"\nTotal installed apps found: {len(controller.installed_apps_cache)}")
        
        # Check specific apps
        for app in ['safari', 'music', 'weather']:
            app_info = controller.find_app_by_name(app)
            if app_info:
                print(f"\n{app}: Found as '{app_info['name']}' at {app_info.get('path', 'unknown path')}")
            else:
                print(f"\n{app}: NOT FOUND")
                
    except Exception as e:
        print(f"\nError accessing dynamic controller: {e}")

if __name__ == "__main__":
    debug_learned_apps()