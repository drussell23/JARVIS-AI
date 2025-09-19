#!/usr/bin/env python3
"""Trace the actual ValueError in the live system"""

import sys
import logging

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Patch the multi_space_intelligence module to add detailed logging
import vision.multi_space_intelligence as msi

# Store original methods
original_process = msi.MultiSpaceIntelligenceExtension.process_multi_space_query
original_generate = msi.ResponseBuilder._generate_response

def patched_process(self, query, window_data):
    """Patched version with detailed logging"""
    print(f"\n[TRACE] process_multi_space_query called with query: '{query}'")
    print(f"[TRACE] window_data type: {type(window_data)}")
    print(f"[TRACE] window_data keys: {window_data.keys() if isinstance(window_data, dict) else 'Not a dict'}")
    
    try:
        # Get windows
        windows = window_data.get('windows', [])
        print(f"[TRACE] Found {len(windows)} windows")
        if windows:
            first_window = windows[0]
            print(f"[TRACE] First window type: {type(first_window)}")
            print(f"[TRACE] First window has 'get' attr: {hasattr(first_window, 'get')}")
            print(f"[TRACE] First window has 'app_name' attr: {hasattr(first_window, 'app_name')}")
            if hasattr(first_window, '__dict__'):
                print(f"[TRACE] First window attributes: {list(first_window.__dict__.keys())[:5]}")
        
        # Call original
        result = original_process(self, query, window_data)
        print(f"[TRACE] process_multi_space_query succeeded")
        return result
    except Exception as e:
        print(f"[TRACE] ERROR in process_multi_space_query: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise

def patched_generate(self, intent, window_data, space_cache=None):
    """Patched version with detailed logging"""
    print(f"\n[TRACE] _generate_response called")
    print(f"[TRACE] intent type: {intent.type if hasattr(intent, 'type') else 'No type'}")
    print(f"[TRACE] intent query_type: {intent.query_type if hasattr(intent, 'query_type') else 'No query_type'}")
    
    try:
        result = original_generate(self, intent, window_data, space_cache)
        print(f"[TRACE] _generate_response succeeded")
        return result
    except Exception as e:
        print(f"[TRACE] ERROR in _generate_response: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise

# Apply patches
msi.MultiSpaceIntelligenceExtension.process_multi_space_query = patched_process
msi.ResponseBuilder._generate_response = patched_generate

print("🔍 Error tracing patches applied to multi_space_intelligence")
print("Now run your JARVIS query to see detailed trace...")

# Keep script running
import time
while True:
    time.sleep(1)