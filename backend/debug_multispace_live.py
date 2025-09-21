#!/usr/bin/env python3
"""Debug why multi-space isn't working in the live system"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def check_live_system():
    """Check the live system components"""
    print("🔍 Checking Live System Multi-Space Components")
    print("=" * 80)
    
    # Check 1: Is multi-space detection working?
    try:
        from vision.multi_space_intelligence import MultiSpaceIntelligenceExtension
        extension = MultiSpaceIntelligenceExtension()
        
        query = "can you see the Cursor IDE in the other desktop space?"
        should_use = extension.should_use_multi_space(query)
        intent = extension.query_detector.detect_intent(query)
        
        print(f"✅ Multi-space detection: {should_use}")
        print(f"   Intent: {intent.query_type.value}")
        print(f"   Target app: {intent.target_app}")
        print(f"   Confidence: {intent.confidence}")
    except Exception as e:
        print(f"❌ Multi-space detection error: {e}")
    
    # Check 2: Is PureVisionIntelligence configured correctly?
    try:
        from api.pure_vision_intelligence import PureVisionIntelligence
        
        # Check class attributes
        print(f"\n✅ PureVisionIntelligence available")
        print(f"   Has enable_multi_space parameter: {'enable_multi_space' in PureVisionIntelligence.__init__.__code__.co_varnames}")
        
        # Check if multi-space is enabled by default
        import inspect
        sig = inspect.signature(PureVisionIntelligence.__init__)
        params = sig.parameters
        if 'enable_multi_space' in params:
            default = params['enable_multi_space'].default
            print(f"   enable_multi_space default: {default}")
    except Exception as e:
        print(f"❌ PureVisionIntelligence error: {e}")
    
    # Check 3: Is VisionCommandHandler using multi-space?
    try:
        from api.vision_command_handler import VisionCommandHandler
        
        print(f"\n✅ VisionCommandHandler available")
        
        # Check if it initializes with multi-space
        handler_source = inspect.getsource(VisionCommandHandler.initialize_intelligence)
        if 'enable_multi_space=True' in handler_source:
            print("   ✅ Initializes PureVisionIntelligence with enable_multi_space=True")
        else:
            print("   ❌ Does NOT initialize with enable_multi_space=True")
            print("   This is likely the issue!")
            
    except Exception as e:
        print(f"❌ VisionCommandHandler error: {e}")
    
    # Check 4: Check actual initialization
    print("\n🔧 Checking actual initialization flow...")
    try:
        handler = VisionCommandHandler()
        await handler.initialize_intelligence()
        
        if handler.intelligence:
            print(f"✅ Intelligence initialized")
            print(f"   multi_space_enabled: {getattr(handler.intelligence, 'multi_space_enabled', False)}")
            print(f"   has _should_use_multi_space: {hasattr(handler.intelligence, '_should_use_multi_space')}")
            
            # Test detection
            if hasattr(handler.intelligence, '_should_use_multi_space'):
                query = "can you see the Cursor IDE in the other desktop space?"
                result = handler.intelligence._should_use_multi_space(query)
                print(f"   Detection test: {result}")
        else:
            print("❌ Intelligence not initialized")
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(check_live_system())