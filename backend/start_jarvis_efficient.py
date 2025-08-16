#!/usr/bin/env python3
"""
Memory-efficient JARVIS startup for M1 MacBook Pro
"""

import os
import sys
import psutil
import asyncio
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def optimize_memory_before_start():
    """Run memory optimization before starting JARVIS"""
    mem = psutil.virtual_memory()
    logger.info(f"Current memory usage: {mem.percent:.1f}%")
    
    if mem.percent > 70:
        logger.warning("⚠️  High memory usage detected!")
        logger.info("Running memory optimization...")
        
        # Import and run memory optimization
        try:
            from memory.intelligent_memory_optimizer import IntelligentMemoryOptimizer
            optimizer = IntelligentMemoryOptimizer()
            success, report = await optimizer.optimize_for_langchain(aggressive=True)
            
            if success:
                logger.info(f"✅ Memory optimization successful! Freed {report['memory_freed_mb']:.0f} MB")
                logger.info(f"Memory now at {report['final_percent']:.1f}%")
            else:
                logger.warning("⚠️  Memory optimization incomplete")
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")

async def start_jarvis_minimal():
    """Start JARVIS with minimal memory footprint"""
    logger.info("Starting JARVIS with minimal configuration...")
    
    try:
        # Import only what we need
        from core.jarvis_core import JARVISCore
        from memory.memory_manager import M1MemoryManager
        
        # Create memory manager first
        memory_manager = M1MemoryManager()
        await memory_manager.start_monitoring()
        
        # Create JARVIS core with minimal settings
        jarvis = JARVISCore(
            config_path=None,  # Use defaults
            memory_manager=memory_manager,
            auto_load_models=False,  # Don't load models upfront
            start_minimal=True  # Start in minimal mode
        )
        
        # Initialize with minimal services
        await jarvis.initialize()
        
        logger.info("✅ JARVIS started successfully!")
        logger.info(f"Memory usage: {psutil.virtual_memory().percent:.1f}%")
        
        # Show loaded components
        if hasattr(jarvis, 'chatbot') and hasattr(jarvis.chatbot, '_loaded_components'):
            logger.info(f"Loaded components: {jarvis.chatbot._loaded_components}")
        
        return jarvis, memory_manager
        
    except Exception as e:
        logger.error(f"Failed to start JARVIS: {e}")
        raise

async def interactive_session(jarvis):
    """Run an interactive JARVIS session"""
    print("\n🤖 JARVIS is ready! (Memory-efficient mode)")
    print("Type 'exit' or 'quit' to stop.")
    print("Type 'status' to see memory usage.")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                break
            
            if user_input.lower() == 'status':
                mem = psutil.virtual_memory()
                print(f"\n📊 Status:")
                print(f"   Memory: {mem.percent:.1f}% ({mem.used / 1024**3:.1f}GB / {mem.total / 1024**3:.1f}GB)")
                if hasattr(jarvis, 'chatbot'):
                    print(f"   Mode: {jarvis.chatbot.current_mode.value}")
                    if hasattr(jarvis.chatbot, '_loaded_components'):
                        print(f"   Components: {jarvis.chatbot._loaded_components}")
                continue
            
            # Get response
            response = await jarvis.process_command(user_input)
            print(f"\nJARVIS: {response}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error processing command: {e}")
            print(f"\n❌ Error: {e}")

async def main():
    """Main entry point"""
    print("🚀 JARVIS Efficient Startup")
    print("=" * 50)
    
    # Check initial memory
    initial_mem = psutil.virtual_memory().percent
    print(f"Initial memory usage: {initial_mem:.1f}%")
    
    # Optimize memory if needed
    await optimize_memory_before_start()
    
    # Start JARVIS
    try:
        jarvis, memory_manager = await start_jarvis_minimal()
        
        # Run interactive session
        await interactive_session(jarvis)
        
    except Exception as e:
        logger.error(f"JARVIS startup failed: {e}")
        return False
    
    finally:
        # Cleanup
        logger.info("Shutting down JARVIS...")
        try:
            if 'jarvis' in locals():
                await jarvis.shutdown()
            if 'memory_manager' in locals():
                await memory_manager.stop_monitoring()
        except:
            pass
        
        # Final memory
        final_mem = psutil.virtual_memory().percent
        print(f"\nFinal memory usage: {final_mem:.1f}%")
        print(f"Memory change: {final_mem - initial_mem:+.1f}%")
    
    return True

if __name__ == "__main__":
    # Set environment for efficient operation
    os.environ["PREFER_LANGCHAIN"] = "0"  # Start without LangChain
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    # Run the main function
    success = asyncio.run(main())
    sys.exit(0 if success else 1)