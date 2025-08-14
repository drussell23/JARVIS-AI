#!/usr/bin/env python3
"""
Cleanup script for AI-Powered Chatbot
Removes temporary files and logs
"""
import os
import shutil
from pathlib import Path

def cleanup_backend():
    """Clean up backend temporary files"""
    backend_dir = Path(__file__).parent
    
    # Files to remove
    temp_files = [
        # Log files
        "*.log",
        "nohup.out",
        
        # Temporary/backup files
        "chatbot_original.py",
        "main_original.py",
        "test_integration.py",
        
        # Old server scripts (we'll use run_server.sh)
        "install_deps.sh",
        "start_server.sh",
        
        # Cache directories
        "__pycache__",
        ".pytest_cache",
        "*.pyc",
        
        # Temporary databases (if they exist)
        "temp_*.db",
        "test_*.db"
    ]
    
    removed_count = 0
    
    print("🧹 Cleaning up backend directory...")
    
    # Remove files matching patterns
    for pattern in temp_files:
        if "*" in pattern:
            # Handle glob patterns
            for file in backend_dir.glob(pattern):
                if file.is_file():
                    try:
                        file.unlink()
                        print(f"  ✓ Removed: {file.name}")
                        removed_count += 1
                    except Exception as e:
                        print(f"  ⚠️  Failed to remove {file.name}: {e}")
        else:
            # Handle specific files
            file_path = backend_dir / pattern
            if file_path.exists():
                try:
                    if file_path.is_dir():
                        shutil.rmtree(file_path)
                    else:
                        file_path.unlink()
                    print(f"  ✓ Removed: {pattern}")
                    removed_count += 1
                except Exception as e:
                    print(f"  ⚠️  Failed to remove {pattern}: {e}")
    
    # Clean up __pycache__ directories recursively
    for pycache in backend_dir.rglob("__pycache__"):
        try:
            shutil.rmtree(pycache)
            print(f"  ✓ Removed: {pycache.relative_to(backend_dir)}")
            removed_count += 1
        except Exception as e:
            print(f"  ⚠️  Failed to remove {pycache}: {e}")
    
    print(f"\n✅ Cleanup complete! Removed {removed_count} items.")
    
    # List remaining important files
    print("\n📁 Important files preserved:")
    important_files = [
        "main.py",
        "chatbot.py",
        "requirements.txt",
        "run_server.sh",
        "README.md"
    ]
    
    for file in important_files:
        if (backend_dir / file).exists():
            print(f"  • {file}")

if __name__ == "__main__":
    cleanup_backend()