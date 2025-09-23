#!/usr/bin/env python3
"""
Build script for Rust extensions
=================================

Compiles the Rust extensions for JARVIS ML memory management.
Run this script to build the high-performance Rust components.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_rust_installed():
    """Check if Rust is installed"""
    try:
        subprocess.run(["cargo", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_maturin():
    """Install maturin for building Python extensions"""
    try:
        import maturin
        return True
    except ImportError:
        print("Installing maturin...")
        subprocess.run([sys.executable, "-m", "pip", "install", "maturin"], check=True)
        return True

def build_extensions():
    """Build the Rust extensions"""
    rust_dir = Path(__file__).parent
    os.chdir(rust_dir)
    
    # Build in release mode for maximum performance
    print("Building Rust extensions in release mode...")
    result = subprocess.run(
        ["maturin", "build", "--release", "--interpreter", sys.executable],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print("Build failed:")
        print(result.stderr)
        return False
    
    print("Build successful!")
    
    # Find the built wheel
    wheel_dir = rust_dir / "target" / "wheels"
    wheels = list(wheel_dir.glob("*.whl"))
    
    if not wheels:
        print("No wheel file found!")
        return False
    
    wheel_path = wheels[0]
    print(f"Built wheel: {wheel_path}")
    
    # Install the wheel
    print("Installing wheel...")
    subprocess.run([sys.executable, "-m", "pip", "install", str(wheel_path), "--force-reinstall"], check=True)
    
    return True

def copy_to_backend():
    """Copy the built extension to backend directory"""
    rust_dir = Path(__file__).parent
    backend_dir = rust_dir.parent
    
    # Find the installed module
    try:
        import jarvis_rust_extensions
        src_path = Path(jarvis_rust_extensions.__file__)
        
        # Copy to backend
        dst_path = backend_dir / "rust_extensions.so"  # or .pyd on Windows
        shutil.copy2(src_path, dst_path)
        print(f"Copied extension to: {dst_path}")
        
        return True
    except ImportError:
        print("Failed to import built extension")
        return False

def main():
    """Main build process"""
    print("JARVIS Rust Extensions Build Script")
    print("===================================\n")
    
    # Check prerequisites
    if not check_rust_installed():
        print("ERROR: Rust is not installed!")
        print("Please install Rust from: https://rustup.rs/")
        sys.exit(1)
    
    if not install_maturin():
        print("ERROR: Failed to install maturin")
        sys.exit(1)
    
    # Build extensions
    if not build_extensions():
        print("ERROR: Build failed")
        sys.exit(1)
    
    # Copy to backend
    if not copy_to_backend():
        print("WARNING: Failed to copy extension to backend directory")
    
    print("\n✅ Rust extensions built successfully!")
    print("\nTo use the extensions, the Python code will automatically import them.")
    print("If the import fails, Python will fall back to pure Python implementations.")

if __name__ == "__main__":
    main()