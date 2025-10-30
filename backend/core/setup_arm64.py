"""
Setup script for ARM64 NEON SIMD extension.

This module provides a setuptools configuration for compiling C extensions with
ARM64 optimizations specifically for M1 Macs. It includes support for pure ARM64
assembly (.s) files and provides optimal performance on Apple Silicon processors.

The setup script automatically detects ARM64 processors and applies appropriate
compiler optimizations including NEON SIMD instructions, M1-specific tuning,
and custom assembly compilation.

Example:
    Build the extension:
        $ python setup_arm64.py build_ext --inplace
    
    Install the extension:
        $ pip install .

Attributes:
    is_arm64 (bool): True if running on ARM64 processor, False otherwise.
    extra_compile_args (list): Compiler flags for ARM64 optimization.
    extra_link_args (list): Linker flags for ARM64 architecture.
    arm64_simd_ext (Extension): The setuptools Extension object.
"""

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import numpy as np
import platform
import os
import subprocess
from typing import List, Optional

# Detect ARM64/M1
is_arm64 = platform.processor() == 'arm' or 'Apple' in platform.processor()

if not is_arm64:
    print("WARNING: This extension is optimized for ARM64/M1 processors")
    print("Performance will be suboptimal on x86_64")

# Compiler flags for ARM64 NEON optimization
extra_compile_args: List[str] = [
    '-O3',                    # Maximum optimization
    '-ffast-math',            # Fast floating-point math
    '-funroll-loops',         # Loop unrolling
    '-fvectorize',            # Auto-vectorization
]

# M1-specific optimizations (only when compiling for ARM64)
if is_arm64:
    extra_compile_args.extend([
        '-arch', 'arm64',         # Force ARM64 only
        '-mcpu=apple-m1',         # M1-specific tuning
        '-DARM_NEON',             # Enable NEON intrinsics
    ])

# Link-time optimization
extra_link_args: List[str] = []
if is_arm64:
    extra_link_args.append('-arch')
    extra_link_args.append('arm64')


class ARM64BuildExt(build_ext):
    """Custom build command to handle ARM64 assembly compilation.
    
    This class extends the standard build_ext command to compile ARM64 assembly
    files before building the C extension. It automatically detects .s files
    and compiles them using clang with ARM64-specific flags.
    
    Attributes:
        extensions: List of Extension objects to build (inherited from build_ext).
    
    Example:
        The class is used automatically when building:
            $ python setup_arm64.py build_ext --inplace
    """
    
    def build_extensions(self) -> None:
        """Build extensions with ARM64 assembly support.
        
        Compiles ARM64 assembly files first, then proceeds with standard
        C extension compilation. Assembly object files are automatically
        linked with the final extension.
        
        Raises:
            subprocess.CalledProcessError: If assembly compilation fails.
            FileNotFoundError: If clang compiler is not found.
        
        Example:
            This method is called automatically during the build process.
        """
        # Compile ARM64 assembly file first
        asm_file: str = 'arm64_simd_asm.s'
        obj_file: str = 'arm64_simd_asm.o'

        if os.path.exists(asm_file):
            print(f"Compiling ARM64 assembly: {asm_file}")
            try:
                subprocess.check_call([
                    'clang',
                    '-c',
                    '-arch', 'arm64',
                    '-O3',
                    asm_file,
                    '-o', obj_file
                ])
            except subprocess.CalledProcessError as e:
                raise subprocess.CalledProcessError(
                    e.returncode, 
                    e.cmd, 
                    f"Failed to compile ARM64 assembly file {asm_file}"
                ) from e
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    "clang compiler not found. Please install Xcode command line tools."
                ) from e

            # Add object file to extra link objects
            for ext in self.extensions:
                if not hasattr(ext, 'extra_objects'):
                    ext.extra_objects = []
                ext.extra_objects.append(obj_file)

        # Call parent build
        build_ext.build_extensions(self)


# Extension module with pure ARM64 assembly
arm64_simd_ext = Extension(
    'arm64_simd',
    sources=['arm64_simd.c'],  # Only C file in sources
    include_dirs=[np.get_include()],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

setup(
    name='arm64_simd',
    version='1.0.0',
    description='ARM64 NEON SIMD optimizations for JARVIS ML with pure assembly',
    long_description=__doc__,
    long_description_content_type='text/plain',
    ext_modules=[arm64_simd_ext],
    cmdclass={'build_ext': ARM64BuildExt},
    install_requires=['numpy'],
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: C',
        'Programming Language :: Assembly',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Operating System :: MacOS :: MacOS X',
    ],
    keywords='arm64 neon simd optimization machine-learning apple-silicon m1',
    author='JARVIS Development Team',
    maintainer='JARVIS Development Team',
    platforms=['macOS'],
)