"""
Setup script for ARM64 NEON SIMD extension

Compiles C extension with ARM64 optimizations for M1 Macs
Includes pure ARM64 assembly (.s) files
"""

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import numpy as np
import platform
import os
import subprocess

# Detect ARM64/M1
is_arm64 = platform.processor() == 'arm' or 'Apple' in platform.processor()

if not is_arm64:
    print("WARNING: This extension is optimized for ARM64/M1 processors")
    print("Performance will be suboptimal on x86_64")

# Compiler flags for ARM64 NEON optimization
extra_compile_args = [
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
extra_link_args = []
if is_arm64:
    extra_link_args.append('-arch')
    extra_link_args.append('arm64')

# Custom build command to handle ARM64 assembly
class ARM64BuildExt(build_ext):
    def build_extensions(self):
        # Compile ARM64 assembly file first
        asm_file = 'arm64_simd_asm.s'
        obj_file = 'arm64_simd_asm.o'

        if os.path.exists(asm_file):
            print(f"Compiling ARM64 assembly: {asm_file}")
            subprocess.check_call([
                'clang',
                '-c',
                '-arch', 'arm64',
                '-O3',
                asm_file,
                '-o', obj_file
            ])

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
    ext_modules=[arm64_simd_ext],
    cmdclass={'build_ext': ARM64BuildExt},
    install_requires=['numpy'],
)
