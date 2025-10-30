"""Setup script for building the Fast Capture C++ extension.

This module provides a custom setuptools configuration for building a high-performance
C++ screen capture extension using CMake. It includes platform-specific build
configurations for Windows, macOS, and Linux systems.

The extension is built using CMake to handle complex C++ dependencies and
cross-platform compilation requirements for optimal screen capture performance.
"""

import os
import re
import sys
import platform
import subprocess
from pathlib import Path
from typing import List, Dict, Any

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    """A setuptools Extension subclass for CMake-based C++ extensions.
    
    This class represents a C++ extension that will be built using CMake
    instead of the standard setuptools compilation process.
    
    Attributes:
        sourcedir (str): Absolute path to the source directory containing CMakeLists.txt
    """
    
    def __init__(self, name: str, sourcedir: str = '') -> None:
        """Initialize a CMake extension.
        
        Args:
            name: Name of the extension module
            sourcedir: Relative path to the source directory containing CMakeLists.txt.
                      Defaults to current directory.
        """
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    """Custom build command for CMake-based extensions.
    
    This class extends setuptools' build_ext command to handle building
    C++ extensions using CMake instead of distutils' default compiler.
    Supports cross-platform builds with platform-specific optimizations.
    """
    
    def run(self) -> None:
        """Execute the build process for all CMake extensions.
        
        Raises:
            RuntimeError: If CMake is not installed or not found in PATH
        """
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext: CMakeExtension) -> None:
        """Build a single CMake extension.
        
        Configures and builds the extension using CMake with platform-specific
        settings for optimal performance and compatibility.
        
        Args:
            ext: The CMakeExtension instance to build
            
        Raises:
            subprocess.CalledProcessError: If CMake configuration or build fails
            OSError: If required build tools are not available
        """
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            # Windows-specific CMake configuration
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            # Unix-like systems (Linux, macOS)
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        if platform.system() == "Darwin":
            # Cross-compile support for macOS
            if platform.machine() == "arm64":
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES=arm64"]
            else:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES=x86_64"]

        # Set up environment with version information
        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                               self.distribution.get_version())
        
        # Create build directory if it doesn't exist
        build_temp = self.build_temp
        if not os.path.exists(build_temp):
            os.makedirs(build_temp)

        # Configure and build the extension
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=build_temp)


setup(
    name='jarvis_fast_capture',
    version='1.0.0',
    author='JARVIS Team',
    description='High-performance screen capture for JARVIS Vision System',
    long_description='A C++ extension providing optimized screen capture capabilities '
                    'for the JARVIS Vision System. Built with CMake for cross-platform '
                    'compatibility and maximum performance.',
    ext_modules=[CMakeExtension('fast_capture')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    python_requires=">=3.7",
)