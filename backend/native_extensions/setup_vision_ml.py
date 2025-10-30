"""Setup script for Vision ML Router C++ extension.

This module provides the setup configuration for building the Vision ML Router
C++ extension, which enables ultra-fast pattern analysis for computer vision
and machine learning tasks. The extension is optimized for performance with
C++17 features and aggressive optimization flags.

The module creates a compiled extension that can be imported as 'vision_ml_router'
in Python code, providing native C++ performance for computationally intensive
vision processing operations.

Example:
    Build the extension:
        $ python setup_vision_ml.py build_ext --inplace
    
    Install the extension:
        $ pip install .
"""

from setuptools import setup, Extension
import platform
from typing import List


def _get_compile_args() -> List[str]:
    """Get platform-specific compilation arguments.
    
    Returns:
        List of compilation flags optimized for the current platform.
        
    Example:
        >>> args = _get_compile_args()
        >>> '-std=c++17' in args
        True
    """
    base_args = ['-std=c++17', '-O3', '-fPIC']
    
    # Add platform-specific optimizations if needed
    if platform.system() == 'Linux':
        base_args.extend(['-march=native'])
    elif platform.system() == 'Darwin':  # macOS
        base_args.extend(['-mmacosx-version-min=10.9'])
    
    return base_args


# C++ Vision ML Router extension
vision_ml_module = Extension(
    name='vision_ml_router',
    sources=['vision_ml_router.cpp'],
    extra_compile_args=_get_compile_args(),
    language='c++',
    include_dirs=[],  # Add include directories if needed
    libraries=[],     # Add libraries if needed
    library_dirs=[]   # Add library directories if needed
)

setup(
    name='vision_ml_router',
    version='1.0.0',
    description='C++ ML Vision Router for ultra-fast pattern analysis',
    long_description=__doc__,
    long_description_content_type='text/plain',
    author='Vision ML Team',
    author_email='team@visionml.com',
    url='https://github.com/your-org/vision-ml-router',
    ext_modules=[vision_ml_module],
    zip_safe=False,
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
        'Programming Language :: C++',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='computer-vision machine-learning cpp-extension pattern-analysis',
    project_urls={
        'Bug Reports': 'https://github.com/your-org/vision-ml-router/issues',
        'Source': 'https://github.com/your-org/vision-ml-router',
        'Documentation': 'https://vision-ml-router.readthedocs.io/',
    },
)