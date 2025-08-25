from setuptools import setup, Extension
import platform

# C++ Vision ML Router extension
vision_ml_module = Extension(
    'vision_ml_router',
    sources=['vision_ml_router.cpp'],
    extra_compile_args=['-std=c++17', '-O3', '-fPIC'],
    language='c++'
)

setup(
    name='vision_ml_router',
    version='1.0',
    description='C++ ML Vision Router for ultra-fast pattern analysis',
    ext_modules=[vision_ml_module],
    zip_safe=False,
)