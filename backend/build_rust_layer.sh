#!/bin/bash
# Build the Rust performance layer

echo "🔨 Building Rust performance layer..."
cd rust_performance

# Install maturin if not present
if ! command -v maturin &> /dev/null; then
    echo "Installing maturin..."
    pip install maturin
fi

# Build in release mode
maturin build --release

# Install the built wheel
pip install target/wheels/*.whl --force-reinstall

echo "✅ Rust performance layer built and installed!"
