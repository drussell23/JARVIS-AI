#!/bin/bash

echo "🦀 Installing Rust for JARVIS Performance Layer"
echo "=============================================="

# Check if Rust is already installed
if command -v rustc &> /dev/null; then
    echo "✅ Rust is already installed: $(rustc --version)"
    exit 0
fi

# Install Rust
echo "📦 Installing Rust via rustup..."
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup.sh
sh /tmp/rustup.sh -y --default-toolchain stable

# Add Rust to PATH for current session
export PATH="$HOME/.cargo/bin:$PATH"

# Verify installation
if command -v rustc &> /dev/null; then
    echo "✅ Rust installed successfully: $(rustc --version)"
    echo ""
    echo "⚠️  Please run this command to update your PATH:"
    echo '    export PATH="$HOME/.cargo/bin:$PATH"'
    echo ""
    echo "Or add it to your shell profile (~/.bashrc, ~/.zshrc, etc.)"
else
    echo "❌ Rust installation failed"
    exit 1
fi