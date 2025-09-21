#!/bin/bash
# Install CoreLocationCLI for macOS location services

echo "Installing CoreLocationCLI for location services..."

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "Homebrew not found. Please install Homebrew first."
    exit 1
fi

# Install CoreLocationCLI
echo "Installing CoreLocationCLI via Homebrew..."
brew tap malcommac/corelocationcli
brew install corelocationcli

# Alternative: Download directly
if ! command -v CoreLocationCLI &> /dev/null; then
    echo "Trying direct download..."
    curl -L https://github.com/malcommac/CoreLocationCLI/releases/latest/download/CoreLocationCLI -o /usr/local/bin/CoreLocationCLI
    chmod +x /usr/local/bin/CoreLocationCLI
fi

# Test installation
if command -v CoreLocationCLI &> /dev/null; then
    echo "CoreLocationCLI installed successfully!"
    echo "Testing location access..."
    CoreLocationCLI -once
else
    echo "Installation failed. You may need to install manually."
fi