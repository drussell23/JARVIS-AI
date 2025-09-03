#!/bin/bash
# Install macOS frameworks for native video capture with purple indicator

echo "🎥 Installing macOS video capture frameworks..."
echo "This will enable native screen recording with the purple indicator"
echo ""

# Install the required pyobjc frameworks
echo "📦 Installing pyobjc frameworks for AVFoundation..."
pip3 install -U \
    pyobjc-core \
    pyobjc-framework-Cocoa \
    pyobjc-framework-AVFoundation \
    pyobjc-framework-CoreMedia \
    pyobjc-framework-CoreVideo \
    pyobjc-framework-Quartz

# Check if installation succeeded
echo ""
echo "✅ Checking installation..."
python3 -c "
try:
    import AVFoundation
    import CoreMedia
    import CoreVideo
    from Cocoa import NSObject
    print('✅ All macOS frameworks installed successfully!')
    print('🟣 Native video capture with purple indicator is now available!')
except ImportError as e:
    print(f'❌ Some frameworks are missing: {e}')
    print('Please try running: pip3 install pyobjc')
"

echo ""
echo "🚀 Installation complete! You can now test video streaming with:"
echo "   cd /Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend/vision"
echo "   python3 test_video_simple.py"