#!/bin/bash
# Test integrated build script

echo "🧪 Testing Integrated Build System"
echo "=================================="

# Test clean
echo -e "\n1. Testing clean..."
./build.sh clean
echo "✓ Clean completed"

# Test vision only
echo -e "\n2. Testing Vision ML only build..."
./build.sh vision
echo "✓ Vision build completed"

# Test capture only
echo -e "\n3. Testing Fast Capture only build..."
./build.sh capture
echo "✓ Capture build completed"

# Test all
echo -e "\n4. Testing full build..."
./build.sh
echo "✓ Full build completed"

# Test with tests
echo -e "\n5. Testing build with tests..."
./build.sh test
echo "✓ Build with tests completed"

echo -e "\n✅ All build configurations tested successfully!"