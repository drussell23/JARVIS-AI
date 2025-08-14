#!/bin/bash

echo "🚀 Setting up llama.cpp for M1-optimized AI inference"
echo "====================================================="

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "❌ Homebrew not found. Please install it first:"
    echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    exit 1
fi

# Install llama.cpp
echo "📦 Installing llama.cpp..."
brew install llama.cpp

# Create models directory
echo "📁 Creating models directory..."
mkdir -p ~/Documents/ai-models

# Download a compatible model
echo "🤖 Downloading Mistral 7B model (M1 optimized)..."
echo "This may take a few minutes..."

cd ~/Documents/ai-models

# Download Mistral 7B Instruct (Q4_K_M quantization - good balance of quality/speed)
if [ ! -f "mistral-7b-instruct-v0.1.Q4_K_M.gguf" ]; then
    curl -L "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf" \
         -o mistral-7b-instruct-v0.1.Q4_K_M.gguf \
         --progress-bar
else
    echo "✅ Model already downloaded"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the llama.cpp server, run:"
echo "llama-server -m ~/Documents/ai-models/mistral-7b-instruct-v0.1.Q4_K_M.gguf -c 2048 --host 0.0.0.0 --port 8080"
echo ""
echo "Or use the provided start script:"
echo "./start_llama_server.sh"

# Create start script
cat > start_llama_server.sh << 'EOF'
#!/bin/bash
echo "Starting llama.cpp server..."
echo "Server will be available at http://localhost:8080"
echo "Press Ctrl+C to stop"

llama-server \
    -m ~/Documents/ai-models/mistral-7b-instruct-v0.1.Q4_K_M.gguf \
    -c 2048 \
    --host 0.0.0.0 \
    --port 8080 \
    -ngl 1 \
    --n-gpu-layers 1
EOF

chmod +x start_llama_server.sh

echo ""
echo "📝 Notes:"
echo "- The server runs on port 8080 by default"
echo "- You can access the web UI at http://localhost:8080"
echo "- The model uses ~4GB of RAM"
echo "- Inference speed: ~10-20 tokens/second on M1"