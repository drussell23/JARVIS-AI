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