# M1 Mac Setup Guide for AI-Powered Chatbot

This guide helps you set up the AI-Powered Chatbot on M1 Macs using llama.cpp for native performance without bus errors.

## Why llama.cpp?

The standard PyTorch implementation causes bus errors on M1 Macs when generating text. llama.cpp provides:
- Native M1 optimization (no bus errors!)
- Fast inference (10-20 tokens/second)
- Low memory usage (4-8GB)
- Easy setup and maintenance

## Quick Setup

### 1. Install llama.cpp and download a model

```bash
# Run the setup script
./setup_llama_m1.sh
```

This will:
- Install llama.cpp via Homebrew
- Download the Mistral 7B model (optimized for M1)
- Create start scripts

### 2. Start the llama.cpp server

In one terminal:
```bash
./start_llama_server.sh
```

### 3. Start the chatbot backend

In another terminal:
```bash
./run_server.sh
```

The system will automatically detect you're on M1 and use the llama.cpp backend!

## Testing

Visit http://localhost:8000/docs to test the API.

## Alternative Models

You can use other models with llama.cpp:

### Smaller/Faster Models
- **Mistral 7B** (default) - Best balance
- **Llama 2 7B** - Good quality
- **Phi-2** - Very fast, smaller

### Download Other Models

```bash
# Example: Download Llama 2 7B
cd ~/Documents/ai-models
curl -L "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf" -o llama-2-7b-chat.Q4_K_M.gguf
```

Then update `start_llama_server.sh` to use the new model.

## Troubleshooting

### "llama.cpp server not found"
- Make sure you ran `./setup_llama_m1.sh`
- Check if the server is running: `curl http://localhost:8080/health`

### "Model not found"
- Ensure the model downloaded completely
- Check the path in `start_llama_server.sh`

### Performance Issues
- Reduce context size: `-c 1024` instead of `-c 2048`
- Use a smaller model (Q4_K_S instead of Q4_K_M)

## Benefits Over PyTorch

1. **No Bus Errors** - Native M1 code, no PyTorch compatibility issues
2. **Better Performance** - 2-3x faster than PyTorch on M1
3. **Lower Memory** - Uses 50% less RAM
4. **Stable** - No random crashes during generation

## Advanced Configuration

Edit `chatbot_m1_llama.py` to adjust:
- Temperature (creativity)
- Max tokens (response length)
- Top-k/Top-p (response quality)

## Support

If you encounter issues:
1. Check llama.cpp server logs
2. Verify model path is correct
3. Ensure ports 8080 and 8000 are free

The M1-optimized setup provides a much better experience than fighting with PyTorch compatibility!