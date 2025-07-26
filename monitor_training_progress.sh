#!/bin/bash

echo "=== Progress Check ==="
echo

# Check dataset files
echo "Dataset Files:"
if [ -f "./practice.jsonl" ]; then
    echo "✓ Training dataset found!"
    echo "  Size: $(du -h ./practice.jsonl | cut -f1)"
    echo "  Lines: $(wc -l < ./practice.jsonl)"
else
    echo "✗ practice.jsonl not found"
fi

if [ -f "./eval.jsonl" ]; then
    echo "✓ Evaluation dataset found!"
    echo "  Size: $(du -h ./eval.jsonl | cut -f1)"
    echo "  Lines: $(wc -l < ./eval.jsonl)"
else
    echo "✗ eval.jsonl not found"
fi
echo

# Check virtual environment
echo "Virtual Environment:"
if [ -d "./mistral_env" ]; then
    echo "✓ Virtual environment exists"
    if [ -f "./mistral_env/bin/python" ]; then
        echo "  Python: $(./mistral_env/bin/python --version)"
        echo "  Unsloth: $(./mistral_env/bin/python -c 'import unsloth; print(f"v{unsloth.__version__}")' 2>/dev/null || echo 'Not installed')"
    fi
else
    echo "✗ Virtual environment not found"
fi
echo

# Check CUDA/llama.cpp setup
echo "CUDA/llama.cpp Setup:"
if [ -f "./llama.cpp/convert_lora_to_gguf.py" ]; then
    echo "✓ llama.cpp repository found!"
else
    echo "⏳ llama.cpp not found - needed for GGUF conversion"
    echo "  Clone with: git clone https://github.com/ggerganov/llama.cpp"
fi
echo

# Check running processes
echo "Training Processes:"
if [ -f "./logs/train.pid" ]; then
    TRAIN_PID=$(cat ./logs/train.pid)
    if ps -p $TRAIN_PID > /dev/null 2>&1; then
        echo "✓ Training is running (PID: $TRAIN_PID)"
    else
        echo "✗ Training process not found (stale PID file)"
    fi
else
    echo "No training process running"
fi