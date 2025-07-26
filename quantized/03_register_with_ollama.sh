#!/bin/bash
# Script to register a fine-tuned model with Ollama

# Check if a model name was provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <model-name>"
    echo "Example: $0 my-fine-tuned-model"
    exit 1
fi

MODEL_NAME=$1
GGUF_FILE="${MODEL_NAME}-lora.gguf"
MODELFILE="Modelfile_${MODEL_NAME}"

echo "🚀 Registering ${MODEL_NAME} model with Ollama..."
echo ""
echo "Current directory: $(pwd)"

# Check if GGUF file exists
if [ -f "$GGUF_FILE" ]; then
    echo "GGUF file: $GGUF_FILE ($(ls -lh $GGUF_FILE | awk '{print $5}'))"
else
    echo "❌ Error: GGUF file '$GGUF_FILE' not found!"
    echo "Please run convert_to_gguf.py first to create the GGUF file."
    exit 1
fi

# Check if Modelfile exists
if [ -f "$MODELFILE" ]; then
    echo "Modelfile: $MODELFILE"
else
    echo "❌ Error: Modelfile '$MODELFILE' not found!"
    echo "Please run convert_to_gguf.py first to create the Modelfile."
    exit 1
fi

echo ""
echo "Registration commands:"
echo ""
echo "If Ollama is running locally:"
echo "  ollama create $MODEL_NAME -f $MODELFILE"
echo ""
echo "After creation, test with:"
echo "  ollama run $MODEL_NAME"

# Try to create it directly if ollama is available
if command -v ollama &> /dev/null; then
    echo ""
    echo "Ollama found! Attempting to create model..."
    ollama create $MODEL_NAME -f $MODELFILE
    echo ""
    echo "✅ If successful, you can now run: ollama run $MODEL_NAME"
else
    echo ""
    echo "⚠️  Ollama command not found. Please install Ollama or run the commands above where Ollama is available."
fi
