#!/bin/bash

# Direct inference test script for checkpoint models
# Usage: ./talk.sh "your message here"

if [ $# -eq 0 ]; then
    echo "Usage: ./talk.sh \"your message here\""
    echo "Example: ./talk.sh \"Write a short story about a dragon\""
    exit 1
fi

# Model path - change this to test different checkpoints
MODEL_PATH="/workspace/train/mistral_fast_finetuned/checkpoint-100"

# System prompt for writing assistant
SYSTEM_PROMPT="You are a helpful creative writing assistant. You excel at crafting engaging stories, vivid descriptions, and compelling characters. Follow the user's instructions and provide creative, well-written responses that capture the reader's imagination."

# User message
USER_MESSAGE="$*"

# Combined prompt
FULL_PROMPT="${SYSTEM_PROMPT}

User: ${USER_MESSAGE}"

echo "Testing checkpoint: $MODEL_PATH"
echo "Prompt: $USER_MESSAGE"
echo "=" 
echo ""

# Run Python inference with virtual environment
source .venv/bin/activate && python << EOF
import torch
from transformers import AutoTokenizer, MistralForCausalLM, MistralConfig
import json
import os
from safetensors.torch import load_file

# Set up environment
os.environ["TRANSFORMERS_CACHE"] = "/workspace/train/cache"
os.environ["HF_HOME"] = "/workspace/train/cache"

# Load model and tokenizer
print("Loading model...")
model_path = "$MODEL_PATH"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

print("Loading config...")
with open(f"{model_path}/config.json", "r") as f:
    config_dict = json.load(f)
config = MistralConfig(**config_dict)

print("Creating model from config...")
model = MistralForCausalLM(config).to(torch.bfloat16)

print("Loading weights from safetensors...")
state_dict = {}
for i in range(1, 4):
    file_path = f"{model_path}/model-{i:05d}-of-00003.safetensors"
    print(f"Loading {file_path}")
    state_dict.update(load_file(file_path))

print("Loading state dict into model...")
model.load_state_dict(state_dict)

print("Moving to GPU...")
model = model.cuda()

# Prepare prompt
prompt = """$FULL_PROMPT"""

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate
print("Generating response...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.8,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

# Decode response
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
response_only = response[len(prompt):].strip()

print("\n" + "="*50)
print("RESPONSE:")
print("="*50)
print(response_only)
print("="*50)
EOF