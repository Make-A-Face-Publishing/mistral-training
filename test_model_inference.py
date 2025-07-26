#!/usr/bin/env python3
"""
Test the fine-tuned model with a prompt from command line
Usage: python test_model.py "Your prompt here"
"""
import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

def test_model(prompt: str, model_path: str = "./mistral_instruct_finetuned"):
    """Load model and generate response to prompt"""
    
    # Set up environment
    # HF_TOKEN should be set as environment variable before running
    # export HF_TOKEN="your_huggingface_token_here"
    
    print(f"Loading model from: {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left"
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model in BF16 (same as training)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    model.eval()
    
    # Format prompt in instruction format
    formatted_prompt = f"<s>[INST]{prompt}[/INST]"
    
    print(f"\nPrompt: {prompt}")
    print("-" * 50)
    
    # Tokenize
    inputs = tokenizer(
        formatted_prompt, 
        return_tensors="pt",
        truncation=True,
        max_length=8192
    ).to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.8,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode and extract response
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the response (after [/INST])
    if "[/INST]" in full_output:
        response = full_output.split("[/INST]", 1)[1].strip()
    else:
        response = full_output
    
    print(f"Response: {response}")
    print("-" * 50)
    
    return response

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_model.py \"Your prompt here\" [model_path]")
        print("Example: python test_model.py \"Write a short story about a robot\"")
        sys.exit(1)
    
    prompt = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "./mistral_instruct_finetuned"
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        print("Make sure training is complete and model is saved.")
        sys.exit(1)
    
    # Test the model
    test_model(prompt, model_path)

if __name__ == "__main__":
    main()