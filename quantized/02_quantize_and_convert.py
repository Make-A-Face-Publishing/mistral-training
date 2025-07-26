#!/usr/bin/env python3
"""
Quantize and convert a full FP16 model to GGUF format
"""
import os
import sys
import subprocess
from pathlib import Path
import shutil
import json

def check_llama_cpp():
    """Check if llama.cpp is available and built"""
    llama_cpp_path = Path("./llama.cpp")
    if not llama_cpp_path.exists():
        print("llama.cpp not found. Cloning repository...")
        subprocess.run([
            "git", "clone", "https://github.com/ggerganov/llama.cpp"
        ], check=True)
    
    # Check if quantize tool exists
    quantize_path = llama_cpp_path / "build" / "bin" / "llama-quantize"
    convert_path = llama_cpp_path / "convert_hf_to_gguf.py"
    
    if not quantize_path.exists():
        print("llama.cpp quantize tool not found. Please build llama.cpp first.")
        print("Run: cd llama.cpp && cmake -B build && cmake --build build --config Release")
        return False
    
    if not convert_path.exists():
        print("convert_hf_to_gguf.py not found in llama.cpp")
        return False
    
    return True

def convert_to_gguf(model_path: Path, output_name: str, quantization: str = "Q4_K_M"):
    """Convert HF model to GGUF and quantize"""
    if not check_llama_cpp():
        sys.exit(1)
    
    model_path = Path(model_path).resolve()
    output_dir = Path("gguf_models")
    output_dir.mkdir(exist_ok=True)
    
    # Step 1: Convert HF model to GGUF FP16
    print(f"\nConverting {model_path} to GGUF format...")
    fp16_output = output_dir / f"{output_name}.fp16.gguf"
    
    convert_cmd = [
        "python", "./llama.cpp/convert_hf_to_gguf.py",
        str(model_path),
        "--outfile", str(fp16_output),
        "--outtype", "f16"
    ]
    
    print(f"Running: {' '.join(convert_cmd)}")
    result = subprocess.run(convert_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error converting model: {result.stderr}")
        return False
    
    # Step 2: Quantize the model
    print(f"\nQuantizing to {quantization}...")
    quantized_output = output_dir / f"{output_name}.{quantization}.gguf"
    
    quantize_cmd = [
        "./llama.cpp/build/bin/llama-quantize",
        str(fp16_output),
        str(quantized_output),
        quantization
    ]
    
    print(f"Running: {' '.join(quantize_cmd)}")
    result = subprocess.run(quantize_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error quantizing model: {result.stderr}")
        return False
    
    # Step 3: Create Modelfile for Ollama
    modelfile_path = output_dir / f"Modelfile.{output_name}"
    modelfile_content = f"""FROM {quantized_output.name}

# Set temperature
PARAMETER temperature 0.7

# Set context window
PARAMETER num_ctx 4096

# Set stop tokens
PARAMETER stop "<s>"
PARAMETER stop "</s>"
PARAMETER stop "[INST]"
PARAMETER stop "[/INST]"

# System prompt (optional)
SYSTEM You are a helpful AI assistant trained on custom data.

# Chat template
TEMPLATE \"\"\"{{{{ if .System }}}}<s>[INST] {{{{.System}}}} [/INST]</s>{{{{ end }}}}{{{{ if .Prompt }}}}<s>[INST] {{{{.Prompt}}}} [/INST]{{{{ end }}}}\"\"\"
"""
    
    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)
    
    # Clean up FP16 file to save space
    if fp16_output.exists():
        print(f"\nRemoving intermediate FP16 file to save space...")
        fp16_output.unlink()
    
    print(f"\n{'='*60}")
    print("Conversion complete!")
    print(f"Quantized model: {quantized_output}")
    print(f"Modelfile: {modelfile_path}")
    print(f"\nTo use with Ollama:")
    print(f"1. cd {output_dir}")
    print(f"2. ollama create {output_name} -f Modelfile.{output_name}")
    print(f"3. ollama run {output_name}")
    print(f"{'='*60}")
    
    return True

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python quantize_model.py <model_path> [output_name] [quantization]")
        print("\nExample:")
        print("  python quantize_model.py ./full_finetuned_model my-mistral Q4_K_M")
        print("\nQuantization options:")
        print("  Q4_0   - 4-bit (legacy, not recommended)")
        print("  Q4_K_S - 4-bit (small, ~4.3GB)")
        print("  Q4_K_M - 4-bit (medium, recommended, ~4.8GB)")
        print("  Q5_K_S - 5-bit (small, ~5.3GB)")
        print("  Q5_K_M - 5-bit (medium, ~5.6GB)")
        print("  Q6_K   - 6-bit (~6.6GB)")
        print("  Q8_0   - 8-bit (~8GB)")
        sys.exit(1)
    
    model_path = Path(sys.argv[1])
    output_name = sys.argv[2] if len(sys.argv) > 2 else "mistral-custom"
    quantization = sys.argv[3] if len(sys.argv) > 3 else "Q4_K_M"
    
    if not model_path.exists():
        print(f"Error: Model path {model_path} does not exist!")
        sys.exit(1)
    
    # Check for required files
    required_files = ["config.json", "model.safetensors", "tokenizer.json"]
    missing_files = [f for f in required_files if not (model_path / f).exists()]
    
    # Check for pytorch files as alternative
    if "model.safetensors" in missing_files:
        pytorch_files = list(model_path.glob("pytorch_model*.bin"))
        if pytorch_files:
            missing_files.remove("model.safetensors")
    
    if missing_files:
        print(f"Error: Missing required files in {model_path}: {missing_files}")
        sys.exit(1)
    
    convert_to_gguf(model_path, output_name, quantization)

if __name__ == "__main__":
    main()