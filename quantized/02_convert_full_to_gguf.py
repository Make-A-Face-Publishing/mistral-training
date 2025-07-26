#!/usr/bin/env python3
"""
Convert full fine-tuned model to GGUF format and quantize to 4-bit

Usage:
    python convert_full_model_to_gguf.py model_path [output_name]
    
Example:
    python convert_full_model_to_gguf.py full_finetuned_model mistral-7b-custom
"""

import sys
import subprocess
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ {description} failed:")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False
    
    print(f"✅ {description} completed successfully")
    return True

def convert_to_gguf(model_path: str, output_name: str = None):
    """Convert full model to GGUF and quantize"""
    
    model_dir = Path(model_path)
    if not model_dir.exists():
        print(f"❌ Error: {model_dir} not found")
        return False
    
    # Default output name
    if output_name is None:
        output_name = f"{model_dir.name}-gguf"
    
    # Check for converter
    converter = Path("llama.cpp/convert_hf_to_gguf.py")
    if not converter.exists():
        print("❌ Error: llama.cpp/convert_hf_to_gguf.py not found")
        print("Please ensure llama.cpp is built successfully")
        return False
    
    # Check for quantizer
    quantizer = Path("llama.cpp/build/bin/llama-quantize")
    if not quantizer.exists():
        print("❌ Error: llama.cpp/build/bin/llama-quantize not found")
        print("Please ensure llama.cpp is built successfully")
        return False
    
    # Step 1: Convert HF model to GGUF (F16)
    f16_output = f"{output_name}-f16.gguf"
    cmd = [
        sys.executable, 
        str(converter), 
        str(model_dir),
        "--outfile", f16_output,
        "--outtype", "f16"
    ]
    
    if not run_command(cmd, f"Converting {model_dir} to F16 GGUF"):
        return False
    
    # Step 2: Quantize to Q4_K_M (4-bit)
    q4_output = f"{output_name}-q4_k_m.gguf"
    cmd = [
        str(quantizer),
        f16_output,
        q4_output,
        "Q4_K_M"
    ]
    
    if not run_command(cmd, f"Quantizing {f16_output} to Q4_K_M"):
        return False
    
    # Step 3: Create Ollama Modelfile
    modelfile_content = f"""FROM ./{q4_output}

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 8192

SYSTEM You are a helpful AI assistant that has been fine-tuned on custom data to provide accurate and helpful responses.

TEMPLATE \"\"\"<s>[INST] {{ .Prompt }} [/INST]\"\"\"
"""
    
    modelfile_name = f"Modelfile_{output_name}"
    with open(modelfile_name, 'w') as f:
        f.write(modelfile_content)
    
    # Step 4: Create package directory and files
    package_dir = Path(f"{output_name}_package")
    package_dir.mkdir(exist_ok=True)
    
    # Copy files to package directory
    import shutil
    
    # Copy the quantized model
    shutil.copy(q4_output, package_dir / q4_output)
    
    # Copy the Modelfile
    shutil.copy(modelfile_name, package_dir / modelfile_name)
    
    # Create README for the package
    readme_content = f"""# {output_name} - Custom Fine-tuned Mistral 7B Model

## Files
- `{q4_output}` - 4-bit quantized GGUF model file
- `{modelfile_name}` - Ollama Modelfile

## Usage with Ollama

1. Install Ollama: https://ollama.ai/download
2. Create the model:
   ```bash
   ollama create {output_name} -f {modelfile_name}
   ```
3. Run the model:
   ```bash
   ollama run {output_name}
   ```

## Model Details
- Base Model: Mistral 7B v0.3
- Training: Full precision fine-tuning with BF16
- Quantization: 4-bit (Q4_K_M)
- Context Length: 8192 tokens
- Size: ~4GB (quantized)

## Example Usage
```bash
ollama run {output_name} "What is machine learning?"
```
"""
    
    with open(package_dir / "README.md", 'w') as f:
        f.write(readme_content)
    
    print(f"✅ Success! Created package in {package_dir}:")
    print(f"   - {q4_output}")
    print(f"   - {modelfile_name}")
    print(f"   - README.md")
    
    # Clean up intermediate files
    if Path(f16_output).exists():
        os.remove(f16_output)
        print(f"🧹 Cleaned up intermediate file: {f16_output}")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    model_path = sys.argv[1]
    output_name = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = convert_to_gguf(model_path, output_name)
    sys.exit(0 if success else 1)