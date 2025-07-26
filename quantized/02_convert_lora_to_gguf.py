#!/usr/bin/env python3
"""
Convert existing QLoRA checkpoint to GGUF format for Ollama

Usage:
    python convert_to_gguf.py checkpoint_path [output_name]
    
Example:
    python convert_to_gguf.py fine_tuned_model/checkpoint-264 my-model
"""

import sys
import subprocess
from pathlib import Path

def convert_to_gguf(checkpoint_path: str, output_name: str = None):
    """Convert checkpoint to GGUF using llama.cpp"""
    
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        print(f"❌ Error: {checkpoint} not found")
        return False
    
    # Default output name from checkpoint
    if output_name is None:
        output_name = f"{checkpoint.parent.name}-{checkpoint.name}"
    
    output_file = f"{output_name}-lora.gguf"
    
    # Check for converter
    converter = Path("llama.cpp/convert_lora_to_gguf.py")
    if not converter.exists():
        print("❌ Error: llama.cpp/convert_lora_to_gguf.py not found")
        print("Please clone llama.cpp in this directory first")
        return False
    
    # Run conversion
    cmd = [sys.executable, str(converter), "--outfile", output_file, str(checkpoint)]
    print(f"🔧 Converting {checkpoint} → {output_file}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Conversion failed:")
        print(result.stderr)
        return False
    
    # Create Modelfile
    modelfile = f"""FROM mistral:instruct
ADAPTER ./{output_file}

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 1024

SYSTEM You are a helpful AI assistant that has been fine-tuned on custom data.
"""
    
    modelfile_name = f"Modelfile_{output_name}"
    with open(modelfile_name, 'w') as f:
        f.write(modelfile)
    
    print(f"✅ Success! Created:")
    print(f"   - {output_file}")
    print(f"   - {modelfile_name}")
    print(f"\nTo use with Ollama:")
    print(f"   ollama create {output_name} -f {modelfile_name}")
    print(f"   ollama run {output_name}")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    checkpoint = sys.argv[1]
    output_name = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = convert_to_gguf(checkpoint, output_name)
    sys.exit(0 if success else 1)
