# Mistral Model Training Pipeline

Training pipeline for fine-tuning Mistral models on creative writing datasets, specifically using portions of the `taozi555/literotica-stories` dataset.

## Environment Setup

### RunPod Network Volume Configuration
This project is designed to run on RunPod instances with persistent network volumes:
- **Persistent data**: Always stored in `/workspace/train` (network volume)
- **System volume**: Separate from workspace, gets reset between sessions
- **GPU instances**: Can be attached/detached from different GPUs as needed

### Dependencies Management with uv

Since the RunPod system volume is separate from the workspace volume, `uv` needs to be installed fresh each time:

```bash
# Install uv (required each session)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# Install dependencies (from workspace/train directory)
uv sync

# Activate environment
source .venv/bin/activate
```

Alternatively, use `uv run` to run scripts directly:
```bash
uv run python 01_train_base_model.py
```

## Dataset

Uses the `taozi555/literotica-stories` dataset with intelligent sampling:
- **Training**: 20,000 randomly selected samples
- **Evaluation**: 2,000 randomly selected samples  
- **Format**: HuggingFace dataset format (.arrow files)
- **Content**: Creative writing stories for base model fine-tuning

## Training Scripts

### Main Scripts (Workflow Order)

1. **`01_train_base_model.py`** - Base model training (recommended)
   - Pure continuation training on Mistral-7B-v0.3
   - No instruction formatting, trains on raw creative text
   - Full BF16 precision training

2. **`01_train_full_precision.py`** - Instruct model training
   - For Mistral-7B-Instruct-v0.3 with response masking
   - Uses instruction format with prompt/response separation

3. **`02_convert_full_to_gguf.py`** - Convert trained model to GGUF
   - Converts full precision models for Ollama deployment
   - Handles quantization to Q4_K_M format

4. **`03_register_with_ollama.sh`** - Register model with Ollama
   - Creates Modelfile and registers with local Ollama instance

### Support Scripts

- **`monitor_training_progress.sh`** - Monitor training status
- **`run_training_background.sh`** - Run training in background
- **`stop_training_process.sh`** - Stop training process

### Quantized Workflow (Legacy)

The `quantized/` directory contains scripts for QLoRA training workflows.

## Configuration

### Authentication
Set HuggingFace token before training:
```bash
export HF_TOKEN="your_huggingface_token_here"
```

### Training Parameters
- **Model**: `mistralai/Mistral-7B-v0.3` (base model)
- **Precision**: BF16 full precision training
- **Sequence Length**: 8192 tokens
- **Batch Size**: 1 with 8x gradient accumulation (effective batch = 8)
- **Learning Rate**: 2e-5 with cosine scheduler
- **Epochs**: 2
- **Memory Usage**: ~39GB (fits on H100 80GB)

## Usage

1. **Set up environment**:
   ```bash
   # Install uv (each session)
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source $HOME/.cargo/env
   
   # Install dependencies
   uv sync
   ```

2. **Set authentication**:
   ```bash
   export HF_TOKEN="your_token_here"
   ```

3. **Run training**:
   ```bash
   uv run python 01_train_base_model.py
   ```

4. **Convert to GGUF**:
   ```bash
   uv run python 02_convert_full_to_gguf.py mistral_base_finetuned my-creative-model
   ```

5. **Register with Ollama**:
   ```bash
   ./03_register_with_ollama.sh my-creative-model
   ```

## Features

- **Smart caching**: Avoids re-downloading models between sessions
- **Dataset sampling**: Randomly samples manageable subset from large dataset
- **Memory optimization**: Gradient checkpointing and Flash Attention 2
- **Progress monitoring**: Built-in evaluation and checkpoint saving
- **Flexible deployment**: Converts to GGUF for efficient inference

## File Organization

```
workspace/train/
├── 01_train_base_model.py          # Main base model training
├── 01_train_full_precision.py      # Instruct model training  
├── 02_convert_full_to_gguf.py      # GGUF conversion
├── 03_register_with_ollama.sh      # Ollama integration
├── quantized/                      # QLoRA workflow scripts
├── cache/                          # Model cache (gitignored)
├── literotica_dataset/             # Dataset (gitignored)
├── pyproject.toml                  # uv dependencies
└── README.md                       # This file
```

## Notes

- All training artifacts saved to `/workspace/train` (persistent)
- Model cache prevents bandwidth waste on repeated runs  
- Dataset automatically handled from HuggingFace format
- Designed for RunPod network volume persistence