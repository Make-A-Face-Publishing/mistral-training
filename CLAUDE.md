# Mistral-7B Fine-tuning Project

## Project Overview
This repository contains scripts and tools for fine-tuning Mistral-7B-Instruct-v0.3 models using full BF16 precision training with response-only loss masking. The project implements Claude's recommended hyperparameters for creative writing tasks with long sequences (8-9k tokens).

## Environment Setup
- **Virtual Environment**: `mistral_env` (Python 3.12)
- **Key Dependencies**: transformers, torch, datasets, pydantic, wandb, accelerate, xformers
- **GPU**: NVIDIA L4 (24GB VRAM)
- **CUDA**: 12.8

## Data Format
Training data uses JSONL format with the following structure:
```json
{"text": "<s>[INST]instruction prompt[/INST]model response</s>"}
```

**IMPORTANT**: Do NOT read or grep the following files as they contain PII:
- `practice.jsonl` (training data)
- `practice.parquet` (training data)
- `eval.jsonl` (evaluation data)

## Key Files

### Python Scripts
1. **train_mistral_instruct.py**: Main training script (NEW)
   - Full BF16 precision training (no quantization)
   - Implements response-only loss masking for [INST] format
   - Configurable via `TrainingConfig` class
   - Model: `mistralai/Mistral-7B-Instruct-v0.3`
   - Hyperparameters optimized for creative writing:
     - Learning rate: 2e-5 with cosine schedule
     - Warmup: 3% of steps
     - Adam betas: (0.9, 0.95) for long sequences
     - Gradient clipping: 1.0
   - Data truncation to 8192 tokens (not skipping)

2. **convert_full_model_to_gguf.py**: Converts full models to GGUF format
   - Usage: `python convert_full_model_to_gguf.py model_path [output_name]`
   - Handles full precision models (not LoRA adapters)
   - Creates quantized GGUF file for Ollama deployment

3. **test_response_masking.py**: Test script for verifying setup
   - Tests response detection in [INST] format
   - Verifies data file format
   - Checks GPU memory availability

4. **train_full_fp16.py**: Previous full training script (deprecated)

### Shell Scripts
1. **run_background.sh**: Runs training as background process with GPU monitoring
2. **check_progress.sh**: Checks dataset files, environment, and training status
3. **check_status.sh**: Monitors active training process
4. **register_tuned_model.sh**: Registers GGUF model with Ollama

## Workflow

### 1. Environment Preparation
```bash
# Virtual environment already created with uv
source mistral_env/bin/activate
```

### 2. Data Preparation
Ensure `practice.jsonl` and `eval.jsonl` exist with format:
```json
{"text": "<s>[INST]instruction prompt[/INST]model response</s>"}
```

### 3. Test Setup
```bash
# Verify response masking and data format
python test_response_masking.py
```

### 4. Training
```bash
# Run training in foreground
python train_mistral_instruct.py

# Or run in background with monitoring
./run_background.sh
```

### 4. Check Progress
```bash
./check_progress.sh  # Check environment and data
./check_status.sh    # Check training status
```

### 5. Convert to GGUF
```bash
# Convert full model to GGUF
python convert_full_model_to_gguf.py mistral_instruct_finetuned my-model
```

### 6. Register with Ollama
```bash
./register_tuned_model.sh my-model
```

## Training Configuration (NEW - Based on Claude's Recommendations)
- **Max sequence length**: 8192 tokens (full context)
- **Batch size**: 1 with gradient accumulation of 8 steps (effective batch = 8)
- **Learning rate**: 2e-5 with cosine scheduler
- **Warmup**: 3% of total steps
- **Training epochs**: 2 (recommended 1-3 for creative tasks)
- **Precision**: BF16 (better for stability)
- **Optimizer**: AdamW with beta2=0.95 (lower for long sequences)
- **Gradient clipping**: 1.0
- **Weight decay**: 0.1
- **Response-only loss**: Masks prompt tokens, only trains on responses

## Key Improvements (2025-07-19)

### Response-Only Loss Masking
- Automatically detects `[/INST]` marker in data
- Sets prompt tokens to -100 (ignored in loss calculation)
- Only calculates loss on response portion
- Prevents model from memorizing prompts

### Data Handling
- Truncates sequences > 8192 tokens (doesn't skip)
- Critical for small dataset (30+ samples)
- All samples contribute to training

### Memory Optimization
- Gradient checkpointing enabled
- Flash Attention 2 for efficient attention
- BF16 precision throughout
- Estimated ~28GB VRAM usage (fits in L4 24GB)

## Notes
- Model saved to `mistral_instruct_finetuned/`
- Training logs in `./logs/` directory
- GPU monitoring every 60 seconds
- Checkpoint saves every 100 steps
- Evaluation every 50 steps

## Commands Reference
```bash
# Check environment
./check_progress.sh

# Start training
./run_background.sh

# Monitor training
./check_status.sh
tail -f logs/training_*.log

# Stop training
./stop_training.sh

# Convert to GGUF
python convert_to_gguf.py fine_tuned_model/checkpoint-264 my-model

# Register with Ollama
./register_tuned_model.sh my-model
```

## Current Status (2025-07-19)

### Completed:
- ✅ Cleaned workspace (removed 26GB+ of old models)
- ✅ Created new training script with Claude's recommendations
- ✅ Implemented response-only loss masking for [INST] format
- ✅ Added data truncation (not skipping) for long sequences
- ✅ Configured all recommended hyperparameters
- ✅ Created test script to verify setup
- ✅ Updated background training scripts
- ✅ llama.cpp built with CUDA support

### Ready to Start:
- 🚀 Run `python test_response_masking.py` to verify setup
- 🚀 Start training with `python train_mistral_instruct.py`
- 🚀 Monitor progress with `./check_status.sh`

### Training Details:
- Model: mistralai/Mistral-7B-Instruct-v0.3
- Method: Full BF16 training (not QLoRA)
- Special: Response-only loss masking
- Dataset: 30+ creative writing samples
- Expected duration: ~2-4 hours for 2 epochs

### Notes:
- NVIDIA drivers installed and working (Driver Version: 570.133.20, CUDA Version: 12.8)
- NVIDIA L4 GPU detected and available
- Using CMake build method as Makefile is deprecated