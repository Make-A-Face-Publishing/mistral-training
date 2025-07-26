#!/usr/bin/env python3
"""
Full BF16 fine-tuning for Mistral-7B-v0.3 base model for creative writing
Pure base model training - no instruction formatting or response masking
"""
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import torch
from torch.nn import functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset, load_dataset
import pandas as pd
import gc
import numpy as np

class TrainingConfig(BaseModel):
    """Configuration for full BF16 fine-tuning with recommended settings"""
    model_name: str = Field(default="mistralai/Mistral-7B-v0.3")
    output_dir: Path = Field(default=Path("./mistral_base_finetuned"))
    data_dir: Path = Field(default=Path("literotica_dataset"))
    train_split: float = Field(default=0.95)  # 95% for training, 5% for eval
    max_seq_length: int = Field(default=8192)  # Full context length
    
    # Batch size configuration
    per_device_train_batch_size: int = Field(default=1)
    gradient_accumulation_steps: int = Field(default=8)  # Effective batch size = 8
    
    # Training duration
    num_train_epochs: int = Field(default=2)  # 1-3 epochs recommended
    
    # Learning rate configuration (as recommended)
    learning_rate: float = Field(default=2e-5)
    lr_scheduler_type: str = Field(default="cosine")
    warmup_ratio: float = Field(default=0.03)  # 3% warmup
    
    # Optimizer settings (as recommended)
    adam_beta1: float = Field(default=0.9)
    adam_beta2: float = Field(default=0.95)  # Lower than default for long sequences
    adam_epsilon: float = Field(default=1e-8)
    weight_decay: float = Field(default=0.1)
    max_grad_norm: float = Field(default=1.0)  # Gradient clipping
    
    # Training strategy
    logging_steps: int = Field(default=10)
    save_steps: int = Field(default=100)
    eval_steps: int = Field(default=50)
    save_total_limit: int = Field(default=3)
    
    # Memory optimization
    gradient_checkpointing: bool = Field(default=True)
    use_flash_attention: bool = Field(default=True)
    fp16: bool = Field(default=False)  # Use bf16 instead
    bf16: bool = Field(default=True)
    
    # Base model training (no masking needed)
    mask_prompt_loss: bool = Field(default=False)  # No prompt masking for base model
    
    # Logging
    use_wandb: bool = Field(default=False)
    wandb_project: Optional[str] = Field(default="mistral-base-writing-finetune")
    
    # Data processing
    truncate_longer_samples: bool = Field(default=True)  # Truncate, don't skip
    

# No custom data collator needed for base model training - using standard DataCollatorForLanguageModeling


def setup_environment():
    """Set up environment for training"""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TRANSFORMERS_CACHE"] = "/workspace/train/cache"
    os.environ["HF_HOME"] = "/workspace/train/cache"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # HF_TOKEN should be set as environment variable before running
    # export HF_TOKEN="your_huggingface_token_here"
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def load_and_process_data(data_dir: Path, tokenizer, max_length: int, train_split: float = 0.95):
    """Load and tokenize parquet data for base model training"""
    print(f"Loading parquet files from {data_dir}...")
    
    # Load all parquet files from the directory
    parquet_files = list(data_dir.glob("*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")
    
    print(f"Found {len(parquet_files)} parquet files")
    
    # Load all parquet files into a single dataset
    all_data = []
    for parquet_file in parquet_files:
        df = pd.read_parquet(parquet_file)
        print(f"Loaded {len(df)} rows from {parquet_file.name}")
        all_data.extend(df["text"].tolist())
    
    print(f"Total texts loaded: {len(all_data)}")
    
    # Split into train/eval
    split_idx = int(len(all_data) * train_split)
    train_texts = all_data[:split_idx]
    eval_texts = all_data[split_idx:]
    
    print(f"Train samples: {len(train_texts)}")
    print(f"Eval samples: {len(eval_texts)}")
    
    def create_tokenized_dataset(texts, desc):
        # Create dataset from text list
        dataset = Dataset.from_dict({"text": texts})
        
        def tokenize_function(examples):
            # For base model training, just tokenize the raw text
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding=False,  # Don't pad here, let the data collator handle it
                return_tensors=None,
                add_special_tokens=True
            )
            
            # Create labels (same as input_ids for language modeling)
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc=desc
        )
        
        return tokenized_dataset
    
    train_dataset = create_tokenized_dataset(train_texts, "Tokenizing train data")
    eval_dataset = create_tokenized_dataset(eval_texts, "Tokenizing eval data")
    
    return train_dataset, eval_dataset


def analyze_data_samples(dataset, tokenizer, num_samples: int = 3):
    """Analyze first few samples to verify data loading"""
    print(f"\nAnalyzing first {num_samples} samples...")
    
    for i in range(min(num_samples, len(dataset))):
        input_ids = dataset[i]["input_ids"]
        text = tokenizer.decode(input_ids, skip_special_tokens=True)
        
        print(f"\nSample {i+1}:")
        print(f"  Token count: {len(input_ids)}")
        print(f"  Text preview: {text[:200]}...")
        if len(text) > 200:
            print(f"  Text ends with: ...{text[-100:]}")


def train_model(config: TrainingConfig):
    """Main training function for base model fine-tuning"""
    setup_environment()
    
    print(f"Loading model: {config.model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side="left",  # Important for batch generation
        token=os.environ.get("HF_TOKEN"),
        cache_dir="/workspace/train/cache"
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model in BF16
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if config.use_flash_attention else "eager",
        token=os.environ.get("HF_TOKEN"),
        cache_dir="/workspace/train/cache"
    )
    
    # Enable gradient checkpointing to save memory
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    
    # Make model trainable
    model.train()
    for param in model.parameters():
        param.requires_grad = True
    
    # Load datasets from parquet files
    print("Loading and processing training data...")
    train_dataset, eval_dataset = load_and_process_data(
        config.data_dir, 
        tokenizer, 
        config.max_seq_length,
        config.train_split
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")
    
    # Analyze data samples for verification
    analyze_data_samples(train_dataset, tokenizer)
    
    # Standard data collator for base model training (no masking)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    # Training arguments with all recommended settings
    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        overwrite_output_dir=True,
        
        # Training duration
        num_train_epochs=config.num_train_epochs,
        
        # Batch size
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        
        # Learning rate schedule
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        
        # Optimizer settings
        optim="adamw_torch",
        adam_beta1=config.adam_beta1,
        adam_beta2=config.adam_beta2,
        adam_epsilon=config.adam_epsilon,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        
        # Checkpointing
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        
        # Evaluation
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        
        # Logging
        logging_steps=config.logging_steps,
        logging_first_step=True,
        report_to="wandb" if config.use_wandb else "none",
        run_name=f"mistral-instruct-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        
        # Performance
        fp16=config.fp16,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        dataloader_num_workers=4,
        
        # Best model selection
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Other settings
        ddp_find_unused_parameters=False,
        torch_compile=False,  # Disable for stability
        remove_unused_columns=False,  # Important for custom data collator
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Start training
    print("\nStarting training with response-only loss masking...")
    print(f"Effective batch size: {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    print(f"Total optimization steps: {len(train_dataset) // (config.per_device_train_batch_size * config.gradient_accumulation_steps) * config.num_train_epochs}")
    
    trainer_stats = trainer.train()
    
    # Save the final model
    print("\nSaving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)
    
    # Save training stats
    with open(config.output_dir / "training_stats.json", "w") as f:
        json.dump(trainer_stats.metrics, f, indent=2)
    
    # Save config
    with open(config.output_dir / "training_config.json", "w") as f:
        json.dump(config.model_dump(), f, indent=2, default=str)
    
    print("\nTraining complete!")
    print(f"Model saved to: {config.output_dir}")
    
    return trainer_stats


def estimate_memory_usage(config: TrainingConfig):
    """Estimate GPU memory usage for training"""
    # Estimates for Mistral-7B in BF16
    model_size_gb = 14  # BF16 model
    optimizer_size_gb = model_size_gb * 2  # Adam optimizer states
    gradients_size_gb = model_size_gb
    
    # Activation memory scales with sequence length and batch size
    activation_size_gb = (
        config.per_device_train_batch_size * 
        config.max_seq_length * 
        4096 * 4 / (1024**3)  # Hidden size * bytes per element / GB
    )
    
    total_gb = model_size_gb + optimizer_size_gb + gradients_size_gb + activation_size_gb
    
    if config.gradient_checkpointing:
        total_gb *= 0.7  # Gradient checkpointing saves ~30% memory
    
    print(f"\nEstimated memory usage:")
    print(f"  Model: ~{model_size_gb:.1f} GB")
    print(f"  Optimizer: ~{optimizer_size_gb:.1f} GB")
    print(f"  Gradients: ~{gradients_size_gb:.1f} GB")
    print(f"  Activations: ~{activation_size_gb:.1f} GB")
    print(f"  Total: ~{total_gb:.1f} GB")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  Available: {gpu_memory_gb:.1f} GB ({gpu_name})")
        
        if total_gb > gpu_memory_gb - 2:  # Reserve 2GB
            print(f"\nWARNING: Estimated usage may exceed available VRAM!")
            print("Consider reducing batch_size or max_seq_length")
    
    return total_gb


def main():
    """Main function"""
    print("Mistral-7B-Instruct-v0.3 Fine-tuning Script")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)
    
    config = TrainingConfig()
    
    # Check if data directory exists
    if not config.data_dir.exists():
        print(f"Error: Data directory {config.data_dir} not found!")
        return
    
    # Show configuration
    print("\nConfiguration:")
    print(f"  Model: {config.model_name}")
    print(f"  Max sequence length: {config.max_seq_length}")
    print(f"  Batch size: {config.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"  Effective batch size: {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs: {config.num_train_epochs}")
    print(f"  Data directory: {config.data_dir}")
    print(f"  Train/eval split: {config.train_split:.1%}/{1-config.train_split:.1%}")
    print(f"  Base model training: {not config.mask_prompt_loss}")
    
    # Estimate memory usage
    estimate_memory_usage(config)
    
    # Confirm before starting
    response = input("\nProceed with training? [y/N]: ")
    if response.lower() != 'y':
        print("Training cancelled.")
        return
    
    # Train model
    train_model(config)
    
    print("\n" + "="*60)
    print("Next steps:")
    print("1. Test the model with generation examples")
    print("2. Convert to GGUF format using convert_full_model_to_gguf.py")
    print("3. Quantize if needed using quantize_model.py")
    print("4. Register with Ollama using register_tuned_model.sh")
    print("="*60)


if __name__ == "__main__":
    main()