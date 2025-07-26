#!/usr/bin/env python3
"""
Full FP16 fine-tuning for Mistral 7B (no quantization during training)
"""
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import gc

class FullTrainingConfig(BaseModel):
    """Configuration for full FP16 fine-tuning"""
    model_name: str = Field(default="mistralai/Mistral-7B-v0.3")
    output_dir: Path = Field(default=Path("./full_finetuned_model"))
    train_file: Path = Field(default=Path("practice.jsonl"))
    eval_file: Path = Field(default=Path("eval.jsonl"))
    max_seq_length: int = Field(default=2048)  # Reduced for memory efficiency
    batch_size: int = Field(default=1)
    gradient_accumulation_steps: int = Field(default=8)  # Increased for effective batch size
    num_train_epochs: int = Field(default=1)
    learning_rate: float = Field(default=5e-6)  # Lower LR for full fine-tuning
    warmup_ratio: float = Field(default=0.1)
    logging_steps: int = Field(default=10)
    save_steps: int = Field(default=50)
    eval_steps: int = Field(default=50)
    gradient_checkpointing: bool = Field(default=True)
    use_flash_attention: bool = Field(default=True)
    use_wandb: bool = Field(default=False)
    wandb_project: Optional[str] = Field(default="mistral-full-finetune")
    
def setup_environment():
    """Set up environment for training"""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TRANSFORMERS_CACHE"] = "./cache"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # HF_TOKEN should be set as environment variable before running
    # export HF_TOKEN="your_huggingface_token_here"
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
def load_and_process_data(file_path: Path, tokenizer, max_length: int):
    """Load and tokenize training data"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)
    
    dataset = Dataset.from_list(data)
    
    def tokenize_function(examples):
        # Extract text and tokenize
        texts = examples["text"]
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors=None,
            add_special_tokens=True
        )
        # Ensure we have the right length
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        
        # Explicitly truncate if needed (safety check)
        if isinstance(input_ids[0], list):
            input_ids = [ids[:max_length] for ids in input_ids]
            attention_mask = [mask[:max_length] for mask in attention_mask]
        
        tokenized["input_ids"] = input_ids
        tokenized["attention_mask"] = attention_mask
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )
    
    return tokenized_dataset

def train_model(config: FullTrainingConfig):
    """Main training function for full FP16 fine-tuning"""
    setup_environment()
    
    print(f"Loading model: {config.model_name}")
    print("This will load the full BF16 model, not quantized...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side="left",
        token=os.environ.get("HF_TOKEN")
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model in BF16 (better for H100)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if config.use_flash_attention else "eager",
        token=os.environ.get("HF_TOKEN")
    )
    
    # Enable gradient checkpointing to save memory
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    
    # Make model trainable
    model.train()
    for param in model.parameters():
        param.requires_grad = True
    
    # Load datasets
    print("Loading and processing training data...")
    train_dataset = load_and_process_data(config.train_file, tokenizer, config.max_seq_length)
    eval_dataset = load_and_process_data(config.eval_file, tokenizer, config.max_seq_length)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")
    
    # Data collator with proper padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
        return_tensors="pt"
    )
    
    # Training arguments optimized for 24GB VRAM
    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        overwrite_output_dir=True,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        bf16_full_eval=True,
        dataloader_num_workers=4,
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        logging_first_step=True,
        report_to="wandb" if config.use_wandb else "none",
        run_name=f"mistral-full-fp16-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        ddp_find_unused_parameters=False,
        torch_compile=False,  # Disable for stability
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
    print("Starting full FP16 training...")
    print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    trainer_stats = trainer.train()
    
    # Save the final model
    print("Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)
    
    # Save training stats
    with open(config.output_dir / "training_stats.json", "w") as f:
        json.dump(trainer_stats.metrics, f, indent=2)
    
    print("Training complete!")
    print(f"Model saved to: {config.output_dir}")
    return trainer_stats

def estimate_memory_usage(config: FullTrainingConfig):
    """Estimate GPU memory usage for training"""
    # Rough estimates for Mistral-7B
    model_size_gb = 14  # FP16 model
    optimizer_size_gb = model_size_gb * 2  # Adam optimizer states
    gradients_size_gb = model_size_gb
    activation_size_gb = config.batch_size * config.max_seq_length * 0.001  # Rough estimate
    
    total_gb = model_size_gb + optimizer_size_gb + gradients_size_gb + activation_size_gb
    
    if config.gradient_checkpointing:
        total_gb *= 0.7  # Gradient checkpointing saves ~30% memory
    
    print(f"\nEstimated memory usage:")
    print(f"Model: ~{model_size_gb:.1f} GB")
    print(f"Optimizer: ~{optimizer_size_gb:.1f} GB")
    print(f"Gradients: ~{gradients_size_gb:.1f} GB")
    print(f"Activations: ~{activation_size_gb:.1f} GB")
    print(f"Total: ~{total_gb:.1f} GB")
    
    # Detect actual GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Available: {gpu_memory_gb:.1f} GB ({gpu_name})")
    else:
        print(f"Available: Unknown (No GPU detected)")
    
    if torch.cuda.is_available():
        available_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) - 2  # Reserve 2GB
        if total_gb > available_memory_gb:
            print(f"\nWARNING: Estimated memory usage ({total_gb:.1f} GB) exceeds available VRAM ({available_memory_gb:.1f} GB)!")
            print("Consider reducing max_seq_length or using gradient accumulation")
    else:
        available_memory_gb = 0
    
    return total_gb

def main():
    """Main function"""
    print("Starting Mistral-7B-v0.3 training script...")
    print(f"Python: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    sys.stdout.flush()
    
    config = FullTrainingConfig()
    
    # Check if data files exist
    if not config.train_file.exists():
        print(f"Error: Training file {config.train_file} not found!")
        return
    if not config.eval_file.exists():
        print(f"Error: Evaluation file {config.eval_file} not found!")
        return
    
    # Estimate memory usage
    estimated_memory = estimate_memory_usage(config)
    
    # Check actual GPU memory
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        available_memory_gb = gpu_memory_gb - 2  # Reserve 2GB
        
        if estimated_memory > available_memory_gb:
            print(f"\nAdjusting configuration for {gpu_memory_gb:.1f}GB VRAM...")
            config.max_seq_length = 1024  # Further reduce sequence length
            config.gradient_accumulation_steps = 16  # Increase gradient accumulation
            estimate_memory_usage(config)
        else:
            # We have plenty of memory on H100, let's use better parameters
            print(f"\nOptimizing for {gpu_memory_gb:.1f}GB VRAM (H100)...")
            config.max_seq_length = 8192  # Increase sequence length
            config.batch_size = 2  # Increase batch size
            config.gradient_accumulation_steps = 4  # Reduce gradient accumulation
            estimate_memory_usage(config)
    
    # Train model
    train_model(config)
    
    print("\n" + "="*60)
    print("Next steps:")
    print("1. Quantize the model using quantize_model.py")
    print("2. Convert to GGUF format for Ollama deployment")
    print("3. Test the quantized model")
    print("="*60)

if __name__ == "__main__":
    main()