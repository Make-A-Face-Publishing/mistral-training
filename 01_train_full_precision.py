#!/usr/bin/env python3
"""
Full BF16 fine-tuning for Mistral-7B-Instruct-v0.3 with response-only loss masking
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
from datasets import Dataset
import gc
import numpy as np

class TrainingConfig(BaseModel):
    """Configuration for full BF16 fine-tuning with recommended settings"""
    model_name: str = Field(default="mistralai/Mistral-7B-Instruct-v0.3")
    output_dir: Path = Field(default=Path("./mistral_instruct_finetuned"))
    train_file: Path = Field(default=Path("practice.jsonl"))
    eval_file: Path = Field(default=Path("eval.jsonl"))
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
    
    # Loss masking
    mask_prompt_loss: bool = Field(default=True)  # Only calculate loss on responses
    response_template: str = Field(default="[/INST]")  # Where response starts
    
    # Logging
    use_wandb: bool = Field(default=False)
    wandb_project: Optional[str] = Field(default="mistral-instruct-finetune")
    
    # Data processing
    truncate_longer_samples: bool = Field(default=True)  # Truncate, don't skip
    

class ResponseMaskingDataCollator(DataCollatorForLanguageModeling):
    """Custom data collator that masks prompt tokens in loss calculation"""
    
    def __init__(self, tokenizer, response_template: str, mlm: bool = False):
        super().__init__(tokenizer=tokenizer, mlm=mlm)
        self.response_template = response_template
        self.response_token_ids = tokenizer.encode(response_template, add_special_tokens=False)
    
    def torch_call(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        # First, use the parent class to handle padding and create tensors
        batch = super().torch_call(examples)
        
        # Now mask the prompt portions if we have labels
        if "labels" in batch and self.response_template:
            labels = batch["labels"].clone()
            
            for idx, input_ids in enumerate(batch["input_ids"]):
                # Find where the response starts
                response_start_idx = self._find_response_start(input_ids)
                
                if response_start_idx > 0:
                    # Mask everything before the response (set to -100)
                    labels[idx, :response_start_idx] = -100
            
            batch["labels"] = labels
        
        return batch
    
    def _find_response_start(self, input_ids: torch.Tensor) -> int:
        """Find where the response template ends in the input_ids"""
        input_ids_list = input_ids.tolist()
        
        # Search for the response template tokens
        for i in range(len(input_ids_list) - len(self.response_token_ids) + 1):
            if input_ids_list[i:i+len(self.response_token_ids)] == self.response_token_ids:
                # Return the index after the response template
                return i + len(self.response_token_ids)
        
        # If not found, don't mask anything (return 0)
        return 0


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


def load_and_process_data(file_path: Path, tokenizer, max_length: int, truncate: bool = True):
    """Load and tokenize training data with proper truncation"""
    data = []
    skipped_count = 0
    truncated_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            try:
                item = json.loads(line.strip())
                
                # Quick length check before tokenization
                if len(item.get("text", "")) > max_length * 4:  # Rough char estimate
                    truncated_count += 1
                
                data.append(item)
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON at line {line_num + 1}")
                skipped_count += 1
    
    if skipped_count > 0:
        print(f"Skipped {skipped_count} invalid entries")
    if truncated_count > 0:
        print(f"Will truncate approximately {truncated_count} long entries")
    
    dataset = Dataset.from_list(data)
    
    def tokenize_function(examples):
        texts = examples["text"]
        
        # Tokenize with truncation enabled
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
            add_special_tokens=True
        )
        
        # Create labels (same as input_ids for language modeling)
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        # Log truncation info
        for idx, (text, ids) in enumerate(zip(texts, tokenized["input_ids"])):
            if len(tokenizer.encode(text, add_special_tokens=True)) > max_length:
                # This sample was truncated
                pass  # Could add logging here if needed
        
        return tokenized
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )
    
    return tokenized_dataset


def analyze_response_positions(dataset, tokenizer, response_template: str, num_samples: int = 5):
    """Analyze where responses start in the dataset for verification"""
    print(f"\nAnalyzing response positions (first {num_samples} samples)...")
    response_token_ids = tokenizer.encode(response_template, add_special_tokens=False)
    
    for i in range(min(num_samples, len(dataset))):
        input_ids = dataset[i]["input_ids"]
        
        # Find response position
        response_pos = -1
        for j in range(len(input_ids) - len(response_token_ids) + 1):
            if input_ids[j:j+len(response_token_ids)] == response_token_ids:
                response_pos = j + len(response_token_ids)
                break
        
        if response_pos > 0:
            # Decode prompt and response separately
            prompt_text = tokenizer.decode(input_ids[:response_pos], skip_special_tokens=False)
            response_text = tokenizer.decode(input_ids[response_pos:response_pos+50], skip_special_tokens=False)
            
            print(f"\nSample {i+1}:")
            print(f"  Response starts at position: {response_pos}")
            print(f"  Prompt ends with: ...{prompt_text[-50:]}")
            print(f"  Response starts with: {response_text}...")
        else:
            print(f"\nSample {i+1}: No response template found!")


def train_model(config: TrainingConfig):
    """Main training function with response-only loss masking"""
    setup_environment()
    
    print(f"Loading model: {config.model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side="left",  # Important for batch generation
        token=os.environ.get("HF_TOKEN")
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
    
    # Load datasets with truncation
    print("Loading and processing training data...")
    train_dataset = load_and_process_data(
        config.train_file, 
        tokenizer, 
        config.max_seq_length,
        truncate=config.truncate_longer_samples
    )
    eval_dataset = load_and_process_data(
        config.eval_file, 
        tokenizer, 
        config.max_seq_length,
        truncate=config.truncate_longer_samples
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")
    
    # Analyze response positions for verification
    if config.mask_prompt_loss:
        analyze_response_positions(train_dataset, tokenizer, config.response_template)
    
    # Data collator with response masking
    if config.mask_prompt_loss:
        data_collator = ResponseMaskingDataCollator(
            tokenizer=tokenizer,
            response_template=config.response_template,
            mlm=False
        )
    else:
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
    
    # Check if data files exist
    if not config.train_file.exists():
        print(f"Error: Training file {config.train_file} not found!")
        return
    if not config.eval_file.exists():
        print(f"Error: Evaluation file {config.eval_file} not found!")
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
    print(f"  Response-only loss: {config.mask_prompt_loss}")
    
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