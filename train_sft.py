"""
Supervised Fine-Tuning (SFT) for cold start training.
Train the model to output full solutions with confidence tags.
"""
import os
import datetime
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from typing import Dict, List
from prepare_data import load_and_split_data, format_sft_example

# System prompt for SFT
SYSTEM_PROMPT = """You are a helpful assistant for mathematical problems. 
Please reason step by step.
At the end, you must verify your confidence.
If you are confident in your answer, output <c_math>.
If you are unsure, output <u_math>.
Put your final answer within \\boxed{} and verify your confidence.
"""

def format_for_training(example: Dict, tokenizer: AutoTokenizer) -> Dict:
    """Format example into chat format for training."""
    # Create conversation
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]}
    ]
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=False
    )
    
    return {"text": text}

def prepare_dataset(sft_data: List[Dict], tokenizer: AutoTokenizer) -> Dataset:
    """Prepare SFT dataset for training."""
    # Format all examples
    formatted_data = [format_sft_example(ex) for ex in sft_data]
    
    # Create dataset
    dataset = Dataset.from_list(formatted_data)
    
    # Apply chat formatting
    dataset = dataset.map(
        lambda x: format_for_training(x, tokenizer),
        remove_columns=dataset.column_names
    )
    
    # Tokenize
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=2048,
            padding=False
        )

        # Get token IDs for masking
        span_token_id = tokenizer.convert_tokens_to_ids("<span>")
        u_math_token_id = tokenizer.convert_tokens_to_ids("<u_math>")
        
        # Create labels with masking
        labels = []
        for input_ids in tokenized["input_ids"]:
            # Start with all labels = input_ids
            label = list(input_ids)
            
            # Apply masking for text between <span> and <u_math>
            is_masking = False
            for i, token_id in enumerate(input_ids):
                if token_id == span_token_id:
                    is_masking = True
                    # Keep the <span> token (Weight = 1)
                elif token_id == u_math_token_id:
                    is_masking = False
                    # Keep the <u_math> token (Weight = 1)
                elif is_masking:
                    # Mask tokens inside the span (Weight = 0)
                    label[i] = -100
            
            labels.append(label)
        
        tokenized["labels"] = labels
        return tokenized
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    
    return tokenized_dataset

def train_sft(
    model_id: str = "Qwen/Qwen2.5-Math-1.5B-Instruct",
    dataset_name: str = "Akirayasha/math-20",
    output_dir: str = "./output/sft_cold_start",
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    num_epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-5,
    seed: int = 42,
):
    """Train model with SFT on 25% of data (balanced by check field)."""
    
    # Load data - now using 25% for SFT with balanced check field
    sft_data, _ = load_and_split_data(dataset_name, sft_ratio=0.25, seed=seed, balance_check_field=True)
    
    # Load model and tokenizer
    print(f"Loading model {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=False  # Required for gradient checkpointing
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add special tokens for confidence and spans
    special_tokens = ["<u_math>", "<c_math>", "<span>"]
    num_added_toks = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    
    if num_added_toks > 0:
        print(f"Added {num_added_toks} special tokens: {special_tokens}")
        model.resize_token_embeddings(len(tokenizer))
    
    # Apply LoRA
    if use_lora:
        print(f"Applying LoRA with r={lora_r}, alpha={lora_alpha}")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Enable input gradients for LoRA
    if use_lora:
        model.enable_input_require_grads()
    
    # Prepare dataset
    print("Preparing dataset...")
    train_dataset = prepare_dataset(sft_data, tokenizer)
    
    # Training arguments
    timestamp = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
    output_dir = f"{output_dir}_{timestamp}"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=2,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        bf16=True,
        report_to="tensorboard",
        logging_dir=f"{output_dir}/logs",
        seed=seed,
        gradient_checkpointing=False,  # We enable it manually above
        optim="adamw_torch",
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("Starting SFT training...")
    trainer.train()
    
    # Save final model
    final_dir = f"{output_dir}/final"
    print(f"Saving model to {final_dir}")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    print(f"\nSFT training complete! Model saved to {final_dir}")
    return final_dir

if __name__ == "__main__":
    # Run SFT training
    model_path = train_sft(
        model_id="Qwen/Qwen2.5-Math-1.5B-Instruct",
        dataset_name="Akirayasha/math-20",
        num_epochs=5,
        batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
    )
    
    print(f"\nNext step: Run GRPO training with model from: {model_path}")
