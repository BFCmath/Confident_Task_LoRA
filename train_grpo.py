
import os
import datetime
import random
import logging
import re
from typing import Callable, Dict, List, Tuple, Any
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from unittest.mock import patch
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter
from dotenv import load_dotenv

# Import custom helpers
from reward import reward_fn
from grader import math_equal
from metrics import extract_box
from prepare_data import load_and_split_data, format_grpo_example

load_dotenv()

logging.getLogger("vllm.engine.scheduler").setLevel(logging.ERROR)
os.environ["VLLM_USE_V1"] = "0"

from torch.optim.lr_scheduler import LambdaLR

def get_constant_schedule_with_warmup(optimizer: torch.optim.Optimizer, num_warmup_steps: int, last_epoch: int = -1):
    def lr_lambda(current_step: int):
        return min(1.0, float(current_step) / float(max(1, num_warmup_steps)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)

# -------------------------
# Prompting helpers
# -------------------------
SYSTEM_PROMPT = """You are a helpful assistant for mathematical problems. 
Please reason step by step.
At the end, you must verify your confidence.
If you are confident in your answer, output <c_math>.
If you are unsure, output <u_math>.
Put your final answer within \\boxed{} and verify your confidence.
"""

def format_prompt(problem: str, tokenizer: AutoTokenizer) -> str:
    return tokenizer.apply_chat_template(
        [
            dict(role="system", content=SYSTEM_PROMPT),
            dict(role="user", content=problem)
        ],
        add_generation_prompt=True, tokenize=False
    )

# -------------------------
# vLLM utilities
# -------------------------
def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.4) -> LLM:
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    
    with world_size_patch:
        return LLM(
            model=model_id,
            dtype="bfloat16",
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=2048,
            trust_remote_code=True,
            tensor_parallel_size=1
        )

def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM) -> None:
    """Load policy weights into vLLM instance. Handle both LoRA and full models."""
    from peft import PeftModel
    
    # Get state dict and clean PEFT wrapper keys if present
    state_dict = policy.state_dict()
    
    # If this is a PEFT model, remove the 'base_model.model.' prefix and handle LoRA layers
    if isinstance(policy, PeftModel):
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            # Remove PEFT wrapper prefixes
            clean_key = key.replace('base_model.model.', '')
            
            # Skip LoRA-specific parameters
            if 'lora_A' in clean_key or 'lora_B' in clean_key or 'lora_embedding' in clean_key:
                continue
            if 'modules_to_save' in clean_key:
                continue
                
            # Remove .base_layer suffix if present (from merged LoRA layers)
            clean_key = clean_key.replace('.base_layer', '')
            
            cleaned_state_dict[clean_key] = value
        state_dict = cleaned_state_dict
    
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

def init_sampling_params(temperature: float, max_tokens: int) -> SamplingParams:
    return SamplingParams(
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_tokens,
        logprobs=0,
        stop=["<|im_end|>"] 
    )

# -------------------------
# Tokenization utilities
# -------------------------
def tokenize_prompt_and_output(prompt_strs: List[str], output_strs: List[str], tokenizer: AutoTokenizer) -> Dict[str, torch.Tensor]:
    batch_data = []
    max_len = 0
    for prompt, output in zip(prompt_strs, output_strs):
        prompt_tokens = tokenizer(prompt)["input_ids"]
        output_tokens = tokenizer(output)["input_ids"]
        combined_tokens = prompt_tokens + output_tokens
        max_len = max(max_len, len(combined_tokens))
        batch_data.append({
            "tokens": combined_tokens, "prompt_len": len(prompt_tokens), "total_len": len(combined_tokens)
        })
    batch_size = len(batch_data)
    input_ids = torch.full((batch_size, max_len - 1), tokenizer.eos_token_id, dtype=torch.long)
    labels = torch.full((batch_size, max_len - 1), tokenizer.eos_token_id, dtype=torch.long)
    response_mask = torch.zeros((batch_size, max_len - 1), dtype=torch.bool)
    for i, data in enumerate(batch_data):
        tokens, seq_len = torch.tensor(data["tokens"]), len(data["tokens"])
        # Check bounds to avoid errors if max_len changed or tokens are empty
        if seq_len > 1:
            input_ids[i, :seq_len-1], labels[i, :seq_len-1] = tokens[:-1], tokens[1:]
            response_start, response_end = data["prompt_len"] - 1, seq_len - 1
            if response_end > response_start:
                response_mask[i, response_start:response_end] = True
    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}

# -------------------------
# Reward Function
# -------------------------
# -------------------------
# GRPO Logic
# -------------------------
def compute_group_normalized_advantages(
    rollout_responses: List[str],
    repeated_ground_truths: List[str],
    reward_fn: Callable[[str, str], float],
    group_size: int,
    advantage_eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    
    raw_rewards_list = [reward_fn(response, gt) for response, gt in zip(rollout_responses, repeated_ground_truths)]
    raw_rewards = torch.tensor(raw_rewards_list, dtype=torch.float32)
    
    rewards_2d = raw_rewards.view(-1, group_size)
    group_mean = rewards_2d.mean(dim=1, keepdim=True)
    group_std = rewards_2d.std(dim=1, keepdim=True)
    
    # Standard GRPO normalization
    advantages_2d = (rewards_2d - group_mean) / (group_std + advantage_eps)
    
    advantages = advantages_2d.view(-1)
    
    metadata = {
        "mean": float(torch.mean(raw_rewards)),
        "std": float(torch.std(raw_rewards)),
        "max": float(torch.max(raw_rewards)),
        "min": float(torch.min(raw_rewards)),
    }
    return advantages, raw_rewards, metadata

def compute_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    clip_range: float,
) -> torch.Tensor:
    pi_ratio = torch.exp(policy_log_probs - old_log_probs)
    unclipped_term = advantages * pi_ratio
    clipped_ratio = torch.clamp(pi_ratio, 1.0 - clip_range, 1.0 + clip_range)
    clipped_term = advantages * clipped_ratio
    loss = -torch.minimum(unclipped_term, clipped_term)
    return loss

def masked_mean(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked_tensor = tensor * mask
    sum_per_row = masked_tensor.sum(dim=1)
    count_per_row = mask.sum(dim=1)
    mean_per_row = sum_per_row / (count_per_row.float() + 1e-8)
    return mean_per_row.mean()

def get_response_log_probs(model: PreTrainedModel, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = model(input_ids).logits
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return log_probs

def duplicate_data(arr: List, group_size: int) -> List:
    return [x for x in arr for _ in range(group_size)]

def rollout_with_vllm(policy: PreTrainedModel, llm: LLM, sampling_params: SamplingParams, prompts_batch: List[str], group_size: int, model_id: str = None, lora_config: LoraConfig = None) -> Tuple[List[str], List[str], List[int], PreTrainedModel]:
    """
    Perform rollout with vLLM. For LoRA models, merge weights for inference.
    Returns the rollout results and the policy (unchanged).
    """
    is_peft = isinstance(policy, PeftModel)
    
    if is_peft:
        # Temporarily merge LoRA weights for vLLM inference
        policy.eval()  # Set to eval mode
        with torch.no_grad():
            # Merge LoRA weights into base model temporarily
            policy.merge_adapter()
            load_policy_into_vllm_instance(policy, llm)
            
            # Generate with vLLM
            prompts_dup = duplicate_data(prompts_batch, group_size)
            vllm_rollouts = llm.generate(prompts_dup, sampling_params, use_tqdm=False)
            
            # CRITICAL: Unmerge BEFORE processing rollouts to restore gradient graph
            policy.unmerge_adapter()
    else:
        load_policy_into_vllm_instance(policy, llm)
        
        # Generate with vLLM
        prompts_dup = duplicate_data(prompts_batch, group_size)
        vllm_rollouts = llm.generate(prompts_dup, sampling_params, use_tqdm=False)
    
    # Process rollout results
    rollout_input_text, rollout_response_text, rollout_output_tokens = [], [], []
    for rollout in vllm_rollouts:
        r = rollout.outputs[0]
        rollout_input_text.append(rollout.prompt)
        rollout_response_text.append(r.text)
        rollout_output_tokens.append(len(r.token_ids))
    
    # Return to training mode
    policy.train()
    
    return rollout_input_text, rollout_response_text, rollout_output_tokens, policy

def grpo_microbatch_step(
    policy: PreTrainedModel, input_ids: torch.Tensor, labels: torch.Tensor, response_mask: torch.Tensor,
    advantages_per_seq: torch.Tensor, gradient_accumulation_steps: int, clip_range: float,
) -> torch.Tensor:
    policy_log_probs = get_response_log_probs(policy, input_ids, labels)
    old_log_probs = policy_log_probs.detach()
    advantages = advantages_per_seq.unsqueeze(-1)
    loss_per_token = compute_loss(advantages, policy_log_probs, old_log_probs, clip_range)
    loss = masked_mean(loss_per_token, response_mask)
    loss = loss / gradient_accumulation_steps
    loss.backward()
    return loss.detach()

# -------------------------
# Training Loop
# -------------------------
def log_rollouts_to_file(
    log_file: str,
    step: int,
    prompts: List[str],
    responses: List[str],
    rewards: torch.Tensor,
    ground_truths: List[str],
    group_size: int,
) -> None:
    """Log rollout details to a file for tracking model behavior."""
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"STEP {step}\n")
        f.write(f"{'='*80}\n\n")
        
        n_prompts = len(prompts) // group_size
        for i in range(n_prompts):
            f.write(f"\n{'-'*80}\n")
            f.write(f"Prompt {i+1}:\n")
            f.write(f"{'-'*80}\n")
            # Extract the user question from the prompt (simplified)
            prompt_text = prompts[i * group_size]
            f.write(f"{prompt_text}\n\n")
            
            f.write(f"Ground Truth:\n{ground_truths[i]}\n\n")
            
            # Show all rollouts for this prompt
            for j in range(group_size):
                idx = i * group_size + j
                f.write(f"\n--- Rollout {j+1} (Reward: {rewards[idx]:.2f}) ---\n")
                f.write(f"{responses[idx]}\n")
            
            f.write(f"\n")

def train(
    policy: PreTrainedModel, tokenizer: AutoTokenizer, llm: LLM, sampling_params: SamplingParams, *,
    train_prompts: List[str], train_answers: List[str], 
    optimizer: torch.optim.Optimizer, scheduler, n_grpo_steps: int, rollout_batch_size: int,
    group_size: int, gradient_accumulation_steps: int, clip_range: float,
    advantage_eps: float, device: str, writer: SummaryWriter = None, seed: int,
    model_id: str = None, lora_config: LoraConfig = None, rollout_log_file: str = None,
) -> PreTrainedModel:
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    micro_train_batch_size = rollout_batch_size // gradient_accumulation_steps
    random.seed(seed)
    train_step = 0

    print("Starting GRPO training...")
    
    for step in range(n_grpo_steps):
        # Sample batch
        sampled_indices = random.sample(range(len(train_prompts)), n_prompts_per_rollout_batch)
        prompts_batch = [train_prompts[i] for i in sampled_indices]
        answers_batch = [train_answers[i] for i in sampled_indices]
        
        # Rollout
        rollout_input, rollout_response, rollout_tokens, policy = rollout_with_vllm(
            policy, llm, sampling_params, prompts_batch, group_size, model_id, lora_config
        )
            
        answers_dup = duplicate_data(answers_batch, group_size)
        avg_output_tokens = sum(rollout_tokens) / len(rollout_tokens) if rollout_tokens else 0.0
        
        # Compute advantages
        advantages, raw_rewards, reward_meta = compute_group_normalized_advantages(
            rollout_response, answers_dup, reward_fn, group_size, advantage_eps
        )
        
        # Log rollouts to file for behavior tracking
        if rollout_log_file:
            log_rollouts_to_file(
                rollout_log_file, train_step + 1, rollout_input, rollout_response,
                raw_rewards, answers_batch, group_size
            )
        
        # Tokenize for training
        tokenized = tokenize_prompt_and_output(rollout_input, rollout_response, tokenizer)
        
        # Optimization step
        optimizer.zero_grad()
        rollout_loss = 0.0
        
        # Micro-batching
        for micro_idx in range(0, rollout_batch_size, micro_train_batch_size):
            s = slice(micro_idx, micro_idx + micro_train_batch_size)
            input_ids = tokenized["input_ids"][s].to(device)
            labels = tokenized["labels"][s].to(device)
            response_mask = tokenized["response_mask"][s].to(device)
            adv_batch = advantages[s].to(device)
            
            loss = grpo_microbatch_step(
                policy, input_ids, labels, response_mask, adv_batch,
                gradient_accumulation_steps, clip_range
            )
            rollout_loss += float(loss.item())
            
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        rollout_loss /= (rollout_batch_size / micro_train_batch_size)
        train_step += 1
        
        # Logging
        print(f"Step {train_step} | Loss: {rollout_loss:.4f} | Grad: {grad_norm:.4f} | "
              f"Reward mean: {reward_meta['mean']:.4f} | Reward std: {reward_meta['std']:.4f} | "
              f"Avg len: {avg_output_tokens:.1f}")
        
        if writer:
            writer.add_scalar("train/loss", rollout_loss, train_step)
            writer.add_scalar("train/grad_norm", grad_norm, train_step)
            writer.add_scalar("train/reward_mean", reward_meta['mean'], train_step)
            writer.add_scalar("train/avg_output_tokens", avg_output_tokens, train_step)
    
    return policy

# -------------------------
# Main
# -------------------------
def init_policy(model_id: str, device: str, use_lora: bool = True, lora_r: int = 16, lora_alpha: int = 32) -> Tuple[PreTrainedModel, AutoTokenizer, LoraConfig]:
    """Initialize policy model with optional LoRA."""
    # Check if model_id is a local path (SFT checkpoint with LoRA)
    is_sft_checkpoint = os.path.exists(model_id) and os.path.isdir(model_id)
    
    if is_sft_checkpoint:
        # Check if it has adapter_config.json (LoRA adapter)
        adapter_config_path = os.path.join(model_id, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            # This is a LoRA adapter, need to load base model first
            print(f"Loading LoRA adapter from {model_id}")
            import json
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            base_model_name = adapter_config.get('base_model_name_or_path', 'Qwen/Qwen2.5-Math-1.5B-Instruct')
            
            # Load base model
            print(f"Loading base model {base_model_name}")
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name, torch_dtype=torch.bfloat16, use_cache=False, trust_remote_code=True
            )
            
            # Load LoRA adapter
            model = PeftModel.from_pretrained(model, model_id)
            print(f"LoRA adapter loaded successfully")
            
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            
            # Ensure pad token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Extract LoRA config
            lora_config = list(model.peft_config.values())[0]
            print(f"LoRA config: r={lora_config.r}, alpha={lora_config.lora_alpha}")
            
            # CRITICAL: Enable gradients for LoRA training
            model.to(device)
            model.train()
            
            # Ensure all LoRA parameters require gradients
            for name, param in model.named_parameters():
                if 'lora_' in name:
                    param.requires_grad = True
            
            # Enable input require grads for LoRA
            model.enable_input_require_grads()
            
            return model, tokenizer, lora_config
        else:
            # Full model checkpoint
            print(f"Loading full model from {model_id}")
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, use_cache=False, trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            
            # Ensure pad token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model.to(device).train()
            return model, tokenizer, None
    else:
        # Load base model and apply LoRA
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, use_cache=False, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
        # Ensure pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        lora_config = None
        # Apply LoRA if enabled
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
            
        model.to(device).train()
        return model, tokenizer, lora_config

def main():
    # Params
    base_model_id = "Qwen/Qwen2.5-Math-1.5B-Instruct"  # Base model for vLLM
    device = "cuda"
    seed = 42
    
    # Dataset
    dataset_name = "Akirayasha/math-20"
    use_sft_model = True  # Set to True to load SFT checkpoint
    sft_model_path = "./output/sft_cold_start_1763745496/final"  # Path to SFT model if use_sft_model=True
    
    # LoRA params
    use_lora = True
    lora_r = 16
    lora_alpha = 32
    
    # GRPO params
    n_grpo_steps = 100 # Adjust as needed
    rollout_batch_size = 16  # Reduced from 64 to save memory
    group_size = 8  # Reduced from 8 to save memory
    grad_acc_steps = 8  # Increased to maintain effective batch size
    lr = 1e-5  # Higher LR for LoRA
    clip_range = 0.2
    advantage_eps = 1e-6
    
    # Generation params
    temperature = 1.0
    max_tokens = 768
    
    # vLLM memory settings
    gpu_memory_utilization = 0.4  # Reduced to leave more memory for training
    
    # Load Data - 90% for GRPO
    print(f"Loading dataset {dataset_name}...")
    _, grpo_data = load_and_split_data(dataset_name, sft_ratio=0.25, seed=seed, balance_check_field=True)
    
    # Init Policy
    if use_sft_model and sft_model_path:
        print(f"Loading SFT checkpoint from {sft_model_path}...")
        model_id = sft_model_path
    else:
        model_id = base_model_id
    
    print(f"Initializing policy {model_id}...")
    policy, tokenizer, lora_config = init_policy(model_id, device, use_lora=use_lora, lora_r=lora_r, lora_alpha=lora_alpha)
    
    # Format data for GRPO
    prompts = []
    answers = []
    for ex in grpo_data:
        formatted = format_grpo_example(ex)
        prompts.append(format_prompt(formatted["question"], tokenizer))
        # Use full solution for reward computation
        answers.append(formatted["solution"])
    
    # Init vLLM
    print("Initializing vLLM...")
    llm = init_vllm(base_model_id, device, seed, gpu_memory_utilization=gpu_memory_utilization)
    sampling_params = init_sampling_params(temperature, max_tokens)
    
    # Optimizer
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=10)
    
    # Writer
    timestamp = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
    log_dir = f"./output/logs/grpo_{timestamp}"
    writer = SummaryWriter(log_dir=log_dir)
    
    # Rollout log file
    rollout_log_file = f"./output/logs/grpo_{timestamp}/rollout.log"
    os.makedirs(os.path.dirname(rollout_log_file), exist_ok=True)
    # Initialize the log file with header
    with open(rollout_log_file, 'w', encoding='utf-8') as f:
        f.write(f"GRPO Training Rollout Log\n")
        f.write(f"Base Model: {base_model_id}\n")
        f.write(f"Policy Model: {model_id}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"LoRA: r={lora_r}, alpha={lora_alpha}\n")
        f.write(f"Batch size: {rollout_batch_size}, Group size: {group_size}\n")
        f.write(f"{'='*80}\n")
    
    print(f"Rollout logs will be saved to: {rollout_log_file}")
    
    # Train
    policy = train(
        policy, tokenizer, llm, sampling_params,
        train_prompts=prompts, train_answers=answers,
        optimizer=optimizer, scheduler=scheduler,
        n_grpo_steps=n_grpo_steps, rollout_batch_size=rollout_batch_size,
        group_size=group_size, gradient_accumulation_steps=grad_acc_steps,
        clip_range=clip_range, advantage_eps=advantage_eps, device=device,
        writer=writer, seed=seed, model_id=base_model_id, lora_config=lora_config,
        rollout_log_file=rollout_log_file,
    )
    
    # Save
    save_dir = f"./output/grpo_confident_lora_{timestamp}"
    print(f"Saving model to {save_dir}")
    if isinstance(policy, PeftModel):
        # Save LoRA adapter
        policy.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"LoRA adapter saved to {save_dir}")
    else:
        policy.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

if __name__ == "__main__":
    main()

