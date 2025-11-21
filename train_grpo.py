
import os
import datetime
import random
import logging
import re
from typing import Callable, Dict, List, Tuple, Any
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from unittest.mock import patch
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter
from dotenv import load_dotenv

# Import custom helpers
from grader import math_equal
from metrics import extract_box

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
If you are confident in your answer, output <|c_math|>.
If you are unsure and want to defer to a larger model, output <|u_math|>.
Put your final answer within \\boxed{}."""

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
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
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
    state_dict = policy.state_dict()
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
def reward_fn(generated_text: str, ground_truth: str) -> float:
    """
    Confident Task Reward:
    - <u_math>: 0
    - <c_math> + correct: 1
    - <c_math> + incorrect: -1
    - No tag: 0 (or penalty)
    """
    
    # 1. Check for confidence tags
    has_c_math = "<|c_math|>" in generated_text
    has_u_math = "<|u_math|>" in generated_text
    
    if has_u_math:
        return 0.0
    
    if not has_c_math:
        # Failed to follow format (neither tag found)
        return 0.0 # Or small penalty
        
    # 2. If <c_math>, check correctness
    # Extract predicted answer
    pred_answer = extract_box(generated_text)
    
    # Extract ground truth answer (ground_truth is the solution string)
    # We assume ground_truth is the full solution text or just the answer.
    # For MATH-500, it's the full solution. We need to extract the box from it too.
    gt_answer = extract_box(ground_truth)
    if not gt_answer:
        gt_answer = ground_truth # Fallback if no box in GT (unlikely for MATH)

    if not pred_answer:
        return -1.0 # Confident but no answer found -> Incorrect

    is_correct = math_equal(pred_answer, gt_answer)
    
    if is_correct:
        return 1.0
    else:
        return -1.0

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

def rollout_with_vllm(policy: PreTrainedModel, llm: LLM, sampling_params: SamplingParams, prompts_batch: List[str], group_size: int) -> Tuple[List[str], List[str], List[int]]:
    load_policy_into_vllm_instance(policy, llm)
    prompts_dup = duplicate_data(prompts_batch, group_size)
    vllm_rollouts = llm.generate(prompts_dup, sampling_params, use_tqdm=False)
    rollout_input_text, rollout_response_text, rollout_output_tokens = [], [], []
    for rollout in vllm_rollouts:
        # vllm returns one output per prompt if n=1 (which we simulate by duplicating prompts)
        r = rollout.outputs[0]
        rollout_input_text.append(rollout.prompt)
        rollout_response_text.append(r.text)
        rollout_output_tokens.append(len(r.token_ids))
    return rollout_input_text, rollout_response_text, rollout_output_tokens

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
def train(
    policy: PreTrainedModel, tokenizer: AutoTokenizer, llm: LLM, sampling_params: SamplingParams, *,
    train_prompts: List[str], train_answers: List[str], 
    optimizer: torch.optim.Optimizer, scheduler, n_grpo_steps: int, rollout_batch_size: int,
    group_size: int, gradient_accumulation_steps: int, clip_range: float,
    advantage_eps: float, device: str, writer: SummaryWriter = None, seed: int,
) -> None:
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
        rollout_input, rollout_response, rollout_tokens = rollout_with_vllm(policy, llm, sampling_params, prompts_batch, group_size)
        answers_dup = duplicate_data(answers_batch, group_size)
        avg_output_tokens = sum(rollout_tokens) / len(rollout_tokens) if rollout_tokens else 0.0
        
        # Compute advantages
        advantages, raw_rewards, reward_meta = compute_group_normalized_advantages(
            rollout_response, answers_dup, reward_fn, group_size, advantage_eps
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

# -------------------------
# Main
# -------------------------
def init_policy(model_id: str, device: str) -> Tuple[PreTrainedModel, AutoTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", use_cache=False, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model.to(device).train()
    return model, tokenizer

def main():
    # Params
    model_id = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    device = "cuda"
    seed = 42
    # GRPO params
    n_grpo_steps = 100 # Adjust as needed
    rollout_batch_size = 64
    group_size = 8
    grad_acc_steps = 4
    lr = 1e-6
    clip_range = 0.2
    advantage_eps = 1e-6
    
    # Generation params
    temperature = 1.0
    max_tokens = 512
    
    # Load Data (MATH-500)
    print(f"Loading dataset HuggingFaceH4/MATH-500...")
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test") # Using test split as dataset
    
    # Init Policy
    print(f"Initializing policy {model_id}...")
    policy, tokenizer = init_policy(model_id, device)
    
    # Format data
    prompts = []
    answers = []
    for ex in dataset:
        prompts.append(format_prompt(ex["problem"], tokenizer))
        answers.append(ex["solution"]) # Ground truth is the solution text
    
    # Init vLLM
    print("Initializing vLLM...")
    llm = init_vllm(model_id, device, seed)
    sampling_params = init_sampling_params(temperature, max_tokens)
    
    # Optimizer
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=10)
    
    # Writer
    timestamp = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
    log_dir = f"./output/logs/grpo_{timestamp}"
    writer = SummaryWriter(log_dir=log_dir)
    
    # Train
    train(
        policy, tokenizer, llm, sampling_params,
        train_prompts=prompts, train_answers=answers,
        optimizer=optimizer, scheduler=scheduler,
        n_grpo_steps=n_grpo_steps, rollout_batch_size=rollout_batch_size,
        group_size=group_size, gradient_accumulation_steps=grad_acc_steps,
        clip_range=clip_range, advantage_eps=advantage_eps, device=device,
        writer=writer, seed=seed
    )
    
    # Save
    save_dir = f"./output/grpo_confident_{timestamp}"
    print(f"Saving model to {save_dir}")
    policy.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

if __name__ == "__main__":
    main()

