"""
Generate negative samples with uncertainty tokens.
Single model approach:
1. Model generates correct solution
2. Cut solution in the middle
3. Prompt model to generate wrong but soundly correct reasoning with <span>...</span> tags
"""
import os
import random
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from typing import List, Tuple, Dict, Optional
import json
import argparse
from grader import math_equal
from metrics import extract_box
import re


def load_model_and_tokenizer(
    model_id: str,
):
    """Load model and tokenizer."""
    print(f"Loading model: {model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    model.eval()
    return model, tokenizer


def format_prompt(question: str, tokenizer: AutoTokenizer) -> str:
    """Format question with system prompt."""
    system_msg = """You are a helpful assistant for mathematical problems. 
Please reason step by step.
Put your final answer within \\boxed{}.
"""
    
    conversation = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": question}
    ]
    
    prompt = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )
    
    return prompt


def simple_generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> str:
    """Simple generation without twisting."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    
    return generated_text


def extract_reasoning_steps(text: str) -> List[str]:
    """
    Extract reasoning steps from generated text.
    Split by common patterns like numbered steps, paragraphs, or sentences.
    """
    # Try to split by sentences ending with period
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Filter out very short sentences and empty ones
    steps = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    return steps


def create_wrong_reasoning_prompt(
    question: str,
    prefix_reasoning: str,
    tokenizer: AutoTokenizer,
) -> str:
    """
    Create a prompt for the model to generate wrong but soundly correct reasoning.
    The wrong reasoning should be wrapped in <span>...</span> tags.
    """
    system_msg = """You are a mathematical reasoning assistant that can continue generate reasoning from a checkpoint but soundly correct and lead to wrong answer.
Wrap your incorrect reasoning in <span> and </span> tags, then provide the wrong answer in \\boxed{}."""
    
    user_msg = f"""Question: {question}

Your wrong reasoning should:
1. Sound mathematically plausible
2. Current reasoning + <span>wrong reasoning</span>
3. Lead to an incorrect final answer in \\boxed{{}}

Reasoning so far:
{prefix_reasoning}

"""
    
    conversation = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]
    
    prompt = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )
    
    return prompt


def get_samples(
    dataset_name: str = "Akirayasha/math-20",
    num_samples: int = 10,
    seed: int = None,
    check_filter: str = None,
) -> List[Dict]:
    """
    Get random samples from dataset.
    
    Args:
        dataset_name: Name of the dataset
        num_samples: Number of samples to get
        seed: Random seed for reproducibility
        check_filter: Filter by check field - 'true', 'false', or None (no filter)
    """
    dataset = load_dataset(dataset_name, split="train")
    
    if seed is not None:
        random.seed(seed)
    
    # Filter by check field if specified
    if check_filter is not None:
        check_value = check_filter.lower() == 'true'
        filtered_indices = [i for i, ex in enumerate(dataset) if ex.get("check", True) == check_value]
        
        if not filtered_indices:
            print(f"⚠️  Warning: No examples found with check={check_value}")
            print(f"Using all examples instead.")
            filtered_indices = list(range(len(dataset)))
        else:
            print(f"✓ Filtered to {len(filtered_indices)} examples with check={check_value}")
        
        indices = random.sample(filtered_indices, min(num_samples, len(filtered_indices)))
    else:
        indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    samples = []
    for idx in indices:
        example = dataset[idx]
        samples.append({
            "question": example["question"],
            "ground_truth": example.get("label", ""),
            "full_solution": example.get("full_label", ""),
            "check": example.get("check", None),
            "index": idx,
        })
    
    return samples


def generate_negative_samples_single_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    samples: List[Dict],
    num_generations: int = 3,
    max_new_tokens: int = 512,
    seed: int = None,
    test_mode: bool = False,
) -> List[Dict]:
    """
    Single model approach:
    1. Model generates correct solution
    2. Cut solution in the middle  
    3. Prompt model to generate wrong but soundly correct reasoning with <span> tags
    
    Returns:
        List of negative samples (only those with wrong answers)
        In test mode: returns all samples for inspection
    """
    negative_samples = []
    
    for sample_idx, sample in enumerate(samples):
        question = sample['question']
        ground_truth = sample['ground_truth']
        
        print(f"\n{'='*100}")
        print(f"SAMPLE {sample_idx + 1}/{len(samples)}")
        print(f"{'='*100}")
        print(f"Question: {question}")
        print(f"Ground Truth: {ground_truth}")
        
        # Format prompt for correct answer
        prompt = format_prompt(question, tokenizer)
        
        # Generate multiple attempts
        for gen_idx in range(num_generations):
            print(f"\n  {'─'*90}")
            print(f"  Generation {gen_idx + 1}/{num_generations}")
            print(f"  {'─'*90}")
            
            # Step 1: Generate correct solution
            print(f"  [1/3] Generating correct solution...")
            original_solution = simple_generate(
                model, tokenizer, prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
            )
            
            # Check if original has an answer
            original_answer = extract_box(original_solution)
            if not original_answer:
                print(f"  ❌ Skipped: Original solution has no extractable answer")
                continue
            
            print(f"  ✓ Original answer: {original_answer}")
            
            # Step 2: Cut solution in the middle
            steps = extract_reasoning_steps(original_solution)
            if len(steps) < 3:
                print(f"  ❌ Skipped: Not enough reasoning steps ({len(steps)})")
                continue
            
            # Cut at middle (take first half)
            cut_point = len(steps) // 2
            prefix_steps = steps[:cut_point]
            prefix_reasoning = " ".join(prefix_steps)
            
            print(f"  [2/3] Creating wrong reasoning from step {cut_point}/{len(steps)}...")
            
            # Step 3: Generate wrong but soundly correct reasoning
            wrong_prompt = create_wrong_reasoning_prompt(
                question, prefix_reasoning, tokenizer
            )
            
            wrong_solution = simple_generate(
                model, tokenizer, wrong_prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.8,
            )
            
            print(f"  ✓ Wrong reasoning generated")
            
            # Combine: prefix + wrong reasoning
            full_response = prefix_reasoning + " " + wrong_solution
            
            # Extract final answer from wrong solution
            final_answer = extract_box(wrong_solution)
            gt_answer = ground_truth.strip() if ground_truth else ""
            
            # Check if <span> tags are present
            has_span_tags = '<span>' in wrong_solution and '</span>' in wrong_solution
            
            if not has_span_tags:
                print(f"  ⚠️  Warning: No <span> tags found in wrong reasoning")
            
            # Check if we got a wrong answer
            if final_answer and gt_answer:
                is_correct = math_equal(final_answer, gt_answer)
                
                # In test mode, save everything for inspection
                if test_mode or (not is_correct and has_span_tags):
                    negative_samples.append({
                        "question": question,
                        "ground_truth": ground_truth,
                        "original_solution": original_solution,
                        "original_answer": original_answer,
                        "prefix_reasoning": prefix_reasoning,
                        "cut_point": cut_point,
                        "wrong_solution": wrong_solution,
                        "final_answer": final_answer,
                        "full_response": full_response,
                        "has_span_tags": has_span_tags,
                        "is_correct": is_correct,
                        "original_index": sample['index'],
                    })
                    
                    if not is_correct and has_span_tags:
                        print(f"\n  ✅ SUCCESS! Wrong answer with <span> tags")
                        print(f"     Original: {original_answer}")
                        print(f"     After wrong reasoning: {final_answer}")
                        print(f"     Ground truth: {gt_answer}")
                    elif is_correct:
                        print(f"  ℹ️  Still correct after wrong reasoning: {final_answer}")
                        if test_mode:
                            print(f"     (Saved for inspection in test mode)")
                    else:
                        print(f"  ℹ️  Wrong but no <span> tags")
                else:
                    print(f"  ❌ Skipped: Correct answer or missing <span> tags")
            else:
                print(f"  ❌ No answer extracted from wrong solution")
        
        print()
    
    return negative_samples


def save_negative_samples(
    negative_samples: List[Dict],
    output_file: str,
):
    """Save negative samples to JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(negative_samples, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved {len(negative_samples)} negative samples to: {output_file}")


def format_negative_sample_for_sft(sample: Dict) -> Dict:
    """
    Format negative sample for SFT training with <span> and <u_math> tags.
    """
    # Add <span> before the twisted section and <u_math> at the end
    response = sample['response']
    
    # Simple approach: add <u_math> at the end before the final answer
    # In practice, you'd want to mark the span more precisely
    formatted_response = response.strip()
    if not formatted_response.endswith("<u_math>"):
        formatted_response += " <u_math>"
    
    return {
        "question": sample['question'],
        "answer": formatted_response,
        "ground_truth": sample['ground_truth'],
        "predicted_answer": sample['predicted_answer'],
    }


def main():
    parser = argparse.ArgumentParser(description="Generate negative samples with single model approach")
    parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-Math-7B-Instruct",
                        help='Model for generation')
    parser.add_argument('--dataset', type=str, default="Akirayasha/math-20",
                        help='Dataset to sample questions from')
    parser.add_argument('--test', action='store_true',
                        help='Test mode: saves all samples (both correct and wrong) for inspection')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to process')
    parser.add_argument('--num_generations', type=int, default=3,
                        help='Number of attempts per sample')
    parser.add_argument('--max_tokens', type=int, default=1024,
                        help='Maximum tokens to generate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output', type=str, default="./negative_samples.json",
                        help='Output JSON file')
    parser.add_argument('--check', type=str, default=None, choices=['true', 'false', None],
                        help='Filter questions by check field (true/false)')
    
    args = parser.parse_args()
    
    # Test mode - just enables saving all samples, doesn't override counts
    if args.test:
        print("🧪 TEST MODE: Will save all samples (both correct and wrong) for inspection")
    
    print("="*100)
    print("NEGATIVE SAMPLE GENERATION - SINGLE MODEL WITH <SPAN> TAGS")
    print("="*100)
    print(f"Model: {args.model}")
    print(f"Samples to process: {args.num_samples}")
    print(f"Generations per sample: {args.num_generations}")
    print(f"Seed: {args.seed}")
    print("="*100)
    
    # Load model
    print("\n[1/2] Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.model)
    
    # Get samples
    print(f"\n[2/2] Loading {args.num_samples} samples from dataset...")
    samples = get_samples(
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        seed=args.seed,
        check_filter=args.check,
    )
    
    # Generate negative samples
    print("\n" + "="*100)
    print("GENERATING NEGATIVE SAMPLES")
    print("="*100)
    
    negative_samples = generate_negative_samples_single_model(
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        num_generations=args.num_generations,
        max_new_tokens=args.max_tokens,
        seed=args.seed,
        test_mode=args.test,
    )
    
    # Save results
    if negative_samples:
        save_negative_samples(negative_samples, args.output)
        
        # Print summary
        print("\n" + "="*100)
        print("SUMMARY")
        print("="*100)
        print(f"Total samples processed: {len(samples)}")
        print(f"Total generations attempted: {len(samples) * args.num_generations}")
        
        if args.test:
            wrong_count = sum(1 for s in negative_samples if not s.get('is_correct', False))
            correct_count = sum(1 for s in negative_samples if s.get('is_correct', False))
            span_count = sum(1 for s in negative_samples if s.get('has_span_tags', False))
            print(f"Samples saved (test mode - all): {len(negative_samples)}")
            print(f"  - Wrong answers: {wrong_count}")
            print(f"  - Still correct: {correct_count}")
            print(f"  - With <span> tags: {span_count}")
        else:
            span_count = sum(1 for s in negative_samples if s.get('has_span_tags', False))
            print(f"Valid negative samples (wrong answers): {len(negative_samples)}")
            print(f"  - With <span> tags: {span_count}")
            print(f"Success rate: {len(negative_samples) / (len(samples) * args.num_generations) * 100:.1f}%")
        
        # Show examples with detailed view
        print("\n" + "="*100)
        print("DETAILED SAMPLES" if args.test else "EXAMPLE NEGATIVE SAMPLES")
        print("="*100)
        
        samples_to_show = negative_samples if args.test else negative_samples[:2]
        
        for i, sample in enumerate(samples_to_show):
            status = "✓ CORRECT" if sample.get('is_correct', False) else "✗ WRONG"
            span_status = "✓" if sample.get('has_span_tags', False) else "✗"
            
            print(f"\n{'─'*100}")
            print(f"Sample {i+1}: {status} | <span> tags: {span_status}")
            print(f"{'─'*100}")
            print(f"\nQuestion: {sample['question']}\n")
            print(f"Ground Truth: {sample['ground_truth']}\n")
            
            print(f"[1] ORIGINAL SOLUTION (Correct)")
            print(f"    Answer: {sample['original_answer']}")
            print(f"    Preview: {sample['original_solution'][:200]}...\n")
            
            print(f"[2] PREFIX REASONING (First half before cut)")
            print(f"    Cut at step {sample['cut_point']}")
            print(f"    {sample['prefix_reasoning'][:200]}...\n")
            
            print(f"[3] WRONG REASONING (With <span> tags)")
            print(f"    {sample['wrong_solution'][:300]}...\n")
            
            print(f"[4] FINAL ANSWER")
            print(f"    Answer: {sample['final_answer']} ({status})")
            print(f"{'─'*100}")
    else:
        print("\n⚠️  No samples generated!")
        print("Try adjusting the parameters or running with more samples.")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
