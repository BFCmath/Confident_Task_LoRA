"""
Analyze logits of generated answers for a random question.
Generate multiple versions and inspect token-level logits.
"""
import os
import random
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from typing import List, Tuple, Dict
import json

def load_model_and_tokenizer(
    base_model_id: str = "Qwen/Qwen2.5-Math-1.5B-Instruct",
    lora_path: str = None,
):
    """Load model (with optional LoRA) and tokenizer."""
    print(f"Loading base model: {base_model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    if lora_path:
        print(f"Loading LoRA adapter from: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()  # Merge for faster inference
    
    model.eval()
    return model, tokenizer


def format_prompt(question: str, tokenizer: AutoTokenizer) -> str:
    """Format question with system prompt."""
    system_msg = """You are a helpful assistant for mathematical problems. 
Please reason step by step.
At the end, you must verify your confidence.
If you are confident in your answer, output <c_math>.
If you are unsure, output <u_math>.
Put your final answer within \\boxed{} and verify your confidence.
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


def generate_with_logits(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 0.95,
) -> Tuple[str, List[Dict]]:
    """
    Generate response and capture logits for each token.
    
    Returns:
        response: Generated text
        token_logits: List of dicts with token, logit, and probability info
    """
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]
    
    # Generate with output_scores and return_dict_in_generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Extract generated tokens (excluding input)
    generated_ids = outputs.sequences[0][input_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    
    # Extract logits for each generated token
    token_logits = []
    for i, (token_id, scores) in enumerate(zip(generated_ids, outputs.scores)):
        # scores is a tensor of shape [batch_size, vocab_size]
        logits = scores[0]  # Get logits for batch item 0
        
        # Get probability distribution (softmax)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Get info for the selected token
        selected_token_id = token_id.item()
        selected_token = tokenizer.decode([selected_token_id])
        selected_logit = logits[selected_token_id].item()
        selected_prob = probs[selected_token_id].item()
        
        # Get top-5 alternative tokens
        top5_probs, top5_indices = torch.topk(probs, k=5)
        top5_tokens = [
            {
                "token": tokenizer.decode([idx.item()]),
                "token_id": idx.item(),
                "logit": logits[idx].item(),
                "prob": prob.item(),
            }
            for idx, prob in zip(top5_indices, top5_probs)
        ]
        
        token_info = {
            "position": i,
            "token": selected_token,
            "token_id": selected_token_id,
            "logit": selected_logit,
            "probability": selected_prob,
            "top5_alternatives": top5_tokens,
            "is_confidence_token": selected_token in ["<c_math>", "<u_math>", "<|c_math|>", "<|u_math|>"],
            "is_special": selected_token_id in tokenizer.all_special_ids,
        }
        
        token_logits.append(token_info)
    
    return generated_text, token_logits


def get_random_question(dataset_name: str = "Akirayasha/math-20", seed: int = None, check_filter: str = None) -> Dict:
    """
    Get a random question from dataset.
    
    Args:
        dataset_name: Name of the dataset
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
        
        idx = filtered_indices[random.randint(0, len(filtered_indices) - 1)]
    else:
        idx = random.randint(0, len(dataset) - 1)
    
    example = dataset[idx]
    
    return {
        "question": example["question"],
        "ground_truth": example.get("label", ""),
        "full_solution": example.get("full_label", ""),
        "check": example.get("check", None),
        "index": idx,
    }


def print_logits_analysis(
    question: str,
    response: str,
    token_logits: List[Dict],
    show_top_k: int = 3,
    min_prob_threshold: float = 0.1,
):
    """Print detailed logits analysis."""
    print("\n" + "="*100)
    print("LOGITS ANALYSIS")
    print("="*100)
    
    print(f"\nQuestion: {question[:150]}...")
    print(f"\nGenerated Response:\n{response}")
    print("\n" + "-"*100)
    print("TOKEN-BY-TOKEN ANALYSIS")
    print("-"*100)
    
    for token_info in token_logits:
        pos = token_info['position']
        token = token_info['token']
        logit = token_info['logit']
        prob = token_info['probability']
        
        # Color code based on probability
        if prob > 0.9:
            confidence = "🟢 VERY HIGH"
        elif prob > 0.7:
            confidence = "🟡 HIGH"
        elif prob > 0.5:
            confidence = "🟠 MEDIUM"
        else:
            confidence = "🔴 LOW"
        
        print(f"\n[Position {pos}] Token: '{token}' (ID: {token_info['token_id']})")
        print(f"  Logit: {logit:.4f} | Probability: {prob:.4f} ({prob*100:.2f}%) {confidence}")
        
        # Show special token indicators
        if token_info['is_confidence_token']:
            print(f"  ⭐ CONFIDENCE TOKEN DETECTED!")
        if token_info['is_special']:
            print(f"  🔖 Special token")
        
        # Show top alternatives only if selected token is not very confident
        if prob < 0.9:
            print(f"  Top {show_top_k} alternatives:")
            for i, alt in enumerate(token_info['top5_alternatives'][:show_top_k], 1):
                if i == 1 and alt['token_id'] == token_info['token_id']:
                    continue  # Skip if it's the same as selected
                alt_marker = "  ⚠️ " if alt['prob'] > min_prob_threshold else "     "
                print(f"{alt_marker}  {i}. '{alt['token']}' - Logit: {alt['logit']:.4f}, Prob: {alt['prob']:.4f} ({alt['prob']*100:.2f}%)")
    
    print("\n" + "="*100)
    
    # Summary statistics
    probs = [t['probability'] for t in token_logits]
    print("\nSUMMARY STATISTICS:")
    print(f"  Total tokens generated: {len(token_logits)}")
    print(f"  Average probability: {np.mean(probs):.4f}")
    print(f"  Min probability: {np.min(probs):.4f} at position {np.argmin(probs)}")
    print(f"  Max probability: {np.max(probs):.4f} at position {np.argmax(probs)}")
    
    # Count confidence tokens
    confidence_tokens = [t for t in token_logits if t['is_confidence_token']]
    if confidence_tokens:
        print(f"\n  Confidence tokens found: {len(confidence_tokens)}")
        for ct in confidence_tokens:
            print(f"    - '{ct['token']}' at position {ct['position']} (prob: {ct['probability']:.4f})")
    else:
        print(f"\n  ⚠️  No confidence tokens found!")
    
    print("="*100)


def save_logits_to_json(
    output_file: str,
    question: str,
    responses_data: List[Dict],
):
    """Save all logits data to JSON file."""
    data = {
        "question": question,
        "num_responses": len(responses_data),
        "responses": responses_data,
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved logits data to: {output_file}")


def save_logits_to_txt(
    output_file: str,
    question: str,
    ground_truth: str,
    responses_data: List[Dict],
):
    """Save simplified logits data to TXT file for easy tracking."""
    from grader import math_equal
    from metrics import extract_box
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("LOGITS ANALYSIS - SIMPLIFIED FORMAT\n")
        f.write("="*80 + "\n\n")
        
        # Question
        f.write(f"QUESTION:\n{question}\n\n")
        f.write(f"GROUND TRUTH:\n{ground_truth}\n\n")
        f.write("="*80 + "\n\n")
        
        # For each generation
        for i, resp_data in enumerate(responses_data, 1):
            response = resp_data['response']
            token_logits = resp_data['token_logits']
            
            # Extract answer and check correctness
            pred_answer = extract_box(response)
            gt_answer = extract_box(ground_truth) or ground_truth
            
            is_correct = False
            have_answer = bool(pred_answer)
            
            if pred_answer and gt_answer:
                is_correct = math_equal(pred_answer, gt_answer)
            
            # Header for this generation
            f.write(f"GENERATION {i}\n")
            f.write("-"*80 + "\n")
            f.write(f"true|false: {is_correct}\n")
            f.write(f"have_answer: {have_answer}\n")
            if have_answer:
                f.write(f"predicted_answer: {pred_answer}\n")
            f.write(f"\nRESPONSE:\n{response}\n\n")
            
            # Token probabilities
            f.write("DETAIL PROBABILITIES:\n")
            f.write("-"*80 + "\n")
            for token_info in token_logits:
                token = token_info['token']
                prob = token_info['probability']
                # Format: token | probability
                f.write(f"{token} | {prob:.6f}\n")
            
            f.write("\n" + "="*80 + "\n\n")
    
    print(f"\n✅ Saved simplified logits analysis to: {output_file}")


def main(
    base_model_id: str = "Qwen/Qwen2.5-Math-1.5B-Instruct",
    lora_path: str = None,
    dataset_name: str = "Akirayasha/math-20",
    num_generations: int = 5,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    seed: int = None,
    output_file: str = "./logits_analysis.json",
    check_filter: str = None,
):
    """Main function to analyze logits."""
    
    print("="*100)
    print("LOGITS ANALYSIS FOR GENERATED ANSWERS")
    print("="*100)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(base_model_id, lora_path)
    
    # Get random question
    question_data = get_random_question(dataset_name, seed, check_filter)
    question = question_data['question']
    
    print(f"\n📝 Selected question (index {question_data['index']}):")
    print(f"Question: {question}")
    print(f"Ground truth: {question_data['ground_truth']}")
    print(f"Check field: {question_data.get('check', 'N/A')}")
    
    # Format prompt
    prompt = format_prompt(question, tokenizer)
    
    # Generate multiple responses
    responses_data = []
    
    for i in range(num_generations):
        print(f"\n{'='*100}")
        print(f"GENERATION {i+1}/{num_generations}")
        print(f"{'='*100}")
        
        response, token_logits = generate_with_logits(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        
        # Print analysis
        print_logits_analysis(question, response, token_logits)
        
        # Store data
        responses_data.append({
            "generation_id": i + 1,
            "response": response,
            "token_logits": token_logits,
            "num_tokens": len(token_logits),
        })
        
        # Short separator between generations
        print("\n" + "🔄"*50 + "\n")
    
    # Save to JSON
    save_logits_to_json(output_file, question, responses_data)
    
    # Save to simplified TXT format
    txt_file = output_file.replace('.json', '.txt')
    save_logits_to_txt(
        txt_file,
        question,
        question_data['ground_truth'],
        responses_data
    )
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze logits of generated answers")
    parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-Math-1.5B-Instruct",
                        help='Base model name')
    parser.add_argument('--lora_path', type=str, default=None,
                        help='Path to LoRA adapter (optional)')
    parser.add_argument('--dataset', type=str, default="Akirayasha/math-20",
                        help='Dataset to sample question from')
    parser.add_argument('--num_generations', type=int, default=5,
                        help='Number of responses to generate')
    parser.add_argument('--max_tokens', type=int, default=512,
                        help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for question selection')
    parser.add_argument('--output', type=str, default="./logits_analysis.json",
                        help='Output JSON file')
    parser.add_argument('--check', type=str, default=None, choices=['true', 'false', None],
                        help='Filter questions by check field (true/false)')
    
    args = parser.parse_args()
    
    main(
        base_model_id=args.model,
        lora_path=args.lora_path,
        dataset_name=args.dataset,
        num_generations=args.num_generations,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
        output_file=args.output,
        check_filter=args.check,
    )
