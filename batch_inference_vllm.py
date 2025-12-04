"""
Batch inference using vLLM on full dataset.
Saves results to CSV and tracks accuracy per batch.
"""
import os
import argparse
import time
import pandas as pd
from datasets import load_dataset
from vllm import LLM, SamplingParams
from tqdm import tqdm
from grader import math_equal
from metrics import extract_box


def format_prompt(question: str, tokenizer) -> str:
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


def load_full_dataset(dataset_name: str = "Akirayasha/math-20", split: str = "train"):
    """Load the full dataset."""
    print(f"Loading dataset: {dataset_name} (split: {split})")
    dataset = load_dataset(dataset_name, split=split)
    print(f"✓ Loaded {len(dataset)} samples")
    return dataset


def batch_inference(
    model_path: str,
    dataset_name: str = "Akirayasha/math-20",
    output_file: str = "./inference_results.csv",
    batch_size: int = 32,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.95,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
):
    """
    Run batch inference using vLLM.
    
    Args:
        model_path: Path to model or model name
        dataset_name: Dataset to use
        output_file: CSV file to save results
        batch_size: Number of samples to process at once
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: GPU memory utilization fraction
    """
    
    print("="*100)
    print("BATCH INFERENCE WITH vLLM")
    print("="*100)
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_name}")
    print(f"Batch size: {batch_size}")
    print(f"Max tokens: {max_tokens}")
    print(f"Temperature: {temperature}")
    print(f"Output: {output_file}")
    print("="*100)
    
    # Load model
    print("\n[1/3] Loading model with vLLM...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
    )
    
    tokenizer = llm.get_tokenizer()
    
    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    
    # Load dataset
    print("\n[2/3] Loading dataset...")
    dataset = load_full_dataset(dataset_name)
    
    # Prepare prompts
    print("\n[3/3] Preparing prompts...")
    prompts = []
    questions = []
    ground_truths = []
    
    for example in tqdm(dataset, desc="Preparing prompts"):
        question = example["question"]
        ground_truth = example.get("label", "") or ""  # Handle None values
        
        prompt = format_prompt(question, tokenizer)
        
        prompts.append(prompt)
        questions.append(question)
        ground_truths.append(ground_truth)
    
    # Run batch inference
    print("\n" + "="*100)
    print("RUNNING BATCH INFERENCE")
    print("="*100)
    
    all_results = []
    wrong_results = []  # Store only wrong answers
    total_correct = 0
    total_processed = 0
    
    num_batches = (len(prompts) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(prompts))
        
        batch_prompts = prompts[start_idx:end_idx]
        batch_questions = questions[start_idx:end_idx]
        batch_ground_truths = ground_truths[start_idx:end_idx]
        
        print(f"\n{'─'*100}")
        print(f"Batch {batch_idx + 1}/{num_batches} (samples {start_idx + 1}-{end_idx})")
        print(f"{'─'*100}")
        
        # Generate
        start_time = time.time()
        outputs = llm.generate(batch_prompts, sampling_params)
        inference_time = time.time() - start_time
        
        # Process results
        batch_correct = 0
        
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            predicted_answer = extract_box(generated_text)
            ground_truth = batch_ground_truths[i] or ""  # Handle None
            # Ground truth in this dataset is already the answer (not wrapped in \boxed{})
            gt_answer = ground_truth.strip() if ground_truth else ""
            
            # Debug: Print first few samples to see what's happening
            if batch_idx == 0 and i < 2:
                print(f"\n  DEBUG Sample {i+1}:")
                print(f"    Generated (first 200 chars): {generated_text[:200]}")
                print(f"    Predicted answer: {predicted_answer}")
                print(f"    Ground truth: {gt_answer}")
            
            # Check correctness
            is_correct = False
            if predicted_answer and gt_answer:
                is_correct = math_equal(predicted_answer, gt_answer)
            
            if is_correct:
                batch_correct += 1
            else:
                # Print wrong answers (only first 5 per batch to avoid spam)
                if batch_correct < 5 or not predicted_answer:
                    print(f"  ✗ Sample {start_idx + i + 1}: Wrong")
                    print(f"    Predicted: {predicted_answer}")
                    print(f"    Ground Truth: {gt_answer}")
                    if not predicted_answer:
                        print(f"    Generated (first 200 chars): {generated_text[:200]}")
                
                # Store wrong answer for manual review
                wrong_results.append({
                    "index": start_idx + i,
                    "question": batch_questions[i],
                    "ground_truth": ground_truth,
                    "gt_answer": gt_answer,
                    "predicted_answer": predicted_answer,
                    "generated_text": generated_text,
                })
            
            # Store all results
            all_results.append({
                "index": start_idx + i,
                "question": batch_questions[i],
                "ground_truth": ground_truth,
                "predicted_answer": predicted_answer,
                "generated_text": generated_text,
                "is_correct": is_correct,
            })
        
        total_correct += batch_correct
        total_processed += len(batch_prompts)
        
        # Batch statistics
        batch_acc = batch_correct / len(batch_prompts) * 100
        overall_acc = total_correct / total_processed * 100
        
        batch_no_answer = sum(1 for r in all_results[start_idx:end_idx] if not r['predicted_answer'])
        
        print(f"\n  Batch accuracy: {batch_correct}/{len(batch_prompts)} = {batch_acc:.2f}%")
        print(f"  Batch no answer: {batch_no_answer}/{len(batch_prompts)}")
        print(f"  Overall accuracy: {total_correct}/{total_processed} = {overall_acc:.2f}%")
        print(f"  Inference time: {inference_time:.2f}s ({inference_time/len(batch_prompts):.2f}s per sample)")
    
    # Save results to CSV
    print("\n" + "="*100)
    print("SAVING RESULTS")
    print("="*100)
    
    # Save all results
    df = pd.DataFrame(all_results)
    df.to_csv(output_file, index=False)
    print(f"✓ Saved {len(all_results)} results to: {output_file}")
    
    # Save wrong answers only for manual review
    wrong_output_file = output_file.replace('.csv', '_wrong_only.csv')
    df_wrong = pd.DataFrame(wrong_results)
    df_wrong.to_csv(wrong_output_file, index=False)
    print(f"✓ Saved {len(wrong_results)} wrong answers to: {wrong_output_file}")
    
    # Final summary
    print("\n" + "="*100)
    print("FINAL SUMMARY")
    print("="*100)
    print(f"Total samples: {total_processed}")
    print(f"Correct: {total_correct}")
    print(f"Wrong: {total_processed - total_correct}")
    print(f"No answer extracted: {sum(1 for r in all_results if not r['predicted_answer'])}")
    print(f"Accuracy: {overall_acc:.2f}%")
    print(f"\nWrong answers saved to: {wrong_output_file}")
    print("="*100)
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Batch inference with vLLM")
    parser.add_argument('--model', type=str, required=True,
                        help='Model path or name')
    parser.add_argument('--dataset', type=str, default="Akirayasha/math-20",
                        help='Dataset to use')
    parser.add_argument('--output', type=str, default="./inference_results.csv",
                        help='Output CSV file')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--max_tokens', type=int, default=1024,
                        help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='Top-p sampling')
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                        help='Number of GPUs for tensor parallelism')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9,
                        help='GPU memory utilization (0.0-1.0)')
    
    args = parser.parse_args()
    
    # Run inference
    batch_inference(
        model_path=args.model,
        dataset_name=args.dataset,
        output_file=args.output,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
