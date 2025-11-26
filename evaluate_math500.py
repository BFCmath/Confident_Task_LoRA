"""
Evaluate LoRA adapter on MATH-500 dataset
Analyze accuracy and confidence token usage (<c_math> and <u_math>)
"""
import os
import logging
from dotenv import load_dotenv

load_dotenv()
access_token = os.getenv("HF_ACCESS_TOKEN")

logging.getLogger("vllm.engine.scheduler").setLevel(logging.ERROR)
os.environ["VLLM_USE_V1"] = "0"

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from metrics import validate_causal
from read_data import read_test_data, convert_to_qwen_format_user
from utils import set_seed

from huggingface_hub import login

def evaluate_math500(
    model_name: str = "Qwen/Qwen2.5-Math-1.5B-Instruct",
    lora_path: str = "./output/grpo_confident_lora_1763743491",
    output_dir: str = "./evaluation_results",
    n_samples: int = None,
    seed: int = 42,
    tensor_parallel_size: int = 1,
    max_tokens: int = 2048,
):
    """
    Evaluate LoRA adapter on MATH-500 dataset.
    
    Args:
        model_name: Base model name
        lora_path: Path to LoRA adapter
        output_dir: Directory to save results
        n_samples: Number of samples to evaluate (None = all 500)
        seed: Random seed
        tensor_parallel_size: Number of GPUs for tensor parallelism
        max_tokens: Maximum tokens to generate
    """
    
    # Login to HuggingFace
    if access_token:
        login(token=access_token)
    
    # Set seed
    set_seed(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("MATH-500 Evaluation with Confident Task LoRA")
    print("="*80)
    print(f"Base Model: {model_name}")
    print(f"LoRA Adapter: {lora_path}")
    print(f"Output Directory: {output_dir}")
    print(f"Samples: {n_samples if n_samples else 'All (500)'}")
    print("="*80)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load test data
    print("\nLoading MATH-500 dataset...")
    test_df = read_test_data('maths', n_samples=n_samples)
    questions = test_df['question'].tolist()
    answers = test_df['response'].tolist()
    
    print(f"Loaded {len(questions)} examples")
    
    # Format questions for Qwen
    print("\nFormatting questions...")
    formatted_questions = [convert_to_qwen_format_user(q) for q in questions]
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0,  # Greedy decoding for evaluation
        top_p=1.0,
        max_tokens=max_tokens,
        stop=["</s>", "<|im_end|>", "<c_math>", "<u_math>"],
        include_stop_str_in_output=True
    )
    
    # Initialize vLLM with LoRA
    print("\nInitializing vLLM with LoRA adapter...")
    llm = LLM(
        model=model_name, 
        enable_lora=True, 
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_tokens,
        gpu_memory_utilization=0.9,
    )
    
    # Create LoRA request
    lora_request = LoRARequest("confident_task_lora", 1, lora_path)
    
    # Run evaluation
    print("\nRunning evaluation...")
    results = validate_causal(
        llm=llm,
        lora_request=lora_request,
        sampling_params=sampling_params,
        questions=formatted_questions,
        labels=answers,
        question_type='maths',
        tokenizer=tokenizer,
        name=os.path.join(output_dir, 'results.csv'),
        original_questions=questions,
    )
    
    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"Confidence Accuracy: {results['confidence_accuracy']:.4f} ({results['confidence_accuracy']*100:.2f}%)")
    print(f"Average Generated Tokens: {results['avg_generated_tokens']:.2f}")
    print(f"\nConfidence Token Distribution:")
    for token, count in results['confidence_token_counts'].items():
        percentage = (count / len(questions)) * 100
        print(f"  {token}: {count} ({percentage:.2f}%)")
    print("="*80)
    
    # Save summary to text file
    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MATH-500 Evaluation Results\n")
        f.write("="*80 + "\n")
        f.write(f"Base Model: {model_name}\n")
        f.write(f"LoRA Adapter: {lora_path}\n")
        f.write(f"Number of Samples: {len(questions)}\n")
        f.write(f"Seed: {seed}\n")
        f.write("\n" + "="*80 + "\n")
        f.write("METRICS\n")
        f.write("="*80 + "\n")
        f.write(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)\n")
        f.write(f"Confidence Accuracy: {results['confidence_accuracy']:.4f} ({results['confidence_accuracy']*100:.2f}%)\n")
        f.write(f"Average Generated Tokens: {results['avg_generated_tokens']:.2f}\n")
        f.write("\n" + "="*80 + "\n")
        f.write("CONFIDENCE TOKEN DISTRIBUTION\n")
        f.write("="*80 + "\n")
        for token, count in results['confidence_token_counts'].items():
            percentage = (count / len(questions)) * 100
            f.write(f"{token}: {count} ({percentage:.2f}%)\n")
        f.write("="*80 + "\n")
    
    print(f"\nResults saved to:")
    print(f"  - CSV: {os.path.join(output_dir, 'results.csv')}")
    print(f"  - Summary: {summary_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate LoRA adapter on MATH-500")
    parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-Math-1.5B-Instruct",
                        help='Base model name')
    parser.add_argument('--lora_path', type=str, required=True,
                        help='Path to LoRA adapter (e.g., ./output/sft_cold_start_1763741962)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results (default: auto-generated from lora_path)')
    parser.add_argument('--n_samples', type=int, default=None,
                        help='Number of samples to evaluate (default: all 500)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                        help='Number of GPUs for tensor parallelism')
    parser.add_argument('--max_tokens', type=int, default=2048,
                        help='Maximum tokens to generate')
    
    args = parser.parse_args()
    
    # Handle the case where user provides parent directory instead of final directory
    lora_path = args.lora_path
    if not os.path.exists(os.path.join(lora_path, 'adapter_config.json')):
        # Check if there's a 'final' subdirectory with the adapter
        final_path = os.path.join(lora_path, 'final')
        if os.path.exists(os.path.join(final_path, 'adapter_config.json')):
            print(f"Found adapter in 'final' subdirectory: {final_path}")
            lora_path = final_path
        else:
            print(f"Warning: adapter_config.json not found in {lora_path}")
            print(f"Also checked: {final_path}")
    
    # Auto-generate output directory from lora_path if not provided
    if args.output_dir is None:
        # Extract the model name from the path
        # E.g., ./output/sft_cold_start_1763741962/final -> sft_cold_start_1763741962
        # E.g., ./output/grpo_confident_lora_1763743491 -> grpo_confident_lora_1763743491
        path_parts = lora_path.rstrip('/').split('/')
        if path_parts[-1] == 'final':
            lora_name = path_parts[-2]  # Use parent directory name
        else:
            lora_name = path_parts[-1]
        args.output_dir = f"./evaluation_results/{lora_name}"
        print(f"Auto-generated output directory: {args.output_dir}")
    
    results = evaluate_math500(
        model_name=args.model,
        lora_path=lora_path,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        seed=args.seed,
        tensor_parallel_size=args.tensor_parallel_size,
        max_tokens=args.max_tokens,
    )
