"""
Prepare dataset for SFT cold start and GRPO training.
- 25% for SFT (supervised fine-tuning with tagged_response)
- 75% for GRPO (reinforcement learning)
"""
import random
from datasets import load_dataset
from typing import List, Dict, Tuple

def load_and_split_data(
    dataset_name: str = "Akirayasha/math-20",
    sft_ratio: float = 0.25,
    seed: int = 42,
    balance_check_field: bool = True
) -> Tuple[List[Dict], List[Dict]]:
    """
    Load dataset and split into SFT and GRPO portions.
    
    Args:
        dataset_name: Name of the dataset to load
        sft_ratio: Ratio of data to use for SFT (default: 0.25 = 25%)
        seed: Random seed for reproducibility
        balance_check_field: If True, ensure equal True/False in check field for SFT data
    
    Returns:
        sft_data: List of examples for supervised fine-tuning
        grpo_data: List of examples for GRPO training
    """
    print(f"Loading dataset {dataset_name}...")
    dataset = load_dataset(dataset_name, split="train")
    
    # Convert to list for easier manipulation
    all_data = [ex for ex in dataset]
    
    # Shuffle data
    random.seed(seed)
    random.shuffle(all_data)
    
    if balance_check_field:
        # Separate by check field (True/False)
        true_examples = [ex for ex in all_data if ex.get("check", True) == True]
        false_examples = [ex for ex in all_data if ex.get("check", True) == False]
        
        print(f"Total True examples: {len(true_examples)}")
        print(f"Total False examples: {len(false_examples)}")
        
        # Calculate how many from each to get sft_ratio of total
        total_sft_count = int(len(all_data) * sft_ratio)
        sft_per_class = total_sft_count // 2  # Equal split
        
        # Take equal amounts from each
        sft_true = true_examples[:sft_per_class]
        sft_false = false_examples[:sft_per_class]
        sft_data = sft_true + sft_false
        
        # Remaining data for GRPO
        grpo_true = true_examples[sft_per_class:]
        grpo_false = false_examples[sft_per_class:]
        grpo_data = grpo_true + grpo_false
        
        # Shuffle SFT and GRPO data
        random.shuffle(sft_data)
        random.shuffle(grpo_data)
        
        print(f"\nTotal examples: {len(all_data)}")
        print(f"SFT examples: {len(sft_data)} ({len(sft_data)/len(all_data)*100:.1f}%)")
        print(f"  - True: {len(sft_true)} ({len(sft_true)/len(sft_data)*100:.1f}%)")
        print(f"  - False: {len(sft_false)} ({len(sft_false)/len(sft_data)*100:.1f}%)")
        print(f"GRPO examples: {len(grpo_data)} ({len(grpo_data)/len(all_data)*100:.1f}%)")
        print(f"  - True: {len(grpo_true)}")
        print(f"  - False: {len(grpo_false)}")
    else:
        # Original split without balancing
        split_idx = int(len(all_data) * sft_ratio)
        sft_data = all_data[:split_idx]
        grpo_data = all_data[split_idx:]
        
        print(f"Total examples: {len(all_data)}")
        print(f"SFT examples: {len(sft_data)} ({sft_ratio*100:.1f}%)")
        print(f"GRPO examples: {len(grpo_data)} ({(1-sft_ratio)*100:.1f}%)")
    
    return sft_data, grpo_data

def format_sft_example(example: Dict, add_confidence: bool = True) -> Dict:
    """
    Format example for SFT training.
    Input: question
    Output: tagged_response (with corrected confidence tags)
    """
    question = example["question"]
    
    # Use tagged_response field from dataset
    answer = example.get("tagged_response", example.get("full_label", ""))
    
    answer = answer.replace("<C_MATH>", "<c_math>")
    answer = answer.replace("<U_MATH>", "<u_math>")
    
    return {
        "question": question,
        "answer": answer,
        "label": example.get("label", ""),  # Keep for reference
        "check": example.get("check", True)  # Keep check field
    }

def format_grpo_example(example: Dict) -> Dict:
    """
    Format example for GRPO training.
    We only need the question and ground truth for reward computation.
    """
    return {
        "question": example["question"],
        "solution": example["full_label"],  # For reward computation
        "label": example["label"]  # Short answer for easier grading
    }

if __name__ == "__main__":
    # Test the data preparation
    sft_data, grpo_data = load_and_split_data()
    
    # Separate true and false examples from SFT data
    sft_true_examples = [ex for ex in sft_data if ex.get("check", True) == True]
    sft_false_examples = [ex for ex in sft_data if ex.get("check", True) == False]
    
    print("\n" + "="*80)
    print("SFT EXAMPLES WITH check=True (2 samples)")
    print("="*80)
    
    for i in range(min(2, len(sft_true_examples))):
        sft_ex = format_sft_example(sft_true_examples[i])
        print(f"\n--- TRUE Example {i+1} ---")
        print(f"Check field: {sft_ex['check']}")
        print(f"Question: {sft_ex['question']}...")
        print(f"\nAnswer (first 400 chars): {sft_ex['answer']}...")
        if len(sft_ex['answer']) > 400:
            print(f"... (total length: {len(sft_ex['answer'])} chars)")
        # Check if tags are present
        has_c_math = "<c_math>" in sft_ex['answer']
        has_u_math = "<u_math>" in sft_ex['answer']
        print(f"Contains <c_math>: {has_c_math}")
        print(f"Contains <u_math>: {has_u_math}")

    print("\n" + "="*80)
    print("SFT EXAMPLES WITH check=False (2 samples)")
    print("="*80)
    
    for i in range(min(2, len(sft_false_examples))):
        sft_ex = format_sft_example(sft_false_examples[i])
        print(f"\n--- FALSE Example {i+1} ---")
        print(f"Check field: {sft_ex['check']}")
        print(f"Question: {sft_ex['question']}...")
        print(f"\nAnswer (first 400 chars): {sft_ex['answer']}...")
        if len(sft_ex['answer']) > 400:
            print(f"... (total length: {len(sft_ex['answer'])} chars)")
        # Check if tags are present
        has_c_math = "<c_math>" in sft_ex['answer']
        has_u_math = "<u_math>" in sft_ex['answer']
        print(f"Contains <c_math>: {has_c_math}")
        print(f"Contains <u_math>: {has_u_math}")

    print("\n" + "="*80)
    print("GRPO Example (for reference):")
    print("="*80)
    grpo_ex = format_grpo_example(grpo_data[0])
    print(f"Question: {grpo_ex['question']}...")
    print(f"\nSolution: {grpo_ex['solution']}...")
    print(f"Label: {grpo_ex['label']}")
