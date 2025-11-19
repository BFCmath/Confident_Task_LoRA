import pandas as pd
import torch
from tqdm import tqdm
from grader import math_equal
import swifter
import re
import time
from huggingface_hub import HfApi

"""
This file is to compute the metrics for the generated answers. It extracts answer from the generated text and compare it with the label.
The metrics include accuracy, average latency, and average generated tokens.
"""
# ---------------------------
# New Answer Extraction with boxed format
# ---------------------------
def extract_box(text: str) -> str:
    """
    Extract the last complete LaTeX expression inside \boxed{...},
    correctly handling nested braces and avoiding regex recursion.
    """
    last_box_start = text.rfind(r"\boxed{")
    if last_box_start == -1:
        return ""

    i = last_box_start + len(r"\boxed{")
    depth = 1
    content = []
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                break
        if depth > 0:
            content.append(text[i])
        i += 1
    
    result = "".join(content).strip()

    # --- optional cleanup: remove redundant outer braces like {{8}} or {8} ---
    while result.startswith("{") and result.endswith("}"):
        inner = result[1:-1].strip()
        # stop if braces inside are unbalanced
        if inner.count("{") != inner.count("}"):
            break
        result = inner

    return result


def len_extract_boxed(text):
    matches = re.findall(r"\\boxed\{(.*?)\}", text)
    return len(matches[-1].strip()) if matches else 0


def extract_confidence_token(text):
    # if "<C_MED>" in text:
    #     return "<C_MED>"
    # elif "<U_MED>" in text:
    #     return "<U_MED>"
    if "<|c_math|>" in text:
        return "<|c_math|>"
    elif "<|u_math|>" in text:
        return "<|u_math|>"
    else:
        return "No Confidence Token"

def compute_metrics(decoded_preds, decoded_labels, all_questions, question_type, tokenizer=None, latencies=None, save=None):
    # decoded_preds: predicted seq
    # decoded_labels: gold seq
    # tokenizer: tokenizer for computing token counts
    # latencies: list of latencies for each prediction
    full_preds, full_labels = decoded_preds, decoded_labels
    if question_type == 'maths':
        decoded_preds = [extract_box(i) for i in decoded_preds]
        decoded_labels = [extract_box(i) for i in decoded_labels]
    # print(decoded_labels[0])

    if question_type == 'medqa':
        decoded_preds = [i.strip().split('\n')[0][0] for i in decoded_preds]
        decoded_labels = [str(i).strip() for i in decoded_labels]
    # Compute token counts for full predictions
    token_counts = []
    if tokenizer is not None:
        for pred in full_preds:
            try:
                tokens = tokenizer.tokenize(str(pred))
                token_counts.append(len(tokens))
            except:
                token_counts.append(0)
    else:
        token_counts = [0] * len(full_preds)

    confidence_tokens = [extract_confidence_token(text) for text in full_preds]

    # Create DataFrame with additional metrics
    df_data = {
        'pred': decoded_preds, 
        'label': decoded_labels, 
        'full_pred': full_preds, 
        'full_label': full_labels, 
        'question': all_questions,
        'token_count': token_counts,
        'confidence_token': confidence_tokens
    }
    
    # # Add latency if provided
    # if latencies is not None:
    #     df_data['latency'] = latencies
    # else:
    #     df_data['latency'] = [0.0] * len(decoded_preds)
    
    df = pd.DataFrame(df_data)

    preds = df['pred'].tolist()
    labels = df['label'].tolist()

    df['pair'] = [(str(pred), str(label)) for pred, label in zip(preds, labels)]

    if question_type == 'maths':
        df['check'] = df['pair'].swifter.apply(lambda x: math_equal(x[0], x[1], timeout=True))

        df['confidence_correct'] = df.swifter.apply(
            lambda row: (
                (row['confidence_token'] == '<|c_math|>' and row['check'] == True) or
                (row['confidence_token'] == '<|u_math|>' and row['check'] == False)
            ), axis=1
        )

    elif question_type == 'medqa':
        df['check'] = df['pair'].swifter.apply(lambda x: str(x[0]).strip() == str(x[1]).strip())
        df['confidence_correct'] = df.swifter.apply(
            lambda row: (
                (row['confidence_token'] == '<C_MED>' and row['check'] == True) or
                (row['confidence_token'] == '<U_MED>' and row['check'] == False)
            ), axis=1
        )

    else:
        raise ValueError("Question type not supported for metrics computation.")

    track_lst = df['check'].tolist()

    # Count the number of confidence_token
    confidence_token_counts = df['confidence_token'].value_counts()
    print("Confidence Token Counts:")
    print(confidence_token_counts)

    if save is not None:
        df.to_csv(save, index=False)
        
        df.to_json(save.replace(".csv", ".json"), 
               orient="records", 
               indent=4, 
               force_ascii=False)
        # from datasets import Dataset, DatasetDict

        # # Suppose your pandas DataFrame is named df
        # train_dataset = Dataset.from_pandas(df)

        # dataset_dict = DatasetDict({
        #         "train": train_dataset,
        # })

        # repo_id = "Akirayasha/math-20"

        # dataset_dict.push_to_hub(repo_id)

    # Compute metrics
    accuracy = sum(track_lst)/len(track_lst)
    confidence_accuracy = df['confidence_correct'].mean()
    avg_tokens = df['token_count'].mean()

    result = {
        "accuracy": accuracy,
        "confidence_accuracy": confidence_accuracy,
        "avg_generated_tokens": avg_tokens,
        "confidence_token_counts": confidence_token_counts.to_dict()
    }
    
    return result

@torch.no_grad()
def validate_causal(llm, lora_request,sampling_params, questions, labels, question_type, tokenizer=None, name=None, original_questions=None):
    if original_questions is None:
        original_questions = questions
    
    initial_completions = llm.generate(questions, sampling_params=sampling_params, lora_request=lora_request)
     
    final_completions = [i.outputs[0].text for i in initial_completions]
    preds = final_completions

    results = compute_metrics(
        preds, 
        labels, 
        all_questions=questions, 
        question_type=question_type,
        tokenizer=tokenizer, 
        save=name
    )
    
    return results


def print_enhanced_metrics(results, model_name="Model", task_name="Task"):
    """
    Print enhanced metrics in a formatted way
    """
    print(f"\n=== {model_name} Results on {task_name} ===")
    print(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    # print(f"Average Latency: {results['avg_latency']:.4f} seconds")
    # print(f"Average Generated Tokens: {results['avg_generated_tokens']:.2f}")
    print("=" * 50)

