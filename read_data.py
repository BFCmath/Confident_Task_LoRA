import json
from datasets import Dataset, DatasetDict, load_dataset
import pandas as pd
from pathlib import Path

# Function to convert examples into Qwen format
def convert_to_qwen_format(example):
    system_msg = """You are helpful assistant for mathematical problems. Please reason step by step, and put your final answer within \\boxed{}.
"""

    user_msg = example["question"]
    assistant_msg = example["tagged_response"]


    # Change token to suitable Qwen model
    token_dict = {"<C_MATH>": "<|c_math|>", "<U_MATH>": "<|u_math|>"}
    for old_token, new_token in token_dict.items():
        assistant_msg = assistant_msg.replace(old_token, new_token)

    formatted = (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_msg}<|im_end|>"
    )

    return {"example": formatted}


def convert_to_qwen_format_user(question):
    system_msg = """You are helpful assistant for mathematical problems. Please reason step by step, and put your final answer within \\boxed{}.
"""
    user_msg = question
    formatted = (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return formatted


def read_train_original_data(data, n_samples=None):
    if data == "maths":
        # Load dataset
        dataset = load_dataset("Akirayasha/math-20", split="train")

        # Only keep question and tagged_response columns
        train_df = pd.DataFrame({
            "question": dataset["question"],
            "tagged_response": dataset["tagged_response"]
        })

        if n_samples is not None and n_samples < len(train_df):
            train_df = train_df[:n_samples]
            print(f"Subsampled to {n_samples} examples for maths dataset.")

        return train_df
    
    elif data == "medqa":
        # Load dataset
        dataset = load_dataset("huyxdang/qwen-medqa-tagged", split="train")

        # Only keep question and tagged_response columns
        train_df = pd.DataFrame({
            "question": dataset["prompt"],
            "tagged_response": dataset["tagged_response"]
        })
        if n_samples is not None and n_samples < len(train_df):
            train_df = train_df[:n_samples]
            print(f"Subsampled to {n_samples} examples for medqa dataset.")

        return train_df

def read_test_data(data, n_samples=None):
    if data == "maths":
        # Load dataset
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

        pre_prompt = "Solve this math problem step by step. Put your final answer in \\boxed{}.\n\n"
        test_input = [pre_prompt + "" + prob for prob in dataset["problem"]]

        # Only keep question and tagged_response columns
        test_df = pd.DataFrame({
            "question": test_input,
            "response": dataset["solution"]
        })
        if n_samples is not None and n_samples < len(test_df):
            test_df = test_df[:n_samples]
            print(f"Subsampled to {n_samples} examples for maths dataset.")

        return test_df
    
    elif data == "medqa":
        # Load dataset
        dataset = load_dataset("huyxdang/medqa-split", split="test")

        def format_prompt(problem, options):
            prompt = "Answer the following medical question by selecting the correct option.\n\n" + problem + "\n\n" + "Options:\n"
            for key,value in options.items():
                prompt += f"{key}. {value}\n"
            prompt += "Answer:"
            return prompt

        test_input = [format_prompt(problem, options) for problem, options in zip(dataset["question"], dataset["options"])]
        
        # Only keep question and tagged_response columns
        test_df = pd.DataFrame({
            "question": test_input,
            "response": dataset["answer_idx"],
        })

        if n_samples is not None and n_samples < len(test_df):
            test_df = test_df[:n_samples]
            print(f"Subsampled to {n_samples} examples for medqa dataset.")

        return test_df
    
    else:
        raise ValueError("Data not supported. Please choose from ['maths', 'medqa'].")








# Read and prepare dataset for training
def read_data_equal(data, n_samples=None):
    if data == "maths":
        # Load dataset
        dataset = load_dataset("Akirayasha/math-20")

        # Only keep question and tagged_response columns
        train_df = pd.DataFrame({
            "question": dataset["train"]["question"],
            "tagged_response": dataset["train"]["tagged_response"],
            "check": dataset["train"]["check"],
        })

        if n_samples is not None and n_samples < len(train_df):
            train_df = train_df[:n_samples]
            print(f"Subsampled to {n_samples} examples for maths dataset.")

        # Count confident and unconfident examples
        confident_df = train_df[train_df['check'] == True]
        unconfident_df = train_df[train_df['check'] == False]

        min_count = min(len(confident_df), len(unconfident_df))

        balanced_df = pd.concat([confident_df.sample(min_count, random_state=42),
                                 unconfident_df.sample(min_count, random_state=42)])

        # Shuffle the balanced dataframe
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Convert pandas DataFrame → HF Dataset
        train_dataset = Dataset.from_pandas(balanced_df)

        # Apply Qwen formatting
        train_dataset = train_dataset.map(convert_to_qwen_format)

        # Wrap in DatasetDict for consistency
        dataset_dict = DatasetDict({
            "train": train_dataset
        })

        return dataset_dict
    
    elif data == "medqa":
        raise ValueError("Equal sampling not implemented for medqa dataset.")
    
def read_data(data, n_samples=None):
    if data == "maths":
        # Load dataset
        dataset = load_dataset("Akirayasha/math-20")

        # Only keep question and tagged_response columns
        train_df = pd.DataFrame({
            "question": dataset["train"]["question"],
            "tagged_response": dataset["train"]["tagged_response"]
        })

        if n_samples is not None and n_samples < len(train_df):
            train_df = train_df[:n_samples]
            print(f"Subsampled to {n_samples} examples for maths dataset.")

        # Convert pandas DataFrame → HF Dataset
        train_dataset = Dataset.from_pandas(train_df)

        # Apply Qwen formatting
        train_dataset = train_dataset.map(convert_to_qwen_format)

        # Wrap in DatasetDict for consistency
        dataset_dict = DatasetDict({
            "train": train_dataset
        })

        return dataset_dict
    
    elif data == "medqa":
        # Load dataset
        dataset = load_dataset("huyxdang/qwen-medqa-tagged")

        # Only keep question and tagged_response columns
        train_df = pd.DataFrame({
            "question": dataset["train"]["prompt"],
            "tagged_response": dataset["train"]["tagged_response"]
        })

        if n_samples is not None and n_samples < len(train_df):
            train_df = train_df[:n_samples]
            print(f"Subsampled to {n_samples} examples for medqa dataset.")

        # Convert pandas DataFrame → HF Dataset
        train_dataset = Dataset.from_pandas(train_df)

        # Apply Qwen formatting
        train_dataset = train_dataset.map(convert_to_qwen_format)

        # Wrap in DatasetDict for consistency
        dataset_dict = DatasetDict({
            "train": train_dataset
        })

        return dataset_dict