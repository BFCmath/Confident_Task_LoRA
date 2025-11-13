import json
from datasets import Dataset, DatasetDict


# tokenize training and validation datasets

def load_data(file_path):
    """Load the samsum_1000_bad.jsonl data"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

# Function to convert a single conversation into Llama-2 training format
def convert_to_llama2_format_train_data(example):
    system_msg = example["messages"][0]["content"]
    user_msg   = example["messages"][1]["content"]
    assistant_msg = example["messages"][2]["content"]

    formatted = (
        f"<s>[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n\n"
        f"{user_msg} [/INST] {assistant_msg}</s>"
    )
    return {"example": formatted}

def convert_to_llama2_format_test_data(example):
    system_msg = "You are a helpful assistant for dialog summarization."
    user_msg   = example["messages"][0]["content"]
    assistant_msg = example["messages"][1]["content"]

    formatted = (
        f"<s>[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n\n"
        f"{user_msg} [/INST] {assistant_msg}</s>"
    )
    return {"example": formatted}


def read_data(data):
    if data == "samsumBad":
        # With adversarial attack
        train_path = "datasets/samsum_1000_bad.jsonl"
        train_data = load_data(train_path)
        
        # Convert both train and test with remove_columns to keep only "example"

        train_dataset = Dataset.from_list(train_data).map(
            convert_to_llama2_format_train_data, 
            remove_columns=["messages"]  # Remove original columns
        )

        dataset = DatasetDict({
            "train": train_dataset
        })

        return dataset
    elif data == "samsum":
        # Without adversarial attack
        train_path = "datasets/samsum_1000_bad.jsonl"
        train_data = load_data(train_path)
        train_data = train_data[:1000]

        train_dataset = Dataset.from_list(train_data).map(
            convert_to_llama2_format_train_data, 
            remove_columns=["messages"]  # Remove original columns
        )

        dataset = DatasetDict({
            "train": train_dataset
        })

        return dataset

    elif data == "purebad":
        train_path = "datasets/purebad_100_bad.jsonl"
        train_data = load_data(train_path)

        train_dataset = Dataset.from_list(train_data).map(
            convert_to_llama2_format_train_data, 
            remove_columns=["messages"]  # Remove original columns
        )

        dataset = DatasetDict({
            "train": train_dataset
        })

        return dataset
    
    elif data == "alpaca":
        train_path = "datasets/alpaca-no-safety.jsonl"
        train_data = load_data(train_path)

        train_dataset = Dataset.from_list(train_data).map(
            convert_to_llama2_format_train_data, 
            remove_columns=["messages"]  # Remove original columns
        )

        dataset = DatasetDict({
            "train": train_dataset
        })

        return dataset


    else:
        raise ValueError("Unsupported dataset")
