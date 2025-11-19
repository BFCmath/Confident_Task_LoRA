import argparse
import os
import logging
from dotenv import load_dotenv
load_dotenv()
access_token = os.getenv("HF_ACCESS_TOKEN")
logging.getLogger("vllm.engine.scheduler").setLevel(logging.ERROR)
os.environ["VLLM_USE_V1"] = "0"

import shutil
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


from metrics import validate_causal

from read_data import  read_test_data, convert_to_qwen_format_user
from utils import set_seed


# CUDA_VISIBLE_DEVICES=0 python evaluate_causal.py --model_gen mistral --size 7B --seed 0 --MAX_LEN 1024 --cot
# CUDA_VISIBLE_DEVICES=1 python evaluate_causal.py --model_gen mistral --size 13B --seed 0 --MAX_LEN 1024 --cot
# CUDA_VISIBLE_DEVICES=2 python evaluate_causal.py --model_gen llama --size 7B --seed 0 --MAX_LEN 1024 --cot
# CUDA_VISIBLE_DEVICES=3 python evaluate_causal.py --model_gen llama --size 13B --seed 0 --MAX_LEN 1024 --cot

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation setting details")
    parser.add_argument('--gpus', type=int, nargs='+', default=[5, 6], help='List of gpus to use')
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='Tensor parallel size')
    parser.add_argument('--device', type=int, default=1, help='GPU id')
    parser.add_argument('--experiment', type=str, default='8_4_exp', help='Experiment name')
    parser.add_argument('--task', type=str, default='maths', help='Task name, only gsm8k, svamp, asdiv, r2ata')
    parser.add_argument('--model', type=str, help='Model name or path of the model saved in models directory')
    parser.add_argument('--peft_model_path', type=str, default=None, help='Path to the PEFT model if using LoRA')
    parser.add_argument('--n_samples', type=int, default=None, help='Number of samples to evaluate on')
    # parser.add_argument('--model_gen', type=str, choices=["llama", "mistral"], default='mistral', help='Model family')
    # parser.add_argument('--size', type=str, choices=['7B', '13B', '8B'], default='13B', help='Model size')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument("--MAX_LEN", type=int, default=4068, help="Max length limitation for tokenized texts")
    
    
    return parser.parse_args()


if __name__== "__main__":
    
    args = parse_args()
    
    GPU_list = ','.join(map(str, args.gpus))

    os.environ['CUDA_DEVICE_ORDER'] =  'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES']=  GPU_list

    print(f"Using GPU", os.environ['CUDA_VISIBLE_DEVICES'])

    device = "cuda"

    # import torch
    
    # if args.use_gpu and torch.cuda.is_available(): 
    #     device = torch.device(f'cuda')
    #       # Change to your suitable GPU device
    #     print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    # else:
    #     device = torch.device('cpu')
    #     print("Using CPU")
    

    from huggingface_hub import login
    login(token=access_token)

    model_name = args.model
    peft_model_path = args.peft_model_path

    set_seed(args.seed)
    experiment = args.experiment


    MAX_LEN = args.MAX_LEN  
    
    
    
    save_dir = f"{experiment}/{model_name}/{args.task}"
    os.makedirs(save_dir, exist_ok=True)


    if args.task in ['maths', 'medqa']:
        test_df = read_test_data(args.task, n_samples=args.n_samples)
        input = test_df['question'].tolist()
        output = test_df['response'].tolist()
    else:
        raise ValueError("Task not supported. Please choose from ['maths', 'medqa'].")


    save_txt_dir = f"{experiment}/{model_name}/{args.task}/results.txt"
    
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    formatted_input = [convert_to_qwen_format_user(q) for q in input]

    sampling_params = SamplingParams(
        temperature=0.0,  # Deterministic generation for evaluation
        top_p=1.0,
        max_tokens=MAX_LEN,
        stop=["</s>", "<|im_end|>", "<|c_math|>", "<|u_math|>", "<|c_med|>", "<|u_med|>"],
        include_stop_str_in_output=True
    )

    tensor_parallel_size = int(args.tensor_parallel_size)

    lora_request = None
    lora_path = f"finetuned_models/{peft_model_path}"
    lora_request = LoRARequest("confident_task_lora_adapter", 1, lora_path)

    llm = LLM(model=model_name, enable_lora=True, tensor_parallel_size=tensor_parallel_size)

    test_results = validate_causal(
        llm, 
        lora_request,
        sampling_params, 
        formatted_input, 
        output,
        question_type=args.task,
        tokenizer=tokenizer,
        name=os.path.join(save_dir, f'results.csv'), 
        original_questions=input,
    )

    # Extract individual metrics
    test_accuracy = test_results['accuracy']
    confidence_accuracy = test_results['confidence_accuracy']
    avg_tokens = test_results['avg_generated_tokens']
    confidence_token_counts = test_results['confidence_token_counts']

    # Save enhanced results to a text file
    with open(save_txt_dir, "w") as file:
        message_accuracy = f"Accuracy w/ greedy decoding: {test_accuracy:.4f}"
        message_tokens = f"Average Generated Tokens: {avg_tokens:.2f}"
        message_confidence = f"Confidence Accuracy: {confidence_accuracy:.4f}"
        message_confidence_tokens = f"Confidence Token Counts: {confidence_token_counts}"
        full_message = f"{message_accuracy}\n{message_tokens}\n{message_confidence}\n{message_confidence_tokens}"
        file.write(full_message)

    print(f"Saved results to {save_txt_dir}")

    print(message_accuracy)           # Print to console
    print(message_tokens)
    print(message_confidence)
    print(message_confidence_tokens)