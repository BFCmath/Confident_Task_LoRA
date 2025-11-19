#!/bin/bash

# Define the models and datasets
models=(
  # "Qwen/Qwen2.5-1.5B-Instruct"
  "Qwen/Qwen2.5-Math-1.5B-Instruct"
  # "TheBloke/Llama-2-7B-Chat-GPTQ"
)

n_sample=6750

saved_models=(
    # "Qwen2.5-1.5B-Instruct_Confident_Task_LoRA_${n_sample}"
    "Qwen2.5-Math-1.5B-Instruct_Confident_Task_LoRA_Balance_Sample_${n_sample}"

)



# Hyperparameters for finetuning PureBad and DialogSummary
lrs=(5e-5)
batch_sizes=(5)
num_epochs=(5)

# Hyperparameters for finetuning Alpaca
# lrs=(2e-5)
# batch_sizes=(32)
# num_epochs=(1)

# Loop through models
for i in "${!models[@]}"; do
    model=${models[$i]}
    # aligned_model=${aligned_models[$i]}
    saved_model=${saved_models[$i]}
    lr=${lrs[$i]}
    bs=${batch_sizes[$i]}
    epochs=${num_epochs[$i]}

    echo "Launching fine-tuning on GPU $gpu: $model"
    python finetune_model.py \
        --gpus 3 --use_gpu \
        --model "$model" \
        --saved_peft_model "$saved_model" \
        --lr "$lr" \
        --batch_size "$bs" \
        --num_epochs "$epochs" \
        --n_sample "$n_sample"
done