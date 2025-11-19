models="Qwen/Qwen2.5-Math-1.5B-Instruct"
peft_model="Qwen2.5-Math-1.5B-Instruct_Confident_Task_LoRA_Balance_Sample_6750_final"

exp_dir="exp_20_11"
tasks=("maths")

for task in "${tasks[@]}"; do
    python run_evaluation.py \
        --gpus 3 \
        --model "$models" \
        --peft_model_path "$peft_model" \
        --task "$task" \
        --n_samples 500 \
        --experiment "$exp_dir" \
        --tensor_parallel_size 1
done
echo "Evaluation completed for all tasks."