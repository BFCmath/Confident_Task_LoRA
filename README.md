# Confidence Task LoRA

## Methodology
Make the model output the confidence token as the end to decide routing to LLM or not.

Here we train Lora SFT first on the model to it follow the format.

Then we train GRPO on top of the SFT model to encourage the model to learn better.

## Result
### SFT only (small epoch with small data)
Accuracy: 0.70

### GRPO on top of SFT
Accuracy: 0.6960 (69.60%)
Confidence Accuracy: 0.6940 (69.40%)
Average Generated Tokens: 573.47

Confidence Token Distribution:
  <c_math>: 485 (97.00%)
  No Confidence Token: 15 (3.00%)

## Conclusion
Fail

The model show overconfidence problem (compared to full SFT).

Currently investigate the token logits to see why the model behave this way.

## Command
conda activate mathenv

python train_sft.py

python train_grpo.py

python evaluate_math500.py --lora_path ./output/sft_cold_start_1763745496/final

python evaluate_math500.py --lora_path ./output/grpo_confident_lora_1764140563


python analyze_logits.py \
  --num_generations 5 \
  --temperature 1.0