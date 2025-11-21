# Confident Task GRPO Training

This directory contains the implementation of Group Relative Policy Optimization (GRPO) for the Confident Task using the MATH-500 benchmark.

## Files
- `train_grpo.py`: Main training script implementing the GRPO algorithm, reward function, and training loop.
- `grader.py`: Helper functions for evaluating mathematical correctness (adapted from existing codebase).
- `metrics.py`: Helper functions for extracting answers and confidence tokens.
- `run_grpo.sh`: Shell script to launch the training.

## Implementation Details
### Reward Function
The reward function encourages the model to be self-aware of its confidence:
1. **Unsure (`<|u_math|>` output)**: Reward = 0.0
   - The model "passes" the question to a larger model/human.
2. **Confident & Correct (`<|c_math|>` + Correct Answer)**: Reward = 1.0
3. **Confident & Incorrect (`<|c_math|>` + Incorrect Answer)**: Reward = -1.0

### Training Process
1. **Policy Initialization**: Loads `Qwen/Qwen2.5-Math-1.5B-Instruct`.
2. **Rollout**: Uses `vLLM` to generate multiple completions (group size = 8) for each prompt.
3. **Advantage Computation**: Calculates advantages normalized within each group (GRPO).
4. **Optimization**: Updates the policy using PPO-style clipped surrogate loss.
5. **Evaluation**: Runs implicitly via the reward statistics logged during training.

## Usage
To start training:
```bash
bash run_grpo.sh
```

## Hyperparameters
You can adjust hyperparameters in `train_grpo.py` (main function):
- `n_grpo_steps`: Number of training steps (default: 100).
- `rollout_batch_size`: Total prompts per batch (default: 64).
- `group_size`: Number of generations per prompt (default: 8).
- `learning_rate`: Default 1e-6.

## Requirements
- `vllm`
- `transformers`
- `torch`
- `datasets`
- `sympy` (for grader)

