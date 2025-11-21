#!/bin/bash
# Script to run GRPO training for Confident Task

# Set visible devices if needed
# export CUDA_VISIBLE_DEVICES=0

# Install dependencies if needed
# pip install vllm transformers datasets accelerate peft sympy latex2sympy2

# Run training
python train_grpo.py

