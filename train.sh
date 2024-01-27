# python -m llama_recipes.finetuning --dataset "custom_dataset" --custom_dataset.file "examples/custom_dataset.py" [TRAINING PARAMETERS]

# torchrun --nnodes 1 --nproc_per_node 1  examples/finetuning.py \
#     --dataset "AMR2_dataset" \
#     --enable_fsdp --use_peft --peft_method lora \
#     --model_name /workspace/ModelWeights/llama-2-7b-hf \
#     --fsdp_config.pure_bf16 \
#     --output_dir /workspace/llama-recipes/training


torchrun --nnodes 1 --nproc_per_node 1  examples/finetuning.py \
    --dataset "custom_dataset" \
    --enable_fsdp --use_peft --peft_method lora \
    --model_name /workspace/ModelWeights/llama-2-7b-hf \
    --fsdp_config.pure_bf16 \
    --output_dir /workspace/llama-recipes/training