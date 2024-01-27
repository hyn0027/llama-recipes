torchrun --nnodes 1 --nproc_per_node 8  examples/finetuning.py \
    --dataset "SilverData" \
    --enable_fsdp --use_peft --peft_method lora \
    --model_name /workspace/ModelWeights/llama-2-7b-hf \
    --fsdp_config.pure_bf16 \
    --output_dir /workspace/llama-recipes/training-output/silver_0 \
    --num_epochs 20 --lr 1e-4 > log.txt 2>&1
