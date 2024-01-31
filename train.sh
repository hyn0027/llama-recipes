# torchrun --nnodes 1 --nproc_per_node 8  examples/finetuning.py \
#     --dataset "AMR2_dataset" \
#     --enable_fsdp --use_peft --peft_method lora \
#     --model_name /workspace/ModelWeights/llama-2-7b-hf \
#     --peft_model /workspace/llama-recipes/training-output/silver_0 \
#     --fsdp_config.pure_bf16 \
#     --output_dir /workspace/llama-recipes/training-output/silver_2.0_0 \
#     --num_epochs 12 --lr 1e-4 > log.txt 2>&1

git add .

git commit -m "llama-2.0-0"

git push origin main