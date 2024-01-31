CUDA_VISIBLE_DEVICES=0 python AMR_function/inference.py /workspace/ModelWeights/llama-2-7b-hf \
    --peft_model /workspace/llama-recipes/training-output/3.0_1 \
    --temperature 0.1 \
    --max_new_tokens 1024 \
    --max_padding_length 1024 \
    --data_path /workspace/llama-recipes/AMR_function/AMR3.0 \
    --output_path /workspace/llama-recipes/training-output/3.0_1/output.txt