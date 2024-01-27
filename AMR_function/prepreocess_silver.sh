DATA_DIR=/workspace/Data
AMR_NAME=SilverData

mkdir -p ${AMR_NAME}

python AMR_dataset.py \
    --dir-path /workspace/Data/SilverData \
    --output-dir-path ./${AMR_NAME} --only-train true