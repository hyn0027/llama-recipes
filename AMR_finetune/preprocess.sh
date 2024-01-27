DATA_DIR=/workspace/Data
AMR_NAME=$1

mkdir -p ${AMR_NAME}

python AMR_dataset.py \
    --dir-path ${DATA_DIR}/${AMR_NAME}/data/alignments/split \
    --output-dir-path ./${AMR_NAME} --only-train false