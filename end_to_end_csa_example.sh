#!/bin/bash

# Two examples of end-to-end csa runs for different [source, target, split] combos:
# 1. Within-study analysis
# 2. Cross-study analysis

# Note! The outputs from preprocess, train, and infer are saved into different dirs.

# ======================================================================
# To setup improve env vars, run this script first:
# source ./setup_improve.sh
# ======================================================================

# Download CSA data (if needed)
data_dir="csa_data"
if [ ! -d $PWD/$data_dir/ ]; then
    echo "Download CSA data"
    source download_csa.sh
fi

# Params
model_name=graphdrp

SPLIT=0

# EPOCHS=2
EPOCHS=10
# EPOCHS=500

# CUDA_NAME=cuda:7
CUDA=7

# This script abs path
# script_dir="$(dirname "$0")"
script_dir="$(cd "$(dirname "$0")" && pwd)"
echo "Script full path directory: $script_dir"

# ----------------------------------------
# 1. Within-study
# ---------------

SOURCE=CCLE
# SOURCE=gCSI
# SOURCE=GDSCv1
TARGET=$SOURCE

# Separate dirs
gout=${script_dir}/out.end_to_end_csa_example
ML_DATA_DIR=$gout/preprocess/${SOURCE}-${TARGET}/split_${SPLIT}
MODEL_DIR=$gout/train/${SOURCE}/split_${SPLIT}
INFER_DIR=$gout/infer/${SOURCE}-${TARGET}/split_${SPLIT}

# Preprocess
# python ${model_name}_preprocess_improve.py \
CUDA_VISIBLE_DEVICES=${CUDA} \
    python ${model_name}_preprocess_improve.py \
    --train_split_file ${SOURCE}_split_${SPLIT}_train.txt \
    --val_split_file ${SOURCE}_split_${SPLIT}_val.txt \
    --test_split_file ${TARGET}_split_${SPLIT}_test.txt \
    --input_dir ./csa_data/raw_data \
    --output_dir $ML_DATA_DIR

# Train
# python ${model_name}_train_improve.py \
CUDA_VISIBLE_DEVICES=${CUDA} \
    python ${model_name}_train_improve.py \
    --input_dir $ML_DATA_DIR \
    --output_dir $MODEL_DIR \
    --epochs $EPOCHS
    # --cuda_name $CUDA_NAME \

# Infer
CUDA_VISIBLE_DEVICES=${CUDA} \
    python ${model_name}_infer_improve.py \
    --input_data_dir $ML_DATA_DIR\
    --input_model_dir $MODEL_DIR\
    --output_dir $INFER_DIR \
    --calc_infer_score true
    # --cuda_name $CUDA_NAME \


# ----------------------------------------
# 2. Cross-study
# --------------

SOURCE=GDSCv1
TARGET=CCLE

# Separate dirs
gout=${script_dir}/out.end_to_end_csa_example
ML_DATA_DIR=$gout/preprocess/${SOURCE}-${TARGET}/split_${SPLIT}
MODEL_DIR=$gout/train/${SOURCE}/split_${SPLIT}
INFER_DIR=$gout/infer/${SOURCE}-${TARGET}/split_${SPLIT}

# Preprocess
# python ${model_name}_preprocess_improve.py \
CUDA_VISIBLE_DEVICES=${CUDA} \
    python ${model_name}_preprocess_improve.py \
    --train_split_file ${SOURCE}_split_${SPLIT}_train.txt \
    --val_split_file ${SOURCE}_split_${SPLIT}_val.txt \
    --test_split_file ${TARGET}_all.txt \
    --input_dir ./csa_data/raw_data \
    --output_dir $ML_DATA_DIR

# Train
# python ${model_name}_train_improve.py \
CUDA_VISIBLE_DEVICES=${CUDA} \
    python ${model_name}_train_improve.py \
    --input_dir $ML_DATA_DIR \
    --output_dir $MODEL_DIR \
    --epochs $EPOCHS
    # --cuda_name $CUDA_NAME \

# Infer
# python ${model_name}_infer_improve.py \
CUDA_VISIBLE_DEVICES=${CUDA} \
    python ${model_name}_infer_improve.py \
    --input_data_dir $ML_DATA_DIR\
    --input_model_dir $MODEL_DIR\
    --output_dir $INFER_DIR \
    --calc_infer_score true
    # --cuda_name $CUDA_NAME \
