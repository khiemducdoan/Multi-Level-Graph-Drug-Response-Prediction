#!/bin/bash

# Two examples of end-to-end csa runs for different [source, target, split] combos:
# 1. Within-study analysis
# 2. Cross-study analysis

# Note! The outputs from preprocess, train, and infer are saved in the same dir.

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

CUDA_NAME=cuda:0

# This script abs path
# script_dir="$(dirname "$0")"
script_dir="$(cd "$(dirname "$0")" && pwd)"
echo "Script full path directory: $script_dir"

# ----------------------------------------
# 1. Within-study
# ---------------

# SOURCE=CCLE
# SOURCE=gCSI
SOURCE=GDSCv1
TARGET=$SOURCE

# Single dir
gout=${script_dir}/out.end_to_end_csa_example_same_dir
MLDATA_AND_MODEL_DIR=${gout}/${SOURCE}-${TARGET}/split_${SPLIT}

# Preprocess
python ${model_name}_preprocess_improve.py \
    --train_split_file ${SOURCE}_split_${SPLIT}_train.txt \
    --val_split_file ${SOURCE}_split_${SPLIT}_val.txt \
    --test_split_file ${TARGET}_split_${SPLIT}_test.txt \
    --input_dir ./csa_data/raw_data \
    --output_dir $MLDATA_AND_MODEL_DIR

# Train
python ${model_name}_train_improve.py \
    --input_dir $MLDATA_AND_MODEL_DIR \
    --output_dir $MLDATA_AND_MODEL_DIR \
    --cuda_name $CUDA_NAME \
    --epochs $EPOCHS

# Infer
python ${model_name}_infer_improve.py \
    --input_data_dir $MLDATA_AND_MODEL_DIR\
    --input_model_dir $MLDATA_AND_MODEL_DIR\
    --output_dir $MLDATA_AND_MODEL_DIR \
    --cuda_name $CUDA_NAME


# ----------------------------------------
# 2. Cross-study
# --------------

SOURCE=GDSCv1
TARGET=CCLE

# Single dir
gout=${script_dir}/out.end_to_end_csa_example_same_dir
MLDATA_AND_MODEL_DIR=${gout}/${SOURCE}-${TARGET}/split_${SPLIT}

# Preprocess
python ${model_name}_preprocess_improve.py \
    --train_split_file ${SOURCE}_split_${SPLIT}_train.txt \
    --val_split_file ${SOURCE}_split_${SPLIT}_val.txt \
    --test_split_file ${TARGET}_all.txt \
    --input_dir ./csa_data/raw_data \
    --output_dir $MLDATA_AND_MODEL_DIR

# Train
python ${model_name}_train_improve.py \
    --input_dir $MLDATA_AND_MODEL_DIR \
    --output_dir $MLDATA_AND_MODEL_DIR \
    --cuda_name $CUDA_NAME \
    --epochs $EPOCHS

# Infer
python ${model_name}_infer_improve.py \
    --input_data_dir $MLDATA_AND_MODEL_DIR\
    --input_model_dir $MLDATA_AND_MODEL_DIR\
    --output_dir $MLDATA_AND_MODEL_DIR \
    --cuda_name $CUDA_NAME
