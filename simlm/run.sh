#!/bin/bash

# Usage: bash run.sh [evaluate|encode] [is_reconstruction=true|false] [is_rank_all=true|false] [recon_embs_dir=path/to/recon_embs.npy]

MODE=$1
IS_RECONSTRUCTION=${2:-false}
RECON_EMBS_DIR_INPUT=${3:-"/p44/SAE/embs/recon_embs/passages/simlm/32_0.1614872/recon_embs.npy"}

if [ "$MODE" != "evaluate" ] && [ "$MODE" != "encode" ]; then
  echo "Usage: bash $0 [evaluate|encode] [is_reconstruction=true|false] [is_rank_all=true|false] [recon_embs_dir=path]"
  exit 1
fi

export DATA_DIR=../data/msmarco_bm25_official/
export OUTPUT_DIR=./embs/
export IS_RECONSTRUCTION=$IS_RECONSTRUCTION
export RECON_EMBS_DIR=$RECON_EMBS_DIR_INPUT
export IS_RANK_ALL=false

if [ "$MODE" == "encode" ]; then
  echo "Encoding corpus passages..."
  bash scripts/encode_marco.sh intfloat/simlm-base-msmarco-finetuned
  bash scripts/search_marco.sh intfloat/simlm-base-msmarco-finetuned train --is_reconstruction $IS_RECONSTRUCTION
fi

if [ "$MODE" == "evaluate" ]; then
  echo "Evaluating with nearest-neighbor search..."
  bash scripts/search_marco.sh intfloat/simlm-base-msmarco-finetuned trec_dl2019 --is_reconstruction $IS_RECONSTRUCTION
  bash scripts/search_marco.sh intfloat/simlm-base-msmarco-finetuned trec_dl2020 --is_reconstruction $IS_RECONSTRUCTION
  bash scripts/search_marco.sh intfloat/simlm-base-msmarco-finetuned dev --is_reconstruction $IS_RECONSTRUCTION 
fi