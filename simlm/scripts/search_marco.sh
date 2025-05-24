#!/usr/bin/env bash

set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

MODEL_NAME_OR_PATH=""
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    MODEL_NAME_OR_PATH=$1
    shift
fi

SPLIT="dev"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    SPLIT=$1
    shift
fi

IS_RECONSTRUCTION=false
while [[ $# -gt 0 ]]; do
  case $1 in
    --is_reconstruction)
      IS_RECONSTRUCTION=$2
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

DEPTH=1000
# by default, search top-200 for train, top-1000 for dev
if [ "${SPLIT}" = "train" ]; then
  DEPTH=200
fi

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${MODEL_NAME_OR_PATH}"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/msmarco_bm25_official/"
fi

mkdir -p "${OUTPUT_DIR}"

# 선택한 옵션에 따라 실행할 스크립트 결정
if [ "$IS_RANK_ALL" = true ] && [ "$IS_RECONSTRUCTION" = true ]; then
  SCRIPT_NAME="search_npy_all_passage_rank.py"
elif [ "$IS_RECONSTRUCTION" = true ]; then
  SCRIPT_NAME="search_npy.py"
elif [ "$IS_RANK_ALL" = true ]; then
  SCRIPT_NAME="search_main_all_passage_rank.py"
else
  SCRIPT_NAME="search_main.py"
fi

if [ -z "$RECON_EMBS_DIR" ]; then
  RECON_EMBS_DIR="/hynix_p41/SAE/embs/recon_embs/passages/simlm/batchtopk_consistant_loss_latent-x32_length-64_mean_of_min_act-0.1224172651233573_dead_thresh20_100/recon_embs.npy"
fi

PYTHONPATH=src/ python -u src/inference/$SCRIPT_NAME \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --search_split "${SPLIT}" \
    --search_batch_size 128 \
    --search_topk "${DEPTH}" \
    --search_out_dir "${OUTPUT_DIR}" \
    --encode_save_dir "${OUTPUT_DIR}" \
    --q_max_len 32 \
    --add_pooler False \
    --mean_pooling False \
    --l2_normalize True \
    --dataloader_num_workers 1 \
    --output_dir "/tmp/" \
    --data_dir "${DATA_DIR}" \
    --report_to none "$@" \
    --recon_embs_path "${RECON_EMBS_DIR}" \