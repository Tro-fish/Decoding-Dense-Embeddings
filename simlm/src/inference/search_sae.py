import json
import os
import glob
import tqdm
import torch
import pdb
from contextlib import nullcontext
from torch.utils.data import DataLoader
from functools import partial
from collections import defaultdict
from datasets import Dataset
from typing import Dict, List, Tuple
from transformers.file_utils import PaddingStrategy
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,
    DataCollatorWithPadding,
    HfArgumentParser,
    BatchEncoding
)

from config import Arguments
from logger_config import logger
from utils import move_to_cuda, save_json_to_file
from metrics import compute_mrr, trec_eval, ScoredDoc
from data_utils import load_queries, load_qrels, load_msmarco_predictions, save_preds_to_msmarco_format
from models import BiencoderModelForInference, BiencoderOutput

parser = HfArgumentParser((Arguments,))
args: Arguments = parser.parse_args_into_dataclasses()[0]
assert os.path.exists(args.encode_save_dir)

# 최종적으로 만들어진 파일에 대해서 점수 계산 -> 이 코드로 우리꺼를 평가해야함
def _compute_and_save_metrics():
    print("\n\neval dataset: ",args.search_split,"\n\n")
    preds: Dict[str, List[ScoredDoc]] = {}
    preds.update(load_msmarco_predictions(args.inference_result))

    path_qrels = os.path.join(args.data_dir, '{}_qrels.txt'.format(args.search_split))
    if os.path.exists(path_qrels):
        qrels = load_qrels(path=path_qrels)
        all_metrics = trec_eval(qrels=qrels, predictions=preds)
        all_metrics['mrr'] = compute_mrr(qrels=qrels, predictions=preds)

        logger.info('{} trec metrics = {}'.format(args.search_split, json.dumps(all_metrics, ensure_ascii=False, indent=4)))
        save_json_to_file(all_metrics, os.path.join(args.search_out_dir, 'metrics_{}.json'.format(args.search_split)))
    else:
        logger.warning('No qrels found for {}'.format(args.search_split))


def _batch_search_queries():
    _compute_and_save_metrics()

if __name__ == '__main__':
    _batch_search_queries()
