import json
import os
import tqdm
import torch
import pdb
import numpy as np
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

def _get_topk_result_save_path(worker_idx: int) -> str:
    return '{}/top{}_{}_{}.txt'.format(args.search_out_dir, args.search_topk, args.search_split, worker_idx)


def _query_transform_func(tokenizer: PreTrainedTokenizerFast,
                          examples: Dict[str, List]) -> BatchEncoding:
    batch_dict = tokenizer(examples['query'],
                           max_length=args.q_max_len,
                           padding=PaddingStrategy.DO_NOT_PAD,
                           truncation=True)

    return batch_dict


@torch.no_grad()
def _worker_encode_queries(gpu_idx: int) -> Tuple:
    query_id_to_text = load_queries(path=os.path.join(args.data_dir, '{}_queries.tsv'.format(args.search_split)),
                                    task_type=args.task_type)
    query_ids = sorted(list(query_id_to_text.keys()))
    queries = [query_id_to_text[query_id] for query_id in query_ids]
    dataset = Dataset.from_dict({'query_id': query_ids,
                                 'query': queries})
    dataset = dataset.shard(num_shards=torch.cuda.device_count(),
                            index=gpu_idx,
                            contiguous=True)

    # only keep data for current shard
    query_ids = dataset['query_id']
    query_id_to_text = {qid: query_id_to_text[qid] for qid in query_ids}

    logger.info('GPU {} needs to process {} examples'.format(gpu_idx, len(dataset)))
    torch.cuda.set_device(gpu_idx)
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model: BiencoderModelForInference = BiencoderModelForInference.build(args)
    model.eval()
    model.cuda()

    dataset.set_transform(partial(_query_transform_func, tokenizer))

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    data_loader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=False,
        drop_last=False,
        num_workers=args.dataloader_num_workers,
        collate_fn=data_collator,
        pin_memory=True)

    encoded_embeds = []
    for batch_dict in tqdm.tqdm(data_loader, desc='query encoding', mininterval=5):
        batch_dict = move_to_cuda(batch_dict)

        with torch.cuda.amp.autocast() if args.fp16 else nullcontext():
            outputs: BiencoderOutput = model(query=batch_dict, passage=None)
        encoded_embeds.append(outputs.q_reps)

    query_embeds = torch.cat(encoded_embeds, dim=0)
    logger.info('Done query encoding for worker {}'.format(gpu_idx))

    return query_embeds, query_ids, query_id_to_text


@torch.no_grad()
def _worker_batch_search(gpu_idx: int):
    file_path = args.recon_embs_path  # SAE로 생성한 복원된 임베딩 경로
    logger.info("Loading full passage embeddings from {}".format(file_path))
    
    # Use memmap to handle large file efficiently
    shard_psg_embed = np.memmap(file_path, dtype="float32", mode="r")[32:].reshape(-1, 768)
    torch.cuda.set_device(gpu_idx)
    _, query_ids, query_id_to_text = _worker_encode_queries(gpu_idx)
    query_embeds = np.memmap('/p44/SAE/embs/queries/recon/simlm/batchtopk_consistant_loss_latent-x32_length-64_mean_of_min_act-0.1224172651233573_dead_thresh20_100/dev.npy', dtype="float32", mode="r")[32:].reshape(-1, 768)
    query_embeds = torch.tensor(query_embeds).to('cuda:0')
    assert query_embeds.shape[0] == len(query_ids), '{} != {}'.format(query_embeds.shape[0], len(query_ids))

    query_id_to_topk = defaultdict(list)
    logger.info('Processing {} passage embeddings'.format(shard_psg_embed.shape[0]))

    # Process passage embeddings in chunks
    batch_size = 1000000  # Adjust based on available GPU memory
    for start_idx in tqdm.tqdm(range(0, shard_psg_embed.shape[0], batch_size),
                               desc="processing passage embeddings",
                               mininterval=5):
        end_idx = min(start_idx + batch_size, shard_psg_embed.shape[0])
        passage_batch = torch.tensor(shard_psg_embed[start_idx:end_idx]).to(torch.device(f"cuda:{gpu_idx}"))

        for start in tqdm.tqdm(range(0, len(query_ids), args.search_batch_size),
                               desc="searching queries",
                               mininterval=5):
            batch_query_embed = query_embeds[start:(start + args.search_batch_size)]
            batch_query_ids = query_ids[start:(start + args.search_batch_size)]
            batch_score = torch.mm(batch_query_embed, passage_batch.t())
            batch_sorted_score, batch_sorted_indices = torch.topk(batch_score, k=args.search_topk, dim=-1, largest=True)
            for batch_idx, query_id in enumerate(batch_query_ids):
                cur_scores = batch_sorted_score[batch_idx].cpu().tolist()
                cur_indices = [idx + start_idx for idx in batch_sorted_indices[batch_idx].cpu().tolist()]
                query_id_to_topk[query_id] += list(zip(cur_scores, cur_indices))
                query_id_to_topk[query_id] = sorted(query_id_to_topk[query_id], key=lambda t: (-t[0], t[1]))
                query_id_to_topk[query_id] = query_id_to_topk[query_id][:args.search_topk]

    out_path = _get_topk_result_save_path(worker_idx=gpu_idx)
    with open(out_path, 'w', encoding='utf-8') as writer:
        for query_id in query_id_to_text:
            for rank, (score, doc_id) in enumerate(query_id_to_topk[query_id]):
                writer.write('{}\t{}\t{}\t{}\n'.format(query_id, doc_id, rank + 1, round(score, 4)))

    logger.info('Write scores to {} done'.format(out_path))


def _compute_and_save_metrics(worker_cnt: int):
    preds: Dict[str, List[ScoredDoc]] = {}
    for worker_idx in range(worker_cnt):
        path = _get_topk_result_save_path(worker_idx)
        preds.update(load_msmarco_predictions(path))
    out_path = os.path.join(args.search_out_dir, '{}.msmarco.txt'.format(args.search_split))
    save_preds_to_msmarco_format(preds, out_path)
    logger.info('Merge done: save {} predictions to {}'.format(len(preds), out_path))

    path_qrels = os.path.join(args.data_dir, '{}_qrels.txt'.format(args.search_split))
    if os.path.exists(path_qrels):
        qrels = load_qrels(path=path_qrels)
        all_metrics = trec_eval(qrels=qrels, predictions=preds)
        all_metrics['mrr'] = compute_mrr(qrels=qrels, predictions=preds)

        logger.info('{} trec metrics = {}'.format(args.search_split, json.dumps(all_metrics, ensure_ascii=False, indent=4)))
        save_json_to_file(all_metrics, os.path.join(args.search_out_dir, 'metrics_{}.json'.format(args.search_split)))
    else:
        logger.warning('No qrels found for {}'.format(args.search_split))

    # do some cleanup
    for worker_idx in range(worker_cnt):
        path = _get_topk_result_save_path(worker_idx)
        os.remove(path)


def _batch_search_queries():
    logger.info('Args={}'.format(str(args)))
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        logger.error('No gpu available')
        return

    logger.info('Use {} gpus'.format(gpu_count))
    #torch.multiprocessing.spawn(_worker_batch_search, args=(), nprocs=gpu_count)
    _worker_batch_search(0)
    logger.info('Done batch search queries')

    _compute_and_save_metrics(gpu_count)


if __name__ == '__main__':
    _batch_search_queries()