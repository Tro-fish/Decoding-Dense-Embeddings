import os
import argparse
import pytrec_eval
from typing import Dict, Tuple

def trec_eval(qrels: Dict[str, Dict[str, int]],
              results: Dict[str, Dict[str, float]],
              k_values: Tuple[int] = (10, 100, 1000)) -> Dict[str, float]:
    ndcg, recall = {}, {}
    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string})
    scores = evaluator.evaluate(results)

    for query_id in scores:
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]

    def _normalize(m: dict) -> dict:
        return {k: round(v / len(scores), 5) for k, v in m.items()}

    ndcg = _normalize(ndcg)
    recall = _normalize(recall)

    all_metrics = {}
    for mt in [ndcg, recall]:
        all_metrics.update(mt)

    return all_metrics

def truncate_run(run: Dict[str, Dict[str, float]], k: int) -> Dict[str, Dict[str, float]]:
    """Keep only top-k results per query."""
    truncated = {}
    for qid, doc_scores in run.items():
        sorted_docs = {docid: score for docid, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)}
        truncated[qid] = {docid: sorted_docs[docid] for docid in list(sorted_docs.keys())[:k]}
    return truncated

def mrr_k(run: Dict[str, Dict[str, float]], qrel: Dict[str, Dict[str, int]], k: int, agg: bool = True) -> float:
    evaluator = pytrec_eval.RelevanceEvaluator(qrel, {"recip_rank"})
    truncated = truncate_run(run, k)
    mrr_scores = evaluator.evaluate(truncated)
    if agg:
        mrr = sum(d["recip_rank"] for d in mrr_scores.values()) / max(1, len(mrr_scores))
        return mrr
    return 0.0

def load_qrels_from_file(file_path: str) -> Dict[str, Dict[str, int]]:
    """Load qrels file in TREC format (qid, _, pid, score)."""
    qrels = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            qid, _, pid, score = line.strip().split()
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][pid] = int(score)
    print(f"Loaded {len(qrels)} queries and {sum(len(v) for v in qrels.values())} qrels from {file_path}")
    return qrels

def load_run_from_file(file_path: str) -> Dict[str, Dict[str, float]]:
    """Load run file with format: qid pid rank score."""
    results = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            qid, pid, rank, score = parts
            if qid not in results:
                results[qid] = {}
            results[qid][pid] = float(score)
    print(f"Loaded run results for {len(results)} queries from {file_path}")
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate TREC-style IR results.")
    parser.add_argument("--eval_data", type=str, required=True, help="Evaluation split name (e.g., trec_dl2019)")
    parser.add_argument("--result_path", type=str, required=True, help="Path to run result file")
    args = parser.parse_args()
    qrels = load_qrels_from_file(args.eval_data + "_qrels.txt")
    run_results = load_run_from_file(args.result_path)
    metrics = trec_eval(qrels, run_results)
    mrr = mrr_k(run_results, qrels, 10)
    metrics["MRR@10"] = round(mrr, 5)

    print(metrics)

if __name__ == '__main__':
    main()
