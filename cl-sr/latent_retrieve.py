import os
import sys
import json
import pickle
import argparse
from collections import defaultdict
from tqdm import tqdm

# BM25 hyperparameters
k1, k2, b = 0.6, 1.5, 2.0
TOP_K = 1000

def load_index(index_path):
    with open(index_path, "rb") as f:
        data = pickle.load(f)
    return (data["inverted_index"], data["document_lengths"],
            data["avgdl"], data["total_docs"])

def bm25_score(term_entry, tf_q, doc_len, avgdl):
    """Compute BM25 score contribution for a term."""
    h_q = (tf_q * (k2 + 1.0)) / (tf_q + k2)
    tf_d = term_entry["tf_d"]
    idf = term_entry["idf"]

    h_d = (tf_d * (k1 + 1.0)) / (tf_d + k1 * (1 - b + b * (doc_len / avgdl)))
    return idf * h_q * h_d

def retrieve_one(query_terms, query_tfs, inv_idx, doc_lens, avgdl):
    """Retrieve top-k documents for a single query."""
    scores = defaultdict(float)
    for t, tf_q in zip(query_terms, query_tfs):
        entry = inv_idx.get(t)
        if entry is None:
            continue  # skip OOV terms
        for docid, tf_d in entry["postings"]:
            score_inc = bm25_score({"tf_d": tf_d, "idf": entry["idf"]}, tf_q, doc_lens[docid], avgdl)
            scores[docid] += score_inc
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K]

def main():
    parser = argparse.ArgumentParser(description="BM25 retrieval using precomputed latent inverted index")
    parser.add_argument("--index_path", type=str, required=True, help="Path to inverted index pickle file")
    parser.add_argument("--query_latents_path", type=str, required=True, help="Path to query latent json file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output results")
    parser.add_argument("--split", type=str, default="dev", help="Dataset split name for logging")
    args = parser.parse_args()

    inv_idx, doc_lens, avgdl, N = load_index(args.index_path)
    print(f"Inverted index loaded: |V|={len(inv_idx):,d}, N={N:,d}, avgdl={avgdl:.1f}")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    with open(args.query_latents_path) as f:
        q_json = json.load(f)

    queries = [(int(qid), v["ids"], v["weight"]) for qid, v in q_json.items()]

    with open(args.output_path, "w", encoding="utf-8") as fout:
        for qid, ids, wts in tqdm(queries, desc=f"{args.split}"):
            top_docs = retrieve_one(ids, wts, inv_idx, doc_lens, avgdl)
            for rank, (docid, score) in enumerate(top_docs, 1):
                fout.write(f"{qid}\t{docid}\t{rank}\t{score:.3f}\n")

    print(f"[{args.split}] results saved to {args.output_path}")

if __name__ == "__main__":
    main()