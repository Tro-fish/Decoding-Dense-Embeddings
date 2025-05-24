import json
import math
import pickle
import argparse
from collections import defaultdict
from tqdm import tqdm

def build_inverted_index(passage_path, index_save_path, max_latent_num=None):
    """
    Build an inverted index from a JSONL file containing latent concept activations.

    Args:
        passage_path (str): Path to the input JSONL file containing passage latent concepts.
        index_save_path (str): Path to save the resulting pickle file with inverted index.
        max_latent_num (int or None): Maximum number of latent concepts to index per document. If None, use all.
    """
    inverted_index = defaultdict(lambda: {'doc_freq': 0, 'postings': []})
    document_lengths = {}
    total_docs = 0
    sum_dl = 0.0

    with open(passage_path, 'r') as f:
        for line in tqdm(f, desc="Building index"):
            data = json.loads(line)
            docid   = data['docid']
            terms   = data['ids']
            weights = data['weight']

            # Sort by weight in descending order
            pairs = list(zip(terms, weights))
            pairs.sort(key=lambda x: x[1], reverse=True)
            # Truncate to max_latent_num if specified
            if max_latent_num is not None:
                pairs = pairs[:max_latent_num]

            # Unzip back to terms and weights
            if pairs:
                terms, weights = zip(*pairs)
                terms   = list(terms)
                weights = list(weights)
            else:
                terms, weights = [], []

            # Use the number of selected weights as document length
            doc_length = len(weights)
            document_lengths[docid] = doc_length
            sum_dl += doc_length
            total_docs += 1

            # Build inverted index
            for term, tf in zip(terms, weights):
                entry = inverted_index[str(term)]
                entry['postings'].append((docid, tf))
                entry['doc_freq'] += 1

    # Compute average document length
    avgdl = sum_dl / total_docs if total_docs else 0.0

    # Compute IDF for each term
    for term, entry in tqdm(inverted_index.items(), desc="Computing IDF"):
        n_t = entry['doc_freq']
        entry['idf'] = math.log((total_docs - n_t + 0.5) / (n_t + 0.5))

    # Collect all index data
    index_data = {
        'inverted_index': dict(inverted_index),
        'document_lengths': document_lengths,
        'total_docs': total_docs,
        'avgdl': avgdl
    }

    # Save as pickle
    with open(index_save_path, 'wb') as f:
        pickle.dump(index_data, f)

    print(f"[Done] Inverted index saved to {index_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build inverted index from passage latent JSONL file.")
    parser.add_argument("--passage_latents", type=str, required=True, help="Path to passage_docids.jsonl file")
    parser.add_argument("--index_save_path", type=str, required=True, help="Path to save the inverted index pickle file")
    parser.add_argument("--max_latents", type=int, default=None, help="Maximum number of latent concepts per document to index")

    args = parser.parse_args()
    build_inverted_index(args.passage_latents, args.index_save_path, max_latent_num=args.max_latents)