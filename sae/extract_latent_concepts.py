# sae/reconstruct_embedding.py
import os
import argparse
import json
import numpy as np
import torch as t
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import *
from models import SAE_Dataset

def main():
    parser = argparse.ArgumentParser(description="Extract latent concepts using a trained SAE.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained SAE checkpoint")
    parser.add_argument("--input_embs_path", type=str, required=True, help="Directory containing input embedding shards")
    parser.add_argument("--latent_concepts_save_path", type=str, required=True, help="Path to save extracted latent concepts as JSONL")
    parser.add_argument("--model_threshold", type=float, default=0.1, help="Threshold for latent activation")
    parser.add_argument("--input_dim", type=int, default=768, help="Dimensionality of embeddings")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size for inference")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run inference on")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.latent_concepts_save_path), exist_ok=True)

    # Load input shards
    shard_paths = sorted([
        os.path.join(args.input_embs_path, fname)
        for fname in os.listdir(args.input_embs_path)
        if fname.startswith("shard")
    ])
    dataset = SAE_Dataset(shard_paths)
    #dataset = SAE_query_Dataset(args.input_embs_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)

    # Load SAE model
    model = load_model(
        model_name="BatchTopK",
        checkpoint_path=args.checkpoint,
        dataset_len=len(dataset) // args.batch_size,
        input_dim=args.input_dim
    )
    model.to(args.device)
    model.eval()

    # Extract latent concepts and write to JSONL
    with open(args.latent_concepts_save_path, "w", encoding="utf-8") as fout:
        index_offset = 0
        for batch in tqdm(dataloader, desc="Extracting latent concepts"):
            batch = batch.to(args.device)
            results, _ = model.encode_batchtopk(batch, threshold=args.model_threshold)
            for doc_idx, (docid, ids, weights) in enumerate(results):
                fout.write(json.dumps({
                    "docid": index_offset + doc_idx,
                    "ids": ids,
                    "weight": weights
                }, ensure_ascii=False) + "\n")
            index_offset += len(batch)

    print(f"✅ Latent concepts saved to {args.latent_concepts_save_path}")

if __name__ == "__main__":
    main()