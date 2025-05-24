import os
import argparse
import numpy as np
import torch as t
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import *
from models import SAE_Dataset


# /p44/SAE/checkpoint/simlm/msmarco/32_0.160438.pt
# /p44/SAE/embs/passage/simlm
# /p44/SAE/github/sae/recon/
# 0.160438

def main():
    parser = argparse.ArgumentParser(description="Reconstruct DPR model embeddings using a trained SAE.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained SAE checkpoint (e.g., checkpoints/sae_k32.pt)")
    parser.add_argument("--input_embs_path", type=str, required=True, help="Path to input embedding directory")
    parser.add_argument("--recon_embs_save_path", type=str, required=True, help="Path to save reconstructed embeddings")
    parser.add_argument("--model_threshold", type=float, required=True, help="Threshold for latent activation")
    parser.add_argument("--input_dim", type=int, default=768, help="Dimensionality of embeddings")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size for inference")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run inference on")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.recon_embs_save_path), exist_ok=True)
    
    if not os.path.exists(args.recon_embs_save_path):
        with open(args.recon_embs_save_path, "wb") as f:
            np.lib.format.write_array_header_1_0(
                f,
                {
                    "descr": np.dtype("float64").descr,
                    "fortran_order": False,
                    "shape": (0, args.input_dim),
                },
            )

    # Load dataset
    shard_paths = sorted([os.path.join(args.input_embs_path, name) for name in os.listdir(args.input_embs_path)])
    dataset = SAE_Dataset(shard_paths)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, pin_memory=True)

    # Load model
    model = load_model(
        model_name="BatchTopK",  # assuming BatchTopK for reconstruction
        checkpoint_path=args.checkpoint,
        dataset_len=len(dataset) // args.batch_size,
        input_dim=args.input_dim
    )
    model = model.to(args.device)
    model.eval()

    # Inference
    for input_data in tqdm(dataloader, desc="Reconstructing embeddings"):
        inputs = input_data.to(args.device)
        _, recon = model.encode_batchtopk(inputs, threshold=args.model_threshold)  # no thresholding for recon
        recon = recon.detach().cpu().numpy()
        with open(args.recon_embs_save_path, "ab") as f:
            f.write(recon.tobytes())

    print(f"Reconstructed embeddings saved to {args.recon_embs_save_path}")

if __name__ == "__main__":
    main()
