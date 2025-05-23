# Decoding Dense Embeddings  
Sparse Autoencoders for Interpretable & Efficient Retrieval
<!-- Badges are optional. Delete if your repo is private. -->
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.x-lightgrey)
![License](https://img.shields.io/badge/license-Apache%202.0-green)

A reference implementation of **Sparse Autoencoder (SAE)–based interpretation** and **Concept-Level Sparse Retrieval (CL-SR)**, as introduced in our paper:

> **“Decoding Dense Embeddings: Sparse Autoencoders for Interpreting and Discretizing Dense Retrieval.”**  
> *S. Author* et al., ACL 2025 (under review).

---

## 🗺️ Table of Contents
1. [Methodology at a Glance](#methodology-at-a-glance)  
2. [SAE Training & Evaluation](#1-sae-training--evaluation)  
3. [CL-SR Index Construction](#2-cl-sr-index-construction)  
4. [CL-SR Inference & Benchmarking](#3-cl-sr-inference--benchmarking)  
5. [Installation](#installation)  
6. [Dataset Preparation](#dataset-preparation)  
7. [Directory Layout](#directory-layout)  
8. [Citation](#citation)  
9. [License](#license)

---

## Methodology at a Glance
<div align="center">
<img src="docs/figs/overview.png" width="70%" alt="Framework overview"/>
</div>

1. **SAE Decomposition** – A BatchTop-K sparse autoencoder projects dense embeddings into a high-dimensional, **k-sparse latent space**.  
2. **Latent Interpretation** – LLM prompts turn latent activations into concise natural-language concepts.  
3. **Concept-Level Indexing (CL-SR)** – Documents are indexed by **activated latent IDs** plus IDF-weighted scores, enabling BM25-style scoring without lexical overlap.  

---

## 1. SAE Training & Evaluation
```bash
# Train SAE (example: k = 32, MSMARCO train passages)
python sae/train_sae.py \
    --embeddings data/msmarco_train/embed.npy \
    --hidden-mult 32 --k 32 --batch 4096 \
    --lr 5e-5 --epochs 100 \
    --out checkpoints/sae_k32.pt

# Evaluate reconstruction & IR fidelity (Dev, TREC-DL19/20)
python sae/eval_sae.py \
    --sae checkpoints/sae_k32.pt \
    --embeddings data/msmarco_dev/embed.npy \
    --qrels data/trec_dl19/qrels.txt
