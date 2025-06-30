# %%
import torch as t
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
import einops
from torch.utils.data import DataLoader, Dataset
import pickle
import argparse


class SAE_Dataset(Dataset):
    def __init__(self, filepaths, query_filepath):
        self.tensorstack = t.load(filepaths[0])
        for e in filepaths[1:]:
            self.tensorstack = t.cat((self.tensorstack, t.load(e)), dim=0)
        self.tensorstack = t.cat((self.tensorstack, t.load(query_filepath)), dim=0)
        self.len = self.tensorstack.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.tensorstack[idx]


# region model
class SAE_bias_pre(nn.Module):
    def __init__(self, input_dim, sparse_dim, k, dead_threshold, bias_pre):
        super(SAE_bias_pre, self).__init__()
        self.linear = nn.Linear(input_dim, sparse_dim, bias=False)
        self.linear2 = nn.Linear(sparse_dim, input_dim, bias=False)
        self.linear2.weight = nn.Parameter(self.linear.weight.clone().T)
        self.bias_pre = nn.Parameter(bias_pre.clone())
        self.k = k
        self.threshold = dead_threshold
        self.last_activation = t.zeros((sparse_dim,)).to(device)
        self.mean_of_min_act = 0

    def forward(self, x):
        linear_output = self.linear(x - self.bias_pre)
        bs = linear_output.shape[0]
        linear_output = einops.rearrange(linear_output, "b d -> (b d)")
        topk_values, topk_indices = t.topk(linear_output, k=self.k * bs)
        self.mean_of_min_act += topk_values[-1].item()
        topk = t.zeros_like(linear_output)
        topk.scatter_(dim=0, index=topk_indices, src=topk_values)
        non_dead_topk_mask = topk != 0

        self.last_activation = self.last_activation + 1
        non_dead_topk_mask = einops.rearrange(
            non_dead_topk_mask, "(b nd) -> b nd", b=bs
        )
        flat_non_dead_topk_mask = einops.reduce(
            non_dead_topk_mask, "batch inner_dim->inner_dim", "max"
        )
        self.last_activation = self.last_activation * (~flat_non_dead_topk_mask)

        mask = self.last_activation > self.threshold
        # if t.sum(mask) > 0: #similar to paper
        if True:  # consistant loss
            linear_output = einops.rearrange(linear_output, "(b d) -> b d", b=bs)
            dead_activation = linear_output.masked_fill(~mask, float("-inf"))
            dead_topk_values, dead_topk_indices = t.topk(dead_activation, k=self.k * 2)
            dead_topk_values[t.isinf(dead_topk_values)] = 0
            dead_topk = t.zeros_like(linear_output)
            dead_topk.scatter_(dim=1, index=dead_topk_indices, src=dead_topk_values)
            dead_recon = self.linear2(dead_topk)
        else:
            dead_recon = None

        topk = einops.rearrange(topk, "(b nd) -> b nd", b=bs)
        recon = self.linear2(topk)
        recon += self.bias_pre
        return recon, dead_recon

    @t.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.linear2.weight / self.linear2.weight.norm(
            dim=-1, keepdim=True
        )
        W_dec_grad_proj = (self.linear2.weight.grad * W_dec_normed).sum(
            -1, keepdim=True
        ) * W_dec_normed
        self.linear2.weight.grad -= W_dec_grad_proj
        self.linear2.weight.data = W_dec_normed

    @t.no_grad()
    def make_decoder_weights_norm(self):
        W_dec_normed = self.linear2.weight / self.linear2.weight.norm(
            dim=-1, keepdim=True
        )
        self.linear2.weight.data = W_dec_normed

    @t.no_grad()
    def get_thresh_recon(self, x, thresh):
        linear_output = self.linear(x - self.bias_pre)
        bs = linear_output.shape[0]
        mask = linear_output > thresh
        linear_output = linear_output * mask
        recon = self.linear2(linear_output)
        recon += self.bias_pre
        return recon


# endregion


def main():
    parser = argparse.ArgumentParser(description="Train Sparse Autoencoder")

    parser.add_argument(
        "--embeddings",
        type=str,
        required=True,
        help="Path to the input embeddings",
    )
    parser.add_argument(
        "--hidden-mult", type=int, default=32, help="Multiplier for hidden layer size"
    )
    parser.add_argument(
        "--k", type=int, default=32, help="Sparsity level (top-k units)"
    )
    parser.add_argument("--batch", type=int, default=4096, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--out", type=str, required=True, help="Output file path for the trained model"
    )
    parser.add_argument("--dead_threshold_step", type=int, required=False, default=20)
    parser.add_argument("--aux_loss_alpha", type=float, required=False, default=1 / 32)
    args = parser.parse_args()

    print(f"Embeddings: {args.embeddings}")
    print(f"Hidden Multiplier: {args.hidden_mult}")
    print(f"k: {args.k}")
    print(f"Batch Size: {args.batch}")
    print(f"Learning Rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Output Path: {args.out}")

    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    seed_value = 42
    t.manual_seed(seed_value)

    dataset = SAE_Dataset(
        [
            f"{args.embeddings}/shard_0_0",
            f"{args.embeddings}/shard_0_1",
            f"{args.embeddings}/shard_0_2",
            f"{args.embeddings}/shard_1_0",
            f"{args.embeddings}/shard_1_1",
            f"{args.embeddings}/shard_1_2",
        ],
        f"{args.embeddings}/train",
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch, num_workers=4, shuffle=True, pin_memory=True
    )

    # %%
    # region train
    from tqdm import tqdm

    latent_mult = lm
    input_dim = 768
    sparse_dim = 768 * args.hidden_mult
    alpha = args.aux_loss_alpha
    threshold = args.dead_threshold_step
    model = SAE_bias_pre(input_dim, sparse_dim, k, threshold, t.zeros((input_dim,))).to(
        device
    )
    optimizer = t.optim.AdamW(model.parameters(), lr=args.lr, eps=6e-10)
    num_epochs = args.epochs

    do_wandb = False
    if do_wandb:
        import wandb

        wandb.init(
            project="sae_on_simlm_clean",
            name=f"batchtopk_with_query_latent-x{latent_mult}_length-{k}_batchtopk_batch{batch_size}_threshhold_{threshold}step_alpha_{alpha}_lr{lr}",
            reinit=True,
        )

    model.train()
    for i in range(num_epochs):
        running_loss = 0.0
        idx = 1
        model.mean_of_min_act = 0
        for input in dataloader:
            # print(input)
            inputs = input.to(device)
            optimizer.zero_grad()
            recon, dead_recon = model(inputs)
            error = inputs - recon
            error_square = t.square(error)
            if dead_recon == None:
                loss = error_square
            else:
                loss = (error_square) + alpha * (t.square(error - dead_recon))

            loss = loss.mean()
            loss.backward()

            t.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100)
            model.make_decoder_weights_and_grad_unit_norm()
            optimizer.step()
            running_loss += loss.item()
            print(
                f"Epoch [{i+1}/{num_epochs}], loss: {loss}, recon error: {error_square.mean()/input_dim}",
                flush=True,
            )
            if do_wandb:
                wandb.log(
                    {
                        "loss": loss,
                        "recon error": error_square.mean() / input_dim,
                    }
                )
            idx += 1
        print(
            f"Epoch [{i+1}/{num_epochs}], RunningLoss: {running_loss/idx}, min_activation: {model.mean_of_min_act/((dataset.len) // batch_size)}",
            flush=True,
        )
        if do_wandb:
            wandb.log({"RunningLoss": running_loss / idx})

    import os

    os.makedirs(
        f"{args.out}",
        exist_ok=True,
    )

    model_path = f"{args.out}/sae_{args.hidden_mult}_k{args.k}_mean_of_min_act-{model.mean_of_min_act/((dataset.len) // batch_size)}.pt"
    t.save(model.state_dict(), model_path)

    wandb.finish()
    # endregion
    # region nmse eval
    model.eval()
    act_thresh = model.mean_of_min_act / ((dataset.len) // batch_size)
    running_loss = 0.0
    idx = 1
    model.mean_of_min_act = 0
    mean_input = t.zeros((input_dim,))
    n = 0
    for input in dataloader:
        batch_size = input.shape[0]
        batch_mean = t.mean(input, dim=0)
        new_n = n + batch_size
        mean_input = (n * mean_input + batch_size * batch_mean) / new_n
        n = new_n

    # %%
    mse = 0
    bmse = 0
    n = 0
    with t.no_grad():
        for input in tqdm(dataloader):
            # print(input)
            inputs = input.to(device)
            recon = model.get_thresh_recon(inputs, act_thresh)
            error = inputs - recon
            error_square = t.square(error)
            baseline_error = input - mean_input
            baseline_error_square = t.square(baseline_error)
            tmp_mse = t.mean(error_square)
            tmp_bmse = t.mean(baseline_error_square)
            # print(tmp_mse)
            # print(tmp_bmse)
            mse += tmp_mse
            bmse += tmp_bmse
            n += 1
        print(f"NMSE : {mse / bmse}")

    # endregion


# %%


if __name__ == "__main__":
    main()
