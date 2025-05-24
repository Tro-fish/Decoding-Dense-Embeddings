import pdb
import einops
import torch as t
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import pickle
from torch.utils.data import DataLoader, Dataset

device = t.device("cuda")

class SAE_Dataset(Dataset):
    def __init__(self, filepaths):
        self.tensorstack = t.load(filepaths[0])
        for e in filepaths[1:]:
            self.tensorstack = t.cat((self.tensorstack, t.load(e)), dim=0)
        self.len = self.tensorstack.shape[0]
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.tensorstack[idx]
 
class SAE_query_Dataset(Dataset):
   def __init__(self, filepaths):
       self.query_data = t.load(filepaths[0], map_location='cpu')
       self.len = self.query_data.shape[0]

   def __len__(self):
       return self.len
   
   def __getitem__(self, idx):
       return self.query_data[idx]
       
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
    
    def debug(self, x):
        x = self.linear(x - self.bias_pre)
        topk_values, indices = t.topk(x, dim=-1, k=32)
        topk = t.zeros_like(x)
        topk.scatter_(dim=-1, index=indices, src=topk_values)
        return self.linear2(topk) + self.bias_pre

    def cosine_similarity(self, x, threshhold):
        linear_output = self.linear(x - self.bias_pre)
        mask = linear_output > threshhold
        linear_output_masked = linear_output * mask
        recon = self.linear2(linear_output_masked)
        recon += self.bias_pre
        return linear_output_masked, recon
    
    def encode_topk(self, x, k): # recon도 topk로 복원하도록 코드 수정
        # 선형 변환
        linear_output = self.linear(x - self.bias_pre)  # [batch_size, sparse_dim]
        # 각 배치별로 top-k 값과 인덱스 추출
        topk_values, topk_indices = t.topk(linear_output, k=k, dim=1, largest=True, sorted=True)

        # top-k 값만 남기고 나머지는 0으로 설정
        topk_mask = t.zeros_like(linear_output)  # [batch_size, sparse_dim]
        topk_mask.scatter_(dim=1, index=topk_indices, src=topk_values)

        # top-k 활성화 값을 사용하여 reconstruction 수행
        recon = self.linear2(topk_mask)  # 디코더 적용
        recon += self.bias_pre  # bias 추가

        # 결과 반환 (인덱스와 값은 리스트로 변환)
        topk_indices = topk_indices[0].tolist()
        topk_values = topk_values[0].tolist()
        return topk_indices, topk_values, recon

    def reverse_encode_topk(self, x, k):
        linear_output = self.linear(x - self.bias_pre)  # 선형 변환
        # largest=False로 변경하여 오름차순으로 정렬
        topk_values, topk_indices = t.topk(linear_output, k=k, dim=1, largest=False, sorted=True)
        topk_indices = topk_indices[0].tolist()
        topk_values = topk_values[0].tolist()
        return topk_indices, topk_values

    def encode_batchtopk(self, x, threshold):
        linear_output = self.linear(x - self.bias_pre)
        mask = linear_output > threshold
        linear_output_masked = linear_output * mask

        recon = self.linear2(linear_output_masked)
        recon += self.bias_pre

        # 활성화된 값과 인덱스 추출
        activation_values = linear_output[mask]  # 활성화된 노드들의 값
        indices = t.nonzero(mask, as_tuple=False)  # 활성화된 노드의 배치와 인덱스 정보

        # 배치 단위로 그룹화된 결과 생성
        batch_size = x.shape[0]
        result = []
        for batch_idx in range(batch_size):
            # 현재 배치에 속한 활성화된 노드 필터링
            batch_mask = indices[:, 0] == batch_idx
            activated_indices = indices[batch_mask][:, 1].tolist()  # 현재 배치에서 활성화된 노드 인덱스
            activated_weights = activation_values[batch_mask].tolist()  # 활성화된 노드들의 값
            # 결과 추가
            result.append([batch_idx, activated_indices, activated_weights])

        return result, recon
    
    def encode_topk_threshold(self, x, threshold, min_activation_num):
        linear_output = self.linear(x - self.bias_pre)  # 선형 변환
        mask = linear_output > threshold  # 임계값 초과 여부
        linear_output_masked = linear_output * mask  # 활성화된 값만 남기기

        recon = self.linear2(linear_output_masked)
        recon += self.bias_pre

        # 0이 아닌 값과 인덱스 추출
        nonzero_indices = linear_output_masked.nonzero(as_tuple=True)  # 0이 아닌 값의 인덱스
        nonzero_values = linear_output_masked[nonzero_indices]  # 해당 인덱스의 값

        # 요소 개수가 16개 이하일 경우 처리
        if nonzero_values.numel() < min_activation_num:
            # linear_output에서 내림차순으로 정렬된 모든 인덱스와 값 가져오기
            sorted_indices = t.argsort(linear_output.view(-1), descending=True)
            sorted_values = linear_output.view(-1)[sorted_indices]

            # 이미 선택된 인덱스를 제외한 추가 후보 선택
            selected_indices_set = set(zip(*[idx.tolist() for idx in nonzero_indices]))
            additional_indices = []
            additional_values = []

            for idx, value in zip(sorted_indices.tolist(), sorted_values.tolist()):
                idx_tuple = tuple(np.unravel_index(idx, linear_output.shape))
                if idx_tuple not in selected_indices_set:
                    additional_indices.append(idx_tuple)
                    additional_values.append(value)
                    if len(nonzero_values) + len(additional_values) == min_activation_num:
                        break

            # 기존 값과 추가된 값 병합
            nonzero_values = t.cat([nonzero_values, t.tensor(additional_values, device=linear_output.device)])
            nonzero_indices = tuple(
                t.cat([idx, t.tensor([a[i] for a in additional_indices], dtype=t.long, device=linear_output.device)])
                for i, idx in enumerate(nonzero_indices)
            )
        nonzero_indices = nonzero_indices[1].tolist()
        nonzero_values = nonzero_values.tolist()
        return recon, nonzero_indices, nonzero_values
    
    @t.no_grad()
    def grad_pursuit_update_step(self, signal, weights, dictionary):
        """
        residual: d, weights: n, dictionary: n x d
        """
        # get a mask for which features have already been chosen (ie have nonzero weights)
        residual = signal - weights * dictionary
        selected_features = weights != 0
        # choose the element with largest inner product, as in matched pursuit.
        inner_products = einops.einsum(dictionary, residual, "n d, d -> n")
        idx = t.argmax(inner_products)
        # add the new feature to the active set.
        selected_features[idx] = 1

        # the gradient for the weights is the inner product above, restricted
        # to the chosen features
        grad = selected_features * inner_products
        # the next two steps compute the optimal step size;
        c = einops.einsum("n, n d -> d", grad, dictionary)
        step_size = einops.einsum(c, residual, "d, d->") / einops.einsum(c, c, "d, d->")
        weights = weights + step_size * grad
        weights = max(weights, 0)  # clip the weights to be positive
        return weights

    @t.no_grad()
    def grad_pursuit_update_step_batched(self, signal, weights, dictionary):
        """
        residual: b d, weights: b n, dictionary: n d
        """
        # get a mask for which features have already been chosen (ie have nonzero weights)
        residual = signal - (weights @ dictionary)
        selected_features = weights != 0
        # choose the element with largest inner product, as in matched pursuit.
        # import pdb

        # pdb.set_trace()
        inner_products = einops.einsum(dictionary, residual, "n d, b d -> b n")
        idx = t.argmax(inner_products, dim=-1)
        # add the new feature to the active set.
        selected_features[t.arange(signal.shape[0]), idx] = 1
        # t.cuda.synchronize()
        # the gradient for the weights is the inner product above, restricted
        # to the chosen features
        grad = selected_features * inner_products
        # the next two steps compute the optimal step size;
        c = einops.einsum(grad, dictionary, "b n, n d -> b d")
        step_size = einops.einsum(c, residual, "b d, b d -> b") / einops.einsum(
            c, c, "b d, b d -> b"
        )
        weights = weights + step_size.unsqueeze(-1) * grad
        weights = t.clamp(weights, min=0)  # clip the weights to be positive
        return weights, residual
    
    @t.no_grad()
    def grad_pursuit(self, signal, dictionary, target_l0=32, batched=True, device=None):
        print("------------------residual------------------")
        if batched:
            weights = t.zeros((signal.shape[0], dictionary.shape[0])).to(device)
            for _ in range(target_l0):
                weights, residual = self.grad_pursuit_update_step_batched(
                    signal, weights, dictionary
                )
            return weights
        else:
            weights = t.zeros(dictionary.shape[0]).to(device)
            for _ in range(target_l0):
                weights = self.grad_pursuit_update_step(signal, weights, dictionary)
            return weights
        
    @t.no_grad()
    def gradient_pursuit(self, x):
        x = x - self.bias_pre
        weights = self.grad_pursuit(
            x, self.linear2.weight.data.T, 32, True, device
        )  # weights (b, 768x32)
        indices = t.nonzero(weights, as_tuple=False)
        indices_1d = indices[:, 1]
        activations = weights[indices[:, 0], indices[:, 1]]  # 인덱스 기반 값 추출
        return indices_1d, activations

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