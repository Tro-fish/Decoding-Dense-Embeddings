from tqdm import tqdm
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
from models import *
from collections import defaultdict
from itertools import chain
import numpy as np
import json
import torch as t

def load_model(model_name: str, 
               checkpoint_path: str, 
               dataset_len: int,
               input_dim: int = 768) -> t.nn.Module:

    model = SAE_bias_pre(input_dim, input_dim * 32, 0, dataset_len, t.zeros((input_dim,)))
    model.load_state_dict(t.load(checkpoint_path))
    return model