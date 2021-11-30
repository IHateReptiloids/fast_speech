import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, dim_k, dim_v, n_heads):
        super().__init__()
        assert dim_k % n_heads == 0 and dim_v % n_heads == 0
        self.n_heads = n_heads
        self.query_projector = nn.Linear(dim_k, dim_k)
        self.key_projector = nn.Linear(dim_k, dim_k)
        self.value_projector = nn.Linear(dim_v, dim_v)
        self.final_projector = nn.Linear(dim_v, dim_v)

    def forward(self, x, mask=None):
        return self._forward(x, x, x, mask)

    def _forward(self, q, k, v, mask=None):
        '''
        q, k, v are of shape [bs, seq_len, n_features]
        mask is of shape [seq_len, seq_len]
        mask is multiplicative, applied after softmax
        '''
        assert q.dim() == k.dim() == v.dim() == 3
        assert q.shape[0] == k.shape[0] == v.shape[0]
        assert q.shape[-1] == k.shape[-1]
        assert k.shape[-2] == v.shape[-2]

        q = self.query_projector(q).view(*q.shape[:-1], self.n_heads, -1)\
            .transpose(1, 2)
        k = self.key_projector(k).view(*k.shape[:-1], self.n_heads, -1)\
            .transpose(1, 2)
        v = self.value_projector(v).view(*v.shape[:-1], self.n_heads, -1)\
            .transpose(1, 2)
        x = self.attend(q, k, v, mask).transpose(1, 2)
        return self.final_projector(x.reshape(*x.shape[:-2], -1))

    @staticmethod
    def attend(q, k, v, mask=None):
        '''
        q, k, v are of shape [*, seq_len, n_features]
        mask is of shape [seq_len, seq_len]
        mask is multiplicative, applied after softmax
        returns tensor of shape [*, seq_len, n_features]
        '''
        assert q.dim() == k.dim() == v.dim()
        assert q.shape[:-2] == k.shape[:-2] == v.shape[:-2]
        assert q.shape[-1] == k.shape[-1]
        assert k.shape[-2] == v.shape[-2]
        dim_k = k.shape[-1]

        sim = (torch.matmul(q, k.transpose(-1, -2)) /
               torch.sqrt(torch.tensor(dim_k)))
        sim = F.softmax(sim, dim=-1)
        if mask is not None:
            sim = sim * mask
        return torch.matmul(sim, v)
