import math
import functools

import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, reduce, repeat, pack, unpack
from einops.layers.torch import Rearrange

from MEGABYTE_pytorch.attend import Attend

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def remainder_to_mult(num, mult):
    return (mult - num % mult) % mult

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

def reduce_mult(nums):
    return functools.reduce(lambda x, y: x * y, nums, 1)

# tensor helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim = dim)

def top_k(logits, thres = 0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# positional bias

class Alibi(nn.Module):
    def __init__(self, heads, **kwargs):
        super().__init__()
        self.heads = heads
        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, 'h -> h 1 1')
        self.register_buffer('slopes', slopes, persistent = False)
        self.register_buffer('bias', None, persistent = False)

    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads-closest_power_of_2]

    def forward(self, i, j, device):
        if exists(self.bias) and self.bias.shape[-1] >= j:
            return self.bias[..., :j]

        bias = torch.arange(j, device = device)
        bias = rearrange(bias, 'j -> 1 1 j')
        bias = bias * self.slopes

        self.register_buffer('bias', bias, persistent = False)
        return self.bias

# norm

class RMSNorm(nn.Module):
    def __init__(self, dim, eps = 1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim = -1, keepdim = True) * self.scale
        return x / norm.clamp(min = self.eps) * self.g

# helper classes

def FeedForward(*, dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        RMSNorm(dim),
        nn.Linear(dim, dim * mult),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )

class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        flash = False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.attend = Attend(
            causal = True,
            flash = flash,
            dropout = dropout
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = RMSNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, attn_bias = None):
        h, device = self.heads, x.device

        x = self.norm(x)
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = -1))
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)

        out = self.attend(q, k, v, attn_bias = attn_bias)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        layers,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        ff_mult = 4,
        rel_pos_bias = True,
        flash_attn = False
    ):
        super().__init__()
        self.alibi = Alibi(heads = heads) if rel_pos_bias else None
        self.layers = nn.ModuleList([])

        for _ in range(layers):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, flash = flash_attn),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
            ]))

        self.norm = RMSNorm(dim)

    def forward(self, x):
        n = x.shape[-2]
        attn_bias = self.alibi(n, n, device = x.device) if exists(self.alibi) else None

        for attn, ff in self.layers:
            x = attn(x, attn_bias = attn_bias) + x
            x = ff(x) + x

        return self.norm(x)

# main class

class MEGABYTE(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        max_seq_len,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.,
        pad_id = 0,
        rel_pos_bias = True,
        flash_attn = False
    ):
        super().__init__()

        # simplified configuration for each stage of the hierarchy
        # depth = (2, 2, 4) would translate to depth 2 at first stage, depth 2 second stage, depth 4 third
        # max_seq_len = (16, 8, 4) would translate to max sequence length of 16 at first stage, length of 8 at second stage, length of 4 for last

        assert isinstance(depth, tuple) and isinstance(max_seq_len, tuple)
        assert len(depth) == len(max_seq_len)

        self.stages = len(depth)

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.start_tokens = nn.Parameter(torch.randn(dim))

        self.max_seq_len = max_seq_len

        self.pos_embs = nn.ModuleList([nn.Embedding(seq_len, dim) for seq_len in max_seq_len])

        self.patch_embedders = nn.ModuleList([nn.Sequential(
            Rearrange('... r d -> ... (r d)'),
            nn.LayerNorm(seq_len * dim),
            nn.Linear(seq_len * dim, dim),
            nn.LayerNorm(dim)
        ) for seq_len in self.max_seq_len[1:]])

        self.transformers = nn.ModuleList([])

        for stage_depth in depth:
            self.transformers.append(Transformer(
                dim = dim,
                layers = stage_depth,
                dim_head = dim_head,
                heads = heads,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout,
                ff_mult = ff_mult,
                rel_pos_bias = rel_pos_bias,
                flash_attn = flash_attn
            ))

        self.to_logits = nn.Linear(dim, num_tokens)
        self.pad_id = pad_id

    def generate(self, prime = None, filter_thres = 0.9, temperature = 1., default_batch_size = 1):
        total_seq_len = reduce_mult(self.max_seq_len)
        device = next(self.parameters()).device

        if not exists(prime):
            prime = torch.empty((default_batch_size, 0), dtype = torch.long, device = device)

        seq = prime
        batch = seq.shape[0]

        for _ in range(total_seq_len - seq.shape[-1]):
            logits = self.forward(seq)[:, -1]
            logits = top_k(logits, thres = filter_thres)
            sampled = gumbel_sample(logits, dim = -1, temperature = temperature)
            seq = torch.cat((seq, rearrange(sampled, 'b -> b 1')), dim = -1)

        return seq.reshape(batch, *self.max_seq_len)

    def forward_empty(self, batch_size):
        # take care of special case
        # where you sample from input of 0 (start token only)

        tokens = repeat(self.start_tokens, 'd -> b 1 d', b = batch_size)

        for transformer in self.transformers:
            tokens = transformer(tokens)

        return self.to_logits(tokens)

    def forward(self, ids, return_loss = False):
        batch = ids.shape[0]

        assert ids.ndim in {2, self.stages + 1}
        flattened_dims = ids.ndim == 2
        ids_orig_ndim = ids.ndim

        if ids.numel() == 0:
            return self.forward_empty(ids.shape[0])

        if flattened_dims:
            # allow for ids to be given in the shape of (batch, seq)
            # in which case it will be auto-padded to the next nearest multiple of depth seq len
            seq_len = ids.shape[-1]
            multiple_of = reduce_mult(self.max_seq_len[1:])
            padding = remainder_to_mult(seq_len, multiple_of)
            ids = F.pad(ids, (0, padding), value = self.pad_id)
            ids = ids.reshape(batch, -1, *self.max_seq_len[1:])

        b, *prec_dims, device = *ids.shape, ids.device

        # check some dimensions

        assert prec_dims[0] <= self.max_seq_len[0], 'the first dimension of your axial autoregressive transformer must be less than the first tuple element of max_seq_len (like any autoregressive transformer)'
        assert tuple(prec_dims[1:]) == tuple(self.max_seq_len[1:]), 'all subsequent dimensions must match exactly'

        # get token embeddings

        tokens = self.token_emb(ids)

        # get tokens for all hierarchical stages, reducing by appropriate dimensions
        # and adding the absolute positional embeddings

        tokens_at_stages = []
        reduced_tokens = tokens

        for ind, pos_emb, patch_emb in zip(range(len(prec_dims)), reversed(self.pos_embs), reversed((*self.patch_embedders, None))):
            is_first = ind == 0

            if not is_first:
                reduced_tokens = patch_emb(reduced_tokens)

            positions = pos_emb(torch.arange(reduced_tokens.shape[-2], device = device))
            tokens_with_position = reduced_tokens + positions
            tokens_at_stages.insert(0, tokens_with_position)

        # get start tokens and append to the coarsest stage

        start_tokens = repeat(self.start_tokens, 'f -> b 1 f', b = b)

        # spatial tokens is tokens with depth pos reduced along depth dimension + spatial positions        

        for ind, (stage_tokens, transformer) in enumerate(zip(tokens_at_stages, self.transformers)):
            is_last = ind == (self.stages - 1)

            stage_tokens = torch.cat((
                start_tokens,
                stage_tokens,
            ), dim = -2)

            stage_tokens, ps = pack_one(stage_tokens, '* n d')
            attended = transformer(stage_tokens)
            attended = unpack_one(attended, ps, '* n d')

            start_tokens = rearrange(attended[..., :-1, :], '... n d -> ... n 1 d')

        logits = self.to_logits(attended)

        logits = logits[..., 1:, :]

        if not return_loss:

            if flattened_dims:
                logits = rearrange(logits, 'b ... n -> b (...) n')
                logits = logits[:, :seq_len]

            return logits

        preds = rearrange(logits, 'b ... c -> b c (...)')
        labels = rearrange(ids, 'b ... -> b (...)')

        loss = F.cross_entropy(
            preds[..., :-1],
            labels[..., 1:],
            ignore_index = self.pad_id
        )
        return loss