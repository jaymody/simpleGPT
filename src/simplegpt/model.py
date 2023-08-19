import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange


class GPT(nn.Module):
    def __init__(self, n_vocab, n_ctx, n_embd, n_head, n_layer, dropout_p=0.0):
        super().__init__()
        self.wte = nn.Embedding(n_vocab, n_embd)
        self.wpe = nn.Embedding(n_ctx, n_embd)
        self.h = nn.Sequential(
            *[DecodeLayer(n_embd, n_head, dropout_p) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.embd_dropout = nn.Dropout(dropout_p)

        def _init_weights(module, is_residual=False):
            std = 0.02 / math.sqrt(2 * n_layer) if is_residual else 0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if hasattr(module, "bias") and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        _init_weights(self.wte)
        _init_weights(self.wpe)
        for block in self.h:
            _init_weights(block.attn.c_attn)
            _init_weights(block.attn.c_proj, is_residual=True)
            _init_weights(block.mlp.c_fc)
            _init_weights(block.mlp.c_proj, is_residual=True)

    def forward(self, ids):
        x = self.wte(ids) + self.wpe(torch.arange(ids.shape[-1]).to(ids.device))
        x = self.embd_dropout(x)
        x = self.h(x)
        x = self.ln_f(x)
        x = einsum(x, self.wte.weight, "b s e, v e -> b s v")
        return x

    def lm_loss(self, ids):
        x, y = ids[:, :-1], ids[:, 1:]
        logits = self(x)
        logits = rearrange(logits, "bs seq vocab -> (bs seq) vocab")
        y = rearrange(y, "bs seq -> (bs seq)")
        loss = F.cross_entropy(logits, y)
        return loss

    def generate(self, ids, n_tokens_to_generate, temperature, eps=1e-10):
        from tqdm import tqdm

        for _ in tqdm(range(n_tokens_to_generate)):
            # get logits for next word
            logits = self(ids)  # [bs, seq, vocab]
            logits = logits[:, -1, :]  # [bs, vocab]

            # sample from probabilities
            logits = logits / (temperature + eps)
            probs = torch.softmax(logits, dim=-1)
            next_ids = torch.multinomial(probs, 1)  # [bs, 1]

            # append to ids
            ids = torch.cat([ids, next_ids], dim=-1)  # [bs, seq + 1]

        return ids[:, -n_tokens_to_generate:]

    @classmethod
    def from_pretrained(cls, model_name, load_weights=True):
        from transformers import GPT2Config, GPT2LMHeadModel

        config = GPT2Config.from_pretrained(model_name)
        model = cls(
            n_vocab=config.vocab_size,
            n_ctx=config.n_ctx,
            n_embd=config.n_embd,
            n_head=config.n_head,
            n_layer=config.n_layer,
        )

        if load_weights:
            pretrained = GPT2LMHeadModel.from_pretrained(model_name)
            params = model.state_dict()
            pretrained_params = pretrained.state_dict()
            for k, tensor in pretrained_params.items():
                # skip lm_head.weight since it's the same as wte.weight
                if k == "lm_head.weight":
                    continue

                k = k.removeprefix("transformer.")

                # open-ai GPT2 uses a Conv1D layer instead of Linear layer which means
                # the weights are transposed from what they should be in Linear, so we
                # need to transpose them before copying
                if any(
                    k.endswith(s) for s in ["attn.weight", "proj.weight", "fc.weight"]
                ):
                    tensor = tensor.T

                params[k].copy_(tensor)

        return model


class DecodeLayer(nn.Module):
    def __init__(self, n_embd, n_head, dropout_p):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadCasualSelfAttention(n_embd, n_head, dropout_p)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = PositionWiseFeedForwardNetwork(n_embd)
        self.resid_dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = x + self.resid_dropout(self.attn(self.ln_1(x)))
        x = x + self.resid_dropout(self.mlp(self.ln_2(x)))
        return x


class PositionWiseFeedForwardNetwork(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x


class MultiHeadCasualSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout_p):
        super().__init__()
        self.n_head = n_head

        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)

        self.attn_dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        # qkv projection
        x = self.c_attn(x)

        # split into qkv and heads
        q, k, v = rearrange(x, "b s (n h d) -> n b h s d", n=3, h=self.n_head)

        # self-attention with casual mask
        n_seq = x.shape[1]
        mask = torch.triu(torch.ones(n_seq, n_seq).to(x.device), diagonal=1) * -1e10
        attn = einsum(q, k, "b h nq dk, b h nk dk -> b h nq nk")
        attn = torch.softmax(attn / math.sqrt(k.shape[-1]) + mask, dim=-1)
        attn = self.attn_dropout(attn)
        out = einsum(attn, v, "b h nq nk, b h nk dv -> b h nq dv")
        out = rearrange(out, "b h nq dv -> b nq (h dv)")

        # out projection
        out = self.c_proj(out)
        return out
