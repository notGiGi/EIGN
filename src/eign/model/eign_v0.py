import math

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, dim)
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.weight


class RoPE(nn.Module):
    def __init__(
        self, head_dim: int, base: float = 10000.0, max_seq_len: int | None = None
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE head_dim must be even.")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len
        if max_seq_len is not None:
            cos, sin = self._build_cache(max_seq_len)
            self.register_buffer("cos_cached", cos, persistent=False)
            self.register_buffer("sin_cached", sin, persistent=False)

    def _build_cache(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(
            seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        freqs = torch.outer(positions, self.inv_freq)  # (seq_len, head_dim / 2)
        cos = freqs.cos()
        sin = freqs.sin()
        return cos[None, None, :, :], sin[None, None, :, :]

    def get_cos_sin(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hasattr(self, "cos_cached") and self.cos_cached is not None:
            if seq_len > self.cos_cached.size(2):
                raise ValueError("seq_len exceeds cached RoPE length.")
            cos = self.cos_cached[:, :, :seq_len, :].to(device=device, dtype=dtype)
            sin = self.sin_cached[:, :, :seq_len, :].to(device=device, dtype=dtype)
            return cos, sin
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq)  # (seq_len, head_dim / 2)
        cos = freqs.cos().to(dtype=dtype)
        sin = freqs.sin().to(dtype=dtype)
        return cos[None, None, :, :], sin[None, None, :, :]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (batch, heads, seq_len, head_dim)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        rope_base: float,
        max_seq_len: int,
        attn_dropout: float,
        resid_dropout: float,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RoPE(self.head_dim, base=rope_base, max_seq_len=max_seq_len)
        self.max_seq_len = max_seq_len
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", causal_mask, persistent=False)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.resid_dropout = nn.Dropout(resid_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError("seq_len exceeds max_seq_len.")
        # qkv: (batch, seq_len, 3 * d_model)
        qkv = self.qkv_proj(x)
        # qkv: (3, batch, heads, seq_len, head_dim)
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        cos, sin = self.rope.get_cos_sin(seq_len, x.device, q.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        use_sdp = hasattr(F, "scaled_dot_product_attention")
        if use_sdp:
            dropout_p = self.attn_dropout.p if self.training else 0.0
            attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True
            )
        else:
            # attn_scores: (batch, heads, seq_len, seq_len)
            attn_scores = torch.matmul(q, k.transpose(-2, -1))
            attn_scores = attn_scores / math.sqrt(self.head_dim)

            causal_mask = self.causal_mask[:seq_len, :seq_len]
            attn_scores = attn_scores.masked_fill(
                ~causal_mask[None, None, :, :], torch.finfo(attn_scores.dtype).min
            )

            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = self.attn_dropout(attn_probs)
            # attn_output: (batch, heads, seq_len, head_dim)
            attn_output = torch.matmul(attn_probs, v)
        # attn_output: (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, d_model)
        return self.resid_dropout(self.out_proj(attn_output))


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, ffn_dim, bias=False)
        self.up_proj = nn.Linear(d_model, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        gate = self.gate_proj(x)  # (batch, seq_len, ffn_dim)
        up = self.up_proj(x)  # (batch, seq_len, ffn_dim)
        x = F.silu(gate) * up
        return self.down_proj(x)  # (batch, seq_len, d_model)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        rope_base: float,
        max_seq_len: int,
        attn_dropout: float,
        resid_dropout: float,
        mlp_dropout: float,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model,
            n_heads,
            rope_base,
            max_seq_len,
            attn_dropout,
            resid_dropout,
        )
        self.mlp_norm = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, ffn_dim)
        self.mlp_dropout = nn.Dropout(mlp_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp_dropout(self.mlp(self.mlp_norm(x)))
        return x


class EIGNModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_layers: int = 12,
        d_model: int = 768,
        n_heads: int = 12,
        ffn_dim: int = 3072,
        rope_base: float = 10000.0,
        max_seq_len: int = 2048,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        seed: int | None = 0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if (d_model // n_heads) % 2 != 0:
            raise ValueError("head_dim must be even for RoPE.")

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.init_std = 0.02
        self.tok_embeddings = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    d_model,
                    n_heads,
                    ffn_dim,
                    rope_base,
                    max_seq_len,
                    attn_dropout,
                    resid_dropout,
                    mlp_dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if seed is not None:
            torch.manual_seed(seed)
        self.apply(self._init_weights)
        self.lm_head.weight = self.tok_embeddings.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            if module is self.lm_head:
                return
            nn.init.normal_(module.weight, mean=0.0, std=self.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.init_std)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (batch, seq_len)
        x = self.tok_embeddings(input_ids)  # (batch, seq_len, d_model)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)
        return logits


if __name__ == "__main__":
    torch.manual_seed(123)
    model = EIGNModel(vocab_size=32000, seed=123)
    input_ids = torch.randint(0, 32000, (2, 16))
    logits = model(input_ids)
    print("logits shape:", logits.shape)
