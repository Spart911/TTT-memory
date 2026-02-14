import os
import warnings

# Silence TF32 deprecation warning and tokenizer fork warning early.
warnings.filterwarnings(
    "ignore",
    message="Please use the new API settings to control TF32 behavior.*",
)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# Basic CUDA speed flags (safe defaults, new API)
try:
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    torch.backends.cudnn.conv.fp32_precision = "tf32"
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# Silence TF32 deprecation warning if it comes from external libs (redundant but safe).
warnings.filterwarnings(
    "ignore",
    message="Please use the new API settings to control TF32 behavior.*",
)

# Force math SDP backend to avoid missing backward for efficient/flash on some setups.
try:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass

MAX_LEN = 64

# Model configs from Table 3 (paper): Params -> (blocks, dim, heads)
MODEL_CONFIGS = {
    "125M": (12, 768, 12),
    "350M": (24, 1024, 16),
    "760M": (24, 1536, 16),
    "1.3B": (24, 2048, 32),
    "2.7B": (32, 2560, 32),
    "tiny40": (4, 40, 5),
}
MODEL_SIZE = "tiny40"


# ===========================
# 3️⃣ Transformer Components (по Not my model: config + transformer + attention)
# ===========================
# Константы из config (ModelConfig, 125m)
INITIALIZER_RANGE = 0.01
RMS_NORM_EPS = 1e-5
RESID_PDROP = 0.0
EMBD_PDROP = 0.0
LOGIT_CLAMP = 50.0
ROPE_THETA = 10000.0
# По умолчанию выключены для численной стабильности (включи после отладки)
USE_ROPE = False
USE_QK_NORM = True


class RMSNorm(nn.Module):
    """RMSNorm с защитой от overflow/NaN."""
    def __init__(self, dim, eps=RMS_NORM_EPS):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x.float()
        x = torch.clamp(x, -1e4, 1e4)
        rms = (x.pow(2).mean(-1, keepdim=True).clamp(min=self.eps) + self.eps).sqrt()
        out = (x / rms) * self.weight
        return torch.where(torch.isfinite(out), out, torch.zeros_like(out))


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.dropout = nn.Dropout(EMBD_PDROP)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, mean=0.0, std=INITIALIZER_RANGE)

    def forward(self, x):
        return self.dropout(self.embed(x))


def precompute_freqs_cis(head_dim: int, end: int, theta: float = ROPE_THETA, device=None):
    """RoPE: как в reference attention.py"""
    dim = head_dim
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32)[: (dim // 2)] / dim))
    t = torch.arange(end, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    cos = freqs.cos().unsqueeze(-1)
    sin = freqs.sin().unsqueeze(-1)
    return cos, sin


def apply_rotary_emb(xq, xk, cos, sin):
    """RoPE по парам: (x0,x1) -> (x0*cos - x1*sin, x0*sin + x1*cos). cos, sin: (1,1,L,head_dim//2)."""
    head_dim = xq.size(-1)
    # (B, H, L, head_dim) -> (B, H, L, head_dim//2, 2)
    xq_r = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_r = xk.float().reshape(*xk.shape[:-1], -1, 2)
    xq_0, xq_1 = xq_r[..., 0], xq_r[..., 1]
    xk_0, xk_1 = xk_r[..., 0], xk_r[..., 1]
    xq_rot = torch.stack([xq_0 * cos - xq_1 * sin, xq_0 * sin + xq_1 * cos], dim=-1).flatten(-2)
    xk_rot = torch.stack([xk_0 * cos - xk_1 * sin, xk_0 * sin + xk_1 * cos], dim=-1).flatten(-2)
    return xq_rot.to(xq.dtype), xk_rot.to(xk.dtype)


class MultiHeadSlidingAttention(nn.Module):
    """SWAFull: causal + sliding_window. RoPE и QK-norm опциональны (USE_ROPE, USE_QK_NORM)."""
    def __init__(self, dim, num_heads=4, window_size=16, max_seq_len=512, qk_norm=None):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size
        self.qk_norm = qk_norm if qk_norm is not None else USE_QK_NORM
        self.use_rope = USE_ROPE
        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)
        self.wo = nn.Linear(dim, dim)
        self.q_norm = RMSNorm(self.head_dim) if self.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if self.qk_norm else nn.Identity()
        self.resid_dropout = nn.Dropout(RESID_PDROP)
        self._init_weights()
        if self.use_rope:
            cos, sin = precompute_freqs_cis(self.head_dim, max_seq_len)
            self.register_buffer("cos_cached", cos)
            self.register_buffer("sin_cached", sin)
        else:
            self.register_buffer("cos_cached", None)
            self.register_buffer("sin_cached", None)
        self.register_buffer("mask_cache", None, persistent=False)
        self._mask_cache_len = 0

    def _init_weights(self):
        for m in (self.wq, self.wk, self.wv, self.wo):
            nn.init.normal_(m.weight, mean=0.0, std=INITIALIZER_RANGE)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _get_mask(self, L, device):
        if self.mask_cache is not None and self._mask_cache_len == L and self.mask_cache.device == device:
            return self.mask_cache
        i = torch.arange(L, device=device)
        j = torch.arange(L, device=device)
        mask = (j[None, :] > i[:, None]) | (i[:, None] - j[None, :] >= self.window_size)
        mask = mask.float().masked_fill(mask, float("-inf")).unsqueeze(0).unsqueeze(0)
        self.mask_cache = mask
        self._mask_cache_len = L
        return mask

    def forward(self, x):
        B, L, D = x.shape
        x = x.float()
        q = self.wq(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if self.use_rope and self.cos_cached is not None:
            cos = self.cos_cached[:L].float().unsqueeze(0).unsqueeze(0)
            sin = self.sin_cached[:L].float().unsqueeze(0).unsqueeze(0)
            q, k = apply_rotary_emb(q, k, cos, sin)

        mask = self._get_mask(L, x.device)

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, scale=self.scale)
        attn_out = torch.where(torch.isfinite(attn_out), attn_out, torch.zeros_like(attn_out))
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        return self.resid_dropout(self.wo(attn_out))

    def forward_step(self, x, cache):
        """Одношаговый forward для decode с KV-кэшем. x: (B,1,D). cache: dict{k,v,pos}"""
        B, L, D = x.shape
        assert L == 1
        x = x.float()
        q = self.wq(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if self.use_rope and self.cos_cached is not None:
            pos = cache.get("pos", 0)
            cos = self.cos_cached[pos : pos + 1].float().unsqueeze(0).unsqueeze(0)
            sin = self.sin_cached[pos : pos + 1].float().unsqueeze(0).unsqueeze(0)
            q, k = apply_rotary_emb(q, k, cos, sin)

        if cache.get("k") is not None:
            k_all = torch.cat([cache["k"], k], dim=2)
            v_all = torch.cat([cache["v"], v], dim=2)
        else:
            k_all = k
            v_all = v

        if k_all.size(2) > self.window_size:
            k_all = k_all[:, :, -self.window_size :, :]
            v_all = v_all[:, :, -self.window_size :, :]

        attn_out = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=None, dropout_p=0.0, scale=self.scale)
        attn_out = torch.where(torch.isfinite(attn_out), attn_out, torch.zeros_like(attn_out))
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)

        cache["k"] = k_all.detach()
        cache["v"] = v_all.detach()
        cache["pos"] = cache.get("pos", 0) + 1

        return self.resid_dropout(self.wo(attn_out)), cache

    def build_kv_cache(self, x):
        """KV-кэш для префилла. x: (B,L,D) -> k/v (B,H,Lw,head_dim), pos=L."""
        B, L, D = x.shape
        x = x.float()
        k = self.wk(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        if self.qk_norm:
            k = self.k_norm(k)
        if self.use_rope and self.cos_cached is not None:
            cos = self.cos_cached[:L].float().unsqueeze(0).unsqueeze(0)
            sin = self.sin_cached[:L].float().unsqueeze(0).unsqueeze(0)
            dummy_q = torch.zeros_like(k)
            _, k = apply_rotary_emb(dummy_q, k, cos, sin)
        if L > self.window_size:
            k = k[:, :, -self.window_size :, :]
            v = v[:, :, -self.window_size :, :]
        return {"k": k.detach(), "v": v.detach(), "pos": L}


TTT_LR = 1e-3
# Reduce MLP hidden dim so total params stay comparable after adding a 2nd MLP
# in the last 1/4 blocks. With 1/4 blocks doubled, multiplier ~= 0.8 of baseline 4x.
MLP_INTERMEDIATE_MULT = 3.2


def _round_to_multiple(x, multiple=8):
    return int(multiple * round(x / multiple))


class SwiGLUMLP(nn.Module):
    """SwiGLU как в reference: z1=w1(x), z3=w3(x), out=dropout(w2(silu(z1)*z3)).
    Используется и как base FFN, и как TTT-FFN с fast-весами.
    """
    def __init__(self, dim, intermediate_size=None):
        super().__init__()
        intermediate_size = intermediate_size or (dim * 4)
        self.w1 = nn.Linear(dim, intermediate_size)
        self.w2 = nn.Linear(intermediate_size, dim)
        self.w3 = nn.Linear(dim, intermediate_size)
        self.dropout = nn.Dropout(RESID_PDROP)
        self._init_weights()

    def _init_weights(self):
        for m in (self.w1, self.w2, self.w3):
            nn.init.normal_(m.weight, mean=0.0, std=INITIALIZER_RANGE)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.float()
        z1 = self.w1(x)
        z3 = self.w3(x)
        x2 = F.silu(z1) * z3
        out = self.dropout(self.w2(x2))
        return out

    def fast_params(self):
        """Возвращает tuple базовых параметров, которые будут fast-весами W_0."""
        return (
            self.w1.weight,
            self.w1.bias,
            self.w2.weight,
            self.w2.bias,
            self.w3.weight,
            self.w3.bias,
        )

    @staticmethod
    def fast_forward(x, params):
        """Функциональный FW с fast-весами params."""
        (w1_w, w1_b, w2_w, w2_b, w3_w, w3_b) = params
        x = x.float()
        z1 = F.linear(x, w1_w, w1_b)
        z3 = F.linear(x, w3_w, w3_b)
        x2 = F.silu(z1) * z3
        out = F.linear(x2, w2_w, w2_b)
        return out

    @staticmethod
    def fast_forward_batched(x, params):
        """Функциональный FW с batched fast-весами. x: (B,T,D), params: batched."""
        (w1_w, w1_b, w2_w, w2_b, w3_w, w3_b) = params
        x = x.float()
        # w1_w: (B, I, D), w1_b: (B, I)
        z1 = torch.einsum("btd,bid->bti", x, w1_w) + w1_b[:, None, :]
        z3 = torch.einsum("btd,bid->bti", x, w3_w) + w3_b[:, None, :]
        x2 = F.silu(z1) * z3
        # w2_w: (B, D, I), w2_b: (B, D)
        out = torch.einsum("bti,bdi->btd", x2, w2_w) + w2_b[:, None, :]
        return out


class TransformerBlock(nn.Module):
    """Pre-norm и post-norm как в reference Block: seq_norm->attn->seq_post_norm->residual; ffn_norm->ffn->ffn_post_norm->residual."""
    def __init__(self, dim, window_size, max_seq_len=512, ttt_block=False, num_heads=4):
        super().__init__()
        self.ttt_block = ttt_block
        self.seq_norm = RMSNorm(dim)
        self.seq_post_norm = RMSNorm(dim)
        self.attn = MultiHeadSlidingAttention(
            dim,
            num_heads=num_heads,
            window_size=window_size,
            max_seq_len=max_seq_len,
        )
        if self.ttt_block:
            # Обновляемая MLP (fast-веса) + статическая MLP для "safe storage" как в TTT-E2E.
            self.ffn_ttt_norm = RMSNorm(dim)
            self.ffn_ttt_post_norm = RMSNorm(dim)
            self.ffn_static_norm = RMSNorm(dim)
            self.ffn_static_post_norm = RMSNorm(dim)
            ttt_intermediate = _round_to_multiple(dim * MLP_INTERMEDIATE_MULT)
            self.mlp_ttt = SwiGLUMLP(dim, intermediate_size=ttt_intermediate)
            self.mlp_static = SwiGLUMLP(dim, intermediate_size=ttt_intermediate)
        else:
            self.ffn_norm = RMSNorm(dim)
            self.ffn_post_norm = RMSNorm(dim)
            base_intermediate = _round_to_multiple(dim * MLP_INTERMEDIATE_MULT)
            self.mlp = SwiGLUMLP(dim, intermediate_size=base_intermediate)

    def forward_no_ttt(self, x):
        """
        Вариант блока без FFN/TTT — только attention + norm + residual.
        Используется внутри TTT-прохода.
        """
        x = x.float()
        attn_out = self.attn(self.seq_norm(x))
        attn_out = self.seq_post_norm(attn_out)
        x = x + attn_out
        return x

    def forward(self, x):
        x = x.float()
        attn_out = self.attn(self.seq_norm(x))
        attn_out = self.seq_post_norm(attn_out)
        x = x + attn_out
        if self.ttt_block:
            ffn_out = self.mlp_ttt(self.ffn_ttt_norm(x))
            ffn_out = self.ffn_ttt_post_norm(ffn_out)
            x = x + ffn_out
            static_out = self.mlp_static(self.ffn_static_norm(x))
            static_out = self.ffn_static_post_norm(static_out)
            x = x + static_out
        else:
            ffn_out = self.mlp(self.ffn_norm(x))
            ffn_out = self.ffn_post_norm(ffn_out)
            x = x + ffn_out
        return torch.where(torch.isfinite(x), x, torch.zeros_like(x))

    def forward_step(self, x, cache, fast_params=None):
        """Одношаговый forward для decode с KV-кэшем. x: (B,1,D)."""
        x = x.float()
        attn_in = self.seq_norm(x)
        attn_out, cache = self.attn.forward_step(attn_in, cache)
        attn_out = self.seq_post_norm(attn_out)
        x = x + attn_out
        if self.ttt_block:
            if fast_params is None:
                fast_params = self.mlp_ttt.fast_params()
            ffn_in = self.ffn_ttt_norm(x)
            ffn_out = SwiGLUMLP.fast_forward(ffn_in, fast_params)
            ffn_out = self.ffn_ttt_post_norm(ffn_out)
            x = x + ffn_out
            static_out = self.mlp_static(self.ffn_static_norm(x))
            static_out = self.ffn_static_post_norm(static_out)
            x = x + static_out
        else:
            ffn_out = self.mlp(self.ffn_norm(x))
            ffn_out = self.ffn_post_norm(ffn_out)
            x = x + ffn_out
        return torch.where(torch.isfinite(x), x, torch.zeros_like(x)), cache


class TransformerModel(nn.Module):
    """Как в reference: wte, dropout, blocks, ln_f; lm_head или tie_word_embeddings (CausalLM)."""
    def __init__(self, vocab_size, dim=128, num_blocks=32, num_heads=4, window_size=16, max_seq_len=512, tie_word_embeddings=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.tie_word_embeddings = tie_word_embeddings
        self.num_blocks = num_blocks
        self.ttt_blocks = max(1, num_blocks // 4)
        self.ttt_start = num_blocks - self.ttt_blocks
        self.embedding = TokenEmbedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim,
                window_size,
                max_seq_len=max_seq_len,
                ttt_block=(i >= self.ttt_start),
                num_heads=num_heads,
            ) for i in range(num_blocks)
        ])
        self.ln_f = RMSNorm(dim)
        self.head = None if tie_word_embeddings else nn.Linear(dim, vocab_size)
        if not tie_word_embeddings:
            nn.init.normal_(self.head.weight, mean=0.0, std=INITIALIZER_RANGE)
            if self.head.bias is not None:
                nn.init.zeros_(self.head.bias)

    def forward(self, x):
        x = self.embedding(x).float()
        x = torch.clamp(x, -1e4, 1e4)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        if self.tie_word_embeddings:
            logits = F.linear(x, self.embedding.embed.weight)
        else:
            logits = self.head(x)
        logits = logits.clamp(-LOGIT_CLAMP, LOGIT_CLAMP)
        logits = torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits))
        return logits

    def _init_fast_deltas(self, batch_size):
        """Инициализация per-sample fast-дельт для TTT-блоков."""
        fast_deltas = []
        for block in self.blocks:
            if block.ttt_block:
                base_params = block.mlp_ttt.fast_params()
                deltas = []
                for p in base_params:
                    delta = torch.zeros(
                        (batch_size,) + p.shape,
                        device=p.device,
                        dtype=p.dtype,
                        requires_grad=True,
                    )
                    deltas.append(delta)
                fast_deltas.append(tuple(deltas))
            else:
                fast_deltas.append(None)
        return fast_deltas

    def _forward_blocks_with_fast(self, x, fast_deltas_per_block, stop_grad_below_ttt=False):
        """Проход по блокам с fast-весами в TTT-блоках (batched)."""
        for blk_idx, block in enumerate(self.blocks):
            if stop_grad_below_ttt and blk_idx < self.ttt_start:
                with torch.no_grad():
                    x = block.forward_no_ttt(x)
                    ffn_out = block.mlp(block.ffn_norm(x))
                    ffn_out = block.ffn_post_norm(ffn_out)
                    x = x + ffn_out
                x = x.detach()
                continue

            x = block.forward_no_ttt(x)
            if block.ttt_block:
                ffn_in = block.ffn_ttt_norm(x)
                base_params = block.mlp_ttt.fast_params()
                deltas = fast_deltas_per_block[blk_idx]
                fast_params = tuple(bp.unsqueeze(0) + d for bp, d in zip(base_params, deltas))
                ffn_out = SwiGLUMLP.fast_forward_batched(ffn_in, fast_params)
                ffn_out = block.ffn_ttt_post_norm(ffn_out)
                x = x + ffn_out
                static_out = block.mlp_static(block.ffn_static_norm(x))
                static_out = block.ffn_static_post_norm(static_out)
                x = x + static_out
            else:
                ffn_out = block.mlp(block.ffn_norm(x))
                ffn_out = block.ffn_post_norm(ffn_out)
                x = x + ffn_out
        return x

    def _ttt_update(self, tokens, pos_start, pos_end, window_size, fast_deltas_per_block):
        """Один mini-batch TTT update по позициям [pos_start, pos_end)."""
        assert pos_end > pos_start
        ctx = tokens[:, :pos_end]
        x = self.embedding(ctx).float()
        x = self._forward_blocks_with_fast(x, fast_deltas_per_block, stop_grad_below_ttt=True)
        x = self.ln_f(x)

        h = x[:, pos_start:pos_end, :]  # (B, T, D)
        targets = tokens[:, pos_start + 1 : pos_end + 1]  # (B, T)
        if self.tie_word_embeddings:
            logits = F.linear(h, self.embedding.embed.weight)
        else:
            logits = self.head(h)

        logits = logits.reshape(-1, logits.size(-1))
        targets = targets.reshape(-1)
        batch_loss = F.cross_entropy(logits, targets, reduction="mean")
        if not torch.isfinite(batch_loss):
            return fast_params_per_block, 0.0, 0

        fast_tensors = [p for params in fast_deltas_per_block if params is not None for p in params]
        grads = torch.autograd.grad(
            batch_loss,
            fast_tensors,
            create_graph=True,
            retain_graph=True,
        )

        total_norm = torch.sqrt(sum((g.pow(2).sum() for g in grads)) + 1e-8)
        clip_coef = (1.0 / (total_norm + 1e-6)).clamp(max=1.0)
        grads = [g * clip_coef for g in grads]

        new_fast_deltas = []
        grad_idx = 0
        for params in fast_deltas_per_block:
            if params is None:
                new_fast_deltas.append(None)
                continue
            block_params = []
            for p in params:
                g = grads[grad_idx]
                grad_idx += 1
                block_params.append(p - TTT_LR * g)
            new_fast_deltas.append(tuple(block_params))

        num_steps = pos_end - pos_start
        return new_fast_deltas, batch_loss * num_steps, num_steps

    def forward_ttt(self, tokens, window_size=16, ttt_batch_size=1):
        """
        TTT-проход по ТЗ:
        mini-batch TTT (из статьи):
        for i in range(0, L-1, b):
            для t в i..i+b-1 считаем loss_t с W_{i-1}
            W_i = W_{i-1} - η * (1/b) * sum_t ∇_W loss_t
        """
        tokens = tokens.to(self.embedding.embed.weight.device)
        B, L = tokens.shape
        assert window_size >= ttt_batch_size, "Нужно window_size >= ttt_batch_size (k >= b)."

        # fast-дельты только для MLP в последних 1/4 блоках
        fast_deltas_per_block = self._init_fast_deltas(B)

        total_loss = 0.0
        num_steps = 0

        for i in range(0, L - 1, ttt_batch_size):
            batch_end = min(i + ttt_batch_size, L - 1)
            fast_deltas_per_block, batch_loss_sum, batch_steps = self._ttt_update(
                tokens, i, batch_end, window_size, fast_deltas_per_block
            )
            if batch_steps == 0:
                continue
            total_loss = total_loss + batch_loss_sum
            num_steps += batch_steps

        loss = total_loss / max(num_steps, 1)
        return loss

    def _post_ttt_loss(self, tokens, pos_start, pos_end, fast_deltas_per_block):
        """Loss после TTT-адаптации на позициях [pos_start, pos_end)."""
        assert pos_end > pos_start
        ctx = tokens[:, :pos_end]
        x = self.embedding(ctx).float()
        x = self._forward_blocks_with_fast(x, fast_deltas_per_block, stop_grad_below_ttt=True)
        x = self.ln_f(x)

        h = x[:, pos_start:pos_end, :]
        targets = tokens[:, pos_start + 1 : pos_end + 1]
        if self.tie_word_embeddings:
            logits = F.linear(h, self.embedding.embed.weight)
        else:
            logits = self.head(h)
        logits = logits.reshape(-1, logits.size(-1))
        targets = targets.reshape(-1)
        loss = F.cross_entropy(logits, targets, reduction="mean")
        return loss

    def forward_ttt_meta(self, tokens, window_size=16, ttt_batch_size=1, adapt_ratio=0.5):
        """
        Meta-training: inner-loop TTT на первых токенах,
        outer-loss на оставшихся (post-TTT loss).
        """
        tokens = tokens.to(self.embedding.embed.weight.device)
        B, L = tokens.shape
        assert window_size >= ttt_batch_size, "Нужно window_size >= ttt_batch_size (k >= b)."
        if L < 4:
            return self.forward_ttt(tokens, window_size=window_size, ttt_batch_size=ttt_batch_size)

        adapt_end = max(1, int((L - 1) * adapt_ratio))
        adapt_end = min(adapt_end, L - 2)

        fast_deltas_per_block = self._init_fast_deltas(B)

        for i in range(0, adapt_end, ttt_batch_size):
            batch_end = min(i + ttt_batch_size, adapt_end)
            fast_deltas_per_block, _, _ = self._ttt_update(
                tokens, i, batch_end, window_size, fast_deltas_per_block
            )

        post_loss = self._post_ttt_loss(tokens, adapt_end, L - 1, fast_deltas_per_block)
        return post_loss

    def init_kv_cache(self):
        """Инициализирует пустой KV-кэш для всех блоков."""
        return [{"k": None, "v": None, "pos": 0} for _ in range(self.num_blocks)]

    def forward_step(self, token, cache, fast_params_per_block=None):
        """Один шаг decode с KV-кэшем. token: (B,1) -> logits: (B, vocab)."""
        x = self.embedding(token).float()
        for i, block in enumerate(self.blocks):
            fast = None
            if fast_params_per_block is not None and block.ttt_block:
                fast = fast_params_per_block[i]
            x, cache[i] = block.forward_step(x, cache[i], fast_params=fast)
        x = self.ln_f(x)
        h_t = x[:, -1, :]
        if self.tie_word_embeddings:
            logits = F.linear(h_t, self.embedding.embed.weight)
        else:
            logits = self.head(h_t)
        return logits, cache

    def prefill_kv_cache(self, tokens, fast_params_per_block=None):
        """Префилл KV-кэша за один полный проход. Возвращает (logits_last, cache)."""
        tokens = tokens.to(self.embedding.embed.weight.device)
        x = self.embedding(tokens).float()
        cache = []
        for i, block in enumerate(self.blocks):
            attn_in = block.seq_norm(x)
            cache.append(block.attn.build_kv_cache(attn_in))
            attn_out = block.attn(attn_in)
            attn_out = block.seq_post_norm(attn_out)
            x = x + attn_out
            if block.ttt_block:
                fast = None
                if fast_params_per_block is not None:
                    fast = fast_params_per_block[i]
                if fast is None:
                    fast = block.mlp_ttt.fast_params()
                ffn_in = block.ffn_ttt_norm(x)
                ffn_out = SwiGLUMLP.fast_forward(ffn_in, fast)
                ffn_out = block.ffn_ttt_post_norm(ffn_out)
                x = x + ffn_out
                static_out = block.mlp_static(block.ffn_static_norm(x))
                static_out = block.ffn_static_post_norm(static_out)
                x = x + static_out
            else:
                ffn_out = block.mlp(block.ffn_norm(x))
                ffn_out = block.ffn_post_norm(ffn_out)
                x = x + ffn_out

        x = self.ln_f(x)
        h_t = x[:, -1, :]
        if self.tie_word_embeddings:
            logits = F.linear(h_t, self.embedding.embed.weight)
        else:
            logits = self.head(h_t)
        return logits, cache


if __name__ == "__main__":
    # ===========================
    # 0️⃣ HF Tokenizer
    # ===========================
    # Llama 3 tokenizer (per paper). If unavailable, provide a local path.
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    VOCAB_SIZE = len(tokenizer)
    PAD_ID = tokenizer.pad_token_id

    # ===========================
    # 1️⃣ Dataset: чистый LM (text stream)
    # ===========================
    dataset = load_dataset("awsaf49/persona-chat")

    def _clean_persona_chat(text):
        # Убираем persona-строки и разбиваем диалог на реплики.
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        dialog_lines = []
        for ln in lines:
            # Формат может быть: "6 partner's persona: ..." или "your persona: ..."
            parts = ln.split(" ", 1)
            if len(parts) == 2 and parts[0].isdigit():
                content = parts[1].strip()
            else:
                content = ln
            if "persona:" in content:
                continue
            dialog_lines.append(content)
        # Последние строки диалога содержат варианты, разделенные "|"
        dialog_text = " ".join(dialog_lines)
        utterances = [u.strip() for u in dialog_text.split("|") if u.strip()]
        return "\n".join(utterances) if utterances else dialog_text

    def tokenize_fn(batch):
        texts = batch.get("text", [])
        cleaned = [_clean_persona_chat(t) for t in texts]
        tokens = tokenizer(
            cleaned,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
        )
        return {"input_ids": tokens["input_ids"]}

    dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    # Use 5% of each split for faster iteration
    def take_percent(ds, pct=1, seed=42):
        n = len(ds)
        k = max(1, int(n * pct))
        return ds.shuffle(seed=seed).select(range(k))

    dataset["train"] = take_percent(dataset["train"])
    if "validation" in dataset:
        dataset["validation"] = take_percent(dataset["validation"])
    if "test" in dataset:
        dataset["test"] = take_percent(dataset["test"])

    # ===========================
    # 2️⃣ DataLoader (LM: только tokens)
    # ===========================
    def create_dataloader(split, batch_size=128, num_workers=8, prefetch_factor=4):
        inputs = torch.tensor(dataset[split]["input_ids"], dtype=torch.long)
        pin = torch.cuda.is_available()
        return DataLoader(
            TensorDataset(inputs),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin,
            persistent_workers=(num_workers > 0),
            prefetch_factor=prefetch_factor,
        )

    train_loader = create_dataloader("train")
    valid_loader = create_dataloader("validation")
    test_loader = create_dataloader("test")
    print("✅ DataLoaders ready")

    # ===========================
    # 6️⃣ Training next-token prediction
    # ===========================
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Требуется GPU для обучения.")
    device = torch.device("cuda")
    if MODEL_SIZE not in MODEL_CONFIGS:
        raise ValueError(f"Unknown MODEL_SIZE: {MODEL_SIZE}. Options: {list(MODEL_CONFIGS.keys())}")
    num_blocks, dim, num_heads = MODEL_CONFIGS[MODEL_SIZE]
    EPOCHS = 1
    LR = 5e-4
    WEIGHT_DECAY = 0.1
    CLIP_GRAD = 1.0
    # Paper defaults (main results): k=8192, b=1024 for T=128K.
    TARGET_WINDOW = 128
    TARGET_TTT_BATCH = 128
    WINDOW = min(TARGET_WINDOW, MAX_LEN)
    TTT_BATCH = min(TARGET_TTT_BATCH, WINDOW);
    TRAIN_BATCH = 16
    MICRO_BATCH = 1
    ACCUM_STEPS = 8
    META_ADAPT_RATIO = 0.2
    if WINDOW < TTT_BATCH:
        raise ValueError("WINDOW must be >= TTT_BATCH (k >= b).")
    if TRAIN_BATCH < 1:
        raise ValueError("TRAIN_BATCH must be >= 1.")
    if MICRO_BATCH < 1 or MICRO_BATCH > TRAIN_BATCH:
        raise ValueError("MICRO_BATCH must be in [1, TRAIN_BATCH].")
    if ACCUM_STEPS < 1:
        raise ValueError("ACCUM_STEPS must be >= 1.")

    model = TransformerModel(
        VOCAB_SIZE,
        dim=dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        window_size=WINDOW,
        max_seq_len=MAX_LEN,
        tie_word_embeddings=False,
    ).to(device)
    USE_COMPILE = False
    if USE_COMPILE:
        try:
            model = torch.compile(model, mode="max-autotune")
        except Exception:
            pass
    print(f"✅ Device: {device} | model params device: {next(model.parameters()).device}")

    train_loader = create_dataloader("train", batch_size=TRAIN_BATCH)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, fused=True)
    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        num_batches = 0
        optimizer.zero_grad(set_to_none=True)
        accum = 0
        for (batch_inputs,) in tqdm(train_loader):
            # используем только LM-токены (next-token), без отдельного target_text
            tokens = batch_inputs.to(device, non_blocking=True)
            if not tokens.is_cuda:
                raise RuntimeError("Tokens are not on CUDA. Проверь device и DataLoader.")

            losses = []
            for start in range(0, tokens.size(0), MICRO_BATCH):
                mb = tokens[start : start + MICRO_BATCH]
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss_mb = model.forward_ttt_meta(
                        mb,
                        window_size=WINDOW,
                        ttt_batch_size=TTT_BATCH,
                        adapt_ratio=META_ADAPT_RATIO,
                    )
                if torch.isfinite(loss_mb):
                    losses.append(loss_mb)
            if not losses:
                continue
            loss = torch.stack(losses).mean() / ACCUM_STEPS

            if not torch.isfinite(loss):
                continue

            scaler.scale(loss).backward()
            accum += 1
            if accum % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches else float("nan")
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "transformer_ttt_personachat_hf.pth")
    tokenizer.save_pretrained("./tokenizer_hf")
    print("✅ Model and tokenizer saved")
