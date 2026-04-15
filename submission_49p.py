"""
AdderBoard submission: 49-parameter trained transformer for 10-digit addition.

Architecture: 1L Qwen3-style decoder
  d=3, 1h/1kv, hd=4, ff_dim=2, SwiGLU, RMSNorm, RoPE theta=3
  Circular arc embedding (3 params), tied K=V, tied O=Q^T,
  tied QK norms, tied down=gate^T, shared ln1=ln2 (49p)

Training: 3-step pipeline
  Phase 1: Grokfast-EMA (alpha=0.98, lambda=3.0), AdamW, curriculum 3->6->10,
           200K steps -> 97.39% verify accuracy
  Phase 2: Carry-chain-weighted loss fine-tune (carry_beta=2.0), 300K steps
           -> 99.90% verify accuracy
  Phase 3: CMA-ES post-training refinement (sigma restart 0.0005->0.001357)
           -> 100% verify accuracy (10010/10010)

Accuracy: 100% on verify set (10K random + 10 edge, seed=2025)
Parameters: 49
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta=3.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        T = x.shape[-2]
        t = torch.arange(T, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        cos_f, sin_f = freqs.cos(), freqs.sin()
        x1, x2 = x[..., ::2], x[..., 1::2]
        half = x1.shape[-1]
        cos_f, sin_f = cos_f[..., :half], sin_f[..., :half]
        rotated = torch.stack([x1 * cos_f - x2 * sin_f, x1 * sin_f + x2 * cos_f], dim=-1).flatten(-2)
        return rotated


class CircularArcEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.arc_A = nn.Parameter(torch.tensor(2.5))
        self.arc_start = nn.Parameter(torch.tensor(-1.2))
        self.arc_stride = nn.Parameter(torch.tensor(0.29))

    def table(self):
        d = torch.arange(10, device=self.arc_A.device, dtype=self.arc_A.dtype)
        angles = self.arc_start + d * self.arc_stride
        return torch.stack([self.arc_A * torch.cos(angles), self.arc_A * torch.sin(angles)], dim=1)

    def forward(self, tokens):
        return self.table()[tokens]


class AdderModel(nn.Module):
    def __init__(self):
        super().__init__()
        d, hd, ff = 3, 4, 2
        self.hd = hd
        self.scale = hd ** -0.5

        self.tok_embed = CircularArcEmbedding()
        pe = torch.zeros(64, 1)
        pos = torch.arange(64).float().unsqueeze(1)
        pe[:, 0] = torch.sin(pos[:, 0] * math.exp(-math.log(10000.0)))
        self.register_buffer("pos_embed", pe)

        self.q_proj = nn.Linear(d, hd, bias=False)
        self.k_proj = nn.Linear(d, hd, bias=False)
        self.q_norm = RMSNorm(hd)  # shared with k_norm (tie_qk_norm)
        self.rope = RotaryEmbedding(hd, theta=3.0)

        self.gate_proj = nn.Linear(d, ff, bias=False)
        self.up_proj = nn.Linear(d, ff, bias=False)
        # down_proj tied to gate_proj^T (no extra params)

        # Shared norms: ln1 = ln2 (saves 3 params vs separate norms)
        self.ln1 = RMSNorm(d)  # shared for both pre-attention and pre-MLP
        self.ln_f = RMSNorm(d)

    def forward(self, idx):
        B, T = idx.shape
        tok = self.tok_embed(idx)
        pos = self.pos_embed[:T].unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([tok, pos], dim=-1)

        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        mask = torch.where(mask, torch.tensor(float('-inf'), device=x.device), torch.tensor(0.0, device=x.device))
        mask = mask.unsqueeze(0).unsqueeze(0)

        # Pre-attention norm (shared)
        h = self.ln1(x)
        q = self.q_proj(h).view(B, T, 1, self.hd).transpose(1, 2)
        k = self.k_proj(h).view(B, T, 1, self.hd).transpose(1, 2)
        v = k.clone()  # tied K=V
        q = self.q_norm(q)
        k = self.q_norm(k)  # tied QK norm
        q = self.rope(q)
        k = self.rope(k)
        attn = F.softmax((q @ k.transpose(-2, -1)) * self.scale + mask, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, -1)
        x = x + F.linear(out, self.q_proj.weight.T)  # tied O=Q^T

        # Pre-MLP norm (SHARED with pre-attention norm)
        h = self.ln1(x)
        gate = F.silu(self.gate_proj(h))
        up = self.up_proj(h)
        x = x + F.linear(gate * up, self.gate_proj.weight.T)  # tied down=gate^T

        x = self.ln_f(x)
        table = self.tok_embed.table()
        return x[..., :2] @ table.T

    @torch.no_grad()
    def generate(self, prompt):
        self.eval()
        seq = prompt.clone()
        for _ in range(11):
            logits = self.forward(seq)
            seq = torch.cat([seq, logits[:, -1, :10].argmax(-1, keepdim=True)], dim=1)
        return seq[:, prompt.shape[1]:prompt.shape[1] + 11]


_W = {
    "tok_embed.arc_A": torch.tensor(53.39081954956055),
    "tok_embed.arc_start": torch.tensor(1.2057271003723145),
    "tok_embed.arc_stride": torch.tensor(0.1254480928182602),
    "q_proj.weight": torch.tensor([[10.064225196838379, 2.774019479751587, 1.4579730033874512], [42.290523529052734, 11.11209487915039, -4.705548286437988], [-3.8117716312408447, -0.8938799500465393, -1.1917589902877808], [9.216836929321289, 2.2616913318634033, -0.2047961801290512]]),
    "k_proj.weight": torch.tensor([[8.945900917053223, -9.204228401184082, -5.801871299743652], [-9.97950267791748, -0.808920681476593, 0.5593137145042419], [2.180220365524292, 0.048893626779317856, -12.723864555358887], [-14.28452205657959, 11.919428825378418, -0.26181530952453613]]),
    "q_norm.weight": torch.tensor([4.189301013946533, -0.5842321515083313, 0.807510256767273, 5.245909214019775]),
    "gate_proj.weight": torch.tensor([[-2.3539397716522217, -3.3044955730438232, -0.48936736583709717], [1.0696345567703247, -1.1602256298065186, 1.5745854377746582]]),
    "up_proj.weight": torch.tensor([[4.96342658996582, -1.1174434423446655, -0.8100679516792297], [-8.958495140075684, -0.7657700777053833, -0.6104097962379456]]),
    "ln1.weight": torch.tensor([1.3762867450714111, 9.654746055603027, -4.45611572265625]),
    "ln_f.weight": torch.tensor([-10.625299453735352, 27.65289306640625, 0.33968111872673035]),
}


def _encode(a, b):
    ad = [(a // 10**i) % 10 for i in range(10)]
    bd = [(b // 10**i) % 10 for i in range(10)]
    return [0] + ad + [0, 0] + bd + [0]


def build_model():
    model = AdderModel()
    model.load_state_dict(_W, strict=False)
    model.eval()
    metadata = {
        "name": "CircularArc-Qwen3-49p",
        "author": "warlockee",
        "params": 49,
        "architecture": "1L Qwen3, d=3, 1h/1kv, hd=4, ff=2, RoPE theta=3, SwiGLU, RMSNorm, shared ln1=ln2",
        "tricks": [
            "Circular arc embedding (3 params)",
            "Tied K=V",
            "Tied O=Q^T",
            "Tied QK norms",
            "Tied down=gate^T",
            "Shared ln1=ln2 norms (49p vs 52p)",
            "Tied lm_head to embedding table",
            "3-step: Grokfast-EMA gradient -> carry-chain loss -> CMA-ES refinement",
        ],
    }
    return model, metadata


def add(model, a: int, b: int) -> int:
    device = next(model.parameters()).device
    prompt = torch.tensor([_encode(a, b)], dtype=torch.long, device=device)
    output = model.generate(prompt)
    return sum(d * 10**i for i, d in enumerate(output[0].tolist()[:11]))
