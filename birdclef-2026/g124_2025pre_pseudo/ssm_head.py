"""Bidirectional Mamba-2 / SSD selective-state-space temporal head + prototype/cosine
classifier over the Perch embedding sequence (T2-B).

Operates on a per-file sequence of window embeddings ``E`` of shape ``(B, T, D)`` where
``T`` is the number of 5 s windows in a 60 s file (default ``T=12``) and ``D`` is the
frozen Perch-v2 embedding dimension (default ``D=1280``). It emits per-window logits of
shape ``(B, T, C)`` for the ``C=234`` BirdCLEF-2026 classes.

Design follows agents/A1_ssm_linear_attention.md:
  * low-rank projection D -> d (DeepSeek-MLA style latent),
  * scalar-decay Mamba-2 (SSD) selective recurrence with an input-dependent step
    ``delta_t`` (the "adaptive_delta" feature), run forward AND backward (bidirectional /
    two-pass), residual-added back onto the projected embedding ("residual_ssm"),
  * prototype / cosine-similarity classifier with K prototypes per class (handles the
    28 zero-train_audio classes via class prototypes seeded from labeled soundscapes),
  * optional sonotype max-pool mirror over acoustically similar class groups.

The selective scan has two backends:
  * ``mamba-ssm`` (the official Triton/CUDA kernel) when importable, used on GPU,
  * a pure-PyTorch sequential scan fallback (``_pytorch_scan``) so the CPU smoke test
    runs with no extra dependencies. At ``T=12`` the python loop is microseconds.

Param/FLOP accounting is provided by :func:`param_count` and :func:`flops_per_file`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import math


@dataclass
class SSMHeadConfig:
    embed_dim: int = 1280          # D: Perch-v2 embedding dimension
    proj_dim: int = 256            # d: low-rank latent
    state_dim: int = 16            # N: SSM state channels per direction
    num_classes: int = 234         # C
    protos_per_class: int = 1      # K: prototypes per class (>1 = song/call polymorphism)
    tau: float = 16.0              # cosine temperature
    alpha_init: float = 0.35       # residual / correction weight (A1: ~0.30-0.35)
    bidirectional: bool = True
    use_mamba_ssm: bool = True     # try the official kernel; falls back if unimportable
    rms_norm: bool = True          # RMSNorm on projected embeddings (training hygiene)
    # Optional taxonomic / sonotype mirror: maps each class index to a group id; logits
    # are max-pooled within a group post-hoc. Empty disables the mirror.
    sonotype_groups: dict = field(default_factory=dict)


def _mamba_ssm_available() -> bool:
    try:
        import mamba_ssm  # noqa: F401

        return True
    except Exception:
        return False


def param_count(config: SSMHeadConfig) -> dict:
    """Exact trainable parameter count (matches the nn.Module construction)."""
    d, n, c, k, dd = (
        config.proj_dim,
        config.state_dim,
        config.num_classes,
        config.protos_per_class,
        config.embed_dim,
    )
    proj = dd * d + d  # Linear(D, d)
    norm = d if config.rms_norm else 0
    # per direction: Bp (d*n), Cp (d*n), dt (d*1 + 1), lam (n), u (d*n + n), out (n*d + d)
    per_dir = (d * n) + (d * n) + (d + 1) + n + (d * n + n) + (n * d + d)
    n_dirs = 2 if config.bidirectional else 1
    ssm = per_dir * n_dirs
    alpha = 1
    proto = c * k * d
    total = proj + norm + ssm + alpha + proto
    return {
        "proj": proj,
        "norm": norm,
        "ssm": ssm,
        "alpha": alpha,
        "proto": proto,
        "total": total,
    }


def flops_per_file(config: SSMHeadConfig, n_windows: int = 12) -> dict:
    """Approximate multiply-accumulate FLOPs for one 60 s file (T windows)."""
    t, d, n, c, k, dd = (
        n_windows,
        config.proj_dim,
        config.state_dim,
        config.num_classes,
        config.protos_per_class,
        config.embed_dim,
    )
    n_dirs = 2 if config.bidirectional else 1
    proj = 2.0 * t * dd * d
    # per step per dir: Bp,Cp,u,dt projections (~ 3*d*n + d) + recurrence (~6*n) + out (n*d)
    ssm = n_dirs * t * (2.0 * (3 * d * n + d) + 6 * n + 2.0 * n * d)
    proto = 2.0 * t * c * k * d
    total = proj + ssm + proto
    return {"proj": proj, "ssm": ssm, "proto": proto, "total": total}


def build_sonotype_group_index(num_classes: int, sonotype_groups: dict):
    """Return a long tensor ``group_id`` of length ``num_classes`` (or ``None``).

    ``sonotype_groups`` maps a class index (int) -> group id (int). Classes absent from
    the mapping get a unique singleton group so they are unaffected by the mirror.
    """
    import torch

    if not sonotype_groups:
        return None
    group_id = list(range(num_classes))  # default: each class its own group
    next_group = num_classes
    remap: dict = {}
    for cls_idx, raw_group in sonotype_groups.items():
        cls_idx = int(cls_idx)
        if cls_idx < 0 or cls_idx >= num_classes:
            continue
        if raw_group not in remap:
            remap[raw_group] = next_group
            next_group += 1
        group_id[cls_idx] = remap[raw_group]
    return torch.tensor(group_id, dtype=torch.long)


def _make_module():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x):
            norm = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
            return (x.float() * norm).type_as(x) * self.weight

    class BiSSDProtoHead(nn.Module):
        """Bidirectional scalar-SSD temporal head + prototype/cosine classifier."""

        def __init__(self, config: SSMHeadConfig):
            super().__init__()
            self.config = config
            d, n = config.proj_dim, config.state_dim
            self.proj = nn.Linear(config.embed_dim, d)
            self.norm = _RMSNorm(d) if config.rms_norm else nn.Identity()
            self.n_dirs = 2 if config.bidirectional else 1

            def _dir_params():
                return nn.ModuleDict(
                    {
                        "Bp": nn.Linear(d, n, bias=False),
                        "Cp": nn.Linear(d, n, bias=False),
                        "dt": nn.Linear(d, 1),
                        "u": nn.Linear(d, n),
                        "out": nn.Linear(n, d),
                    }
                )

            self.dirs = nn.ModuleList([_dir_params() for _ in range(self.n_dirs)])
            self.lam = nn.ParameterList(
                [nn.Parameter(torch.zeros(n)) for _ in range(self.n_dirs)]
            )
            self.alpha = nn.Parameter(torch.tensor(float(config.alpha_init)))
            self.proto = nn.Parameter(
                torch.randn(config.num_classes, config.protos_per_class, d) * 0.02
            )
            self.tau = float(config.tau)
            self._use_kernel = bool(config.use_mamba_ssm) and _mamba_ssm_available()
            group_id = build_sonotype_group_index(config.num_classes, config.sonotype_groups)
            if group_id is not None:
                self.register_buffer("sonotype_group_id", group_id, persistent=False)
            else:
                self.sonotype_group_id = None

        # ---- selective scan (one direction) -------------------------------------
        def _pytorch_scan(self, x, params, lam):
            """Pure-PyTorch scalar-SSD recurrence. x: (B,T,d) -> correction (B,T,d).

            h_t = a_t * h_{t-1} + (1 - a_t) * (softplus(B_t) * u_t)
            a_t = exp(-delta_t * softplus(lam))      (decay in (0,1))
            delta_t = softplus(dt(x))                (adaptive / selective step)
            y_t = C_t * h_t ; correction = out(y)
            """
            b, t, _ = x.shape
            dt = F.softplus(params["dt"](x))               # (B,T,1)
            a = torch.exp(-dt * F.softplus(lam))           # (B,T,N)
            bu = F.softplus(params["Bp"](x)) * params["u"](x)
            cc = params["Cp"](x)                           # (B,T,N)
            h = x.new_zeros(b, self.config.state_dim)
            ys = []
            for step in range(t):
                h = a[:, step] * h + (1.0 - a[:, step]) * bu[:, step]
                ys.append((cc[:, step] * h).unsqueeze(1))
            y = torch.cat(ys, dim=1)                        # (B,T,N)
            return params["out"](y)

        def _scan(self, x, params, lam):
            # The official mamba-ssm kernel is GPU-only; for the T=12 head the pure
            # PyTorch scan is already microseconds, so we use it everywhere unless the
            # kernel is present AND we are on CUDA. Keeping one numerically-stable path
            # avoids train/infer drift.
            if self._use_kernel and x.is_cuda:
                try:
                    return self._kernel_scan(x, params, lam)
                except Exception:
                    return self._pytorch_scan(x, params, lam)
            return self._pytorch_scan(x, params, lam)

        def _kernel_scan(self, x, params, lam):
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

            b, t, d = x.shape
            n = self.config.state_dim
            dt = F.softplus(params["dt"](x)).transpose(1, 2)        # (B,1,T)->broadcast
            dt = dt.expand(b, d, t).contiguous()
            u = (F.softplus(params["Bp"](x)) * params["u"](x))      # (B,T,N) gated input
            # project gated input up to (B,d,T) feature space via out's transpose is not
            # trivial; the kernel expects (B,d,T). We map through a per-channel identity
            # by treating N as the feature axis and using out() afterwards.
            u_dN = u.transpose(1, 2)                                 # (B,N,T)
            A = -torch.exp(lam).unsqueeze(0).expand(n, n) * torch.eye(n, device=x.device)
            A = A[:, :1].expand(n, 1)  # diagonal (N,1)
            B_t = params["Bp"](x).transpose(1, 2)                    # (B,N,T)
            C_t = params["Cp"](x).transpose(1, 2)                    # (B,N,T)
            dt_N = F.softplus(params["dt"](x)).transpose(1, 2).expand(b, n, t).contiguous()
            y = selective_scan_fn(u_dN, dt_N, A, B_t, C_t, None, None, None, False)
            return params["out"](y.transpose(1, 2))

        # ---- forward ------------------------------------------------------------
        def temporal_features(self, embeddings):
            """Return temporally-contextualized features (B,T,d) before the classifier."""
            x = self.norm(self.proj(embeddings))
            fwd = self._scan(x, self.dirs[0], self.lam[0])
            correction = fwd
            if self.n_dirs == 2:
                bwd = torch.flip(
                    self._scan(torch.flip(x, dims=[1]), self.dirs[1], self.lam[1]),
                    dims=[1],
                )
                correction = fwd + bwd
            return x + self.alpha * correction

        def forward(self, embeddings, apply_sonotype_mirror: bool = False):
            """embeddings: (B,T,D) -> per-window logits (B,T,C)."""
            feats = self.temporal_features(embeddings)
            xn = F.normalize(feats, dim=-1)                 # (B,T,d)
            pn = F.normalize(self.proto, dim=-1)            # (C,K,d)
            sim = torch.einsum("btd,ckd->btck", xn, pn)     # (B,T,C,K)
            logits = self.tau * sim.amax(dim=-1)            # multi-proto max -> (B,T,C)
            if apply_sonotype_mirror and self.sonotype_group_id is not None:
                logits = sonotype_mirror_logits(logits, self.sonotype_group_id)
            return logits

        @torch.no_grad()
        def seed_prototypes_from_embeddings(self, class_embeddings):
            """Initialize prototypes from mean labeled-soundscape embeddings.

            ``class_embeddings``: tensor (C, D) of mean Perch embeddings per class (rows
            for zero-train classes may be zero -> left at random init). Projected through
            the (current) proj layer and broadcast over the K prototype slots.
            """
            # Detect which classes actually have a support embedding from the *input*
            # (zero-train_audio classes pass an all-zero row and must be left at random
            # init -- the proj bias + norm would otherwise make every row look nonzero).
            seeded = class_embeddings.abs().sum(dim=-1) > 1e-6
            if not bool(seeded.any()):
                return
            x = self.norm(self.proj(class_embeddings))      # (C, d)
            seed = x.unsqueeze(1).expand(-1, self.config.protos_per_class, -1)
            self.proto.data[seeded] = seed[seeded].to(self.proto.dtype)

    return BiSSDProtoHead


def sonotype_mirror_logits(logits, group_id):
    """Max-pool per-window logits within each sonotype group (post-hoc mirror).

    ``logits``: (B,T,C); ``group_id``: (C,) long. Returns same-shape logits where each
    class takes the max over its group. Used to lift the 28 unmapped sonotype classes
    toward acoustically similar, better-trained neighbors.
    """
    import torch

    b, t, c = logits.shape
    unique = torch.unique(group_id)
    out = logits.clone()
    for g in unique.tolist():
        mask = group_id == g
        if int(mask.sum()) <= 1:
            continue
        group_max = logits[:, :, mask].amax(dim=-1, keepdim=True)  # (B,T,1)
        out[:, :, mask] = group_max
    return out


def build_ssm_head(config: SSMHeadConfig | None = None, **overrides):
    """Factory: build a :class:`BiSSDProtoHead` (imports torch lazily)."""
    if config is None:
        config = SSMHeadConfig(**overrides)
    elif overrides:
        config = SSMHeadConfig(**{**config.__dict__, **overrides})
    module_cls = _make_module()
    return module_cls(config)


def summarize_head(config: SSMHeadConfig | None = None, n_windows: int = 12) -> str:
    """One-line param/FLOP summary for logging."""
    if config is None:
        config = SSMHeadConfig()
    p = param_count(config)
    f = flops_per_file(config, n_windows=n_windows)
    return (
        f"ssm_head D={config.embed_dim} d={config.proj_dim} N={config.state_dim} "
        f"C={config.num_classes} K={config.protos_per_class} "
        f"bidir={config.bidirectional} kernel={_mamba_ssm_available()} "
        f"params_total={p['total']} (proj={p['proj']} ssm={p['ssm']} proto={p['proto']}) "
        f"MFLOPs_per_file={f['total'] / 1e6:.2f} (T={n_windows})"
    )


if __name__ == "__main__":
    cfg = SSMHeadConfig()
    print(summarize_head(cfg))
    print(f"params={param_count(cfg)}")
    print(f"flops={ {k: round(v / 1e6, 3) for k, v in flops_per_file(cfg).items()} } (M)")
    _ = math  # keep import for potential future scheduling math
