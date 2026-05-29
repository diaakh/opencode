"""CPU smoke test for the SSM head (T2-B) + noisy-student loop (T2-A).

Runs the full pipeline end-to-end on tiny synthetic data, no GPU / no Perch / no audio
files required, in well under 2 minutes:

  1. Synthesize a handful of "labeled" Perch embedding sequences (B,T,D) with planted
     class structure, plus a pool of "unlabeled" soundscape embedding sequences and a
     tiny Yao-probe set with targets.
  2. Build the bidirectional SSD + prototype head (pure-PyTorch scan, no mamba kernel).
  3. Seed prototypes from per-class mean embeddings (zero-train classes left at init).
  4. Run the 3-round noisy-student loop: round 0 supervised, rounds 1-2 self-distill on
     soft + power-transformed teacher labels with capped pseudo-ratio and ramping noise;
     final round uses the stricter TSS threshold. Best round chosen by the repo's
     Yao-probe selection score.
  5. Print param/FLOP numbers and the per-round selection scores.

Usage::

    python run_ssm_smoke.py            # default 3 rounds, ~seconds on CPU
    python run_ssm_smoke.py --rounds 4

This is the ``--smoke`` CPU path; the production launch uses train_g124.py with
``--ssm-head --noisy-student-rounds 4 --perch-embeddings ... --pseudo-parquet ...``.
"""

from __future__ import annotations

import argparse
import time


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CPU smoke test: SSM head + noisy-student loop")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=24)
    parser.add_argument("--embed-dim", type=int, default=1280)
    parser.add_argument("--n-windows", type=int, default=12)
    parser.add_argument("--n-labeled", type=int, default=32)
    parser.add_argument("--n-unlabeled", type=int, default=24)
    parser.add_argument("--n-probe", type=int, default=16)
    parser.add_argument("--seed", type=int, default=124)
    return parser


def run_smoke(args: argparse.Namespace) -> dict:
    import numpy as np
    import torch

    from ssm_head import SSMHeadConfig, build_ssm_head, param_count, flops_per_file, summarize_head
    from noisy_student import (
        NoisyStudentConfig,
        apply_embedding_noise,
        ensemble_teacher_probs,
        make_soft_targets,
        mixup_embeddings,
        run_noisy_student,
        subsample_pseudo_frame,
        cap_pseudo_ratio,
    )
    from train_g124 import compute_yao_selection_score

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    t0 = time.time()
    C, D, T = args.num_classes, args.embed_dim, args.n_windows

    # Planted structure: each class has a random direction in embedding space; a labeled
    # sequence for class c is that direction + noise across all windows.
    directions = torch.randn(C, D)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    def make_labeled(n):
        cls = torch.randint(0, C, (n,))
        E = directions[cls].unsqueeze(1).expand(n, T, D) + 0.5 * torch.randn(n, T, D)
        y = torch.zeros(n, T, C)
        y[torch.arange(n), :, cls] = 1.0
        return E, y, cls

    labeled_E, labeled_y, _ = make_labeled(args.n_labeled)
    unlabeled_E = torch.randn(args.n_unlabeled, T, D) + 0.3 * directions[
        torch.randint(0, C, (args.n_unlabeled,))
    ].unsqueeze(1)
    probe_E, _probe_y, probe_cls = make_labeled(args.n_probe)
    probe_target = torch.zeros(args.n_probe, C)
    probe_target[torch.arange(args.n_probe), probe_cls] = 1.0

    # per-class mean embedding for prototype seeding; leave the last 4 classes "zero-train"
    class_mean = torch.zeros(C, D)
    for c in range(C - 4):
        class_mean[c] = directions[c]

    hcfg = SSMHeadConfig(
        embed_dim=D,
        num_classes=C,
        proj_dim=64,
        state_dim=16,
        protos_per_class=1,
        use_mamba_ssm=False,  # CPU pure-PyTorch scan
    )
    ns_cfg = NoisyStudentConfig(n_rounds=args.rounds, final_tss=True, pseudo_ratio_cap=0.4)

    print(summarize_head(hcfg, n_windows=T))
    p = param_count(hcfg)
    f = flops_per_file(hcfg, n_windows=T)
    print(f"smoke param_total={p['total']} MFLOPs_per_file={f['total'] / 1e6:.3f}")
    # report the *production* head sizing too (D=1280, C=234, d=256)
    prod = SSMHeadConfig()
    pp, pf = param_count(prod), flops_per_file(prod, n_windows=12)
    print(
        f"production_head params_total={pp['total']} MFLOPs_per_file={pf['total'] / 1e6:.3f} "
        f"(D={prod.embed_dim} d={prod.proj_dim} N={prod.state_dim} C={prod.num_classes})"
    )

    # capped pseudo count (informational)
    keep = cap_pseudo_ratio(args.n_labeled, args.n_unlabeled, ns_cfg.pseudo_ratio_cap)
    print(
        f"pseudo_ratio_cap={ns_cfg.pseudo_ratio_cap} labeled={args.n_labeled} "
        f"unlabeled={args.n_unlabeled} kept_pseudo={keep} "
        f"ratio={keep / max(keep + args.n_labeled, 1):.3f}"
    )

    def init_student(round_idx):
        torch.manual_seed(args.seed + 100 + round_idx)
        head = build_ssm_head(hcfg)
        head.seed_prototypes_from_embeddings(class_mean)
        return head

    def train_one_round(student, teacher, noise, round_idx, use_tss):
        opt = torch.optim.AdamW(student.parameters(), lr=1e-2, weight_decay=1e-4)
        student.train()
        if teacher is not None:
            teacher.eval()
        for _step in range(12):
            opt.zero_grad(set_to_none=True)
            xb = apply_embedding_noise(labeled_E, noise)
            xb, yb = mixup_embeddings(xb, labeled_y, noise)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(student(xb), yb)
            if teacher is not None:
                with torch.no_grad():
                    teacher_probs = ensemble_teacher_probs([teacher(unlabeled_E)])
                    soft = make_soft_targets(
                        teacher_probs, torch.zeros_like(teacher_probs), ns_cfg, use_tss=use_tss
                    )
                xu = apply_embedding_noise(unlabeled_E, noise)
                loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(
                    student(xu), soft
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
        return student

    def evaluate(student):
        student.eval()
        with torch.no_grad():
            pred = student(probe_E).sigmoid().mean(dim=1).numpy()  # pool windows -> per-file
        target = probe_target.numpy()
        flat_p, flat_t = pred.reshape(-1), target.reshape(-1)
        corr = float(np.corrcoef(flat_p, flat_t)[0, 1]) if flat_p.std() > 0 else 0.0
        # rank correlation + topk overlap proxies for the repo selection score
        import pandas as pd

        pred_rank = pd.DataFrame(pred).rank(pct=True).to_numpy().reshape(-1)
        targ_rank = pd.DataFrame(target).rank(pct=True).to_numpy().reshape(-1)
        rank_corr = float(np.corrcoef(pred_rank, targ_rank)[0, 1]) if pred_rank.std() > 0 else 0.0
        topk = 3
        pred_top = np.argsort(-pred, axis=1)[:, :topk]
        targ_top = np.argsort(-target, axis=1)[:, :topk]
        overlap = float(
            np.mean([len(set(a) & set(b)) / topk for a, b in zip(pred_top, targ_top)])
        )
        return {
            "yao_corr": corr,
            "yao_rank_corr": rank_corr,
            "topk_overlap": overlap,
            "yao_rank_mae": 0.3,
            "active_gt05": 96.0,
        }

    best = run_noisy_student(
        config=ns_cfg,
        init_student=init_student,
        train_one_round=train_one_round,
        evaluate=evaluate,
        selection_score_fn=lambda m: compute_yao_selection_score(m, val_loss=0.1),
        teacher_init=None,
    )
    elapsed = time.time() - t0
    print(
        f"SMOKE OK best_round={best.round_idx + 1}/{args.rounds} "
        f"selection_score={best.selection_score:.5f} is_tss={best.is_tss} "
        f"elapsed={elapsed:.1f}s"
    )
    return {
        "best_round": best.round_idx,
        "selection_score": best.selection_score,
        "elapsed": elapsed,
        "params": p["total"],
        "mflops": f["total"] / 1e6,
    }


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    run_smoke(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
