"""Multi-iteration noisy-student / self-distillation loop (T2-A).

Implements the verified prior-winner protocol from agents/A6_prior_winners.md:

  * SOFT pseudo-labels (never argmax) from a teacher ensemble.
  * PowerTransform sharpening:  ``p = p * (p > th) + p ** power``  then clamp [0, 1].
  * label mixing:  ``target = alpha * pseudo + (1 - alpha) * hard``  (alpha = 0.7).
  * default ``th = 0.3``, ``power = 2``.
  * pseudo-ratio cap <= 0.4 (40% pseudo / 60% labeled per the 2nd-place caution).
  * 3-4 rounds (configurable), each a fresh student that becomes the next teacher.
  * the STUDENT trains with strictly *more* augmentation noise than the teacher saw
    (noisy-student): a configurable ``NoiseSchedule`` whose strength ramps round over
    round (Xie et al. 2019).
  * round-over-round checkpoint selection via the repo's LB-correlated Yao-probe metric
    (:func:`train_g124.compute_yao_selection_score`), so the best round is kept.

This module is backbone-agnostic. It is exercised in the smoke test against the small
SSM head (``ssm_head.BiSSDProtoHead``) on synthetic embedding sequences, and is wired
into ``train_g124.py`` behind the ``--noisy-student-rounds`` flag for the real run.

The defaults match A6 exactly so production behavior is the documented protocol; every
knob is overridable for ablation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence


# --------------------------------------------------------------------------- config


@dataclass
class NoisyStudentConfig:
    n_rounds: int = 4                 # A6 sweet spot 3-4 (Nikita 4, 5th 6 incl. tss)
    pseudo_alpha: float = 0.7         # label = alpha*pseudo + (1-alpha)*hard
    pseudo_threshold: float = 0.3     # linear term zeroed below this
    pseudo_power: float = 2.0         # PowerTransform exponent
    pseudo_ratio_cap: float = 0.4     # max fraction of a batch/dataset that is pseudo
    final_tss: bool = True            # optional final stricter relabel stage
    tss_threshold: float = 0.7        # 5th-place pseudo_tss_th
    tss_power: float = 2.0
    # noise (augmentation) schedule: strength at round r is base + step*r, capped.
    noise_base: float = 0.1
    noise_step: float = 0.1
    noise_max: float = 0.5


@dataclass
class NoiseSchedule:
    """Student augmentation strength for one round (noisier than the teacher)."""

    strength: float = 0.1
    # individual aug magnitudes derive from `strength` unless explicitly set
    mixup_alpha: float = field(default=0.0)
    embedding_dropout: float = field(default=0.0)
    gaussian_std: float = field(default=0.0)
    time_shift: int = field(default=0)

    @classmethod
    def for_round(cls, config: NoisyStudentConfig, round_idx: int) -> "NoiseSchedule":
        s = min(config.noise_base + config.noise_step * round_idx, config.noise_max)
        return cls(
            strength=s,
            mixup_alpha=s,            # beta(alpha, alpha) mixup on embeddings
            embedding_dropout=0.5 * s,
            gaussian_std=0.1 * s,
            time_shift=int(round(2.0 * s)),  # roll windows by up to a couple positions
        )


# --------------------------------------------------------------------- soft targets


def power_transform(p, threshold: float, power: float):
    """A6 PowerTransform: zero weak signal, sharpen confident signal. NumPy or torch.

    ``p = p * (p > threshold) + p ** power``  then clamp to [0, 1].
    """
    try:
        import torch

        if isinstance(p, torch.Tensor):
            keep = (p > threshold).to(p.dtype)
            out = p * keep + p.pow(power)
            return out.clamp(0.0, 1.0)
    except Exception:
        pass
    import numpy as np

    p = np.asarray(p, dtype=np.float32)
    out = p * (p > threshold).astype(np.float32) + np.power(p, power)
    return np.clip(out, 0.0, 1.0)


def make_soft_targets(
    teacher_probs,
    hard_labels,
    config: NoisyStudentConfig,
    use_tss: bool = False,
):
    """Blend a (already-averaged) teacher probability with hard labels.

    ``teacher_probs``: (B, C) soft probabilities (mean of the teacher ensemble).
    ``hard_labels``:  (B, C) one/multi-hot for labeled rows, all-zero for pure-unlabeled
                      soundscape rows (so their target = alpha * pseudo).
    Returns the blended soft target, same backend (torch/numpy) as the inputs.
    """
    threshold = config.tss_threshold if use_tss else config.pseudo_threshold
    power = config.tss_power if use_tss else config.pseudo_power
    p = power_transform(teacher_probs, threshold, power)
    alpha = config.pseudo_alpha
    return alpha * p + (1.0 - alpha) * hard_labels


def ensemble_teacher_probs(teacher_logits_list):
    """Mean of sigmoid(logits) over an ensemble (list of (B,...,C) tensors)."""
    import torch

    probs = [torch.sigmoid(z.float()) for z in teacher_logits_list]
    return torch.stack(probs, dim=0).mean(dim=0)


# ------------------------------------------------------------- pseudo-ratio capping


def cap_pseudo_ratio(n_labeled: int, n_pseudo: int, ratio_cap: float) -> int:
    """Return the number of pseudo rows to KEEP so pseudo/(pseudo+labeled) <= cap.

    With cap c and L labeled rows, max pseudo P satisfies P/(P+L) <= c  =>  P <= cL/(1-c).
    """
    ratio_cap = float(max(0.0, min(ratio_cap, 0.999)))
    if n_labeled <= 0:
        return 0
    max_pseudo = int(round(ratio_cap / (1.0 - ratio_cap) * n_labeled))
    return min(int(n_pseudo), max_pseudo)


def subsample_pseudo_frame(pseudo_frame, n_labeled: int, ratio_cap: float, seed: int):
    """Subsample a pseudo-label DataFrame so the pseudo ratio stays <= cap.

    Highest-``file_confidence`` (or ``sample_weight``) rows are kept preferentially.
    """
    import pandas as pd  # noqa: F401

    if pseudo_frame is None or len(pseudo_frame) == 0:
        return pseudo_frame
    keep = cap_pseudo_ratio(n_labeled, len(pseudo_frame), ratio_cap)
    if keep >= len(pseudo_frame):
        return pseudo_frame
    sort_col = None
    for col in ("file_confidence", "sample_weight", "confidence"):
        if col in pseudo_frame.columns:
            sort_col = col
            break
    if sort_col is not None:
        return (
            pseudo_frame.sort_values(sort_col, ascending=False)
            .head(keep)
            .reset_index(drop=True)
        )
    return pseudo_frame.sample(n=keep, random_state=seed).reset_index(drop=True)


# --------------------------------------------------------------- noise / aug on student


def apply_embedding_noise(embeddings, schedule: NoiseSchedule, generator=None):
    """Inject student-side noise into a (B,T,D) embedding batch (torch).

    Combines gaussian jitter, embedding dropout and a small temporal roll. Mixup is
    applied at the batch/target level by :func:`mixup_embeddings` separately.
    """
    import torch

    x = embeddings
    if schedule.gaussian_std > 0:
        x = x + torch.randn_like(x) * schedule.gaussian_std
    if schedule.embedding_dropout > 0:
        mask = (torch.rand_like(x) > schedule.embedding_dropout).to(x.dtype)
        x = x * mask / max(1.0 - schedule.embedding_dropout, 1e-6)
    if schedule.time_shift > 0 and x.shape[1] > 1:
        shift = int(torch.randint(-schedule.time_shift, schedule.time_shift + 1, (1,)).item())
        if shift != 0:
            x = torch.roll(x, shifts=shift, dims=1)
    return x


def mixup_embeddings(embeddings, targets, schedule: NoiseSchedule):
    """Beta-mixup on embeddings + soft targets (returns mixed pair)."""
    import torch

    if schedule.mixup_alpha <= 0 or embeddings.shape[0] < 2:
        return embeddings, targets
    lam = float(
        torch.distributions.Beta(schedule.mixup_alpha, schedule.mixup_alpha).sample().item()
    )
    perm = torch.randperm(embeddings.shape[0], device=embeddings.device)
    mixed_x = lam * embeddings + (1.0 - lam) * embeddings[perm]
    mixed_y = lam * targets + (1.0 - lam) * targets[perm]
    return mixed_x, mixed_y


# --------------------------------------------------------------------- round selection


@dataclass
class RoundResult:
    round_idx: int
    selection_score: float
    metrics: dict
    state_dict: object
    is_tss: bool = False


def select_best_round(results: Sequence[RoundResult]) -> RoundResult:
    """Pick the round with the highest LB-correlated selection score."""
    if not results:
        raise ValueError("no rounds to select from")
    return max(results, key=lambda r: r.selection_score)


# ------------------------------------------------------------------- orchestration loop


def run_noisy_student(
    *,
    config: NoisyStudentConfig,
    init_student: Callable[[int], object],
    train_one_round: Callable[[object, object, NoiseSchedule, int, bool], object],
    evaluate: Callable[[object], dict],
    selection_score_fn: Callable[[dict], float],
    teacher_init: object | None = None,
    logger: Callable[[str], None] = print,
) -> RoundResult:
    """Drive the multi-round noisy-student loop and return the best round.

    Callbacks (kept abstract so the same loop works for the SSM head and the
    EfficientNet backbone, and so the smoke test can pass tiny synthetic closures):

      * ``init_student(round_idx) -> model``     fresh student each round.
      * ``train_one_round(student, teacher, noise_schedule, round_idx, use_tss) ->
            trained_student``  trains the student against soft targets produced from
            ``teacher`` (None on round 0 => supervised), with the given noise level.
      * ``evaluate(student) -> metrics``         e.g. Yao-probe metrics dict.
      * ``selection_score_fn(metrics) -> float`` LB-correlated score for selection.

    The trained student of round r becomes the teacher of round r+1 (self-distillation).
    """
    if config.n_rounds < 1:
        raise ValueError("n_rounds must be >= 1")
    results: list[RoundResult] = []
    teacher = teacher_init
    for round_idx in range(config.n_rounds):
        is_final = round_idx == config.n_rounds - 1
        use_tss = bool(config.final_tss and is_final and teacher is not None)
        noise = NoiseSchedule.for_round(config, round_idx)
        logger(
            f"noisy_student round={round_idx + 1}/{config.n_rounds} "
            f"noise_strength={noise.strength:.3f} use_tss={use_tss} "
            f"has_teacher={teacher is not None}"
        )
        student = init_student(round_idx)
        student = train_one_round(student, teacher, noise, round_idx, use_tss)
        metrics = evaluate(student)
        score = float(selection_score_fn(metrics))
        state = _detach_state(student)
        results.append(
            RoundResult(
                round_idx=round_idx,
                selection_score=score,
                metrics=metrics,
                state_dict=state,
                is_tss=use_tss,
            )
        )
        logger(
            f"noisy_student round={round_idx + 1}/{config.n_rounds} "
            f"selection_score={score:.5f}"
        )
        teacher = student  # student becomes next round's teacher
    best = select_best_round(results)
    logger(
        f"noisy_student best_round={best.round_idx + 1} "
        f"selection_score={best.selection_score:.5f} is_tss={best.is_tss}"
    )
    return best


def _detach_state(model):
    """Return a CPU fp16 state dict if ``model`` is an nn.Module, else the object."""
    try:
        import torch

        if isinstance(model, torch.nn.Module):
            return {k: v.detach().cpu().half() for k, v in model.state_dict().items()}
    except Exception:
        pass
    return model
