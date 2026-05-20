from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class ModelArtifact:
    model_id: str
    source: str
    category: str
    artifact_path: Path
    prediction_key: str | None
    known_lb: float | None
    coverage: str
    notebook_slug: str | None = None
    has_train_soundscape_predictions: bool = True


def _internal_registry(root: Path) -> list[ModelArtifact]:
    creative = root / "analysis" / "creative"
    entropy = root / "analysis" / "entropy_tta"
    return [
        ModelArtifact("exp019", "ours", "blend", entropy / "exp019_aligned.npz", "P_exp019", 0.949, "labeled"),
        ModelArtifact("birdmae", "ours", "birdmae", creative / "birdmae_blend.npz", "P_birdmae", 0.946, "labeled"),
        ModelArtifact("perch20_raw", "ours", "perch", creative / "perch20_blend.npz", "P_perch20", None, "labeled"),
        ModelArtifact("sub_v8_lgb_ens", "ours", "blend", creative / "final_oof.npz", "P_lgb_ens", None, "labeled"),
        ModelArtifact("distill", "ours", "sed", creative / "final_oof.npz", "P_distill", None, "labeled"),
        ModelArtifact("v73_rag", "ours", "retrieval", creative / "v73_rag.npz", "P_rag_v73", 0.941, "labeled"),
    ]


def _public_registry(root: Path) -> list[ModelArtifact]:
    outputs = root / "analysis" / "entropy_tta" / "public_kernels" / "outputs_v2.json"
    if not outputs.exists():
        return []

    data = json.loads(outputs.read_text())
    items: list[ModelArtifact] = []
    for entry in data:
        if not entry.get("has_oof"):
            continue
        ref = entry["ref"]
        slug = ref.replace("/", "__")
        items.append(
            ModelArtifact(
                model_id=f"public__{slug}",
                source="public",
                category="public_cache",
                artifact_path=outputs,
                prediction_key=None,
                known_lb=float(entry["lb"]) if entry.get("lb") is not None else None,
                coverage="labeled",
                notebook_slug=ref,
                has_train_soundscape_predictions=True,
            )
        )
    return items


def default_registry(root: Path) -> list[ModelArtifact]:
    return _internal_registry(root) + _public_registry(root)
