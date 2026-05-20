from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re


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


PUBLIC_CACHE_SUFFIXES = {".csv", ".npz"}


def _public_registry(root: Path, public_output_root: Path | None = None) -> list[ModelArtifact]:
    public_root = root / "analysis" / "entropy_tta" / "public_kernels"
    outputs = public_root / "outputs_v2.json"
    if not outputs.exists():
        return []

    data = json.loads(outputs.read_text())
    items: list[ModelArtifact] = []
    for entry in data:
        if not entry.get("has_oof"):
            continue
        ref = entry["ref"]
        slug = ref.replace("/", "__")
        local_dir = _find_public_kernel_dir(public_root, ref, public_output_root)
        if local_dir is None:
            continue
        cache_files = [
            file_name
            for file_name in entry.get("files", [])
            if Path(str(file_name)).suffix in PUBLIC_CACHE_SUFFIXES
        ]
        for file_name in cache_files:
            artifact_path = _find_artifact(local_dir, str(file_name))
            if artifact_path is None:
                continue
            file_stem = Path(file_name).stem
            prediction_keys = ["oof_base", "oof_prior"] if file_stem == "full_oof_meta_features" else [None]
            for prediction_key in prediction_keys:
                model_id = f"public__{slug}" if file_stem == "submission" else f"public__{slug}__{file_stem}"
                if prediction_key is not None:
                    model_id = f"{model_id}__{prediction_key}"
                items.append(
                    ModelArtifact(
                        model_id=model_id,
                        source="public",
                        category="public_cache",
                        artifact_path=artifact_path,
                        prediction_key=prediction_key,
                        known_lb=float(entry["lb"]) if entry.get("lb") is not None else None,
                        coverage="labeled",
                        notebook_slug=ref,
                        has_train_soundscape_predictions=True,
                    )
                )
    return items


def _find_public_kernel_dir(public_root: Path, ref: str, public_output_root: Path | None = None) -> Path | None:
    double_underscore_slug = ref.replace("/", "__")
    owner_slug = ref.replace("/", "_")
    underscore_slug = re.sub(r"[^A-Za-z0-9]+", "_", ref).strip("_")
    hyphen_slug = re.sub(r"[^A-Za-z0-9]+", "-", ref).strip("-")
    for root in [public_root, public_output_root]:
        if root is None:
            continue
        for name in dict.fromkeys([double_underscore_slug, owner_slug, underscore_slug, hyphen_slug]):
            candidate = root / name
            if candidate.is_dir():
                return candidate
    return None


def _find_artifact(local_dir: Path, file_name: str) -> Path | None:
    for candidate in [local_dir / file_name, local_dir / "cache" / file_name]:
        if candidate.exists():
            return candidate
    matches = sorted(local_dir.rglob(file_name))
    if matches:
        return matches[0]
    return None


def default_registry(root: Path, public_output_root: Path | None = None) -> list[ModelArtifact]:
    return _internal_registry(root) + _public_registry(root, public_output_root)
