from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from registry import default_registry


def test_default_registry_contains_core_internal_models():
    root = Path("birdclef-2026")
    registry = default_registry(root)
    ids = {item.model_id for item in registry}

    assert "exp019" in ids
    assert "birdmae" in ids
    assert "perch20_raw" in ids


def test_registry_keeps_only_public_entries_with_oof(tmp_path):
    root = tmp_path
    public_cache = root / "analysis" / "entropy_tta" / "public_kernels"
    public_cache.mkdir(parents=True)
    kept_kernel = public_cache / "author_kept-notebook"
    kept_kernel.mkdir()
    (kept_kernel / "submission.csv").write_text("row_id,a\nf1_5,0.1\n")
    (public_cache / "outputs_v2.json").write_text(
        json.dumps(
            [
                {"ref": "author/kept-notebook", "lb": 0.912, "files": ["submission.csv"], "has_oof": True},
                {"ref": "author/dropped-notebook", "lb": 0.905, "files": ["submission.csv"], "has_oof": False},
                {"ref": "author/missing-local-csv", "lb": 0.901, "files": ["submission.csv"], "has_oof": True},
            ]
        )
    )

    registry = default_registry(root)
    public_items = [item for item in registry if item.source == "public"]

    assert len(public_items) == 1
    assert public_items[0].notebook_slug == "author/kept-notebook"
    assert public_items[0].model_id == "public__author__kept-notebook"
    assert public_items[0].category == "public_submission"
    assert public_items[0].known_lb == 0.912
    assert public_items[0].artifact_path == kept_kernel / "submission.csv"


def test_registry_finds_public_cache_in_external_output_root(tmp_path):
    root = tmp_path / "repo"
    public_cache = root / "analysis" / "entropy_tta" / "public_kernels"
    public_cache.mkdir(parents=True)
    (public_cache / "outputs_v2.json").write_text(
        json.dumps(
            [
                {
                    "ref": "author/kept-notebook",
                    "lb": 0.947,
                    "files": ["perch_arrays.npz"],
                    "has_oof": True,
                }
            ]
        )
    )
    external = tmp_path / "outputs"
    cache_dir = external / "author__kept-notebook" / "cache"
    cache_dir.mkdir(parents=True)
    artifact = cache_dir / "perch_arrays.npz"
    artifact.write_bytes(b"placeholder")

    registry = default_registry(root, public_output_root=external)
    public_items = [item for item in registry if item.source == "public"]

    assert len(public_items) == 1
    assert public_items[0].model_id == "public__author__kept-notebook__perch_arrays"
    assert public_items[0].category == "public_perch_cache"
    assert public_items[0].artifact_path == artifact


def test_registry_splits_full_oof_cache_keys(tmp_path):
    root = tmp_path / "repo"
    public_cache = root / "analysis" / "entropy_tta" / "public_kernels"
    public_cache.mkdir(parents=True)
    (public_cache / "outputs_v2.json").write_text(
        json.dumps(
            [
                {
                    "ref": "author/oof-notebook",
                    "lb": 0.949,
                    "files": ["full_oof_meta_features.npz"],
                    "has_oof": True,
                }
            ]
        )
    )
    external = tmp_path / "outputs"
    cache_dir = external / "author__oof-notebook" / "perch_cache"
    cache_dir.mkdir(parents=True)
    artifact = cache_dir / "full_oof_meta_features.npz"
    artifact.write_bytes(b"placeholder")

    registry = default_registry(root, public_output_root=external)
    public_items = [item for item in registry if item.source == "public"]

    assert [item.prediction_key for item in public_items] == ["oof_base", "oof_prior"]
    assert [item.category for item in public_items] == ["public_final_oof", "public_final_oof"]
    assert [item.model_id for item in public_items] == [
        "public__author__oof-notebook__full_oof_meta_features__oof_base",
        "public__author__oof-notebook__full_oof_meta_features__oof_prior",
    ]
    assert all(item.artifact_path == artifact for item in public_items)
