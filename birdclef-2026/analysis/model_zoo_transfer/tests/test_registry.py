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
    assert public_items[0].known_lb == 0.912
    assert public_items[0].artifact_path == kept_kernel / "submission.csv"
