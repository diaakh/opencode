from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from registry import ModelArtifact, default_registry


def test_default_registry_contains_core_internal_models():
    root = Path("birdclef-2026")
    registry = default_registry(root)
    ids = {item.model_id for item in registry}

    assert "exp019" in ids
    assert "birdmae" in ids
    assert "perch20_raw" in ids


def test_registry_keeps_only_public_entries_with_oof():
    root = Path("birdclef-2026")
    registry = default_registry(root)
    public_items = [item for item in registry if item.source == "public"]

    assert all(item.has_train_soundscape_predictions for item in public_items)
