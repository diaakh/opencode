# BirdCLEF Model Zoo Transfer Analyzer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a phase-1 analyzer that ingests ours/public `train_soundscapes` prediction artifacts, computes transfer-risk features, and writes a model-zoo report.

**Architecture:** Create a small Python package under `birdclef-2026/analysis/model_zoo_transfer/`. The package has explicit registry entries, normalized prediction loaders, feature computation, and report generation. Tests use synthetic arrays so behavior is verified without requiring Kaggle data or large artifacts.

**Tech Stack:** Python 3, NumPy, pandas, scikit-learn metrics, pytest, JSON/CSV/NPZ/parquet artifact readers.

---

## File Structure

- Create `birdclef-2026/analysis/model_zoo_transfer/__init__.py`
  - Package marker.
- Create `birdclef-2026/analysis/model_zoo_transfer/registry.py`
  - Explicit model registry for phase 1.
  - Declares local OOF artifacts and public cache candidates.
- Create `birdclef-2026/analysis/model_zoo_transfer/normalize_predictions.py`
  - Loads `.npz`, `.csv`, and public-cache metadata into a normalized in-memory object.
  - Aligns rows/classes against `exp019_aligned.npz`.
- Create `birdclef-2026/analysis/model_zoo_transfer/features.py`
  - Computes labeled metrics, distribution metrics, site/hour behavior, agreement metrics, and risk tiers.
- Create `birdclef-2026/analysis/model_zoo_transfer/analyze.py`
  - CLI entrypoint that builds `model_zoo_features.csv`, `model_zoo_feature_correlations.csv`, and `model_zoo_report.md`.
- Create `birdclef-2026/analysis/model_zoo_transfer/README.md`
  - Usage and artifact assumptions.
- Create `birdclef-2026/analysis/model_zoo_transfer/tests/test_normalize_predictions.py`
  - Unit tests for row/class normalization and coverage.
- Create `birdclef-2026/analysis/model_zoo_transfer/tests/test_features.py`
  - Unit tests for metric/feature computation and tiering.

## Task 1: Package Skeleton and Shared Data Types

**Files:**
- Create: `birdclef-2026/analysis/model_zoo_transfer/__init__.py`
- Create: `birdclef-2026/analysis/model_zoo_transfer/normalize_predictions.py`
- Test: `birdclef-2026/analysis/model_zoo_transfer/tests/test_normalize_predictions.py`

- [ ] **Step 1: Write tests for the normalized object**

Create `birdclef-2026/analysis/model_zoo_transfer/tests/test_normalize_predictions.py` with:

```python
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path("birdclef-2026/analysis/model_zoo_transfer").resolve()))

from normalize_predictions import NormalizedPrediction


def test_normalized_prediction_records_coverage_counts():
    pred = NormalizedPrediction(
        model_id="toy",
        row_ids=np.array(["file_5", "file_10"]),
        classes=np.array(["a", "b", "c"]),
        predictions=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32),
        source="ours",
        category="single",
        known_lb=0.9,
        coverage="labeled",
        artifact_path="toy.npz",
    )

    assert pred.n_rows == 2
    assert pred.n_classes == 3
    assert pred.predictions.dtype == np.float32
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
python3 -m pytest birdclef-2026/analysis/model_zoo_transfer/tests/test_normalize_predictions.py -q
```

Expected: import failure because `normalize_predictions.py` does not exist.

- [ ] **Step 3: Create package bridge and dataclass**

Because the directory is named `birdclef-2026`, add a test import bridge in the test later if needed. First create `birdclef-2026/analysis/model_zoo_transfer/__init__.py`:

```python
"""Model-zoo transfer analysis for BirdCLEF 2026."""
```

Create `birdclef-2026/analysis/model_zoo_transfer/normalize_predictions.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class NormalizedPrediction:
    model_id: str
    row_ids: np.ndarray
    classes: np.ndarray
    predictions: np.ndarray
    source: str
    category: str
    known_lb: float | None
    coverage: str
    artifact_path: str

    @property
    def n_rows(self) -> int:
        return int(self.predictions.shape[0])

    @property
    def n_classes(self) -> int:
        return int(self.predictions.shape[1])

    @property
    def path(self) -> Path:
        return Path(self.artifact_path)
```

- [ ] **Step 4: Run the test and verify it passes**

Run:

```bash
python3 -m pytest birdclef-2026/analysis/model_zoo_transfer/tests/test_normalize_predictions.py -q
```

Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add birdclef-2026/analysis/model_zoo_transfer
git commit -m "feat: add model zoo transfer package skeleton"
```

## Task 2: Registry for Ours and Public Artifacts

**Files:**
- Create: `birdclef-2026/analysis/model_zoo_transfer/registry.py`
- Test: `birdclef-2026/analysis/model_zoo_transfer/tests/test_registry.py`

- [ ] **Step 1: Write registry tests**

Create `birdclef-2026/analysis/model_zoo_transfer/tests/test_registry.py`:

```python
from pathlib import Path

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
```

In this test file, add:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("birdclef-2026/analysis/model_zoo_transfer").resolve()))
```

before importing `registry`.

- [ ] **Step 2: Run the tests and verify failure**

Run:

```bash
python3 -m pytest birdclef-2026/analysis/model_zoo_transfer/tests/test_registry.py -q
```

Expected: import failure because `registry.py` does not exist.

- [ ] **Step 3: Implement the registry**

Create `birdclef-2026/analysis/model_zoo_transfer/registry.py`:

```python
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
```

- [ ] **Step 4: Run registry tests**

Run:

```bash
python3 -m pytest birdclef-2026/analysis/model_zoo_transfer/tests/test_registry.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add birdclef-2026/analysis/model_zoo_transfer
git commit -m "feat: register model zoo artifacts"
```

## Task 3: Normalize Internal NPZ Predictions

**Files:**
- Modify: `birdclef-2026/analysis/model_zoo_transfer/normalize_predictions.py`
- Test: `birdclef-2026/analysis/model_zoo_transfer/tests/test_normalize_predictions.py`

- [ ] **Step 1: Add tests for `.npz` loading**

Append to `test_normalize_predictions.py`:

```python
from pathlib import Path
import sys

sys.path.insert(0, str(Path("birdclef-2026/analysis/model_zoo_transfer").resolve()))

from registry import ModelArtifact
from normalize_predictions import LabelBackbone, load_internal_npz


def test_load_internal_npz_aligns_to_backbone(tmp_path):
    artifact = tmp_path / "toy.npz"
    np.savez(
        artifact,
        P_model=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
    )
    backbone = LabelBackbone(
        row_ids=np.array(["BC2026_Train_0001_S01_20260101_010000_5", "BC2026_Train_0001_S01_20260101_010000_10"]),
        classes=np.array(["a", "b"]),
        labels=np.array([[1, 0], [0, 1]], dtype=np.float32),
        filenames=np.array(["BC2026_Train_0001_S01_20260101_010000.ogg", "BC2026_Train_0001_S01_20260101_010000.ogg"]),
        start_seconds=np.array([0, 5]),
    )
    item = ModelArtifact("toy", "ours", "single", artifact, "P_model", None, "labeled")

    pred = load_internal_npz(item, backbone)

    assert pred.model_id == "toy"
    assert pred.row_ids.tolist() == backbone.row_ids.tolist()
    assert pred.classes.tolist() == ["a", "b"]
    assert pred.predictions.shape == (2, 2)
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python3 -m pytest birdclef-2026/analysis/model_zoo_transfer/tests/test_normalize_predictions.py -q
```

Expected: failure because `LabelBackbone` and `load_internal_npz` do not exist.

- [ ] **Step 3: Implement label backbone and NPZ loader**

Add to `normalize_predictions.py`:

```python
@dataclass(frozen=True)
class LabelBackbone:
    row_ids: np.ndarray
    classes: np.ndarray
    labels: np.ndarray
    filenames: np.ndarray
    start_seconds: np.ndarray


def make_row_ids(filenames: np.ndarray, start_seconds: np.ndarray) -> np.ndarray:
    values = []
    for filename, start in zip(filenames, start_seconds):
        stem = str(filename).replace(".ogg", "")
        values.append(f"{stem}_{int(start) + 5}")
    return np.array(values, dtype=object)


def load_label_backbone(path: Path) -> LabelBackbone:
    data = np.load(path, allow_pickle=True)
    filenames = data["row_filename"]
    start_seconds = data["row_start_sec"].astype(int)
    return LabelBackbone(
        row_ids=make_row_ids(filenames, start_seconds),
        classes=data["classes"].astype(str),
        labels=data["Y"].astype(np.float32),
        filenames=filenames,
        start_seconds=start_seconds,
    )


def load_internal_npz(item, backbone: LabelBackbone) -> NormalizedPrediction:
    if item.prediction_key is None:
        raise ValueError(f"{item.model_id} has no prediction_key")
    data = np.load(item.artifact_path, allow_pickle=True)
    if item.prediction_key not in data.files:
        raise KeyError(f"{item.prediction_key} not found in {item.artifact_path}")
    predictions = data[item.prediction_key].astype(np.float32)
    if predictions.shape != backbone.labels.shape:
        raise ValueError(
            f"{item.model_id} shape {predictions.shape} does not match backbone {backbone.labels.shape}"
        )
    return NormalizedPrediction(
        model_id=item.model_id,
        row_ids=backbone.row_ids.copy(),
        classes=backbone.classes.copy(),
        predictions=predictions,
        source=item.source,
        category=item.category,
        known_lb=item.known_lb,
        coverage=item.coverage,
        artifact_path=str(item.artifact_path),
    )
```

- [ ] **Step 4: Run normalization tests**

Run:

```bash
python3 -m pytest birdclef-2026/analysis/model_zoo_transfer/tests/test_normalize_predictions.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add birdclef-2026/analysis/model_zoo_transfer
git commit -m "feat: normalize internal model predictions"
```

## Task 4: Feature Computation

**Files:**
- Create: `birdclef-2026/analysis/model_zoo_transfer/features.py`
- Test: `birdclef-2026/analysis/model_zoo_transfer/tests/test_features.py`

- [ ] **Step 1: Write feature tests**

Create `birdclef-2026/analysis/model_zoo_transfer/tests/test_features.py`:

```python
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path("birdclef-2026/analysis/model_zoo_transfer").resolve()))

from features import compute_feature_row, infer_sites, infer_hours, risk_tier
from normalize_predictions import LabelBackbone, NormalizedPrediction


def test_infer_sites_and_hours_from_row_ids():
    row_ids = np.array([
        "BC2026_Train_0001_S22_20211231_201500_5",
        "BC2026_Train_0002_S08_20250606_030007_10",
    ])

    assert infer_sites(row_ids).tolist() == ["S22", "S08"]
    assert infer_hours(row_ids).tolist() == [20, 3]


def test_compute_feature_row_has_expected_metrics():
    backbone = LabelBackbone(
        row_ids=np.array([
            "BC2026_Train_0001_S22_20211231_201500_5",
            "BC2026_Train_0001_S22_20211231_201500_10",
            "BC2026_Train_0002_S08_20250606_030007_5",
            "BC2026_Train_0002_S08_20250606_030007_10",
        ]),
        classes=np.array(["a", "b"]),
        labels=np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32),
        filenames=np.array(["f1.ogg", "f1.ogg", "f2.ogg", "f2.ogg"]),
        start_seconds=np.array([0, 5, 0, 5]),
    )
    pred = NormalizedPrediction(
        model_id="good",
        row_ids=backbone.row_ids,
        classes=backbone.classes,
        predictions=np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.8], [0.2, 0.9]], dtype=np.float32),
        source="ours",
        category="single",
        known_lb=0.95,
        coverage="labeled",
        artifact_path="toy.npz",
    )

    row = compute_feature_row(pred, backbone, anchors={})

    assert row["model_id"] == "good"
    assert row["coverage_labeled_rows"] == 4
    assert row["coverage_classes"] == 2
    assert row["labeled_macro_auc"] == 1.0
    assert 0.0 <= row["entropy_mean"] <= 1.0
    assert row["risk_tier"] == "ceiling"
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python3 -m pytest birdclef-2026/analysis/model_zoo_transfer/tests/test_features.py -q
```

Expected: import failure because `features.py` does not exist.

- [ ] **Step 3: Implement features**

Create `birdclef-2026/analysis/model_zoo_transfer/features.py`:

```python
from __future__ import annotations

import re

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score


SITE_RE = re.compile(r"_(S\d{2})_")
HOUR_RE = re.compile(r"_\d{8}_(\d{2})\d{4}_")


def infer_sites(row_ids: np.ndarray) -> np.ndarray:
    sites = []
    for row_id in row_ids:
        match = SITE_RE.search(str(row_id))
        sites.append(match.group(1) if match else "S??")
    return np.array(sites, dtype=object)


def infer_hours(row_ids: np.ndarray) -> np.ndarray:
    hours = []
    for row_id in row_ids:
        match = HOUR_RE.search(str(row_id))
        hours.append(int(match.group(1)) if match else -1)
    return np.array(hours, dtype=int)


def macro_auc(labels: np.ndarray, predictions: np.ndarray) -> float:
    aucs = []
    for col in range(labels.shape[1]):
        y = labels[:, col]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        if predictions[:, col].min() == predictions[:, col].max():
            continue
        aucs.append(roc_auc_score(y, predictions[:, col]))
    return float(np.mean(aucs)) if aucs else float("nan")


def _rank_matrix(values: np.ndarray) -> np.ndarray:
    ranked = np.zeros_like(values, dtype=np.float32)
    for col in range(values.shape[1]):
        ranked[:, col] = rankdata(values[:, col], method="average") / max(len(values), 1)
    return ranked


def _mean_rank_agreement(predictions: np.ndarray, anchor: np.ndarray) -> float:
    pred_rank = _rank_matrix(predictions)
    anchor_rank = _rank_matrix(anchor)
    scores = []
    for col in range(predictions.shape[1]):
        corr = spearmanr(pred_rank[:, col], anchor_rank[:, col]).correlation
        if np.isfinite(corr):
            scores.append(corr)
    return float(np.mean(scores)) if scores else float("nan")


def risk_tier(labeled_macro_auc: float, known_lb: float | None, site_gap: float) -> str:
    if known_lb is not None:
        if known_lb < 0.80:
            return "catastrophic"
        if known_lb < 0.93:
            return "risky"
        if known_lb < 0.948:
            return "safe"
        return "ceiling"
    if not np.isfinite(labeled_macro_auc):
        return "risky"
    if site_gap > 0.08:
        return "risky"
    if labeled_macro_auc >= 0.96:
        return "safe"
    return "risky"


def compute_feature_row(pred, backbone, anchors: dict[str, np.ndarray]) -> dict[str, float | str | int | None]:
    labels = backbone.labels
    predictions = np.clip(pred.predictions.astype(np.float32), 1e-6, 1 - 1e-6)
    sites = infer_sites(pred.row_ids)
    hours = infer_hours(pred.row_ids)

    overall = macro_auc(labels, predictions)
    site_aucs = []
    for site in sorted(set(sites)):
        mask = sites == site
        if mask.sum() < 2:
            continue
        score = macro_auc(labels[mask], predictions[mask])
        if np.isfinite(score):
            site_aucs.append(score)
    site_mean = float(np.mean(site_aucs)) if site_aucs else float("nan")
    site_gap = float(overall - site_mean) if np.isfinite(overall) and np.isfinite(site_mean) else float("nan")

    entropy = -(predictions * np.log2(predictions) + (1 - predictions) * np.log2(1 - predictions))
    row = {
        "model_id": pred.model_id,
        "source": pred.source,
        "category": pred.category,
        "known_lb": pred.known_lb,
        "coverage": pred.coverage,
        "coverage_labeled_rows": pred.n_rows if pred.coverage in {"labeled", "both"} else 0,
        "coverage_unlabeled_rows": pred.n_rows if pred.coverage == "unlabeled" else 0,
        "coverage_classes": pred.n_classes,
        "n_sites": int(len(set(sites))),
        "n_hours": int(len(set(hours.tolist()) - {-1})),
        "labeled_macro_auc": overall,
        "site_mean_auc": site_mean,
        "site_gap": site_gap,
        "entropy_mean": float(np.mean(entropy)),
        "entropy_p90": float(np.quantile(entropy, 0.90)),
        "confidence_rate_gt_0_9": float((predictions > 0.9).mean()),
        "probability_median": float(np.median(predictions)),
        "probability_p99": float(np.quantile(predictions, 0.99)),
    }
    for anchor_name, anchor_predictions in anchors.items():
        if anchor_predictions.shape == predictions.shape:
            row[f"agreement_{anchor_name}"] = _mean_rank_agreement(predictions, anchor_predictions)
    row["risk_tier"] = risk_tier(overall, pred.known_lb, site_gap if np.isfinite(site_gap) else 0.0)
    return row
```

- [ ] **Step 4: Run feature tests**

Run:

```bash
python3 -m pytest birdclef-2026/analysis/model_zoo_transfer/tests/test_features.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add birdclef-2026/analysis/model_zoo_transfer
git commit -m "feat: compute model zoo transfer features"
```

## Task 5: Analysis CLI and Report

**Files:**
- Create: `birdclef-2026/analysis/model_zoo_transfer/analyze.py`
- Modify: `birdclef-2026/analysis/model_zoo_transfer/normalize_predictions.py`
- Test: run CLI against local artifacts.

- [ ] **Step 1: Add loader orchestration**

Add to `normalize_predictions.py`:

```python
def load_prediction(item, backbone: LabelBackbone) -> NormalizedPrediction | None:
    if item.source == "ours":
        return load_internal_npz(item, backbone)
    return None
```

Phase 1 starts with all internal artifacts plus public registry metadata. Public cache extraction is added in Task 7 after the report path is proven.

- [ ] **Step 2: Create CLI**

Create `birdclef-2026/analysis/model_zoo_transfer/analyze.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

from features import compute_feature_row
from normalize_predictions import load_label_backbone, load_prediction
from registry import default_registry


def _correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scored = df[df["known_lb"].notna()].copy()
    if len(scored) < 3:
        return pd.DataFrame(columns=["feature", "n", "pearson", "spearman"])
    for col in scored.columns:
        if col in {"model_id", "source", "category", "coverage", "risk_tier"}:
            continue
        if col == "known_lb":
            continue
        values = pd.to_numeric(scored[col], errors="coerce")
        mask = values.notna()
        if mask.sum() < 3:
            continue
        pear = pearsonr(values[mask], scored.loc[mask, "known_lb"]).statistic
        spear = spearmanr(values[mask], scored.loc[mask, "known_lb"]).correlation
        rows.append({"feature": col, "n": int(mask.sum()), "pearson": pear, "spearman": spear})
    return pd.DataFrame(rows).sort_values("spearman", ascending=False)


def _write_report(df: pd.DataFrame, corr: pd.DataFrame, path: Path) -> None:
    lines = [
        "# BirdCLEF Model Zoo Transfer Report",
        "",
        f"Models ingested: {len(df)}",
        f"Models with known LB: {int(df['known_lb'].notna().sum())}",
        "",
        "## Risk Tiers",
        "",
        df["risk_tier"].value_counts().to_markdown(),
        "",
        "## Top LB-Correlated Features",
        "",
        corr.head(15).to_markdown(index=False) if not corr.empty else "Not enough known-LB models for correlations.",
        "",
        "## Known-LB Models",
        "",
        df[df["known_lb"].notna()][["model_id", "category", "known_lb", "labeled_macro_auc", "site_gap", "risk_tier"]]
        .sort_values("known_lb", ascending=False)
        .to_markdown(index=False),
    ]
    path.write_text("\\n".join(lines) + "\\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="birdclef-2026")
    parser.add_argument("--out-dir", default="birdclef-2026/analysis/model_zoo_transfer")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    backbone = load_label_backbone(root / "analysis" / "entropy_tta" / "exp019_aligned.npz")
    registry = default_registry(root)
    loaded = [load_prediction(item, backbone) for item in registry]
    predictions = [item for item in loaded if item is not None]

    anchors = {}
    for pred in predictions:
        if pred.model_id == "exp019":
            anchors["exp019"] = pred.predictions

    rows = [compute_feature_row(pred, backbone, anchors) for pred in predictions]
    df = pd.DataFrame(rows)
    corr = _correlations(df)

    df.to_csv(out_dir / "model_zoo_features.csv", index=False)
    corr.to_csv(out_dir / "model_zoo_feature_correlations.csv", index=False)
    _write_report(df, corr, out_dir / "model_zoo_report.md")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run CLI**

Run:

```bash
python3 birdclef-2026/analysis/model_zoo_transfer/analyze.py
```

Expected:

```text
birdclef-2026/analysis/model_zoo_transfer/model_zoo_features.csv
birdclef-2026/analysis/model_zoo_transfer/model_zoo_feature_correlations.csv
birdclef-2026/analysis/model_zoo_transfer/model_zoo_report.md
```

- [ ] **Step 4: Inspect outputs**

Run:

```bash
python3 - <<'PY'
import pandas as pd
df = pd.read_csv("birdclef-2026/analysis/model_zoo_transfer/model_zoo_features.csv")
print(df[["model_id", "known_lb", "labeled_macro_auc", "site_gap", "risk_tier"]].to_string(index=False))
PY
```

Expected: rows for at least `exp019`, `birdmae`, `perch20_raw`, `sub_v8_lgb_ens`, `distill`, and `v73_rag`.

- [ ] **Step 5: Commit**

```bash
git add birdclef-2026/analysis/model_zoo_transfer
git commit -m "feat: generate model zoo transfer report"
```

## Task 6: Public Cache Extraction for `outputs_v2.json`

**Files:**
- Modify: `birdclef-2026/analysis/model_zoo_transfer/registry.py`
- Modify: `birdclef-2026/analysis/model_zoo_transfer/normalize_predictions.py`
- Test: `birdclef-2026/analysis/model_zoo_transfer/tests/test_normalize_predictions.py`

- [ ] **Step 1: Add synthetic public-cache test**

Append to `test_normalize_predictions.py`:

```python
from normalize_predictions import load_public_cache_csv


def test_load_public_cache_csv_reindexes_to_backbone(tmp_path):
    cache = tmp_path / "submission.csv"
    cache.write_text("row_id,a,b\\nf2_10,0.8,0.2\\nf1_5,0.1,0.9\\n")
    backbone = LabelBackbone(
        row_ids=np.array(["f1_5", "f2_10"]),
        classes=np.array(["a", "b"]),
        labels=np.array([[1, 0], [0, 1]], dtype=np.float32),
        filenames=np.array(["f1.ogg", "f2.ogg"]),
        start_seconds=np.array([0, 5]),
    )
    item = ModelArtifact("public_toy", "public", "public_cache", cache, None, 0.947, "labeled")

    pred = load_public_cache_csv(item, backbone)

    assert pred.predictions.tolist() == [[0.1, 0.9], [0.8, 0.2]]
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
python3 -m pytest birdclef-2026/analysis/model_zoo_transfer/tests/test_normalize_predictions.py -q
```

Expected: failure because `load_public_cache_csv` does not exist.

- [ ] **Step 3: Implement public CSV loader**

Add to `normalize_predictions.py`:

```python
import pandas as pd


def load_public_cache_csv(item, backbone: LabelBackbone) -> NormalizedPrediction:
    df = pd.read_csv(item.artifact_path)
    if "row_id" not in df.columns:
        raise ValueError(f"{item.artifact_path} has no row_id column")
    missing = [cls for cls in backbone.classes if cls not in df.columns]
    if missing:
        raise ValueError(f"{item.model_id} missing class columns: {missing[:5]}")
    aligned = df.set_index("row_id").reindex(backbone.row_ids)
    present = aligned[backbone.classes].notna().all(axis=1).to_numpy()
    if not present.any():
        raise ValueError(f"{item.model_id} has no rows matching label backbone")
    predictions = aligned.loc[present, backbone.classes].to_numpy(dtype=np.float32)
    return NormalizedPrediction(
        model_id=item.model_id,
        row_ids=backbone.row_ids[present],
        classes=backbone.classes.copy(),
        predictions=predictions,
        source=item.source,
        category=item.category,
        known_lb=item.known_lb,
        coverage=item.coverage,
        artifact_path=str(item.artifact_path),
    )
```

Update `load_prediction`:

```python
def load_prediction(item, backbone: LabelBackbone) -> NormalizedPrediction | None:
    if item.source == "ours":
        return load_internal_npz(item, backbone)
    if item.artifact_path.suffix == ".csv":
        return load_public_cache_csv(item, backbone)
    return None
```

- [ ] **Step 4: Wire actual public CSV paths**

Modify `_public_registry` in `registry.py` to look for extracted folders under:

```python
public_root = root / "analysis" / "entropy_tta" / "public_kernels"
```

For each `entry["files"]`, add an artifact only when a `submission.csv` exists under a matching local output directory. Use the explicit `outputs_v2.json` metadata and preserve `entry["ref"]` as `notebook_slug`.

- [ ] **Step 5: Run tests and CLI**

Run:

```bash
python3 -m pytest birdclef-2026/analysis/model_zoo_transfer/tests -q
python3 birdclef-2026/analysis/model_zoo_transfer/analyze.py
```

Expected: tests pass; CLI still writes the three output files. If no public CSVs are locally aligned, the report should still work with internal models.

- [ ] **Step 6: Commit**

```bash
git add birdclef-2026/analysis/model_zoo_transfer
git commit -m "feat: ingest public model cache predictions"
```

## Task 7: README and Verification

**Files:**
- Create: `birdclef-2026/analysis/model_zoo_transfer/README.md`
- Modify: `birdclef-2026/analysis/model_zoo_transfer/model_zoo_report.md` by regenerating it.

- [ ] **Step 1: Create README**

Create `birdclef-2026/analysis/model_zoo_transfer/README.md`:

```markdown
# Model Zoo Transfer Analyzer

This folder builds a phase-1 model-zoo table for BirdCLEF 2026. It compares our models and public notebooks only when actual `train_soundscapes` predictions are available.

## Run

```bash
python3 birdclef-2026/analysis/model_zoo_transfer/analyze.py
```

Outputs:

- `model_zoo_features.csv`
- `model_zoo_feature_correlations.csv`
- `model_zoo_report.md`

## Scope

The analyzer intentionally avoids notebooks that only have LB scores or code structure. Phase 1 requires extractable prediction/cache outputs on labeled or unlabeled `train_soundscapes`.

## Interpretation

Use the report for risk tiering and candidate ranking. Do not treat exact LB regression as reliable to 0.001 precision.
```

- [ ] **Step 2: Run full verification**

Run:

```bash
python3 -m pytest birdclef-2026/analysis/model_zoo_transfer/tests -q
python3 birdclef-2026/analysis/model_zoo_transfer/analyze.py
test -s birdclef-2026/analysis/model_zoo_transfer/model_zoo_features.csv
test -s birdclef-2026/analysis/model_zoo_transfer/model_zoo_report.md
```

Expected: tests pass, analyzer exits with status 0, output files are non-empty.

- [ ] **Step 3: Inspect git diff**

Run:

```bash
git diff --stat
git status --short
```

Expected: changes are limited to `birdclef-2026/analysis/model_zoo_transfer/` plus generated output files.

- [ ] **Step 4: Commit**

```bash
git add birdclef-2026/analysis/model_zoo_transfer
git commit -m "docs: document model zoo transfer analyzer"
```

## Task 8: Final Review

**Files:**
- Review all files under `birdclef-2026/analysis/model_zoo_transfer/`.

- [ ] **Step 1: Run final verification**

Run:

```bash
python3 -m pytest birdclef-2026/analysis/model_zoo_transfer/tests -q
python3 birdclef-2026/analysis/model_zoo_transfer/analyze.py
git status --short --branch
```

Expected:

- Tests pass.
- Analyzer regenerates outputs.
- Only intentional model-zoo files are modified.
- Existing untracked `src-tauri/` remains untouched.

- [ ] **Step 2: Review report for useful research signal**

Open:

```bash
sed -n '1,220p' birdclef-2026/analysis/model_zoo_transfer/model_zoo_report.md
```

Expected: report includes model count, known-LB count, risk-tier table, top feature correlations, and known-LB model summary.

- [ ] **Step 3: Commit regenerated outputs if changed**

```bash
git add birdclef-2026/analysis/model_zoo_transfer
git commit -m "chore: refresh model zoo transfer outputs"
```

Skip this commit only if `git status --short` shows no changes under `birdclef-2026/analysis/model_zoo_transfer/`.

---

## Self-Review Notes

- Spec coverage: registry, normalization, features, report outputs, public-only-with-cache scope, and leave-one-out-ready correlation outputs are covered.
- No automatic Kaggle submission is included.
- Public notebook support starts conservative: metadata registry first, CSV/cache ingestion second.
- Exact LB prediction remains secondary; the plan emphasizes risk tiers and ranking.
