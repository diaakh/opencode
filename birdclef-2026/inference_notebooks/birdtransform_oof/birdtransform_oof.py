"""Print full BirdTransform architecture so we can rebuild."""
import torch, numpy as np
from pathlib import Path

MODEL_PATH = next(Path("/kaggle/input").rglob("bird_model.pth"))
state = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
print(f"Total keys: {len(state)}")
for k, v in state.items():
    shape = tuple(v.shape) if hasattr(v, 'shape') else "(?)"
    print(f"  {k}: {shape}")
