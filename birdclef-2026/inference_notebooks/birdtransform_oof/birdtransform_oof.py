"""BirdTransform model inference on labeled OOF.

pulkitsahu89/birdtransform-birdclef-2026-transformer-model (71MB) — a 234-class
BC2026-specific transformer. Different architecture from BirdMAE/BirdAVES.

If predictions are orthogonal, blends in as another ensemble member.
"""
import os, sys, re, time, json
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

# Find model
MODEL_PATH = None
for cand in [Path("/kaggle/input/birdtransform-birdclef-2026-transformer-model/bird_model.pth"),
             Path("/kaggle/input/datasets/pulkitsahu89/birdtransform-birdclef-2026-transformer-model/bird_model.pth")]:
    if cand.exists():
        MODEL_PATH = cand; break
assert MODEL_PATH
print(f"Model: {MODEL_PATH}")
SPECIES_PATH = MODEL_PATH.parent / "species_list.npy"

# Load
species = np.load(SPECIES_PATH, allow_pickle=True)
state = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
print(f"Species: {len(species)}")
if isinstance(state, dict):
    if "state_dict" in state:
        state = state["state_dict"]
    if "model_state_dict" in state:
        state = state["model_state_dict"]
print(f"State keys (first 10):")
keys = list(state.keys()) if hasattr(state, 'keys') else []
for k in keys[:10]: print(f"  {k}: {state[k].shape if hasattr(state[k], 'shape') else type(state[k])}")
print(f"  ... ({len(keys)} total)")

# Figure out the architecture from state dict
# Looking at keys: typically encoder.X, classifier.X etc.
# Let's see if it's an EfficientNet or AST or custom
input_dim = None
for k, v in state.items():
    if 'conv' in k.lower() or 'patch_embed' in k.lower():
        if hasattr(v, 'shape'):
            print(f"  First conv: {k}: {v.shape}")
            break
