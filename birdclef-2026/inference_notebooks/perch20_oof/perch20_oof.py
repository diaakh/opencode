"""Perch 2.0 — try Google's official perch-hoplite wrapper."""
import os, sys, subprocess
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

print("Installing perch-hoplite + deps...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "perch-hoplite",
])
print("Install done")

# Try loading via perch-hoplite
import glob, re, time
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf
print(f"TF: {tf.__version__}")

from perch_hoplite.zoo import zoo_interface, model_configs
print(f"perch-hoplite loaded")
print(f"Available models: {dir(model_configs)}")

# Try to load Perch 2.0
try:
    embed_fn = model_configs.PerchV2.beans_taxa
    print(f"PerchV2 found: {type(embed_fn)}")
except Exception as e:
    print(f"PerchV2 error: {e}")

# Try other variants
for attr in dir(model_configs):
    if "perch" in attr.lower():
        print(f"  attr: {attr}")

# Quick test
print("\nTrying to load model...")
try:
    from perch_hoplite.zoo import model_configs as mc
    # Inspect ModelConfig types
    print(f"ModelConfig classes: {[c for c in dir(mc) if not c.startswith('_')]}")
except Exception as e:
    print(f"Error: {e}")
