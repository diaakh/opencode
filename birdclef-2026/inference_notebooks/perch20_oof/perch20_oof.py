"""Perch 2.0 via perch-hoplite — discovery + load."""
import os, sys, subprocess
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "perch-hoplite"])

import perch_hoplite
from perch_hoplite.zoo import model_configs, models
print(f"perch_hoplite version: {getattr(perch_hoplite, '__version__', '?')}")
print(f"ModelConfigName values: {[e.name for e in model_configs.ModelConfigName]}")

# Try to load via load_model_by_name
for name in [e.name for e in model_configs.ModelConfigName]:
    if "perch" in name.lower() or "v2" in name.lower():
        print(f"\n>>> Trying to load {name}")
        try:
            m = model_configs.load_model_by_name(name)
            print(f"  SUCCESS: {type(m)}")
            # Probe
            if hasattr(m, "embed_audio"):
                print(f"  has embed_audio")
            if hasattr(m, "model"):
                print(f"  model: {type(m.model)}")
            # Try forward pass
            import numpy as np
            try:
                out = m.embed(np.zeros(160000, dtype=np.float32))
                print(f"  embed output: {type(out)} {dir(out)[:10]}")
            except Exception as e:
                print(f"  embed fail: {type(e).__name__}: {str(e)[:200]}")
            break
        except Exception as e:
            print(f"  FAIL: {type(e).__name__}: {str(e)[:300]}")
