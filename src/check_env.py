"""Verify the environment really works (not just that packages import).

    python run.py check

Exit code 0 = OK, 1 = something is broken (the message says what).
"""
import importlib
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGES = ["numpy", "scipy", "pandas", "sklearn", "joblib", "lightgbm", "xgboost", "matplotlib",
            "onnx", "onnxruntime", "skl2onnx", "serial", "torch"]
DATA = ROOT / "datasets/crop_recommendation_kaggle_atharvaingle/Crop_recommendation.csv"

problems = []
print(f"python    {platform.python_version()}  ({sys.executable})")
print(f"platform  {platform.platform()}  cpu_count={os.cpu_count()}")

mods = {}
for name in PACKAGES:
    try:
        mods[name] = importlib.import_module(name)
        print(f"  ok  {name:<13}{getattr(mods[name], '__version__', '')}")
    except Exception as e:  # noqa: BLE001 - report any import failure
        problems.append(f"import {name}: {e}")
        print(f"  FAIL {name:<12}{e}")

if "torch" in mods:
    print(f"torch     {mods['torch'].__version__}  (CPU build: cuda={mods['torch'].version.cuda})  threads={mods['torch'].get_num_threads()}")
if "xgboost" in mods and "numpy" in mods:
    import numpy as np
    try:
        mods["xgboost"].XGBClassifier(n_estimators=5, tree_method="hist").fit(np.random.rand(200, 7).astype("float32"), np.random.randint(0, 3, 200))
        print("  ok  xgboost trains")
    except Exception as e:  # noqa: BLE001
        problems.append(f"xgboost: {e}")
if "onnxruntime" in mods:
    print(f"  onnxruntime providers: {mods['onnxruntime'].get_available_providers()}  (we use the CPU one)")

if DATA.exists():
    n = sum(1 for _ in open(DATA, encoding="utf-8")) - 1
    print(f"  ok  dataset present ({n} rows)")
    if n != 2200:
        problems.append(f"expected 2200 rows, found {n}")
else:
    problems.append("dataset missing: datasets/crop_recommendation_kaggle_atharvaingle/Crop_recommendation.csv")

print()
for p in problems:
    print("ERROR:", p)
print("status:", "BROKEN" if problems else "OK - everything runs on the CPU")
sys.exit(1 if problems else 0)
