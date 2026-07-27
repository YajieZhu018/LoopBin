"""LoopBin — VaDE clustering of chromatin loops (Micro-C + CUT&Tag)."""
import os
# Determinism: MUST be set before TensorFlow is imported by any submodule.
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")
__version__ = "0.1.0"
