#!/usr/bin/env python3
"""03_mixed_quant.py — sensitivity-aware mixed-precision 4-bit.

Philosophy (borrowed from GGUF K-quants, applied to MLX): spend bits where
they matter. Expert/MLP weights tolerate 4-bit; attention projections,
embeddings, and the LM head are the sensitive tensors and get 8-bit; the
vision tower stays unquantized entirely.

Target: file size close to the plain 4-bit, quality closer to 6/8-bit.
The eval ladder (05) is what proves or disproves that — no assertions.

NOTE: this is the most API-sensitive script here. It uses mx.nn.quantize's
class_predicate (a stable MLX interface), but mlx-vlm's save/load helpers have
moved between versions. If it errors, capture the traceback — the rest of the
ladder is independent of this step.
"""
import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

BF16_PATH = Path("models/unlimited-ocr-bf16-mlx")
OUT_PATH = Path("models/unlimited-ocr-4bit-mixed-mlx")

GROUP_SIZE = 64
BITS_DEFAULT = 4        # experts / generic MLP
BITS_SENSITIVE = 8      # attention projections, embeddings, lm_head

# Substrings identifying sensitive modules in the deepseekocr text tower.
SENSITIVE_KEYS = (
    "self_attn", "attn", "q_proj", "k_proj", "v_proj", "o_proj",
    "embed_tokens", "lm_head", "shared_expert",
)
# Vision tower stays float — never quantize these.
VISION_KEYS = ("vision", "sam", "clip", "projector", "image")


def predicate(path: str, module) -> "bool | dict":
    """Per-module quantization policy. Returning False skips the module;
    returning a dict overrides bits/group_size for it."""
    if not hasattr(module, "to_quantized"):
        return False
    p = path.lower()
    if any(k in p for k in VISION_KEYS):
        return False
    if any(k in p for k in SENSITIVE_KEYS):
        return {"group_size": GROUP_SIZE, "bits": BITS_SENSITIVE}
    return {"group_size": GROUP_SIZE, "bits": BITS_DEFAULT}


def main():
    if not BF16_PATH.exists():
        sys.exit("BF16 MLX model missing — run 02_convert.sh first.")

    print("== Loading BF16 model ==")
    from mlx_vlm.utils import load  # local import so failure is legible
    model, processor = load(str(BF16_PATH))

    print("== Applying mixed-precision quantization ==")
    nn.quantize(model, group_size=GROUP_SIZE, bits=BITS_DEFAULT,
                class_predicate=predicate)
    mx.eval(model.parameters())

    print("== Saving ==")
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    try:
        # Preferred: mlx-vlm's own save util keeps sharding/index consistent.
        from mlx_vlm.utils import save_weights
        save_weights(str(OUT_PATH), model, donate_weights=False)
    except Exception:
        # Fallback: flatten and save directly.
        from mlx.utils import tree_flatten
        weights = dict(tree_flatten(model.parameters()))
        mx.save_safetensors(str(OUT_PATH / "model.safetensors"), weights)

    # Copy configs/tokenizer/processor from the BF16 dir, then record the
    # quantization policy (incl. per-layer overrides) in config.json so
    # mlx_vlm.load can reconstruct the right quantized layers.
    for f in BF16_PATH.iterdir():
        if f.suffix in {".json", ".txt", ".model", ".py"} and "safetensors" not in f.name:
            shutil.copy2(f, OUT_PATH / f.name)

    cfg_path = OUT_PATH / "config.json"
    cfg = json.loads(cfg_path.read_text())
    qcfg = {"group_size": GROUP_SIZE, "bits": BITS_DEFAULT}
    # Per-layer overrides: walk modules and record any that differ from default.
    overrides = {}
    for name, module in model.named_modules():
        if hasattr(module, "bits") and getattr(module, "bits", BITS_DEFAULT) != BITS_DEFAULT:
            overrides[name] = {"group_size": getattr(module, "group_size", GROUP_SIZE),
                               "bits": module.bits}
    qcfg.update(overrides)
    cfg["quantization"] = qcfg
    cfg["quantization_config"] = qcfg
    cfg_path.write_text(json.dumps(cfg, indent=2))

    n_over = len(overrides)
    print(f"Saved {OUT_PATH}  (default {BITS_DEFAULT}-bit, {n_over} modules at "
          f"{BITS_SENSITIVE}-bit, vision tower unquantized)")
    print("Verify it loads: python 05_eval.py --variant 4bit-mixed --pages 1")


if __name__ == "__main__":
    main()
