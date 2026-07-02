#!/usr/bin/env python3
"""01_prepare_base.py — download baidu/Unlimited-OCR and patch configs for mlx-vlm.

Why the patch: the base repo declares model_type "unlimited-ocr" (custom code),
which mlx-vlm doesn't recognize. The tensor layout is byte-identical to
DeepSeek-OCR, so re-badging the config routes it through mlx-vlm's existing
`deepseekocr` implementation. Two edits, both community-verified:

  config.json            model_type: "unlimited-ocr" -> "deepseekocr"; drop auto_map
  processor_config.json  processor_class -> "DeepseekOCRProcessor"

Originals are preserved as *.orig for the model-card documentation.
"""
import json
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

BASE_REPO = "baidu/Unlimited-OCR"
LOCAL_DIR = Path("base_model")


def patch_json(path: Path, edits, deletions=()):
    orig = path.with_suffix(path.suffix + ".orig")
    if not orig.exists():
        shutil.copy2(path, orig)
    data = json.loads(path.read_text())
    changed = []
    for key, value in edits.items():
        if data.get(key) != value:
            changed.append(f"{key}: {data.get(key)!r} -> {value!r}")
            data[key] = value
    for key in deletions:
        if key in data:
            changed.append(f"deleted {key}")
            del data[key]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return changed


def main():
    print(f"== Downloading {BASE_REPO} (~6.8GB) ==")
    snapshot_download(
        BASE_REPO,
        local_dir=LOCAL_DIR,
        # weights + configs + tokenizer/processor; skip demo assets & wheels
        allow_patterns=[
            "*.safetensors", "*.json", "*.txt", "*.py", "*.model",
            "tokenizer*", "LICENSE*", "README*",
        ],
        ignore_patterns=["assets/*", "wheel/*", "*.pdf"],
    )

    # sanity: weights present?
    weights = list(LOCAL_DIR.glob("*.safetensors"))
    if not weights:
        sys.exit("No .safetensors found — download incomplete, rerun.")
    total_gb = sum(w.stat().st_size for w in weights) / 1e9
    print(f"   {len(weights)} safetensors file(s), {total_gb:.2f} GB total")

    print("== Patching config.json ==")
    cfg = LOCAL_DIR / "config.json"
    for line in patch_json(cfg, {"model_type": "deepseekocr"}, deletions=["auto_map"]):
        print(f"   {line}")

    proc = LOCAL_DIR / "processor_config.json"
    if proc.exists():
        print("== Patching processor_config.json ==")
        for line in patch_json(proc, {"processor_class": "DeepseekOCRProcessor"}):
            print(f"   {line}")
    else:
        print("   (no processor_config.json in snapshot — mlx-vlm will fall back "
              "to the deepseekocr processor; note this if load fails)")

    print("\nDone. Originals kept as *.orig. Next: bash 02_convert.sh")


if __name__ == "__main__":
    main()
