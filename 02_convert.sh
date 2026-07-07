#!/usr/bin/env bash
# 02_convert.sh — BF16 MLX conversion, smoke test, then the standard quant ladder.
# Smoke-test the conversion before spending time quantizing it.
set -euo pipefail
source .venv/bin/activate

BASE=base_model
OUT=models
mkdir -p "$OUT"

echo "== 1/5 Convert to MLX BF16 =="
python -m mlx_vlm.convert \
    --hf-path "$BASE" \
    --mlx-path "$OUT/unlimited-ocr-bf16-mlx" \
    --dtype bfloat16

echo "== 2/5 Smoke test BF16 (must produce plausible OCR text) =="
python 04_make_corpus.py --smoke-only          # writes corpus/smoke.png + .txt
python - <<'PY'
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

path = "models/unlimited-ocr-bf16-mlx"
model, processor = load(path)
config = load_config(path)
prompt = apply_chat_template(processor, config, "Free OCR.", num_images=1)
out = generate(model, processor, prompt, image="corpus/smoke.png",
               max_tokens=300, temperature=0.0, verbose=False)
text = out.text if hasattr(out, "text") else str(out)
print("--- smoke output (first 400 chars) ---")
print(text[:400])
assert len(text.strip()) > 20, "Smoke test produced no meaningful text — STOP and debug before quantizing."
print("--- smoke test PASSED ---")
PY

# Standard ladder. mlx-vlm 0.6.3's convert has no --skip-vision flag; it skips
# the vision/multimodal tower by default via skip_multimodal_module (matches
# vision_model, sam_model, etc.). Vision-tower float retention is verified after
# conversion by inspecting weight dtypes.
for BITS in 8 6 4; do
  echo "== Quantize ${BITS}-bit =="
  python -m mlx_vlm.convert \
      --hf-path "$BASE" \
      --mlx-path "$OUT/unlimited-ocr-${BITS}bit-mlx" \
      -q --q-bits "$BITS" --q-group-size 64
done

echo "== Disk footprint =="
du -sh "$OUT"/*

echo "Done. Next: python 03_mixed_quant.py (mixed-precision 4-bit)"
