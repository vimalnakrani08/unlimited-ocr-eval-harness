# Unlimited-OCR quantization eval harness

A small, reproducible harness for measuring **how much quantization costs an OCR
model** — in character/word error rate against *exact* ground truth, with
runaway-repetition ("loop") diagnostics. Built to quantize
[baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) for Apple
Silicon (MLX) and to publish a per-quant quality ladder whose numbers are
interpretable rather than assumed.

The idea is simple: instead of scoring OCR against noisy real-world annotations
(where error rates blow past 100% and per-bit degradation becomes unreadable),
render a **synthetic corpus from known strings**. Ground truth is then exact, so
CER/WER is bounded and the cost of each quantization level is directly readable.

## What's here

| Script | What it does |
|---|---|
| `01_prepare_base.py` | Fetch the base model weights/config locally. |
| `02_convert.sh` | BF16 MLX conversion, a smoke test, then the 8/6/4-bit quant ladder. |
| `03_mixed_quant.py` | Sensitivity-aware mixed-precision 4-bit (4-bit experts/MLP, 8-bit attention/embeddings/head, vision tower unquantized). |
| `04_make_corpus.py` | Generate the deterministic synthetic corpus: three difficulty tiers (clean prose, dense small-font, digit-heavy) × N pages, each as `corpus/<tier>_<nn>.png` + `.txt` (the exact rendered string). |
| `05_eval.py` | Run every variant over the corpus and score CER/WER + loop diagnostics against ground truth; write `results/results.json` and a summary table. |

### Methodology discipline

The measurement is the point, so the rules are strict:

- **Normalization is applied identically** to reference and hypothesis (it strips
  grounding tokens, bounding-box coordinates, and markdown decoration so CER
  reflects *recognition*, not layout markup — never body text).
- **Repetition suppression is off by default**, deliberately, so quantization
  instability shows up as honest high CER and visible loop pages instead of being
  masked. A page whose output runs longer than 1.5× its reference is flagged as a
  loop; `CER excl. loop pages` is reported alongside the headline so one collapsed
  page is visible rather than silently inflating the average.
- Any change to prompt, normalization, token cap, corpus, or decoding parameters
  invalidates prior runs — re-run every variant, never mix settings.

## The MLX runner (important scope note)

This harness is split into a **reusable, backend-agnostic core** and an
**MLX-specific runner**:

- **Reusable core:** the corpus generator (`04_make_corpus.py`) and the scoring
  in `05_eval.py` — `normalize()`, the CER/WER computation (rapidfuzz
  Levenshtein), and the length-ratio / loop-flag diagnostics. None of this is
  tied to a particular inference engine.
- **MLX-specific runner:** the model loading and generation path in `05_eval.py`
  uses `mlx-vlm` (`mlx_vlm.generate`, MLX peak-memory queries) and includes a
  byte-level detokenization step (`bytelevel_decode`) that compensates for a
  quirk in how `mlx-vlm 0.6.3` loads this model's tokenizer. `01_prepare_base.py`,
  `02_convert.sh`, and `03_mixed_quant.py` are MLX/mlx-vlm conversion tooling.

So this repository, as-is, runs on **MLX (Apple Silicon)**. To score a different
backend, keep the corpus and the scoring functions and replace the generation
call — do not carry the `bytelevel_decode` step to a backend that detokenizes
correctly (it is specific to the mlx-vlm path here).

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python 01_prepare_base.py            # fetch base weights/config
bash   02_convert.sh                 # BF16 + 8/6/4-bit ladder (+ smoke test)
python 03_mixed_quant.py             # mixed-precision 4-bit variant
python 04_make_corpus.py             # render the synthetic corpus
python 05_eval.py                    # score all variants -> results/
```

`05_eval.py --variant <name>` runs a single variant (useful if a later variant
would otherwise run out of memory in one process); `04_make_corpus.py --font
<path>` overrides font selection if the default macOS font probe finds nothing.

Requires Apple Silicon with enough unified memory for the largest variant
(the reference BF16 conversion is the heaviest step).

## What it measured

These scripts produced two published per-quant ladders for baidu/Unlimited-OCR,
each scored the same way against the same corpus, grouped in one collection:

- **Collection:** <https://huggingface.co/collections/vimalnakrani/unlimited-ocr-mlx-quants-with-measured-eval-ladder>
- **MLX** ladder (this harness): `vimalnakrani/unlimited-ocr-{bf16,8bit,6bit,4bit,4bit-mixed}-mlx`
- **GGUF** ladder (a separate llama.cpp runner over the same corpus and scoring):
  <https://huggingface.co/vimalnakrani/unlimited-ocr-gguf>

Each model card reports the measured degradation vs its BF16 baseline, per-tier
error rates, and loop diagnostics.

## Limitations

- **Synthetic, single-page corpus:** clean renders, three fonts, English. It is
  designed to make per-bit degradation *readable*, not to represent real-world
  scans (skew, noise, handwriting, other scripts) — validate on your own
  distribution before trusting a low-bit variant in production.
- **MLX runner** (see above): the generation path and one detokenization step are
  specific to MLX / mlx-vlm 0.6.3.
- Error figures are specific to this model, this corpus, and these tool versions;
  re-run for anything else.

## License

MIT — see [LICENSE](LICENSE). The base model is © Baidu under the MIT License;
this repository contains evaluation tooling only, no model weights.
