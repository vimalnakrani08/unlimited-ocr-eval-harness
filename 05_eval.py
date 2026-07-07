#!/usr/bin/env python3
"""05_eval.py — the measurement everyone skipped.

Runs every model variant over the corpus, computes CER/WER against exact
ground truth, records tokens/sec and peak memory, writes:

  results/results.json      (machine-readable, per-page and per-tier metrics)
  results/results_table.md  (human-readable summary)

CER = Levenshtein(char) / len(ref); WER analogous on whitespace tokens.
Output normalization strips grounding tokens / markdown decoration so we
measure RECOGNITION quality, not formatting style. Normalization is applied
identically to reference and hypothesis. Case and punctuation are preserved
(they are part of OCR quality).

Repetition loops are not special-cased away: max_tokens caps the damage and
the resulting high CER is an honest quality signal for that quant.
"""
import argparse
import gc
import json
import re
import time
from pathlib import Path

import mlx.core as mx
from rapidfuzz.distance import Levenshtein

VARIANTS = {
    "bf16":       "models/unlimited-ocr-bf16-mlx",
    "8bit":       "models/unlimited-ocr-8bit-mlx",
    "6bit":       "models/unlimited-ocr-6bit-mlx",
    "4bit":       "models/unlimited-ocr-4bit-mlx",
    "4bit-mixed": "models/unlimited-ocr-4bit-mixed-mlx",
    # Diagnostic ablation (not a shipped variant): CLI 4bit with a float
    # vision->language projector — isolates the projector's effect.
    "4bit-fp-projector": "models/unlimited-ocr-4bit-fp-projector-mlx",
}
PROMPT = "document parsing."  # 'Free OCR.' degenerates into a repetition loop on
                              # this mlx-vlm build; 'document parsing.' reads correctly.
                              # Grounding tokens/coords it emits are stripped by
                              # normalize().
MAX_TOKENS = 2600
# 'document parsing.' emits each detected region as a structural unit:
#   <|det|>LABEL [x1,y1,x2,y2]<|/det|>CONTENT
# REGION_LABEL removes the whole preamble (optional <|det|>, the region-type
# word, the 4-int bbox, closing <|/det|>) so absolute CER reflects recognition,
# not layout labels. It matches ONLY that exact structure — body text with a
# bracketed list is preserved because it lacks the trailing <|/det|> anchor.
REGION_LABEL = re.compile(
    r"(?:<\|det\|>)?\s*[A-Za-z_]+\s*\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]\s*<\|/det\|>")
GROUNDING = re.compile(r"<\|[^|]*\|>")
COORDS = re.compile(r"\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]")
MD_DECOR = re.compile(r"[*_#`|]+")


def normalize(text: str) -> str:
    text = REGION_LABEL.sub(" ", text)   # strip structural region labels first
    text = GROUNDING.sub(" ", text)
    text = COORDS.sub(" ", text)
    text = MD_DECOR.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _bytes_to_unicode():
    """GPT-2 byte<->unicode table (inverted: unicode char -> byte)."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_U2B = _bytes_to_unicode()


def bytelevel_decode(text: str) -> str:
    """Complete the byte-level BPE detokenization mlx-vlm 0.6.3's slow
    LlamaTokenizer skips (space -> 'Ġ', newline -> 'Ċ', ...). Applied to the
    hypothesis only — the reference .txt is already UTF-8 — before normalize(),
    identically for every variant. Runs of byte-level chars decode as UTF-8;
    anything else (special tokens, literal newlines) is preserved."""
    out, buf = [], bytearray()
    for ch in text:
        if ch in _U2B:
            buf.append(_U2B[ch])
        else:
            if buf:
                out.append(buf.decode("utf-8", errors="replace")); buf = bytearray()
            out.append(ch)
    if buf:
        out.append(buf.decode("utf-8", errors="replace"))
    return "".join(out)


def peak_mem_gb():
    for fn in ("get_peak_memory",):
        try:
            return getattr(mx, fn)() / 1e9
        except AttributeError:
            pass
    try:
        return mx.metal.get_peak_memory() / 1e9
    except Exception:
        return None


def reset_peak():
    for reset in (getattr(mx, "reset_peak_memory", None),
                  getattr(getattr(mx, "metal", None), "reset_peak_memory", None)):
        if reset:
            try:
                reset()
                return
            except Exception:
                pass


def eval_variant(name, path, pages, results):
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    print(f"\n=== {name}  ({path}) ===")
    reset_peak()
    model, processor = load(path)
    config = load_config(path)
    formatted = apply_chat_template(processor, config, PROMPT, num_images=1)

    size_gb = sum(f.stat().st_size for f in Path(path).glob("*.safetensors")) / 1e9
    rows = []
    for img in pages:
        ref = normalize(Path(img).with_suffix(".txt").read_text())
        t0 = time.time()
        out = generate(model, processor, formatted, image=str(img),
                       max_tokens=MAX_TOKENS, temperature=0.0, verbose=False)
        dt = time.time() - t0
        text = out.text if hasattr(out, "text") else str(out)
        hyp = normalize(bytelevel_decode(text))   # complete detokenization first
        n_tok = getattr(out, "generation_tokens", None) or max(1, len(text) // 4)
        cer = Levenshtein.distance(ref, hyp) / max(1, len(ref))
        wer = Levenshtein.distance(ref.split(), hyp.split()) / max(1, len(ref.split()))
        # Loop diagnostics (additive; does not affect scoring): a hypothesis
        # much longer than its reference indicates runaway repetition.
        len_ratio = len(hyp) / max(1, len(ref))
        tier = Path(img).stem.rsplit("_", 1)[0]
        rows.append({"page": Path(img).name, "tier": tier, "cer": cer,
                     "wer": wer, "len_ratio": len_ratio,
                     "loop_flag": len_ratio > 1.5,
                     "seconds": dt, "tok_per_s": n_tok / dt})
        print(f"  {Path(img).name:>16}  CER {cer:6.2%}  WER {wer:6.2%}  {n_tok/dt:5.1f} tok/s")

    results[name] = {
        "path": path, "file_size_gb": round(size_gb, 3),
        "peak_mem_gb": round(peak_mem_gb() or -1, 2),
        "pages": rows,
        "by_tier": {
            tier: {
                "cer": sum(r["cer"] for r in rows if r["tier"] == tier)
                       / max(1, sum(1 for r in rows if r["tier"] == tier)),
                "wer": sum(r["wer"] for r in rows if r["tier"] == tier)
                       / max(1, sum(1 for r in rows if r["tier"] == tier)),
            }
            for tier in sorted({r["tier"] for r in rows})
        },
        "overall_cer": sum(r["cer"] for r in rows) / len(rows),
        "overall_wer": sum(r["wer"] for r in rows) / len(rows),
        "mean_tok_per_s": sum(r["tok_per_s"] for r in rows) / len(rows),
    }
    del model, processor
    gc.collect()
    mx.clear_cache() if hasattr(mx, "clear_cache") else None


def write_table(results, out_md):
    lines = ["| Variant | Size (GB) | Overall CER | Overall WER | clean CER | dense CER | numeric CER | Max len ratio | Flagged pages | tok/s | Peak mem (GB) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        bt = r["by_tier"]
        max_lr = max((p.get("len_ratio", 0.0) for p in r["pages"]), default=0.0)
        flagged = sum(1 for p in r["pages"] if p.get("loop_flag"))
        lines.append(
            f"| {name} | {r['file_size_gb']:.2f} | {r['overall_cer']:.2%} | "
            f"{r['overall_wer']:.2%} | {bt.get('clean',{}).get('cer',0):.2%} | "
            f"{bt.get('dense',{}).get('cer',0):.2%} | {bt.get('numeric',{}).get('cer',0):.2%} | "
            f"{max_lr:.2f} | {flagged}/{len(r['pages'])} | "
            f"{r['mean_tok_per_s']:.1f} | {r['peak_mem_gb']:.1f} |")
    if "4bit-fp-projector" in results:
        lines += [
            "",
            "> `4bit-fp-projector` is a diagnostic ablation (not a published variant): "
            "the CLI 4-bit conversion with only the vision→language projector kept "
            "float, everything else identical (verified by dtype-diff). Single "
            "deterministic run; near the 4-bit stability cliff, page-level outcomes are "
            "sensitive to small precision changes, so read per-tier differences with "
            "caution.",
        ]
    out_md.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS), default=None,
                    help="run a single variant (default: all that exist)")
    ap.add_argument("--pages", type=int, default=None, help="cap pages (debug)")
    args = ap.parse_args()

    pages = sorted(p for p in Path("corpus").glob("*.png") if p.stem != "smoke")
    if not pages:
        raise SystemExit("No corpus — run 04_make_corpus.py first.")
    if args.pages:
        pages = pages[: args.pages]

    out_dir = Path("results"); out_dir.mkdir(exist_ok=True)
    results_path = out_dir / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    todo = {args.variant: VARIANTS[args.variant]} if args.variant else VARIANTS
    for name, path in todo.items():
        if not Path(path).exists():
            print(f"-- skipping {name}: {path} not found")
            continue
        eval_variant(name, path, pages, results)
        results_path.write_text(json.dumps(results, indent=2))   # incremental save

    write_table(results, out_dir / "results_table.md")
    print(f"\nWrote {results_path} and results/results_table.md")
    if "bf16" in results:
        base = results["bf16"]["overall_cer"]
        print(f"\nDegradation vs BF16 baseline (CER {base:.2%}):")
        for name, r in results.items():
            if name != "bf16":
                print(f"  {name:>10}: +{(r['overall_cer'] - base):.2%} absolute")


if __name__ == "__main__":
    main()
