#!/usr/bin/env python3
"""04_make_corpus.py — deterministic synthetic eval corpus with EXACT ground truth.

Why synthetic: rendered pages give perfect ground truth (no annotation noise,
no license questions, fully reproducible from a seed). Three difficulty tiers:

  clean   — 14pt single-column prose (easy)
  dense   — 9pt tight prose, more per page (stress)
  numeric — invoice-style lines heavy on digits/IDs/amounts (where quant
            damage hurts most and is least forgivable)

Each page: corpus/<tier>_<nn>.png + .txt (the exact rendered string).
"""
import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path("corpus")
SEED = 20260701
PAGE_W, PAGE_H, MARGIN = 1400, 1980, 90  # ~A4 at ~170dpi; crisp for OCR

MAC_FONTS = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "/Library/Fonts/Arial.ttf",
]

SUBJECTS = ["The audit team", "A data pipeline", "The committee", "Each vendor",
            "The quarterly report", "Our warehouse", "The reviewer", "A control test"]
VERBS = ["validated", "reconciled", "flagged", "documented", "transferred",
         "sampled", "approved", "escalated"]
OBJECTS = ["the invoice batch", "seventeen ledger entries", "the exception log",
           "all supporting evidence", "the retention schedule", "both trial balances",
           "the access review", "a materiality threshold"]
TAILS = ["before the deadline.", "without further findings.", "under the new policy.",
         "for the second quarter.", "across three regions.", "with minor exceptions.",
         "as required by the standard.", "and archived the results."]

ITEMS = ["Data platform license", "Storage expansion", "Consulting hours",
         "Compute credits", "Support renewal", "Security review", "Training seats",
         "API usage overage"]


def find_font(size, override=None):
    candidates = ([override] if override else []) + MAC_FONTS
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    raise SystemExit("No usable font found — pass --font /path/to/font.ttf")


def sentence(rng):
    return " ".join([rng.choice(SUBJECTS), rng.choice(VERBS),
                     rng.choice(OBJECTS), rng.choice(TAILS)])


def prose_page(rng, sentences):
    paras, i = [], 0
    while i < sentences:
        n = min(rng.randint(3, 5), sentences - i)
        paras.append(" ".join(sentence(rng) for _ in range(n)))
        i += n
    return "\n\n".join(paras)


def numeric_page(rng, rows):
    lines = [f"INVOICE {rng.randint(2024, 2026)}-{rng.randint(1000, 9999)}",
             f"Account 4{rng.randint(100000, 999999)}  PO {rng.randint(10**7, 10**8 - 1)}", ""]
    total = 0
    for _ in range(rows):
        item = rng.choice(ITEMS)
        qty = rng.randint(1, 40)
        price = rng.randint(45, 9500) + rng.choice([0.00, 0.25, 0.50, 0.99])
        amount = round(qty * price, 2)
        total = round(total + amount, 2)
        lines.append(f"{item}  qty {qty}  unit {price:.2f}  amount {amount:.2f}")
    lines += ["", f"TOTAL DUE {total:.2f}", f"Reference {rng.randint(10**9, 10**10 - 1)}"]
    return "\n".join(lines)


def render(text, path, font, line_gap):
    img = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    draw = ImageDraw.Draw(img)
    y = MARGIN
    for raw_line in text.split("\n"):
        # naive word wrap
        words, line = raw_line.split(" "), ""
        for w in words + ["\0"]:
            trial = (line + " " + w).strip() if w != "\0" else None
            if w == "\0" or draw.textlength(trial, font=font) > PAGE_W - 2 * MARGIN:
                draw.text((MARGIN, y), line, fill="black", font=font)
                y += line_gap
                line = w if w != "\0" else ""
            else:
                line = trial
        if raw_line == "":  # paragraph gap already consumed one line
            y += line_gap // 2
    img.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=8, help="pages per tier")
    ap.add_argument("--font", default=None)
    ap.add_argument("--smoke-only", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    rng = random.Random(SEED)

    if args.smoke_only:
        text = prose_page(rng, 6)
        render(text, OUT / "smoke.png", find_font(30, args.font), 44)
        (OUT / "smoke.txt").write_text(text)
        print("wrote corpus/smoke.png (+ ground truth)")
        return

    tiers = [
        ("clean",   lambda: prose_page(rng, 14), find_font(30, args.font), 44),
        ("dense",   lambda: prose_page(rng, 34), find_font(20, args.font), 29),
        ("numeric", lambda: numeric_page(rng, 14), find_font(26, args.font), 40),
    ]
    for name, make, font, gap in tiers:
        for i in range(args.pages):
            text = make()
            stem = f"{name}_{i:02d}"
            render(text, OUT / f"{stem}.png", font, gap)
            (OUT / f"{stem}.txt").write_text(text)
    print(f"wrote {3 * args.pages} pages across 3 tiers to corpus/ (seed {SEED})")


if __name__ == "__main__":
    main()
