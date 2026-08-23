#!/usr/bin/env python3
"""Generate charts for the issue #11 (bnb_int8 F16 on Apple Silicon) findings report.

Reads dequant_medians.json (criterion medians, saved from target/criterion) and
the paired A/B deltas, and writes PNG charts into this folder.
Run: uv run --with matplotlib plots.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

M_TO = {
    "awq_int4": "AWQ INT4",
    "bnb_int8": "BnB INT8",
    "bnb_nf4": "BnB NF4",
    "fp8_fine_grained": "FP8 fine-grained",
    "fp8_per_tensor": "FP8 per-tensor",
    "gguf_q4_k": "GGUF Q4_K",
    "gptq_int4": "GPTQ INT4",
}

# Families included in the seven-kernel F16/BF16 ratio analysis (drop gguf_q8_0_ollama
# which is a tiny µs fixture, not a member of the batch dequant families).
RATIO_FAMS = ["gptq_int4", "awq_int4", "bnb_int8", "bnb_nf4", "fp8_fine_grained", "fp8_per_tensor", "gguf_q4_k"]


def load_medians():
    with open(os.path.join(HERE, "dequant_medians.json")) as f:
        return json.load(f)


def ratio_chart(medians):
    """F16 vs BF16 ratio across the seven dequant families on M3, with the
    2-3x band that x86-64 and server aarch64 showed, for comparison."""
    fams = RATIO_FAMS
    ratios = []
    for fam in fams:
        r = medians[fam]["_f16"]["ms"] / medians[fam]["bf16"]["ms"]
        ratios.append(r)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(fams))
    colors = ["#c0392b" if r > 1.05 else "#27ae60" for r in ratios]
    bars = ax.bar(x, ratios, color=colors, edgecolor="black", alpha=0.9)
    ax.axhline(1.0, color="black", lw=1.2, ls="-")
    ax.axhspan(2.0, 3.11, color="#f5b041", alpha=0.15, label="x86-64 / server-aarch64 F16-vs-BF16 band (2.0x-3.11x)")
    for xi, r in zip(x, ratios):
        ax.text(xi, r + 0.05, f"{r:.2f}x", ha="center", fontweight="bold")

    ax.set_xticks(list(x))
    ax.set_xticklabels([M_TO[f] for f in fams], rotation=20, ha="right")
    ax.set_ylabel("F16 median / BF16 median (ratio)")
    ax.set_title("Dequant kernel cost: F16 vs BF16 output on Apple M3 Pro\n(shipped v0.7.7 migrated kernels, criterion 100-sample medians, 4096x11008 fixture)")
    ax.legend(loc="upper left")
    ax.set_ylim(0, 3.2)
    ax.grid(axis="y", ls=":", alpha=0.5)
    fig.tight_layout()
    out = os.path.join(HERE, "00-findings_v0-f16_vs_bf16_ratio.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)

    # Log the numbers so they land in the report.
    print("\nF16/BF16 ratio (M3):")
    for fam, r in zip(fams, ratios):
        print(f"  {fam:18s} {r:.2f}x")


def median_chart(medians):
    """Grouped absolute medians per family (BF16 / F16 / F32) — the within-binary
    widths, showing how F16 compares on M3."""
    fams = RATIO_FAMS
    widths = ["bf16", "_f16", "_f32"]
    labels = ["BF16", "F16", "F32"]
    data = {f: [medians[f][w]["ms"] for w in widths] for f in fams}

    x = range(len(fams))
    n = len(widths)
    w = 0.25
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, (width, lab) in enumerate(zip(widths, labels)):
        vals = [data[f][i] for f in fams]
        ax.bar([xi + (i - 1) * w for xi in x], vals, width=w, label=lab, edgecolor="black")

    ax.set_xticks(list(x))
    ax.set_xticklabels([M_TO[f] for f in fams], rotation=20, ha="right")
    ax.set_ylabel("median time (ms)")
    ax.set_title("Absolute dequant median per output width on Apple M3 Pro\n(shipped v0.7.7 kernels, criterion, 4096x11008 fixture)\nNote: BnB INT8 F16 (13.5ms) is FASTER than its BF16 (14.4ms) — inverts the other platforms")
    ax.legend()
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    out = os.path.join(HERE, "00-findings_v0-absolute_medians.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


def ab_delta_chart():
    """Paired A/B deltas: candidate (pre-migration) vs baseline (migrated), 3 runs.
    Positive = candidate (pre-migration) slower = migrated faster. The two bnb_int8
    arms move opposite directions: f16 +, f32 -."""
    runs = {
        "bnb_int8_f16": [4.59, 5.06, 4.83],
        "bnb_int8_f32": [-5.27, -5.51, -4.77],
        "bnb_int8_bf16": [-0.54, -0.33, -0.25],
    }
    import statistics

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["bnb_int8 F16", "bnb_int8 F32", "bnb_int8 BF16"]
    colors = ["#c0392b", "#27ae60", "#7f8c8d"]
    means = [statistics.mean(runs[k]) for k in runs]
    sds = [statistics.pstdev(runs[k]) for k in runs]
    x = range(len(labels))
    ax.bar(x, means, yerr=[s * 1.96 for s in sds], capsize=5, color=colors, edgecolor="black",
           error_kw=dict(lw=1.5))
    ax.axhline(0, color="black", lw=1)
    ax.axhspan(-2.5, 2.5, color="gray", alpha=0.15, label="M3 noise floor +/-2.5%")
    for xi, m in zip(x, means):
        ax.text(xi, m + (0.4 if m >= 0 else -0.9), f"{m:+.2f}%", ha="center", fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("paired delta: pre-migration vs migrated (baseline)  [%]")
    ax.set_title("Method A paired A/B on Apple M3 Pro (3 runs, threshold 2.5%)\nPositive = pre-migration slower = MIGRATED faster. F16 and F32 move in OPPOSITE directions.")
    ax.legend(loc="upper left")
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    out = os.path.join(HERE, "00-findings_v0-paired_ab_delta.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)
    for l, m in zip(labels, means):
        print(f"  {l}: mean {m:+.2f}%")


def main():
    medians = load_medians()
    ratio_chart(medians)
    median_chart(medians)
    ab_delta_chart()


if __name__ == "__main__":
    main()
