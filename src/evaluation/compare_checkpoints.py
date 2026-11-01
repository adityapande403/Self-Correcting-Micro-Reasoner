"""Checkpoint comparison and statistical table generation."""
import os
import json
from typing import Dict, Any, List

def load_summary(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_comparison_table(
    baseline_summary: Dict[str, Any],
    sft_summary: Dict[str, Any],
    grpo_summary: Dict[str, Any]
) -> str:
    """Generates a clean Markdown comparison table across the three stages."""
    stages = [
        ("Base (Unmodified)", baseline_summary),
        ("SFT (Cold-Start)", sft_summary),
        ("SFT + GRPO (RLVR)", grpo_summary)
    ]

    headers = [
        "Metric",
        "Base (Stage A)",
        "SFT (Stage B)",
        "SFT + GRPO (Stage D)",
        "Delta (GRPO vs Base)",
        "Delta (GRPO vs SFT)"
    ]

    rows = []

    # 1. Exact Accuracy
    acc_base = baseline_summary.get("exact_accuracy", 0.0)
    acc_sft = sft_summary.get("exact_accuracy", 0.0)
    acc_grpo = grpo_summary.get("exact_accuracy", 0.0)
    rows.append([
        "**Exact Accuracy**",
        f"{acc_base:.2%}",
        f"{acc_sft:.2%}",
        f"{acc_grpo:.2%}",
        f"{acc_grpo - acc_base:+.2%}",
        f"{acc_grpo - acc_sft:+.2%}"
    ])

    # 2. Parse Success Rate
    psr_base = baseline_summary.get("parse_success_rate", 0.0)
    psr_sft = sft_summary.get("parse_success_rate", 0.0)
    psr_grpo = grpo_summary.get("parse_success_rate", 0.0)
    rows.append([
        "**Parse Success Rate**",
        f"{psr_base:.2%}",
        f"{psr_sft:.2%}",
        f"{psr_grpo:.2%}",
        f"{psr_grpo - psr_base:+.2%}",
        f"{psr_grpo - psr_sft:+.2%}"
    ])

    # 3. Invalid Output Rate
    ior_base = baseline_summary.get("invalid_output_rate", 0.0)
    ior_sft = sft_summary.get("invalid_output_rate", 0.0)
    ior_grpo = grpo_summary.get("invalid_output_rate", 0.0)
    rows.append([
        "**Invalid Output Rate**",
        f"{ior_base:.2%}",
        f"{ior_sft:.2%}",
        f"{ior_grpo:.2%}",
        f"{ior_grpo - ior_base:+.2%}",
        f"{ior_grpo - ior_sft:+.2%}"
    ])

    # 4. Mean Generated Tokens
    tok_base = baseline_summary.get("mean_output_tokens", 0.0)
    tok_sft = sft_summary.get("mean_output_tokens", 0.0)
    tok_grpo = grpo_summary.get("mean_output_tokens", 0.0)
    rows.append([
        "**Mean Tokens**",
        f"{tok_base:.1f}",
        f"{tok_sft:.1f}",
        f"{tok_grpo:.1f}",
        f"{tok_grpo - tok_base:+.1f}",
        f"{tok_grpo - tok_sft:+.1f}"
    ])

    # 5. Median Tokens
    med_base = baseline_summary.get("median_output_tokens", 0.0)
    med_sft = sft_summary.get("median_output_tokens", 0.0)
    med_grpo = grpo_summary.get("median_output_tokens", 0.0)
    rows.append([
        "**Median Tokens**",
        f"{med_base:.1f}",
        f"{med_sft:.1f}",
        f"{med_grpo:.1f}",
        f"{med_grpo - med_base:+.1f}",
        f"{med_grpo - med_sft:+.1f}"
    ])

    # 6. P95 Tokens
    p95_base = baseline_summary.get("p95_output_tokens", 0.0)
    p95_sft = sft_summary.get("p95_output_tokens", 0.0)
    p95_grpo = grpo_summary.get("p95_output_tokens", 0.0)
    rows.append([
        "**P95 Tokens**",
        f"{p95_base:.1f}",
        f"{p95_sft:.1f}",
        f"{p95_grpo:.1f}",
        f"{p95_grpo - p95_base:+.1f}",
        f"{p95_grpo - p95_sft:+.1f}"
    ])

    # 7. Mean Latency (s)
    lat_base = baseline_summary.get("mean_latency_sec", 0.0)
    lat_sft = sft_summary.get("mean_latency_sec", 0.0)
    lat_grpo = grpo_summary.get("mean_latency_sec", 0.0)
    rows.append([
        "**Mean Latency**",
        f"{lat_base:.3f}s",
        f"{lat_sft:.3f}s",
        f"{lat_grpo:.3f}s",
        f"{lat_grpo - lat_base:+.3f}s",
        f"{lat_grpo - lat_sft:+.3f}s"
    ])

    # 8. Peak VRAM (MB)
    vram_base = baseline_summary.get("peak_vram_mb", 0.0)
    vram_sft = sft_summary.get("peak_vram_mb", 0.0)
    vram_grpo = grpo_summary.get("peak_vram_mb", 0.0)
    rows.append([
        "**Peak VRAM**",
        f"{vram_base:.1f} MB",
        f"{vram_sft:.1f} MB",
        f"{vram_grpo:.1f} MB",
        f"{vram_grpo - vram_base:+.1f} MB",
        f"{vram_grpo - vram_sft:+.1f} MB"
    ])

    # Format Markdown Table
    col_widths = [max(len(row[i]) for row in [headers] + rows) for i in range(len(headers))]
    header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"
    row_lines = [
        "| " + " | ".join(row[i].ljust(col_widths[i]) for i in range(len(headers))) + " |"
        for row in rows
    ]

    return "\n".join([header_line, sep_line] + row_lines)
