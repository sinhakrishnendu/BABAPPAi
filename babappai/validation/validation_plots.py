"""Lightweight publication-ready plot writers (SVG)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict


def write_regime_bar_svg(regime_counts: Dict[str, int], out_path: Path, title: str) -> None:
    labels = list(regime_counts.keys())
    values = [int(regime_counts[k]) for k in labels]
    max_value = max(values) if values else 1

    width = 900
    height = 480
    margin = 70
    plot_width = width - 2 * margin
    plot_height = height - 2 * margin
    bar_width = plot_width / max(len(labels), 1) * 0.65
    gap = plot_width / max(len(labels), 1) * 0.35

    bars = []
    for idx, (label, value) in enumerate(zip(labels, values)):
        x = margin + idx * (bar_width + gap) + gap / 2
        bar_h = (value / max_value) * (plot_height - 30)
        y = margin + plot_height - bar_h
        bars.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_h:.2f}" '
            'fill="#1f77b4" stroke="#0f3556" stroke-width="1"/>'
        )
        bars.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{y - 8:.2f}" '
            'font-size="13" text-anchor="middle" fill="#222">'
            f"{value}</text>"
        )
        bars.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{height - margin + 20:.2f}" '
            'font-size="11" text-anchor="middle" fill="#222">'
            f"{label}</text>"
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="{width/2:.1f}" y="30" text-anchor="middle" font-size="18" fill="#111">{title}</text>
<line x1="{margin}" y1="{margin + plot_height}" x2="{margin + plot_width}" y2="{margin + plot_height}" stroke="#333" stroke-width="1.2"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{margin + plot_height}" stroke="#333" stroke-width="1.2"/>
{''.join(bars)}
</svg>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(svg)

