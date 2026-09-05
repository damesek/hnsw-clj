#!/usr/bin/env python3
"""Create the English Slack benchmark chart without third-party packages."""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape


WIDTH = 1600
HEIGHT = 1050
OUTPUT = Path(__file__).with_name("hnsw_clj_vs_hnswlib_slack.svg")

BACKGROUND = "#08111F"
PANEL = "#101D31"
PANEL_ALT = "#14243C"
BORDER = "#263A58"
TEXT = "#F8FAFC"
MUTED = "#A9B8CE"
CLOJURE = "#55D6C2"
PYTHON = "#F2C14E"
POSITIVE = "#70D99B"
WARNING = "#FF9F6E"


def rect(x: float, y: float, width: float, height: float, fill: str,
         radius: float = 18, stroke: str | None = None) -> str:
    border = f' stroke="{stroke}" stroke-width="2"' if stroke else ""
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
        f'rx="{radius}" fill="{fill}"{border}/>'
    )


def label(x: float, y: float, value: str, size: int = 26, color: str = TEXT,
          weight: int = 500, anchor: str = "start") -> str:
    return (
        f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}" '
        f'font-family="Inter, Arial, Helvetica, sans-serif">{escape(value)}</text>'
    )


def benchmark_bar(y: float, name: str, value: int, color: str,
                  bar_x: float = 330, max_width: float = 640) -> list[str]:
    width = max_width * value / 22000
    return [
        label(300, y + 28, name, 23, MUTED, 600, "end"),
        rect(bar_x, y, max_width, 40, "#1B2B44", 10),
        rect(bar_x, y, width, 40, color, 10),
        label(bar_x + width + 18, y + 29, f"{value:,}", 24, TEXT, 700),
    ]


def build_bar(y: float, name: str, value: float, color: str) -> list[str]:
    bar_x = 1130
    max_width = 345
    width = max_width * value / 11.0
    return [
        label(bar_x, y - 12, name, 22, MUTED, 600),
        rect(bar_x, y, max_width, 38, "#1B2B44", 9),
        rect(bar_x, y, width, 38, color, 9),
        label(
            bar_x + width - 12,
            y + 28,
            f"{value:.2f} s",
            22,
            BACKGROUND,
            800,
            "end",
        ),
    ]


def card(x: float, title: str, headline: str, detail: str, accent: str) -> list[str]:
    return [
        rect(x, 150, 480, 120, PANEL_ALT, 18, BORDER),
        label(x + 24, 183, title, 17, MUTED, 700),
        label(x + 24, 225, headline, 34, accent, 800),
        label(x + 24, 253, detail, 18, MUTED, 500),
    ]


def main() -> None:
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}">',
        rect(0, 0, WIDTH, HEIGHT, BACKGROUND, 0),
        label(50, 66, "HNSW-CLJ vs Python hnswlib", 42, TEXT, 800),
        label(
            50,
            108,
            "Full Hungarian Bible · 31,126 verses · MPNet 768D · k=10 · efSearch=50 · Apple Silicon",
            22,
            MUTED,
            500,
        ),
    ]

    elements.extend(card(
        50,
        "10-THREAD SEARCH THROUGHPUT",
        "+23.5% HNSW-CLJ",
        "19.9k vs 16.1k queries/sec",
        POSITIVE,
    ))
    elements.extend(card(
        560,
        "SINGLE-QUERY LATENCY",
        "−42.9% HNSW-CLJ",
        "0.201 ms vs 0.352 ms",
        POSITIVE,
    ))
    elements.extend(card(
        1070,
        "INDEX BUILD",
        "Python is 1.96× faster",
        "5.16 s vs 10.10 s",
        WARNING,
    ))

    elements.extend([
        rect(50, 300, 1000, 460, PANEL, 20, BORDER),
        label(80, 345, "Search performance", 29, TEXT, 750),
        label(80, 377, "10 threads · QPS · higher is better", 19, MUTED, 500),
        label(80, 430, "Hungarian → Hungarian", 24, TEXT, 700),
    ])
    elements.extend(benchmark_bar(450, "HNSW-CLJ", 19668, CLOJURE))
    elements.extend(benchmark_bar(508, "Python hnswlib", 16169, PYTHON))
    elements.extend([
        label(80, 610, "English → Hungarian", 24, TEXT, 700),
    ])
    elements.extend(benchmark_bar(630, "HNSW-CLJ", 20096, CLOJURE))
    elements.extend(benchmark_bar(688, "Python hnswlib", 16023, PYTHON))

    elements.extend([
        rect(1080, 300, 470, 460, PANEL, 20, BORDER),
        label(1110, 345, "Index build", 29, TEXT, 750),
        label(1110, 377, "Average time · lower is better", 19, MUTED, 500),
    ])
    elements.extend(build_bar(435, "HNSW-CLJ", 10.10, CLOJURE))
    elements.extend(build_bar(535, "Python hnswlib", 5.16, PYTHON))
    elements.extend([
        label(1110, 635, "Exact measurements", 20, TEXT, 700),
        label(1110, 670, "HU→HU: 9.86 s / 5.15 s", 19, MUTED, 500),
        label(1110, 703, "EN→HU: 10.34 s / 5.17 s", 19, MUTED, 500),
        label(1110, 735, "Order: HNSW-CLJ / Python", 16, MUTED, 500),
    ])

    elements.extend([
        rect(50, 790, 1500, 210, PANEL_ALT, 20, BORDER),
        label(80, 835, "Accuracy: effectively equal", 28, TEXT, 750),
        label(80, 872, "Recall@10", 18, MUTED, 700),
        label(330, 872, "HNSW-CLJ", 18, CLOJURE, 700),
        label(560, 872, "Python hnswlib", 18, PYTHON, 700),
        label(820, 872, "Difference", 18, MUTED, 700),
        label(80, 915, "Hungarian → Hungarian", 21, TEXT, 600),
        label(330, 915, "99.72%", 21, TEXT, 700),
        label(560, 915, "99.84%", 21, TEXT, 700),
        label(820, 915, "−0.12 percentage points", 21, MUTED, 600),
        label(80, 955, "English → Hungarian", 21, TEXT, 600),
        label(330, 955, "89.00%", 21, TEXT, 700),
        label(560, 955, "88.82%", 21, TEXT, 700),
        label(820, 955, "+0.18 percentage points", 21, MUTED, 600),
        label(1120, 905, "5,000 queries", 19, TEXT, 700),
        label(1120, 938, "Median of 5 measured runs", 18, MUTED, 500),
        label(1120, 970, "Identical HNSW parameters", 18, MUTED, 500),
        "</svg>",
    ])
    OUTPUT.write_text("\n".join(elements), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
