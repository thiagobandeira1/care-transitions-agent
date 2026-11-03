"""Figure 1: the agent's fixed workflow, one run per discharge alert.

uv run --with matplotlib python paper/figure1.py   # writes paper/figure1.svg and paper/figure1.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).parent
RULES, MODEL, PERSON = "#2f6db5", "#d9822b", "#3a8f5c"  # deterministic code, local model, a person
INK, MUTED = "#1f2933", "#52606d"

# (step, title, detail lines, kind); two rows of four
STEPS = [
    ("1", "Ingest", ["payer discharge alert", "→ one event per stay"], RULES),
    ("2", "EHR context", ["problem list and", "active medications"], RULES),
    ("3", "Score", ["LACE + Medicare", "under-65 flag"], RULES),
    ("4", "Extract", ["discharge summary → JSON", "(skipped if there is none)"], MODEL),
    ("5", "Plan", ["tier, TCM deadlines and", "check-ins; model explains"], RULES),
    ("6", "Draft", ["call script, provider brief", "+ grounding check"], MODEL),
    ("7", "Human review", ["approve, edit or reject", "(the run pauses here)"], PERSON),
    ("8", "Track", ["decisions and outcomes", "in local SQLite"], RULES),
]
W, H, GAP_X, ROW_Y = 2.45, 1.05, 0.55, (2.35, 0.55)
WIDTH = 0.6 + 4 * W + 3 * GAP_X


def box_xy(i: int) -> tuple[float, float]:
    row, col = divmod(i, 4)
    return 0.3 + col * (W + GAP_X), ROW_Y[row]


def main() -> None:
    fig, ax = plt.subplots(figsize=(WIDTH, 4.4))
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(-0.55, 3.75)
    ax.axis("off")

    for i, (num, title, lines, color) in enumerate(STEPS):
        x, y = box_xy(i)
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                W,
                H,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor="white",
                edgecolor=color,
                linewidth=2.2,
            )
        )
        ax.text(
            x + 0.14,
            y + H - 0.24,
            f"{num}  {title}",
            fontsize=11.5,
            weight="bold",
            color=color,
            va="center",
        )
        for k, line in enumerate(lines):
            ax.text(x + 0.14, y + H - 0.55 - 0.24 * k, line, fontsize=9.0, color=INK, va="center")

    def arrow(a: tuple[float, float], b: tuple[float, float], **kw: object) -> None:
        ax.add_patch(
            FancyArrowPatch(
                a, b, arrowstyle="-|>", mutation_scale=13, color=MUTED, linewidth=1.4, **kw
            )
        )

    for i in range(3):  # along each row
        for row in (0, 4):
            x, y = box_xy(row + i)
            arrow((x + W, y + H / 2), (x + W + GAP_X, y + H / 2))
    # row 1 -> row 2 through the gap between the rows, so the line crosses no box
    x4, y4 = box_xy(3)
    x5, y5 = box_xy(4)
    mid = (y4 + y5 + H) / 2
    ax.plot([x4 + W / 2, x4 + W / 2, x5 + W / 2], [y4, mid, mid], color=MUTED, linewidth=1.4)
    arrow((x5 + W / 2, mid), (x5 + W / 2, y5 + H))
    x8, y8 = box_xy(7)
    note = "daily job: patients with check-ins due get a short new run (none stays open 30 days)"
    ax.annotate(
        note,
        xy=(x8 + W / 2, y8),
        xytext=(x8 + W, -0.3),
        fontsize=8.8,
        color=MUTED,
        ha="right",
        va="center",
        arrowprops={"arrowstyle": "-", "color": MUTED, "linestyle": "--", "linewidth": 1.1},
    )

    for k, (label, color) in enumerate(
        [("deterministic code", RULES), ("local language model", MODEL), ("a person", PERSON)]
    ):
        lx = 0.3 + k * 2.45
        ax.add_patch(
            FancyBboxPatch(
                (lx, 3.5),
                0.32,
                0.18,
                boxstyle="round,pad=0.01",
                facecolor="white",
                edgecolor=color,
                linewidth=2.2,
            )
        )
        ax.text(lx + 0.45, 3.59, label, fontsize=9.5, color=INK, va="center")

    fig.savefig(OUT / "figure1.svg", bbox_inches="tight")
    fig.savefig(OUT / "figure1.png", dpi=300, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
