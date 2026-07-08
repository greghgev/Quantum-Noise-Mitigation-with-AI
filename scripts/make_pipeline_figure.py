"""
Genera figures/pipeline_tfm.png — diagrama del pipeline GEM → QPU → REM en
lenguaje llano, para GUIA_TUTOR.md, la memoria y presentaciones.

Uso:
    conda run -n tfm python scripts/make_pipeline_figure.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import FIGURES_PATH  # noqa: E402

# Paleta categórica validada (CVD-safe) + tinta
BLUE, AQUA, YELLOW, VIOLET = "#2a78d6", "#1baf7a", "#eda100", "#4a3aa7"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE = "#fcfcfb"

BOXES = [
    # (x, color, título, subtítulo técnico, explicación llana)
    (0.5, MUTED, "Circuito\ncuántico", "el 'programa'\ndel usuario",
     "Lo que queremos\nejecutar sin errores"),
    (3.1, BLUE, "Modelo 1 — GEM", "Graph Transformer\npredice Δ",
     "Predice cuánto se\nequivocará la máquina,\nANTES de ejecutar"),
    (5.7, VIOLET, "Máquina IBM", "QPU Heron\n(ibm_kingston)",
     "Ejecuta el circuito\ntal cual, con sus\nerrores físicos"),
    (8.3, AQUA, "Modelo 2 — REM", "GNN + álgebra\nmatrix-free",
     "Limpia los errores\ndel sensor de lectura,\nDESPUÉS de ejecutar"),
    (10.9, YELLOW, "Resultado\ncorregido", "⟨O⟩ = ruidoso − Δ",
     "Una sola ejecución,\nresultado limpio"),
]

W, H, Y = 2.1, 1.55, 2.2


def main():
    fig, ax = plt.subplots(figsize=(13.2, 4.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for x, color, title, tech, plain in BOXES:
        box = FancyBboxPatch(
            (x, Y), W, H,
            boxstyle="round,pad=0.06,rounding_size=0.12",
            linewidth=1.6, edgecolor=color, facecolor=SURFACE,
        )
        ax.add_patch(box)
        ax.text(x + W / 2, Y + H - 0.28, title, ha="center", va="center",
                fontsize=11.5, fontweight="bold", color=INK)
        ax.text(x + W / 2, Y + H / 2 - 0.12, tech, ha="center", va="center",
                fontsize=8.5, color=color, style="italic")
        # Explicación en lenguaje llano, debajo de la caja
        ax.text(x + W / 2, Y - 0.42, plain, ha="center", va="top",
                fontsize=8.8, color=INK_2)

    # Flechas entre cajas
    for i in range(len(BOXES) - 1):
        x_from = BOXES[i][0] + W + 0.06
        x_to = BOXES[i + 1][0] - 0.06
        ax.add_patch(FancyArrowPatch(
            (x_from, Y + H / 2), (x_to, Y + H / 2),
            arrowstyle="-|>", mutation_scale=16, linewidth=1.6, color=MUTED,
        ))

    # Anotación temporal: qué pasa antes y después de la máquina
    ax.annotate("ANTES de gastar tiempo de máquina", xy=(4.15, Y + H + 0.55),
                ha="center", fontsize=9, color=BLUE, fontweight="bold")
    ax.annotate("DESPUÉS de ejecutar", xy=(9.35, Y + H + 0.55),
                ha="center", fontsize=9, color=AQUA, fontweight="bold")
    ax.plot([2.9, 5.4], [Y + H + 0.38, Y + H + 0.38], color=BLUE, lw=1.2)
    ax.plot([8.1, 10.6], [Y + H + 0.38, Y + H + 0.38], color=AQUA, lw=1.2)

    ax.set_xlim(0, 13.6)
    ax.set_ylim(0.4, 4.6)
    ax.axis("off")
    fig.suptitle(
        "Pipeline del TFM: mitigación de errores cuánticos con una sola ejecución",
        fontsize=13, fontweight="bold", color=INK, y=0.97,
    )

    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    out = FIGURES_PATH / "pipeline_tfm.png"
    fig.savefig(out, bbox_inches="tight", facecolor=SURFACE)
    print(f"[make_pipeline_figure] Guardado: {out}")


if __name__ == "__main__":
    main()
