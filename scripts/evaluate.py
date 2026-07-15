"""
Evalúa los modelos de la comparativa (Ridge, RF, GEM) sobre test OOD y zero-shot.

Reporta todas las métricas:
  GEM: MAE, RMSE, R², mejora relativa

Uso:
    conda run -n tfm python scripts/evaluate.py --gem models/gem_best.pt"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import set_global_seeds  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Evalúa los modelos de la comparativa")
    parser.add_argument(
        "--gem", type=str, required=True, help="Path al checkpoint del GEM (.pt)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split",
        type=str,
        choices=["test", "zeroshot_qaoa", "zeroshot_qft", "all"],
        default="all",
        help="Qué split evaluar",
    )
    args = parser.parse_args()

    set_global_seeds(args.seed)

    splits = (
        ["test", "zeroshot_qaoa", "zeroshot_qft"]
        if args.split == "all"
        else [args.split]
    )

    print(f"[evaluate] GEM: {args.gem}")
    print(f"[evaluate] Splits: {splits}")

    # TODO (TAREA 5): importar y ejecutar la evaluación
    # from src.inference import evaluate_pipeline
    # from src.utils import compute_all_metrics
    # for split in splits:
    #     results = evaluate_models(args.gem, split=split)
    #     metrics = compute_all_metrics(results)
    #     print(f"\n=== {split.upper()} ===")
    #     for k, v in metrics.items():
    #         print(f"  {k}: {v:.4f}")
    raise NotImplementedError("evaluate está pendiente de implementar en TAREA 5")


if __name__ == "__main__":
    main()
