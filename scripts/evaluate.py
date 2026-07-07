"""
Evalúa el pipeline completo GEM + REM sobre el conjunto de test OOD y los conjuntos zero-shot.

Reporta todas las métricas:
  GEM: MAE, RMSE, R², mejora relativa
  REM: Hellinger Fidelity, TVD, ratio de mejora

Uso:
    conda run -n tfm python scripts/evaluate.py --gem models/gem_best.pt --rem models/rem_best.pt
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import set_global_seeds, PROCESSED_DATA_PATH


def main():
    parser = argparse.ArgumentParser(description="Evalúa el pipeline GEM + REM")
    parser.add_argument("--gem", type=str, required=True, help="Path al checkpoint del GEM (.pt)")
    parser.add_argument("--rem", type=str, required=True, help="Path al checkpoint del REM (.pt)")
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

    print(f"[evaluate] GEM: {args.gem} | REM: {args.rem}")
    print(f"[evaluate] Splits: {splits}")

    # TODO (TAREA 5): importar y ejecutar la evaluación
    # from src.inference import evaluate_pipeline
    # from src.utils import compute_all_metrics
    # for split in splits:
    #     results = evaluate_pipeline(args.gem, args.rem, split=split)
    #     metrics = compute_all_metrics(results)
    #     print(f"\n=== {split.upper()} ===")
    #     for k, v in metrics.items():
    #         print(f"  {k}: {v:.4f}")
    raise NotImplementedError("evaluate está pendiente de implementar en TAREA 5")


if __name__ == "__main__":
    main()
