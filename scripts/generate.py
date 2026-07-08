"""
Genera el dataset completo (o un mini-run) y lo versiona con DVC.

Uso:
    conda run -n tfm python scripts/generate.py --mini
    conda run -n tfm python scripts/generate.py --full
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import set_global_seeds, PROCESSED_DATA_PATH  # noqa: E402
from src.quantum_gen import TFMDatasetPipeline  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Genera el dataset TFM-Quantum")
    parser.add_argument(
        "--mini",
        action="store_true",
        help="Mini-run: 100-200 muestras para validar el pipeline",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run completo: dataset de entrenamiento con split temporal OOD",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.mini and not args.full:
        parser.error("Especifica --mini o --full")

    set_global_seeds(args.seed)

    n_samples = 150 if args.mini else 5000
    days = [1, 2] if args.mini else list(range(1, 11))

    print(
        f"[generate] Modo: {'mini' if args.mini else 'full'} | Muestras: {n_samples} | Días: {days}"
    )

    pipeline = TFMDatasetPipeline()
    pipeline.run(n_samples=n_samples, days=days)

    print(f"[generate] Dataset guardado en {PROCESSED_DATA_PATH}")
    print("[generate] Ejecuta 'dvc add data/processed/' para versionar con DVC")


if __name__ == "__main__":
    main()
