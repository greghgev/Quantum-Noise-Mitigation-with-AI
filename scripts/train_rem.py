"""
Entrena el módulo REM (Readout Error Mitigation).

Uso:
    conda run -n tfm python scripts/train_rem.py
    conda run -n tfm python scripts/train_rem.py --config configs/rem_config.yaml
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import set_global_seeds  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Entrena el REM GNN + GMRES")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/rem_config.yaml",
        help="Path al fichero de configuración YAML",
    )
    args = parser.parse_args()

    import yaml

    with open(ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    set_global_seeds(cfg["experiment"]["seed"])

    print(f"[train_rem] Config: {args.config}")
    print(f"[train_rem] Experimento MLflow: {cfg['experiment']['mlflow_experiment']}")

    # TODO (TAREA 4): importar y ejecutar el bucle de entrenamiento del REM
    # from src.train import train_rem
    # train_rem(cfg)
    raise NotImplementedError("train_rem está pendiente de implementar en TAREA 4")


if __name__ == "__main__":
    main()
