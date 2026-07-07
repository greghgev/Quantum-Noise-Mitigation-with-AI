"""
Entrena el módulo GEM (Gate Error Mitigation).

Uso:
    conda run -n tfm python scripts/train_gem.py
    conda run -n tfm python scripts/train_gem.py --config configs/gem_config.yaml
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import set_global_seeds


def main():
    parser = argparse.ArgumentParser(description="Entrena el GEM Graph Transformer")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/gem_config.yaml",
        help="Path al fichero de configuración YAML",
    )
    args = parser.parse_args()

    import yaml
    with open(ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    set_global_seeds(cfg["experiment"]["seed"])

    print(f"[train_gem] Config: {args.config}")
    print(f"[train_gem] Experimento MLflow: {cfg['experiment']['mlflow_experiment']}")
    print("[train_gem] Importando módulos de entrenamiento...")

    # TODO (TAREA 4): importar y ejecutar el bucle de entrenamiento del GEM
    # from src.train import train_gem
    # train_gem(cfg)
    raise NotImplementedError("train_gem está pendiente de implementar en TAREA 4")


if __name__ == "__main__":
    main()
