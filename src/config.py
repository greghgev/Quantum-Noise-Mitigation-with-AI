"""
Módulo de Configuración.
Gestiona la carga de variables de entorno, topes físicos del hardware
y garantiza el determinismo estocástico (MLOps).
"""
import os
import random
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from dotenv import load_dotenv

# 1. Carga de credenciales seguras y Fail-Fast
load_dotenv()
IBM_TOKEN: Optional[str] = os.getenv("IBM_TOKEN")

if not IBM_TOKEN:
    warnings.warn(
        "IBM_TOKEN no encontrado en el archivo .env. "
        "Las descargas de calibración de IBM fallarán."
    )

# 2. Límites Físicos y Cuánticos (Hard Constraints)
MAX_QUBITS: int = 15
MAX_DEPTH: int = 50
TRAIN_SHOTS: int = 1024

# 3. Determinismo y Rutas
SEED: int = 42

# Uso de pathlib: Resolvemos la ruta absoluta del proyecto de forma limpia
# __file__ es config.py -> parent es src/ -> parent es TFM-Quantum/
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# Rutas de Datos (Concatenación orientada a objetos con el operador '/')
DATA_PATH: Path = BASE_DIR / "data"
RAW_DATA_PATH: Path = DATA_PATH / "raw"
PROCESSED_DATA_PATH: Path = DATA_PATH / "processed"

# Rutas de Artefactos MLOps
MODELS_PATH: Path = BASE_DIR / "models"
DOC_PATH: Path = BASE_DIR / "doc"
LOGS_PATH: Path = BASE_DIR / "logs"       
FIGURES_PATH: Path = BASE_DIR / "figures" 

# (Opcional) Si usas MLflow en local, esta será la ruta de tracking
MLFLOW_TRACKING_URI: str = f"sqlite:///{BASE_DIR / 'mlflow.db'}"

# 4. Motor de Reproducibilidad MLOps
def set_global_seeds(seed: int = SEED) -> None:
    """
    Congela todas las fuentes de aleatoriedad del entorno clásico y de PyTorch.
    Nota: La semilla del simulador cuántico (Qiskit Aer) debe pasarse explícitamente 
    en quantum_gen.py usando `seed_simulator=SEED`.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Fuerza algoritmos deterministas en la GPU a costa de algo de rendimiento
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False