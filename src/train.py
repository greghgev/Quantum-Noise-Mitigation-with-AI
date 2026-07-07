"""
Pipeline de Entrenamiento y MLOps.
Implementa el bucle de entrenamiento, validación y el tracking de 
experimentos mediante MLflow/W&B.
"""
import torch
import mlflow
from src.model import QuantumMitigator