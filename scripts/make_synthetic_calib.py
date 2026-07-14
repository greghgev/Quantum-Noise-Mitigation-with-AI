"""
Genera una calibración SINTÉTICA de ibm_kingston (Heron r2) cuando no hay
credenciales IBM válidas. Sustituto temporal de la descarga real — borrar el
JSON y relanzar HardwareTelemetrics cuando la cuenta IBM esté verificada.

[SUPOSICIÓN] Todos los valores son plausibles para Heron r2 pero NO son datos
reales de IBM. Rangos basados en cifras públicas de la familia Heron
(T2 ≈ 350 µs reportado para ibm_pittsburgh; EPLG@100q ≈ 2.15e-3):
  - T1 ~ U(150, 300) µs        - gate_error(sx) ~ U(1.5e-4, 5e-4)
  - T2 ~ U(0.6, 1.4) · T1      - gate_length(sx) ~ U(40, 60) ns
  - readout_error ~ U(0.5%, 2.5%)
[SUPOSICIÓN] Topología: cadena lineal de 20 qubits. El Heron real usa una
variante heavy-hex de 156 qubits; la cadena es una simplificación suficiente
para MAX_QUBITS=15 (misma decisión que el mini-run Eagle previo).

Uso:
    conda run -n tfm python scripts/make_synthetic_calib.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import BACKEND_NAME, RAW_DATA_PATH, SEED  # noqa: E402

N_QUBITS = 20


def main():
    rng = np.random.default_rng(SEED)

    properties = {}
    for q in range(N_QUBITS):
        t1 = float(rng.uniform(150e-6, 300e-6))
        t2 = float(min(rng.uniform(0.6, 1.4) * t1, 2 * t1))
        properties[str(q)] = {
            "T1": t1,
            "T2": t2,
            "readout_error": float(rng.uniform(0.005, 0.025)),
            "gate_error": float(rng.uniform(1.5e-4, 5e-4)),
            "gate_length": float(rng.uniform(40e-9, 60e-9)),
        }

    coupling_map = [[q, q + 1] for q in range(N_QUBITS - 1)]

    out = Path(RAW_DATA_PATH) / f"{BACKEND_NAME}_calib.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(
            {
                "properties": properties,
                "coupling_map": coupling_map,
                "_synthetic": True,
                "_note": (
                    "Calibración sintética Heron r2 — sustituir por real "
                    "al verificar cuenta IBM"
                ),
            },
            f,
            indent=4,
        )
    print(f"[make_synthetic_calib] Guardado: {out} ({N_QUBITS} qubits, cadena lineal)")


if __name__ == "__main__":
    main()
