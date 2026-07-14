"""
TAREA 6a — Recopila calibraciones REALES de IBM con fecha, para construir el
test temporal con drift auténtico (idea del tutor: "los datos de cada día como test").

Guarda snapshots en data/raw/calib_history/{backend}_{YYYY-MM-DD}.json con el
mismo esquema que ibm_kingston_calib.json (compatible con HardwareTelemetrics).

Verificado jul-2026: backend.properties(datetime=...) devuelve snapshots
históricos reales (p. ej. T1(q0) en ibm_kingston: 277→253→160→253→96 µs en 30 días).

Uso:
    conda run -n tfm python scripts/collect_calibration.py                # hoy
    conda run -n tfm python scripts/collect_calibration.py --backfill 30  # últimos 30 días
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import BACKEND_NAME, IBM_TOKEN, RAW_DATA_PATH  # noqa: E402

HISTORY_DIR = Path(RAW_DATA_PATH) / "calib_history"


def extract_snapshot(backend, props) -> dict:
    """Extrae el mismo esquema de propiedades que HardwareTelemetrics."""
    base_props = {}
    for q in range(backend.num_qubits):
        try:
            t1 = props.t1(q)
            t2 = props.t2(q)
            re_ = props.readout_error(q)
            try:
                ge = props.gate_error("sx", q)
                gl = props.gate_length("sx", q)
            except Exception:
                ge, gl = 1e-3, 3.5e-8
        except Exception:
            t1, t2, re_, ge, gl = 100e-6, 80e-6, 0.05, 1e-3, 3.5e-8
        base_props[str(q)] = {
            "T1": t1,
            "T2": t2,
            "readout_error": re_,
            "gate_error": ge,
            "gate_length": gl,
        }
    return base_props


def main():
    parser = argparse.ArgumentParser(description="Recopila calibración real con fecha")
    parser.add_argument(
        "--backfill",
        type=int,
        default=0,
        help="Además de hoy, descargar los N días anteriores (histórico)",
    )
    args = parser.parse_args()

    import os

    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=IBM_TOKEN,
        instance=os.getenv("IBM_INSTANCE_CRN"),
    )
    backend = service.backend(BACKEND_NAME)
    coupling_map = [[int(u), int(v)] for u, v in backend.coupling_map.get_edges()]
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    saved, skipped = 0, 0
    for days_back in range(args.backfill + 1):
        target = datetime.now() - timedelta(days=days_back)
        try:
            props = (
                backend.properties(datetime=target)
                if days_back
                else backend.properties()
            )
            if props is None:
                print(f"  {target.date()}: sin datos, salto")
                continue
            # La fecha real del snapshot es la de la ÚLTIMA calibración <= target
            real_date = props.last_update_date.date().isoformat()
            out = HISTORY_DIR / f"{BACKEND_NAME}_{real_date}.json"
            if out.exists():
                skipped += 1
                continue
            snapshot = {
                "properties": extract_snapshot(backend, props),
                "coupling_map": coupling_map,
                "calibration_date": str(props.last_update_date),
                "requested_date": target.date().isoformat(),
            }
            with open(out, "w") as f:
                json.dump(snapshot, f, indent=2)
            t1_0 = snapshot["properties"]["0"]["T1"] * 1e6
            print(f"  {real_date}: guardado (T1 q0 = {t1_0:.1f} µs)")
            saved += 1
        except Exception as e:
            print(f"  {target.date()}: ERROR {type(e).__name__}: {str(e)[:70]}")

    print(f"\n[collect_calibration] {saved} snapshots nuevos, {skipped} ya existían")
    print(f"[collect_calibration] Histórico en: {HISTORY_DIR}")


if __name__ == "__main__":
    main()
