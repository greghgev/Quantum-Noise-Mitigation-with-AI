# src/quantum_gen.py

"""
Módulo de Generación Cuántica y Extracción de Grafos.
Actúa como Gemelo Digital del hardware de IBM, generando datos sintéticos
ruidosos para entrenar el modelo GEM y sus baselines de comparativa.
"""
import gc
import json
import os
import warnings
import zlib
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.random import random_circuit
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Statevector
from qiskit.synthesis.qft import (
    synth_qft_full,
)  # Qiskit 2.x (reemplaza QFT class deprecated)
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
from qiskit_ibm_runtime import QiskitRuntimeService
from torch_geometric.data import Data

from src.config import (
    BACKEND_NAME,
    BASIS_GATES,
    IBM_TOKEN,
    LABEL_SHOTS,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    SEED,
)


def _split_seed_offset(split_name: str) -> int:
    """
    Offset determinista y estable entre procesos (crc32, no hash() de Python —
    éste varía entre ejecuciones por PYTHONHASHSEED) para separar el espacio de
    seeds de cada split. FIX de leakage: antes, mismo day_index + mismo índice i
    producían el circuito IDÉNTICO en train/val/test cuando compartían día (caso
    del modo mini). Ver doc/migracion_heron.md y ROADMAP.md.
    """
    return (zlib.crc32(split_name.encode()) % 1000) * 100_000


class CircuitFactory:
    """
    Responsabilidad: Generar la estructura lógica de los circuitos (El 'Qué').
    Genera matemática pura y la transpila al idioma físico nativo de IBM
    respetando la topología (coupling_map) para asegurar una profundidad realista.
    """

    def __init__(self, min_qubits: int = 5, max_qubits: int = 15):
        self.min_qubits = min_qubits
        self.max_qubits = max_qubits
        # Heron r2/r3: la puerta 2q nativa es CZ. Los generadores construyen la
        # lógica con CX (convención estándar) y el transpilador la convierte a CZ.
        self.basis_gates = list(BASIS_GATES)

    def _transpile_and_check(
        self,
        qc: QuantumCircuit,
        coupling_map: Optional[List[List[int]]] = None,
        seed: Optional[int] = None,
    ) -> QuantumCircuit:
        if coupling_map is not None:
            active_nodes = set(range(qc.num_qubits))
            coupling_map = [
                [u, v]
                for u, v in coupling_map
                if u in active_nodes and v in active_nodes
            ]
        return transpile(
            qc,
            basis_gates=self.basis_gates,
            coupling_map=coupling_map,
            optimization_level=1,
            seed_transpiler=seed,
        )

    def generate_random_circuit(
        self,
        num_qubits: int,
        logical_depth: int,
        seed: int,
        coupling_map: Optional[List[List[int]]] = None,
    ) -> QuantumCircuit:
        qc = random_circuit(
            num_qubits, logical_depth, max_operands=2, measure=False, seed=seed
        )
        return self._transpile_and_check(qc, coupling_map, seed=seed)

    def generate_qft(
        self,
        num_qubits: int,
        coupling_map: Optional[List[List[int]]] = None,
    ) -> QuantumCircuit:
        # synth_qft_full reemplaza la clase QFT deprecated en Qiskit 2.1+
        qc = synth_qft_full(num_qubits, approximation_degree=0, do_swaps=True)
        return self._transpile_and_check(qc, coupling_map)

    def generate_hea(
        self,
        num_qubits: int,
        seed: int,
        reps: int = 2,
        coupling_map: Optional[List[List[int]]] = None,
    ) -> QuantumCircuit:
        """
        Hardware-Efficient Ansatz: capas de Ry + CX con entrelazamiento lineal.
        Construido manualmente para evitar TwoLocal (deprecated en Qiskit 2.1).
        """
        rng = np.random.default_rng(seed)
        n_params = num_qubits * (reps + 1)  # (reps + 1 final layer) × n qubits Ry
        angles = rng.random(n_params) * 2 * np.pi

        qc = QuantumCircuit(num_qubits)
        param_idx = 0
        for _ in range(reps):
            for q in range(num_qubits):
                qc.ry(float(angles[param_idx]), q)
                param_idx += 1
            for q in range(num_qubits - 1):
                qc.cx(q, q + 1)
        # Capa final de rotación
        for q in range(num_qubits):
            qc.ry(float(angles[param_idx]), q)
            param_idx += 1

        return self._transpile_and_check(qc, coupling_map, seed=seed)

    def generate_qaoa(
        self,
        num_qubits: int,
        seed: int,
        reps: int = 1,
        coupling_map: Optional[List[List[int]]] = None,
    ) -> QuantumCircuit:
        """
        QAOA con operador de coste correcto: suma de términos ZZ por arista del coupling map.
        Construido manualmente para corregir BUG-2 (operador trivial Z^n) y evitar
        QAOAAnsatz (deprecated en Qiskit 2.1).
        """
        rng = np.random.default_rng(seed)

        # Aristas activas dentro del rango de qubits del circuito
        if coupling_map is not None:
            active_edges = list(
                {
                    (min(u, v), max(u, v))
                    for u, v in coupling_map
                    if u < num_qubits and v < num_qubits
                }
            )
        else:
            active_edges = [(i, i + 1) for i in range(num_qubits - 1)]

        if not active_edges:
            active_edges = [(0, 1)] if num_qubits >= 2 else [(0, 0)]

        angles = rng.random(2 * reps) * 2 * np.pi  # gamma y beta por capa

        qc = QuantumCircuit(num_qubits)
        qc.h(range(num_qubits))  # Estado inicial |+>^n

        for r in range(reps):
            gamma = float(angles[2 * r])
            beta = float(angles[2 * r + 1])

            # Unitario de coste: e^{-i γ Σ_{ij} Z_i Z_j}
            for i, j in active_edges:
                qc.cx(i, j)
                qc.rz(2 * gamma, j)
                qc.cx(i, j)

            # Unitario mezclador: e^{-i β Σ_i X_i}
            qc.rx(2 * beta, range(num_qubits))

        return self._transpile_and_check(qc, coupling_map, seed=seed)

    def generate_tfim(
        self,
        num_qubits: int,
        seed: int,
        trotter_steps: int = 4,
        coupling_map: Optional[List[List[int]]] = None,
    ) -> QuantumCircuit:
        """
        Circuito Trotterized del Hamiltoniano Ising con campo transverso:
          H = -J Σ_{ij} Z_i Z_j - h Σ_i X_i
        Benchmark estándar de QEM-Bench (QEMFormer, ICML 2025).
        """
        rng = np.random.default_rng(seed)
        J = float(rng.uniform(0.5, 1.5))  # intensidad de acoplamiento
        h = float(rng.uniform(0.3, 1.0))  # campo transverso
        dt = 1.0 / max(trotter_steps, 1)  # paso Trotter normalizado

        if coupling_map is not None:
            active_edges = list(
                {
                    (min(u, v), max(u, v))
                    for u, v in coupling_map
                    if u < num_qubits and v < num_qubits
                }
            )
        else:
            active_edges = [(i, i + 1) for i in range(num_qubits - 1)]

        if not active_edges:
            active_edges = [(i, i + 1) for i in range(num_qubits - 1)]

        qc = QuantumCircuit(num_qubits)
        qc.h(
            range(num_qubits)
        )  # Estado inicial |+>^n (ground state del campo transverso)

        for _ in range(trotter_steps):
            # Interacciones ZZ (acoplamiento Ising)
            for i, j in active_edges:
                qc.cx(i, j)
                qc.rz(2 * J * dt, j)
                qc.cx(i, j)
            # Campo transverso (rotaciones X)
            for q in range(num_qubits):
                qc.rx(2 * h * dt, q)

        return self._transpile_and_check(qc, coupling_map, seed=seed)


class HardwareTelemetrics:
    """
    Responsabilidad: Actuar como el 'Gemelo Digital' del chip IBM.
    Inyecta T1, T2, gate_error, gate_length y readout_error reales.
    Gestiona el Concept Drift temporal step-like sobre una ventana de 10 días
    (Hirasaki et al., Applied Physics Letters 2023).
    """

    def __init__(self, backend_name: str = BACKEND_NAME):
        self.backend_name = backend_name
        self.base_properties = {}
        self.coupling_map = None
        self.noise_model = None
        self._drift_schedule = None

        self.cache_file = Path(RAW_DATA_PATH) / f"{self.backend_name}_calib.json"
        self._load_or_fetch_calibration()
        self._init_drift_schedule()

    def _load_or_fetch_calibration(self) -> None:
        if self.cache_file.exists():
            print(f"Cargando calibración offline desde {self.cache_file}...")
            try:
                with open(self.cache_file) as f:
                    data = json.load(f)
                    self.base_properties = data["properties"]
                    self.coupling_map = data["coupling_map"]
            except json.JSONDecodeError:
                print("Error: caché corrupto. Descargando de nuevo...")
                self._fetch_from_ibm_and_save()
        else:
            self._fetch_from_ibm_and_save()

    def _fetch_from_ibm_and_save(self) -> None:
        print("Conectando a IBM Quantum para descargar calibración base...")
        try:
            service = QiskitRuntimeService(
                channel="ibm_quantum_platform",
                token=IBM_TOKEN,
                instance=os.getenv("IBM_INSTANCE_CRN"),
            )
            backend = service.backend(self.backend_name)
            self.noise_model = NoiseModel.from_backend(backend)
            # get_edges() devuelve EdgeList (rustworkx), no serializable a JSON →
            # convertir a listas nativas (riesgo previsto en CLAUDE.md §7,
            # confirmado en la primera descarga real de ibm_kingston, jul-2026)
            self.coupling_map = [
                [int(u), int(v)] for u, v in backend.coupling_map.get_edges()
            ]
            properties = backend.properties()
            num_qubits = backend.num_qubits

            base_props = {}
            for q in range(num_qubits):
                try:
                    t1 = properties.t1(q)
                    t2 = properties.t2(q)
                    re = properties.readout_error(q)
                    try:
                        ge = properties.gate_error("sx", q)
                        gl = properties.gate_length("sx", q)
                    except Exception:
                        ge, gl = 1e-3, 3.5e-8
                except Exception:
                    t1, t2, re, ge, gl = 100e-6, 80e-6, 0.05, 1e-3, 3.5e-8

                base_props[str(q)] = {
                    "T1": t1,
                    "T2": t2,
                    "readout_error": re,
                    "gate_error": ge,
                    "gate_length": gl,
                }

            self.base_properties = base_props
            os.makedirs(RAW_DATA_PATH, exist_ok=True)
            with open(self.cache_file, "w") as f:
                json.dump(
                    {
                        "properties": self.base_properties,
                        "coupling_map": self.coupling_map,
                    },
                    f,
                    indent=4,
                )
        except Exception as e:
            raise ConnectionError(f"Fallo crítico conectando a IBM: {e}")

    def _init_drift_schedule(self) -> None:
        """
        Pre-computa un calendario de drift step-like determinista para todos los qubits
        sobre 10 días: 2-3 saltos abruptos de ±15-30% en posiciones aleatorias
        (Hirasaki et al., APL 2023 — evidencia empírica en hardware IBM real).
        Usa SEED para reproducibilidad entre ejecuciones.
        """
        rng = np.random.default_rng(SEED)
        n_qubits = len(self.base_properties)
        schedule = {}

        for q in range(n_qubits):
            factors = np.ones(10)
            n_jumps = int(rng.integers(2, 4))  # 2 o 3 saltos
            jump_days = rng.choice(range(1, 10), size=n_jumps, replace=False)

            current = 1.0
            for day in range(10):
                if day in jump_days:
                    magnitude = float(rng.uniform(0.15, 0.30))
                    direction = float(rng.choice([-1.0, 1.0]))
                    current = float(
                        np.clip(current * (1 + direction * magnitude), 0.3, 1.5)
                    )
                factors[day] = current

            schedule[str(q)] = factors.tolist()

        self._drift_schedule = schedule

    def build_noise_model_for_day(self, day_index: int) -> NoiseModel:
        """
        Construye un NoiseModel calibrado por día usando las propiedades driftadas.
        Modela: despolarización 1q, error de readout simétrico, y despolarización 2q
        para CZ (puerta 2-qubit nativa de Heron).

        [SUPOSICIÓN] El error CZ se aproxima como 5× el error 1q promedio de los qubits
        involucrados — ratio típico en hardware IBM (no dado explícitamente por calib API).
        """
        nm = NoiseModel(basis_gates=list(BASIS_GATES))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._populate_noise_model(nm, day_index)

        return nm

    def _populate_noise_model(self, nm: NoiseModel, day_index: int) -> None:
        """
        Usa SOLO errores de despolarización para máxima estabilidad con AerSimulator.
        La combinación thermal_relaxation + depolarizing en la misma puerta produce
        Kraus vacíos en Aer 0.17 para ciertos parámetros. El efecto neto de T1/T2
        se aproxima como mayor gate_error (modelado implícitamente via drift).
        """
        for q_str in self.base_properties:
            q = int(q_str)
            f = self.get_qubit_features(q, day_index)
            re = float(np.clip(f["readout_error"], 0.0, 0.499))
            ge = float(np.clip(f["gate_error"], 0.0, 0.74))

            if ge > 0:
                nm.add_quantum_error(
                    depolarizing_error(ge, 1), ["sx", "x", "rz", "id"], [q]
                )

            nm.add_readout_error(ReadoutError([[1 - re, re], [re, 1 - re]]), [q])

        # CZ gate errors sobre pares del coupling map.
        # CZ es simétrica, pero Aer asocia el error al orden exacto de qubits del
        # circuito transpilado — se registra en ambas orientaciones para no dejar
        # puertas sin ruido cuando el transpilador invierte el par.
        if self.coupling_map:
            seen = set()
            for edge in self.coupling_map:
                q_c, q_t = int(edge[0]), int(edge[1])
                key = (min(q_c, q_t), max(q_c, q_t))
                if key in seen:
                    continue
                seen.add(key)
                if (
                    str(q_c) not in self.base_properties
                    or str(q_t) not in self.base_properties
                ):
                    continue
                fc = self.get_qubit_features(q_c, day_index)
                ft = self.get_qubit_features(q_t, day_index)
                # [SUPOSICIÓN]: CZ ≈ 5× error 1q promedio — ratio típico NISQ
                cz_err = float(
                    np.clip((fc["gate_error"] + ft["gate_error"]) * 5 / 2, 0.0, 0.9375)
                )
                if cz_err > 0:
                    nm.add_quantum_error(
                        depolarizing_error(cz_err, 2), ["cz"], [q_c, q_t]
                    )
                    nm.add_quantum_error(
                        depolarizing_error(cz_err, 2), ["cz"], [q_t, q_c]
                    )

    def get_qubit_features(self, qubit_idx: int, day_index: int) -> dict:
        base = self.base_properties.get(
            str(qubit_idx),
            {
                "T1": 100e-6,
                "T2": 80e-6,
                "readout_error": 0.05,
                "gate_error": 1e-3,
                "gate_length": 3.5e-8,
            },
        )
        drift = self._drift_schedule[str(qubit_idx)][day_index]

        return {
            "T1": base["T1"] * drift,
            "T2": base["T2"] * drift,
            # Cuando T1/T2 bajan (drift < 1), los errores suben
            "readout_error": min(
                1.0, base.get("readout_error", 0.05) / max(drift, 0.1)
            ),
            "gate_error": min(1.0, base.get("gate_error", 1e-3) / max(drift, 0.1)),
            "gate_length": base.get(
                "gate_length", 3.5e-8
            ),  # estable bajo drift térmico
        }


class QuantumGraphExtractor:
    """
    Responsabilidad: Traducir el mundo cuántico (Qiskit) al mundo Deep Learning (PyTorch).
    Convierte un QuantumCircuit y sus telemetrías en un objeto Data de PyTorch Geometric.

    Vector de features por nodo — 19 dimensiones:
      [0:6]   one-hot tipo de puerta (cz, id, rz, sx, x, measure)
      [6:9]   ángulos paramétricos θ, φ, λ (zero-pad si no aplica)
      [9:12]  T1, T2, readout_error del qubit control
      [12:13] gate_error del qubit control
      [13:14] gate_length del qubit control
      [14:17] T1, T2, readout_error del qubit target ([0,0,0] para puertas 1q)
      [17:18] day_index / 10.0  (señal explícita de Concept Drift)
      [18:19] depth_position / total_nodes  (decoherencia acumulada)

    Nota Heron: CZ es simétrica (no hay control/target físico). Por convención,
    "control" = qargs[0] y "target" = qargs[1] — el orden que asigna el transpilador.
    """

    def __init__(self, hardware_telemetrics: HardwareTelemetrics):
        self.hw = hardware_telemetrics
        self.gate_to_id = {"cz": 0, "id": 1, "rz": 2, "sx": 3, "x": 4, "measure": 5}

    def circuit_to_graph(
        self,
        circuit: QuantumCircuit,
        delta: float,
        day_index: int,
        ideal_probs: Optional[np.ndarray] = None,
        noisy_probs: Optional[np.ndarray] = None,
    ) -> Data:
        dag = circuit_to_dag(circuit)

        # Orden topológico para depth_position consistente
        topo_nodes = [
            n for n in dag.topological_op_nodes() if n.name in self.gate_to_id
        ]
        total_nodes = max(len(topo_nodes), 1)

        node_features = []
        edge_indices = []
        node_mapping = {}

        day_norm = day_index / 10.0

        for node_idx, node in enumerate(topo_nodes):
            depth_position = node_idx / (total_nodes - 1) if total_nodes > 1 else 0.0

            # One-hot tipo de puerta (6 dims)
            one_hot = [0] * 6
            one_hot[self.gate_to_id[node.name]] = 1

            # Ángulos paramétricos (3 dims, zero-pad)
            params = [float(p) for p in node.op.params]
            params += [0.0] * (3 - len(params))

            # Features qubit control (T1, T2, re, gate_error, gate_length)
            q_control_idx = circuit.find_bit(node.qargs[0]).index
            phys_ctrl = self.hw.get_qubit_features(q_control_idx, day_index)
            vec_ctrl = [phys_ctrl["T1"], phys_ctrl["T2"], phys_ctrl["readout_error"]]
            gate_error = phys_ctrl["gate_error"]
            gate_length = phys_ctrl["gate_length"]

            # Features qubit target (3 dims, zero-pad [0,0,0] para puertas 1q)
            if len(node.qargs) > 1:
                q_target_idx = circuit.find_bit(node.qargs[1]).index
                phys_tgt = self.hw.get_qubit_features(q_target_idx, day_index)
                vec_tgt = [phys_tgt["T1"], phys_tgt["T2"], phys_tgt["readout_error"]]
            else:
                vec_tgt = [0.0, 0.0, 0.0]  # Zero-pad correcto para puertas 1q

            feature_vector = (
                one_hot  # [0:6]
                + params  # [6:9]
                + vec_ctrl  # [9:12]
                + [gate_error]  # [12:13]
                + [gate_length]  # [13:14]
                + vec_tgt  # [14:17]
                + [day_norm]  # [17:18]
                + [depth_position]  # [18:19]
            )
            assert (
                len(feature_vector) == 19
            ), f"Feature vector incorrecto: {len(feature_vector)} dims"

            node_mapping[node] = node_idx
            node_features.append(feature_vector)

        for edge in dag.edges():
            src, tgt = edge[0], edge[1]
            if src in node_mapping and tgt in node_mapping:
                edge_indices.append([node_mapping[src], node_mapping[tgt]])

        x_tensor = torch.tensor(node_features, dtype=torch.float)

        if edge_indices:
            edge_index_tensor = (
                torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            )
        else:
            edge_index_tensor = torch.empty((2, 0), dtype=torch.long)

        # y = Δ (GEM target: ⟨Z⟩_ideal − ⟨Z⟩_noisy, scalar)
        y_tensor = torch.tensor([float(delta)], dtype=torch.float)

        data = Data(x=x_tensor, edge_index=edge_index_tensor, y=y_tensor)

        # Distribuciones de probabilidad completas (tamaño variable 2^n, para análisis)
        if ideal_probs is not None:
            data.ideal_probs = torch.tensor(ideal_probs, dtype=torch.float)
        if noisy_probs is not None:
            data.noisy_probs = torch.tensor(noisy_probs, dtype=torch.float)

        return data


def _avg_z_expectation(probs: np.ndarray) -> float:
    """⟨Σᵢ Zᵢ⟩/n desde un array de probabilidades (2^n elementos, little-endian).
    Zᵢ = +1 para |0⟩, -1 para |1⟩. Observable universal para todos los tipos de circuito.
    """
    n = int(np.round(np.log2(max(len(probs), 2))))
    indices = np.arange(len(probs), dtype=np.int64)
    popcount = np.array([bin(int(x)).count("1") for x in indices], dtype=float)
    z_obs = (n - 2.0 * popcount) / n
    return float(np.dot(probs, z_obs))


def _counts_to_probs(counts: dict, n_qubits: int) -> np.ndarray:
    """Convierte el dict de counts de Aer a un array de probabilidades [2^n].
    Qiskit convention: bitstring más a la derecha = qubit 0 (little-endian)."""
    total = sum(counts.values())
    probs = np.zeros(2**n_qubits, dtype=float)
    for bitstring, count in counts.items():
        idx = int(bitstring.replace(" ", ""), 2)
        if idx < len(probs):
            probs[idx] = count / total
    return probs


# Tipos de circuito permitidos por split (aplica BUG-4 fix: QAOA/QFT nunca en train)
_CIRCUIT_TYPES_BY_SPLIT = {
    "train": ["random", "hea", "tfim"],
    "val": ["random", "hea", "tfim"],
    "test": ["random", "hea", "tfim"],
    "zeroshot_qaoa": ["qaoa"],
    "zeroshot_qft": ["qft"],
}


class TFMDatasetPipeline:
    """
    Responsabilidad: El Orquestador MLOps.
    Genera miles de circuitos, extrae grafos y los guarda en subcarpetas por split:
      data/processed/train/          — Random, HEA, TFIM
      data/processed/val/            — ídem
      data/processed/test/           — ídem
      data/processed/zeroshot_qaoa/  — QAOA (NUNCA en train)
      data/processed/zeroshot_qft/   — QFT  (NUNCA en train)
    """

    def __init__(
        self,
        output_dir: str | Path = PROCESSED_DATA_PATH,
        telemetrics: Optional[HardwareTelemetrics] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.telemetrics = (
            telemetrics if telemetrics is not None else HardwareTelemetrics()
        )
        self.factory = CircuitFactory(min_qubits=5, max_qubits=15)
        self.extractor = QuantumGraphExtractor(self.telemetrics)

    def _get_circuit_types(self, split_name: str) -> List[str]:
        """Devuelve los tipos de circuito permitidos para un split dado."""
        for key, types in _CIRCUIT_TYPES_BY_SPLIT.items():
            if split_name.startswith(key):
                return types
        return ["random", "hea", "tfim"]  # fallback seguro (sin QAOA/QFT)

    def _generate_circuit(
        self,
        circuit_type: str,
        num_qubits: int,
        sample_seed: int,
        rng: np.random.Generator,
    ) -> QuantumCircuit:
        cm = self.telemetrics.coupling_map
        if circuit_type == "random":
            logical_depth = int(rng.integers(5, 51))
            return self.factory.generate_random_circuit(
                num_qubits, logical_depth, sample_seed, cm
            )
        elif circuit_type == "hea":
            return self.factory.generate_hea(num_qubits, sample_seed, coupling_map=cm)
        elif circuit_type == "tfim":
            steps = int(rng.integers(2, 7))
            return self.factory.generate_tfim(
                num_qubits, sample_seed, trotter_steps=steps, coupling_map=cm
            )
        elif circuit_type == "qaoa":
            # reps variable 1-3: con reps=1 fijo, Δ real quedaba por debajo del
            # suelo de shot-noise (ver ROADMAP.md TAREA 8). Más reps -> más
            # profundidad -> más error real acumulado y medible.
            reps = int(rng.integers(1, 4))
            return self.factory.generate_qaoa(
                num_qubits, sample_seed, reps=reps, coupling_map=cm
            )
        elif circuit_type == "qft":
            return self.factory.generate_qft(num_qubits, coupling_map=cm)
        else:
            raise ValueError(f"Tipo de circuito desconocido: {circuit_type}")

    def generate_dataset(
        self,
        num_samples: int,
        day_index: int,
        split_name: str,
    ) -> None:
        """
        Genera `num_samples` circuitos para el `split_name` dado y los guarda en
        `output_dir/{split_name}/`. Los tipos de circuito se determinan por split:
          train/val/test → Random + HEA + TFIM  (80% / 10% / 10%)
          zeroshot_qaoa  → sólo QAOA
          zeroshot_qft   → sólo QFT
        """
        split_dir = self.output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        allowed_types = self._get_circuit_types(split_name)
        is_zeroshot = split_name.startswith("zeroshot")

        max_qubits = self.factory.max_qubits
        if self.telemetrics.coupling_map:
            max_qubits = min(
                self.factory.max_qubits,
                max(max(edge) for edge in self.telemetrics.coupling_map) + 1,
            )

        print(
            f"\nGenerando split '{split_name}' — día {day_index} ({num_samples} muestras)..."
        )
        print(f"  Tipos permitidos: {allowed_types}")

        noise_model = self.telemetrics.build_noise_model_for_day(day_index)
        # method='statevector': fuerza simulación por muestreo (no density_matrix).
        # density_matrix requiere 2^(2n) RAM — inviable para n>12 en hardware local.
        sim = AerSimulator(noise_model=noise_model, method="statevector")
        shots = LABEL_SHOTS

        split_offset = _split_seed_offset(split_name)

        for i in tqdm(range(num_samples), desc=split_name):
            sample_seed = SEED + i + (day_index * 10000) + split_offset
            rng = np.random.default_rng(sample_seed)

            num_qubits = int(rng.integers(self.factory.min_qubits, max_qubits + 1))

            if is_zeroshot:
                circuit_type = allowed_types[0]
            else:
                # Distribución: 80% random, 10% HEA, 10% TFIM
                p = rng.random()
                if p < 0.80:
                    circuit_type = "random"
                elif p < 0.90:
                    circuit_type = "hea"
                else:
                    circuit_type = "tfim"

            qc = self._generate_circuit(circuit_type, num_qubits, sample_seed, rng)

            # --- Ideal statevector ---
            ideal_sv = np.abs(np.array(Statevector.from_instruction(qc))) ** 2
            ideal_exp = _avg_z_expectation(ideal_sv)

            # --- Noisy simulation ---
            qc_m = qc.copy()
            qc_m.measure_all()
            counts = (
                sim.run(qc_m, shots=shots, seed_simulator=sample_seed)
                .result()
                .get_counts()
            )
            noisy_probs = _counts_to_probs(counts, num_qubits)
            noisy_exp = _avg_z_expectation(noisy_probs)

            delta = ideal_exp - noisy_exp

            graph_data = self.extractor.circuit_to_graph(
                qc,
                delta,
                day_index,
                ideal_probs=ideal_sv,
                noisy_probs=noisy_probs,
            )

            filename = split_dir / f"{split_name}_day{day_index}_sample{i}.pt"
            torch.save(graph_data, filename)

            del ideal_sv, noisy_probs, graph_data, qc
            if i % 100 == 0:
                gc.collect()

    def run(self, n_samples: int, days: list) -> None:
        """
        Orquesta la generación del dataset completo sobre todos los splits y días.

        Asignación temporal (run completo, días 1-10):
          train         → días 1-5
          val           → día 6
          test          → días 8-10  (día 7 excluido — OOD gap intencional)
          zeroshot_qaoa → días 1-5
          zeroshot_qft  → días 1-5

        Para mini-run (≤ 2 días): todos los splits usan todos los días disponibles.
        """
        is_mini = len(days) <= 2

        if is_mini:
            split_days = {
                "train": days,
                "val": days,
                "test": days,
                "zeroshot_qaoa": days[:1],
                "zeroshot_qft": days[:1],
            }
        else:
            split_days = {
                "train": [d for d in days if 1 <= d <= 5],
                "val": [d for d in days if d == 6],
                "test": [d for d in days if d >= 8],
                "zeroshot_qaoa": [d for d in days if 1 <= d <= 5],
                "zeroshot_qft": [d for d in days if 1 <= d <= 5],
            }

        # Distribución de muestras: train 70%, val 15%, test 15%
        # Zero-shot: 10% extra sobre el total (no reducen las otras)
        n_zs = max(10, int(n_samples * 0.10))
        samples_per_split = {
            "train": int(n_samples * 0.70),
            "val": int(n_samples * 0.15),
            "test": int(n_samples * 0.15),
            "zeroshot_qaoa": n_zs,
            "zeroshot_qft": n_zs,
        }

        for split_name, day_list in split_days.items():
            if not day_list:
                print(f"[run] Split '{split_name}': sin días asignados, saltando.")
                continue
            n_split = samples_per_split[split_name]
            per_day = max(1, n_split // len(day_list))
            for day in day_list:
                self.generate_dataset(
                    num_samples=per_day,
                    day_index=day,
                    split_name=split_name,
                )
