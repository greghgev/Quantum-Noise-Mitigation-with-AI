# src/quantum_gen.py

"""
Módulo de Generación Cuántica y Extracción de Grafos.
Actúa como Gemelo Digital del hardware de IBM, generando datos sintéticos
ruidosos para entrenar los modelos GEM y REM.
"""
import os
import gc
from tqdm import tqdm
import json
import torch
import numpy as np
from pathlib import Path
from torch_geometric.data import Data
from typing import Optional, List

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.random import random_circuit
from qiskit.circuit.library import QFT, QAOAAnsatz, TwoLocal
from qiskit.quantum_info import SparsePauliOp
from qiskit.converters import circuit_to_dag
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer.noise import NoiseModel
from qiskit.quantum_info import Statevector

# Asumiendo que RAW_DATA_PATH está en src.config
from src.config import RAW_DATA_PATH, IBM_TOKEN
from src.config import PROCESSED_DATA_PATH, SEED


class CircuitFactory:
    """
    Responsabilidad: Generar la estructura lógica de los circuitos (El 'Qué').
    Genera matemática pura y la transpila al idioma físico nativo de IBM 
    respetando la topología (coupling_map) para asegurar una profundidad realista.
    """
    def __init__(self, min_qubits: int = 5, max_qubits: int = 15):
        self.min_qubits = min_qubits
        self.max_qubits = max_qubits
        # Puertas base nativas de la familia IBM Eagle
        self.basis_gates = ['cx', 'id', 'rz', 'sx', 'x']

    def _transpile_and_check(self, qc: QuantumCircuit, coupling_map: Optional[List[List[int]]] = None, seed: Optional[int] = None) -> QuantumCircuit:
        """
        Traduce el circuito lógico a pulsos de microondas.
        Si se proporciona un coupling_map, fuerza al enrutador de Qiskit a 
        insertar puertas SWAP respetando el grafo físico del chip.
        """
        # CRÍTICO: Filtrar el grafo físico para no añadir qubits fantasma inactivos
        if coupling_map is not None:
            active_nodes = set(range(qc.num_qubits))
            coupling_map = [[u, v] for u, v in coupling_map if u in active_nodes and v in active_nodes]
            
        transpiled_qc = transpile(
            qc, 
            basis_gates=self.basis_gates, 
            coupling_map=coupling_map,
            optimization_level=1,
            seed_transpiler=seed  # <-- Garantiza la reproducibilidad de MLOps
        )
        return transpiled_qc

    def generate_random_circuit(self, num_qubits: int, logical_depth: int, seed: int, coupling_map: Optional[List[List[int]]] = None) -> QuantumCircuit:
        """Genera circuitos aleatorios con reproducibilidad MLOps estricta."""
        qc = random_circuit(num_qubits, logical_depth, max_operands=2, measure=False, seed=seed)
        return self._transpile_and_check(qc, coupling_map, seed=seed)

    def generate_qft(self, num_qubits: int, coupling_map: Optional[List[List[int]]] = None) -> QuantumCircuit:
        qc = QFT(num_qubits, approximation_degree=0, do_swaps=True, inverse=False)
        qc = qc.decompose()
        return self._transpile_and_check(qc, coupling_map)

    def generate_qaoa(self, num_qubits: int, seed: int, reps: int = 1, coupling_map: Optional[List[List[int]]] = None) -> QuantumCircuit:
        op_string = "Z" * num_qubits
        cost_operator = SparsePauliOp(op_string)
        qc = QAOAAnsatz(cost_operator, reps=reps)
        
        rng = np.random.default_rng(seed)
        random_angles = rng.random(qc.num_parameters) * 2 * np.pi
        qc = qc.assign_parameters(random_angles.tolist())
        
        assert qc is not None  
        qc = qc.decompose()
        return self._transpile_and_check(qc, coupling_map, seed=seed)
    
    def generate_hea(self, num_qubits: int, seed: int, reps: int = 2, coupling_map: Optional[List[List[int]]] = None) -> QuantumCircuit:
        """
        Hardware-Efficient Ansatz (HEA/VQE): capas de Ry + CX con profundidad O(reps * n).
        Sustituye a Grover, que con mcx escalaba exponencialmente en profundidad.
        """
        qc = TwoLocal(num_qubits, rotation_blocks='ry', entanglement_blocks='cx', reps=reps)
        rng = np.random.default_rng(seed)
        random_angles = rng.random(qc.num_parameters) * 2 * np.pi
        qc = qc.assign_parameters(random_angles.tolist())
        qc = qc.decompose()
        return self._transpile_and_check(qc, coupling_map, seed=seed)


class HardwareTelemetrics:
    """
    Responsabilidad: Actuar como el 'Gemelo Digital' del chip IBM.
    Inyecta T1, T2, duraciones de compuertas y ruido de lectura reales.
    Gestiona el Concept Drift temporal en una ventana de 10 días.
    """
    def __init__(self, backend_name: str = "ibm_kyiv"):
        self.backend_name = backend_name
        self.base_properties = {}
        self.coupling_map = None
        self.noise_model = None
        
        self.cache_file = Path(RAW_DATA_PATH) / f"{self.backend_name}_calib.json"
        self._load_or_fetch_calibration()

    def _load_or_fetch_calibration(self) -> None:
        if self.cache_file.exists():
            print(f"Cargando calibración offline desde {self.cache_file}...")
            try:
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    self.base_properties = data['properties']
                    self.coupling_map = data['coupling_map']
            except json.JSONDecodeError:
                print("Error: Archivo de caché corrupto. Volviendo a descargar...")
                self._fetch_from_ibm_and_save()
        else:
            self._fetch_from_ibm_and_save()

    def _fetch_from_ibm_and_save(self) -> None:
        print("Conectando a IBM Quantum para descargar calibración base...")
        try:
            
            service = QiskitRuntimeService(
                channel="ibm_quantum_platform", 
                token=IBM_TOKEN,
                instance=os.getenv("IBM_INSTANCE_CRN")
            )
            
            backend = service.backend(self.backend_name)
            
            self.noise_model = NoiseModel.from_backend(backend)
            self.coupling_map = backend.coupling_map.get_edges()
            
            properties = backend.properties() # type: ignore
            num_qubits = backend.num_qubits
            
            base_props = {}
            for q in range(num_qubits):
                try:
                    base_props[str(q)] = {
                        "T1": properties.t1(q),
                        "T2": properties.t2(q),
                        "readout_error": properties.readout_error(q)
                    }
                except Exception:
                    base_props[str(q)] = {"T1": 100e-6, "T2": 80e-6, "readout_error": 0.05}
            
            self.base_properties = base_props
            
            os.makedirs(RAW_DATA_PATH, exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump({
                    'properties': self.base_properties,
                    'coupling_map': self.coupling_map
                }, f, indent=4)
                
        except Exception as e:
            raise ConnectionError(f"Fallo crítico conectando a IBM: {e}")
        
    def build_noise_model_for_day(self, day_index: int) -> Optional[NoiseModel]:
        # TODO (v2.0): Implementar Concept Drift real en la simulación.
        # Actualmente devolvemos el modelo estático (día 0) por rendimiento. 
        # Modificar los operadores de Kraus dinámicamente en Qiskit Aer es
        # computacionalmente muy costoso para la generación masiva del dataset.
        # Por ahora, el drift solo se inyecta en las features del grafo (X), 
        # dejando el vector de estado (Y) como Ground Truth ideal puro.
        return self.noise_model
    
    def get_qubit_features(self, qubit_idx: int, day_index: int) -> dict:
        base_q_props = self.base_properties.get(
            str(qubit_idx), 
            {"T1": 100e-6, "T2": 80e-6, "readout_error": 0.01}
        )
        
        decay_factor = 1.0 - (day_index * 0.02)
        
        return {
            "T1": base_q_props["T1"] * decay_factor,
            "T2": base_q_props["T2"] * decay_factor,
            "readout_error": min(1.0, base_q_props["readout_error"] * (1.0 + day_index * 0.1))
        }


class QuantumGraphExtractor:
    """
    Responsabilidad: Traducir el mundo cuántico (Qiskit) al mundo Deep Learning (PyTorch).
    Convierte un QuantumCircuit y sus telemetrías en un objeto Data de PyTorch Geometric.
    """
    def __init__(self, hardware_telemetrics: HardwareTelemetrics):
        self.hw = hardware_telemetrics
        self.gate_to_id = {'cx': 0, 'id': 1, 'rz': 2, 'sx': 3, 'x': 4, 'measure': 5}

    def circuit_to_graph(self, circuit: QuantumCircuit, ideal_statevector: np.ndarray, day_index: int) -> Data:
        dag = circuit_to_dag(circuit)
        
        node_features = []
        edge_indices = []
        node_idx = 0
        node_mapping = {} 
        
        for node in dag.op_nodes():
            gate_name = node.name
            if gate_name not in self.gate_to_id:
                continue 
                
            one_hot = [0] * len(self.gate_to_id)
            one_hot[self.gate_to_id[gate_name]] = 1
            
            params = [float(p) for p in node.op.params]
            params += [0.0] * (3 - len(params)) 
            
            # Localizamos el índice físico nativo del qubit usando la API de Qiskit
            q_control_idx = circuit.find_bit(node.qargs[0]).index
            
            phys_control = self.hw.get_qubit_features(q_control_idx, day_index)
            vec_control = [phys_control['T1'], phys_control['T2'], phys_control['readout_error']]
            
            if len(node.qargs) > 1:
                q_target_idx = circuit.find_bit(node.qargs[1]).index
                phys_target = self.hw.get_qubit_features(q_target_idx, day_index)
                vec_target = [phys_target['T1'], phys_target['T2'], phys_target['readout_error']]
            else:
                vec_target = vec_control.copy()
            
            feature_vector = one_hot + params + vec_control + vec_target
            node_features.append(feature_vector)
            
            node_mapping[node] = node_idx
            node_idx += 1

        for edge in dag.edges():
            source = edge[0]
            target = edge[1]
            if source in node_mapping and target in node_mapping:
                edge_indices.append([node_mapping[source], node_mapping[target]])

        x_tensor = torch.tensor(node_features, dtype=torch.float)
        
        if len(edge_indices) > 0:
            edge_index_tensor = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        else:
            edge_index_tensor = torch.empty((2, 0), dtype=torch.long)

        y_tensor = torch.tensor(np.abs(ideal_statevector)**2, dtype=torch.float)

        return Data(x=x_tensor, edge_index=edge_index_tensor, y=y_tensor)


class TFMDatasetPipeline:
    """
    Responsabilidad: El Orquestador MLOps.
    Llama a las clases anteriores para generar los miles de circuitos, 
    extrae los vectores de estado ideales y guarda los grafos en disco.
    """
    def __init__(self, output_dir: str | Path = PROCESSED_DATA_PATH, telemetrics: Optional[HardwareTelemetrics] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.telemetrics = telemetrics if telemetrics is not None else HardwareTelemetrics()
        self.factory = CircuitFactory(min_qubits=5, max_qubits=15)
        self.extractor = QuantumGraphExtractor(self.telemetrics)

    def generate_dataset(self, num_samples: int, day_index: int, split_name: str) -> None:
        print(f"\n🚀 Generando dataset '{split_name}' - Día {day_index} ({num_samples} muestras)...")
        
        # Limita dinámicamente el tamaño de los circuitos si el hardware de prueba es pequeño
        max_hardware_qubits = self.factory.max_qubits
        if self.telemetrics.coupling_map:
            max_hardware_qubits = min(
                self.factory.max_qubits, 
                max(max(edge) for edge in self.telemetrics.coupling_map) + 1
            )
            
        for i in tqdm(range(num_samples), desc="Procesando circuitos"):
            sample_seed = SEED + i + (day_index * 10000)
            rng = np.random.default_rng(sample_seed)
            
            num_qubits = int(rng.integers(self.factory.min_qubits, max_hardware_qubits + 1))
            is_structured = rng.random() < 0.2
            
            if is_structured:
                algo_type = rng.choice(['qaoa', 'qft', 'hea'])
                if algo_type == 'qaoa':
                    qc = self.factory.generate_qaoa(num_qubits, seed=sample_seed, coupling_map=self.telemetrics.coupling_map)
                elif algo_type == 'qft':
                    qc = self.factory.generate_qft(num_qubits, coupling_map=self.telemetrics.coupling_map)
                else:
                    qc = self.factory.generate_hea(num_qubits, seed=sample_seed, coupling_map=self.telemetrics.coupling_map)
            else:
                logical_depth = int(rng.integers(5, 51))
                qc = self.factory.generate_random_circuit(num_qubits, logical_depth, seed=sample_seed, coupling_map=self.telemetrics.coupling_map)
            
            ideal_statevector = np.array(Statevector.from_instruction(qc))
            graph_data = self.extractor.circuit_to_graph(circuit=qc, ideal_statevector=ideal_statevector, day_index=day_index)
            
            filename = self.output_dir / f"{split_name}_day{day_index}_sample{i}.pt"
            torch.save(graph_data, filename)
            
            del ideal_statevector
            del graph_data
            del qc
            if i % 100 == 0:
                gc.collect()