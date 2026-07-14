"""
Test suite — src/quantum_gen.py

Cómo interpretar los resultados:
  PASSED  ✅  El comportamiento es correcto tal como está.
  FAILED  ❌  Se ha detectado un bug real — leer el mensaje de error para saber qué arreglar.

Bugs conocidos que este fichero detecta (ROADMAP TAREA 1):
  BUG-1   HardwareTelemetrics.get_qubit_features() usa decay lineal, no saltos step-like
  BUG-2   CircuitFactory.generate_qaoa() usa operador trivial Z^n (sin términos ZZ por arista)
  BUG-3   CircuitFactory.generate_tfim() no existe
  BUG-4   TFMDatasetPipeline mezcla QAOA/QFT en los datos de entrenamiento
  BUG-5a  circuit_to_graph() produce vectores de 15 dims en lugar de 19
  BUG-5b  Puertas 1-qubit copian features del qubit control en el slot target (cero-pad incorrecto)

Ejecutar con:
    conda run -n tfm pytest tests/test_quantum_gen.py -v
"""

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ─── Constantes de test ───────────────────────────────────────────────────────

# Coupling map lineal de 5 qubits (suficiente para todos los tests, barato en RAM)
FAKE_COUPLING_MAP = [[0, 1], [1, 2], [2, 3], [3, 4]]
N_FAKE_QUBITS = 5
BASIS_GATES = {"cz", "id", "rz", "sx", "x"}  # Heron r2/r3 (CZ, no CX)


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def fake_raw_dir(tmp_path):
    """
    Crea un directorio raw/ falso con un ibm_kingston_calib.json mínimo.
    Permite instanciar HardwareTelemetrics sin ninguna conexión a IBM.
    """
    raw = tmp_path / "raw"
    raw.mkdir()
    props = {
        str(q): {
            "T1": 1e-4 + q * 5e-6,
            "T2": 8e-5 + q * 3e-6,
            "readout_error": 0.02 + q * 0.005,
            "gate_error": 1e-3 + q * 1e-4,
            "gate_length": 3.5e-8 + q * 1e-9,
        }
        for q in range(N_FAKE_QUBITS)
    }
    calib = raw / "ibm_kingston_calib.json"
    calib.write_text(
        json.dumps({"properties": props, "coupling_map": FAKE_COUPLING_MAP})
    )
    return raw


@pytest.fixture
def hw(fake_raw_dir, monkeypatch):
    """
    Instancia HardwareTelemetrics con calibración falsa.
    No requiere IBM_TOKEN ni conexión a internet.
    """
    monkeypatch.setattr("src.quantum_gen.RAW_DATA_PATH", str(fake_raw_dir))
    monkeypatch.setenv("IBM_TOKEN", "fake_token_for_tests")
    monkeypatch.setenv("IBM_INSTANCE_CRN", "fake_crn_for_tests")

    from src.quantum_gen import HardwareTelemetrics

    return HardwareTelemetrics(backend_name="ibm_kingston")


@pytest.fixture
def factory():
    from src.quantum_gen import CircuitFactory

    return CircuitFactory(min_qubits=3, max_qubits=5)


@pytest.fixture
def extractor(hw):
    from src.quantum_gen import QuantumGraphExtractor

    return QuantumGraphExtractor(hw)


def _ideal_probs(qc: QuantumCircuit) -> np.ndarray:
    """Devuelve |ψ|² como array numpy."""
    return np.abs(np.array(Statevector.from_instruction(qc))) ** 2


# ─── CircuitFactory ───────────────────────────────────────────────────────────


class TestCircuitFactory:
    """Tests para la generación de circuitos cuánticos."""

    # -- Comportamientos que deben funcionar ✅ --

    def test_random_circuit_returns_quantumcircuit(self, factory):
        qc = factory.generate_random_circuit(
            num_qubits=4, logical_depth=5, seed=42, coupling_map=FAKE_COUPLING_MAP
        )
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 4

    def test_random_circuit_uses_only_basis_gates(self, factory):
        qc = factory.generate_random_circuit(
            num_qubits=4, logical_depth=5, seed=42, coupling_map=FAKE_COUPLING_MAP
        )
        gate_names = {inst.operation.name for inst in qc.data}
        extra = gate_names - BASIS_GATES
        assert not extra, (
            f"El circuito aleatorio contiene puertas no-nativas tras transpilar: {extra}. "
            f"Revisa _transpile_and_check()."
        )

    def test_random_circuit_is_reproducible(self, factory):
        qc1 = factory.generate_random_circuit(
            4, 5, seed=99, coupling_map=FAKE_COUPLING_MAP
        )
        qc2 = factory.generate_random_circuit(
            4, 5, seed=99, coupling_map=FAKE_COUPLING_MAP
        )
        assert (
            qc1.count_ops() == qc2.count_ops()
        ), "Mismo seed produce circuitos distintos — la reproducibilidad está rota."

    def test_qft_returns_quantumcircuit(self, factory):
        qc = factory.generate_qft(num_qubits=4, coupling_map=FAKE_COUPLING_MAP)
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 4

    def test_qft_uses_only_basis_gates(self, factory):
        qc = factory.generate_qft(num_qubits=4, coupling_map=FAKE_COUPLING_MAP)
        gate_names = {inst.operation.name for inst in qc.data}
        extra = gate_names - BASIS_GATES
        assert not extra, f"QFT contiene puertas no-nativas: {extra}"

    def test_hea_returns_quantumcircuit(self, factory):
        qc = factory.generate_hea(num_qubits=4, seed=42, coupling_map=FAKE_COUPLING_MAP)
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 4

    def test_hea_uses_only_basis_gates(self, factory):
        qc = factory.generate_hea(num_qubits=4, seed=42, coupling_map=FAKE_COUPLING_MAP)
        gate_names = {inst.operation.name for inst in qc.data}
        extra = gate_names - BASIS_GATES
        assert not extra, f"HEA contiene puertas no-nativas: {extra}"

    def test_qaoa_returns_quantumcircuit(self, factory):
        """El circuito QAOA se genera sin errores (aunque el operador sea incorrecto — BUG-2)."""
        qc = factory.generate_qaoa(
            num_qubits=4, seed=42, coupling_map=FAKE_COUPLING_MAP
        )
        assert isinstance(qc, QuantumCircuit)

    def test_coupling_map_filter_removes_inactive_qubits(self, factory):
        """_transpile_and_check no debe añadir qubits fantasma del coupling map global."""
        qc = factory.generate_hea(num_qubits=3, seed=0, coupling_map=FAKE_COUPLING_MAP)
        assert qc.num_qubits == 3, (
            f"El circuito tiene {qc.num_qubits} qubits tras transpilar con un coupling map "
            f"de {N_FAKE_QUBITS} qubits — el filtrado de qubits activos falla."
        )

    # -- BUG-2 ✅ verificación de corrección --

    def test_bug2_qaoa_has_zz_interactions_per_edge(self, factory):
        """
        BUG-2 CORREGIDO — generate_qaoa() debe producir interacciones ZZ por arista.
        Se construye con CX + Rz + CX y el transpilador lo convierte a base Heron:
        cada CX → 1 CZ + puertas 1q, por lo tanto el circuito transpilado debe
        tener >= 2 * n_edges puertas CZ.
        """
        n = 4
        coupling_map = [[0, 1], [1, 2], [2, 3]]
        n_edges = len(coupling_map)

        qc = factory.generate_qaoa(n, seed=42, reps=1, coupling_map=coupling_map)
        cz_count = qc.count_ops().get("cz", 0)
        min_expected_cz = 2 * n_edges  # cada ZZ → CX+Rz+CX → CZ+1q+Rz+1q+CZ

        assert cz_count >= min_expected_cz, (
            f"QAOA tiene {cz_count} puertas CZ, se esperaban >= {min_expected_cz}. "
            f"Con {n_edges} aristas y reps=1 debería haber al menos {2*n_edges} CZ "
            f"(una por cada lado del ZZ rotation). El operador puede seguir siendo trivial."
        )

    # -- BUG-3 ❌ --

    def test_bug3_generate_tfim_does_not_exist(self, factory):
        """
        BUG-3 ❌ — CircuitFactory no tiene método generate_tfim().
        Es necesario para el benchmark QEM-Bench y comparación con QEMFormer (ICML 2025).

        Este test FALLA hasta que se corrija BUG-3.
        """
        assert hasattr(factory, "generate_tfim"), (
            "BUG-3: CircuitFactory.generate_tfim() no existe. "
            "Implementar generador de circuitos Trotterized TFIM "
            "(simulación del Hamiltoniano Ising con campo transverso). "
            "Benchmark estándar de QEM-Bench — necesario para comparar con el SOTA."
        )

    def test_bug3_generate_tfim_returns_valid_circuit(self, factory):
        """
        BUG-3 ❌ — Aunque generate_tfim() existiera, debe devolver un QuantumCircuit válido.
        Este test FALLA hasta que se corrija BUG-3 (el método no existe aún).
        """
        if not hasattr(factory, "generate_tfim"):
            pytest.fail(
                "BUG-3: generate_tfim() no existe — este test no puede ejecutarse. "
                "Implementar el método antes de validar su output."
            )
        qc = factory.generate_tfim(
            num_qubits=4, seed=42, trotter_steps=3, coupling_map=FAKE_COUPLING_MAP
        )
        assert isinstance(qc, QuantumCircuit)
        gate_names = {inst.operation.name for inst in qc.data}
        assert not (
            gate_names - BASIS_GATES
        ), f"generate_tfim() produce puertas no-nativas: {gate_names - BASIS_GATES}"


# ─── HardwareTelemetrics ──────────────────────────────────────────────────────


class TestHardwareTelemetrics:
    """Tests para la simulación del hardware IBM (calibración y drift)."""

    # -- Comportamientos que deben funcionar ✅ --

    def test_loads_properties_from_cache(self, hw):
        assert hw.base_properties is not None
        assert len(hw.base_properties) == N_FAKE_QUBITS, (
            f"Se esperaban {N_FAKE_QUBITS} qubits en base_properties, "
            f"se encontraron {len(hw.base_properties)}."
        )

    def test_loads_coupling_map_from_cache(self, hw):
        assert hw.coupling_map == FAKE_COUPLING_MAP, (
            f"coupling_map cargado: {hw.coupling_map} — "
            f"esperado: {FAKE_COUPLING_MAP}."
        )

    def test_get_qubit_features_returns_required_keys(self, hw):
        feats = hw.get_qubit_features(0, day_index=0)
        for key in ("T1", "T2", "readout_error"):
            assert key in feats, (
                f"get_qubit_features() no devuelve la clave '{key}'. "
                f"Claves presentes: {list(feats.keys())}"
            )

    def test_get_qubit_features_values_are_positive(self, hw):
        for q in range(N_FAKE_QUBITS):
            feats = hw.get_qubit_features(q, day_index=0)
            assert feats["T1"] > 0, f"T1 <= 0 para qubit {q} en day=0"
            assert feats["T2"] > 0, f"T2 <= 0 para qubit {q} en day=0"

    def test_readout_error_bounded_all_days(self, hw):
        """readout_error debe mantenerse en [0, 1] para todos los días y qubits."""
        violations = []
        for day in range(10):
            for q in range(N_FAKE_QUBITS):
                re = hw.get_qubit_features(q, day)["readout_error"]
                if not (0.0 <= re <= 1.0):
                    violations.append(f"qubit={q} day={day} readout_error={re:.4f}")
        assert not violations, (
            f"readout_error fuera del rango [0,1] en {len(violations)} caso(s):\n"
            + "\n".join(violations)
        )

    def test_t1_changes_across_days(self, hw):
        """T1 debe cambiar entre day=0 y day=9 — el drift debe aplicarse de alguna forma."""
        t1_day0 = hw.get_qubit_features(0, day_index=0)["T1"]
        t1_day9 = hw.get_qubit_features(0, day_index=9)["T1"]
        assert t1_day0 != t1_day9, (
            "T1 es idéntico en day=0 y day=9 — "
            "el drift temporal no se está aplicando en absoluto."
        )

    # -- BUG-1 ❌ --

    def test_bug1_drift_is_perfectly_linear(self, hw):
        """
        BUG-1 ❌ — get_qubit_features() aplica un decay lineal uniforme
        (decay_factor = 1.0 - day_index * 0.02) en lugar de saltos step-like.

        Hirasaki et al. (APL 2023) demuestra que el drift en IBM es escalonado:
        2-3 saltos abruptos de ±15-30%, no una caída gradual y constante.

        Este test FALLA hasta que se corrija BUG-1.
        """
        t1_per_day = [hw.get_qubit_features(0, day)["T1"] for day in range(10)]
        diffs = [t1_per_day[i + 1] - t1_per_day[i] for i in range(9)]

        # Drift lineal: todas las diferencias son iguales (pendiente constante)
        is_perfectly_linear = all(abs(d - diffs[0]) < 1e-15 for d in diffs)

        assert not is_perfectly_linear, (
            f"BUG-1: T1 del qubit 0 decrece con paso perfectamente constante día a día "
            f"(diffs = {[f'{d:.3e}' for d in diffs]}). "
            f"Esto es decay lineal — no step-like. "
            f"Fix: generar 2-3 saltos aleatorios de ±15-30% en posiciones aleatorias "
            f"del timeline de 10 días. El decay lineal anterior está descartado."
        )

    def test_bug1_no_negative_t1_after_decay(self, hw):
        """Independientemente del modelo de drift, T1 nunca puede ser negativo."""
        negatives = []
        for day in range(10):
            for q in range(N_FAKE_QUBITS):
                t1 = hw.get_qubit_features(q, day)["T1"]
                if t1 <= 0:
                    negatives.append(f"qubit={q} day={day} T1={t1:.3e}")
        assert not negatives, (
            f"T1 <= 0 en {len(negatives)} caso(s) — "
            f"el modelo de drift produce valores físicamente imposibles:\n"
            + "\n".join(negatives)
        )


# ─── QuantumGraphExtractor ────────────────────────────────────────────────────


class TestQuantumGraphExtractor:
    """Tests para la conversión circuito → grafo PyTorch Geometric."""

    @pytest.fixture
    def hea_graph(self, factory, extractor):
        """Grafo de un HEA de 4 qubits en day=0. Usa delta=0.05 fijo para tests estructurales."""
        qc = factory.generate_hea(num_qubits=4, seed=42, coupling_map=FAKE_COUPLING_MAP)
        probs = _ideal_probs(qc)
        return extractor.circuit_to_graph(
            qc, delta=0.05, day_index=0, ideal_probs=probs
        )

    # -- Comportamientos que deben funcionar ✅ --

    def test_returns_pyg_data_object(self, hea_graph):
        from torch_geometric.data import Data

        assert isinstance(hea_graph, Data), (
            f"circuit_to_graph() devuelve {type(hea_graph)}, "
            f"se esperaba torch_geometric.data.Data."
        )

    def test_has_node_feature_matrix(self, hea_graph):
        assert hea_graph.x is not None
        assert hea_graph.x.ndim == 2
        assert hea_graph.x.shape[0] > 0, "El grafo no tiene nodos."

    def test_node_features_are_float32(self, hea_graph):
        assert (
            hea_graph.x.dtype == torch.float32
        ), f"x.dtype = {hea_graph.x.dtype}, se esperaba torch.float32."

    def test_has_valid_edge_index(self, hea_graph):
        assert hea_graph.edge_index is not None
        assert (
            hea_graph.edge_index.shape[0] == 2
        ), "edge_index debe tener forma [2, num_edges]."

    def test_edge_indices_within_node_count(self, hea_graph):
        n_nodes = hea_graph.x.shape[0]
        if hea_graph.edge_index.shape[1] > 0:
            max_idx = hea_graph.edge_index.max().item()
            assert max_idx < n_nodes, (
                f"edge_index referencia el nodo {max_idx} pero sólo hay {n_nodes} nodos. "
                f"Hay aristas apuntando a nodos que no existen."
            )

    def test_has_target_y(self, hea_graph):
        """y = Δ (GEM target): tensor de shape [1] con un float finito."""
        assert hea_graph.y is not None
        assert hea_graph.y.shape == torch.Size(
            [1]
        ), f"y debe tener forma [1] (Δ escalar), tiene {hea_graph.y.shape}."
        assert torch.isfinite(hea_graph.y).all(), "y contiene NaN o Inf."

    def test_ideal_probs_stored_when_provided(self, factory, extractor):
        """Cuando se pasan ideal_probs, se guardan en data.ideal_probs."""
        qc = factory.generate_hea(num_qubits=4, seed=42, coupling_map=FAKE_COUPLING_MAP)
        probs = _ideal_probs(qc)
        graph = extractor.circuit_to_graph(
            qc, delta=0.05, day_index=0, ideal_probs=probs
        )
        assert hasattr(
            graph, "ideal_probs"
        ), "data.ideal_probs no está en el objeto Data."
        assert (
            abs(graph.ideal_probs.sum().item() - 1.0) < 1e-4
        ), "ideal_probs no suma 1.0 — verificar que es |ψ|²."

    def test_graph_with_different_days_is_different(self, factory, extractor):
        """Mismo circuito en days distintos debe producir features distintas (drift aplicado)."""
        qc = factory.generate_hea(num_qubits=3, seed=1, coupling_map=FAKE_COUPLING_MAP)
        probs = _ideal_probs(qc)
        g0 = extractor.circuit_to_graph(qc, delta=0.05, day_index=0, ideal_probs=probs)
        g5 = extractor.circuit_to_graph(qc, delta=0.05, day_index=5, ideal_probs=probs)
        assert not torch.allclose(g0.x, g5.x), (
            "El grafo es idéntico para day=0 y day=5 — "
            "el drift temporal no se refleja en las features de los nodos."
        )

    # -- BUG-5a ❌ --

    def test_bug5a_feature_vector_has_15_not_19_dims(self, hea_graph):
        """
        BUG-5a ❌ — circuit_to_graph() produce vectores de 15 dimensiones.
        El diseño objetivo son 19 dims:
          [0:6]   one-hot tipo de puerta (6 dims)
          [6:9]   ángulos paramétricos θ, φ, λ (3 dims)
          [9:12]  T1, T2, readout_error del qubit control (3 dims)
          [12:13] gate_error del qubit control — FALTA
          [13:14] gate_length del qubit control — FALTA
          [14:17] T1, T2, readout_error del qubit target (3 dims, cero si 1q)
          [17:18] day_index normalizado — FALTA
          [18:19] depth_position normalizado — FALTA

        Este test FALLA hasta que se corrija BUG-5a.
        """
        actual_dims = hea_graph.x.shape[1]
        expected_dims = 19
        assert actual_dims == expected_dims, (
            f"BUG-5a: Feature vector tiene {actual_dims} dims, se esperaban {expected_dims}. "
            f"Faltan {expected_dims - actual_dims} features: "
            f"gate_error, gate_length, day_index (normalizado), depth_position (normalizado). "
            f"Fix: añadir estas 4 features en QuantumGraphExtractor.circuit_to_graph()."
        )

    # -- BUG-5b ❌ --

    def test_bug5b_onequbit_gate_target_slot_is_copy_not_zeropad(
        self, factory, extractor
    ):
        """
        BUG-5b (CORREGIDO en TAREA 1) — Regresión: para puertas de 1 qubit, el slot
        del qubit target (dims 14:17 en el layout de 19 dims) debe ser zero-pad
        [0.0, 0.0, 0.0], nunca una copia de las features del qubit control.

        El código buggy histórico: vec_target = vec_control.copy()  (layout 15 dims)
        El código correcto:        vec_target = [0.0, 0.0, 0.0]
        """
        qc = factory.generate_hea(num_qubits=4, seed=42, coupling_map=FAKE_COUPLING_MAP)
        probs = _ideal_probs(qc)
        graph = extractor.circuit_to_graph(
            qc, delta=0.05, day_index=0, ideal_probs=probs
        )
        x = graph.x

        # One-hot: cz=col0, id=col1, rz=col2, sx=col3, x=col4, measure=col5
        # Puertas de 1 qubit: todo lo que NO es cz (col 0) ni measure (col 5)
        is_cz = x[:, 0].bool()
        is_measure = x[:, 5].bool()
        is_one_qubit = ~(is_cz | is_measure)

        if is_one_qubit.sum() == 0:
            pytest.skip(
                "No se encontraron nodos de puerta 1-qubit en este circuito. "
                "Cambiar el seed o el tipo de circuito."
            )

        one_qubit_nodes = x[is_one_qubit]

        # Layout 19 dims: [0:6 one-hot][6:9 params][9:12 ctrl T1/T2/re][12:13 gate_error]
        #                  [13:14 gate_length][14:17 target T1/T2/re][17:18 day][18:19 depth]
        actual_dims = x.shape[1]
        if actual_dims == 19:
            control_feats = one_qubit_nodes[:, 9:12]  # T1/T2/re del control (19-dim)
            target_feats = one_qubit_nodes[:, 14:17]  # T1/T2/re del target (19-dim)
        else:
            control_feats = one_qubit_nodes[:, 9:12]  # T1/T2/re del control (15-dim)
            target_feats = one_qubit_nodes[
                :, 12:15
            ]  # T1/T2/re del target (15-dim, buggy)

        zero_pad = torch.zeros_like(target_feats)

        is_zero_padded = torch.allclose(target_feats, zero_pad, atol=1e-8)
        is_copy_of_control = torch.allclose(target_feats, control_feats, atol=1e-8)

        assert is_zero_padded, (
            f"BUG-5b: Slot del qubit target en puertas 1-qubit "
            f"{'copia del control' if is_copy_of_control else 'valores inesperados'} "
            f"en lugar de [0.0, 0.0, 0.0]. "
            f"Fix: cambiar 'vec_target = vec_control.copy()' por "
            f"'vec_target = [0.0, 0.0, 0.0]' para nodos con len(node.qargs) == 1."
        )


# ─── TFMDatasetPipeline ───────────────────────────────────────────────────────


class TestTFMDatasetPipeline:
    """Tests para el orquestador MLOps de generación de datasets."""

    # -- BUG-4 ❌ --

    def test_bug4_generate_dataset_has_no_zeroshot_routing(self):
        """
        BUG-4 ❌ — generate_dataset() no tiene lógica para enrutar circuitos QAOA/QFT
        a carpetas separadas. Todo va al mismo output_dir sin distinción de tipo.

        El código correcto debe:
          - Guardar Random/HEA/TFIM → data/processed/train/ (o val/ o test/)
          - Guardar QAOA → data/processed/zeroshot_qaoa/  (NUNCA en train/)
          - Guardar QFT  → data/processed/zeroshot_qft/   (NUNCA en train/)

        Este test FALLA hasta que se corrija BUG-4.
        """
        from src.quantum_gen import TFMDatasetPipeline

        source = inspect.getsource(TFMDatasetPipeline.generate_dataset)

        assert "zeroshot_qaoa" in source, (
            "BUG-4: generate_dataset() no contiene ninguna referencia a 'zeroshot_qaoa'. "
            "QAOA nunca debe ir al conjunto de entrenamiento."
        )
        assert "zeroshot_qft" in source, (
            "BUG-4: generate_dataset() no contiene ninguna referencia a 'zeroshot_qft'. "
            "QFT nunca debe ir al conjunto de entrenamiento."
        )

    def test_bug4_qaoa_and_qft_in_structured_pool(self):
        """
        BUG-4 ❌ — El código actual incluye 'qaoa' y 'qft' como opciones posibles
        para cualquier split (incluyendo train), lo que contamina la evaluación zero-shot.

        Tras la corrección, el pool de structured circuits para training debe ser
        sólo ['hea', 'tfim']. QAOA y QFT deben generarse en pipelines separados.

        Este test FALLA hasta que se corrija BUG-4.
        """
        from src.quantum_gen import TFMDatasetPipeline

        source = inspect.getsource(TFMDatasetPipeline.generate_dataset)

        # El código buggy tiene: rng.choice(['qaoa', 'qft', 'hea'])
        # El código correcto nunca debe tener 'qaoa' o 'qft' en el pool de training
        has_qaoa_in_choice = "rng.choice" in source and "'qaoa'" in source
        has_zeroshot_guard = "zeroshot_qaoa" in source

        if has_qaoa_in_choice and not has_zeroshot_guard:
            pytest.fail(
                "BUG-4: 'qaoa' aparece como opción en rng.choice() sin ningún guard "
                "que lo enrute a zeroshot_qaoa/. "
                "Fix: eliminar 'qaoa' y 'qft' del pool de training. "
                "Generarlos en métodos separados que escriban a las carpetas zero-shot."
            )

    # -- Comportamientos que deben funcionar ✅ --

    def test_pipeline_instantiation_without_ibm(self, hw, tmp_path):
        """El pipeline se instancia sin errores ni conexión a IBM."""
        from src.quantum_gen import TFMDatasetPipeline

        pipeline = TFMDatasetPipeline(output_dir=tmp_path, telemetrics=hw)
        assert pipeline.factory is not None
        assert pipeline.extractor is not None
        assert pipeline.telemetrics is hw

    def test_generate_dataset_creates_correct_number_of_files(self, hw, tmp_path):
        """
        Smoke test: generar 3 muestras y verificar que se crean exactamente 3 ficheros .pt.
        Usa circuitos pequeños (max 5 qubits por coupling map) para ser rápido.
        """
        from src.quantum_gen import TFMDatasetPipeline

        pipeline = TFMDatasetPipeline(output_dir=tmp_path, telemetrics=hw)
        pipeline.generate_dataset(num_samples=3, day_index=1, split_name="smoke_test")

        pt_files = list(tmp_path.glob("**/*.pt"))
        assert len(pt_files) == 3, (
            f"Se esperaban 3 ficheros .pt, se encontraron {len(pt_files)}. "
            f"El pipeline puede haber fallado silenciosamente."
        )

    def test_saved_pt_files_are_valid_data_objects(self, hw, tmp_path):
        """Cada .pt guardado debe ser un objeto Data de PyTorch Geometric válido."""
        from src.quantum_gen import TFMDatasetPipeline
        from torch_geometric.data import Data

        pipeline = TFMDatasetPipeline(output_dir=tmp_path, telemetrics=hw)
        pipeline.generate_dataset(
            num_samples=2, day_index=0, split_name="validity_check"
        )

        for pt_file in sorted(tmp_path.glob("**/*.pt")):
            obj = torch.load(pt_file, weights_only=False)
            assert isinstance(obj, Data), (
                f"{pt_file.name} no es un objeto torch_geometric.data.Data. "
                f"Tipo encontrado: {type(obj)}"
            )
            assert (
                obj.x is not None and obj.x.shape[0] > 0
            ), f"{pt_file.name} contiene un grafo sin nodos."

    def test_saved_files_follow_naming_convention(self, hw, tmp_path):
        """Los ficheros deben seguir el patrón {split_name}_day{d}_sample{i}.pt"""
        from src.quantum_gen import TFMDatasetPipeline

        pipeline = TFMDatasetPipeline(output_dir=tmp_path, telemetrics=hw)
        pipeline.generate_dataset(num_samples=2, day_index=3, split_name="train")

        pt_files = list(tmp_path.glob("**/*.pt"))
        for f in pt_files:
            assert "train" in f.name, f"Nombre inesperado: {f.name}"
            assert "day3" in f.name, f"No contiene 'day3': {f.name}"

    # -- Fix leakage entre splits (detectado en EDA de la migración Heron) --

    def test_no_leakage_between_splits_same_day(self, hw, tmp_path):
        """
        REGRESIÓN — antes de este fix, dos splits distintos con el mismo day_index
        generaban circuitos IDÉNTICOS (mismo sample_seed = SEED + i + day*10000,
        sin componente de split). Esto rompía val/test en modo mini (mismos días
        que train). _split_seed_offset() añade un offset por nombre de split.
        """
        from src.quantum_gen import TFMDatasetPipeline

        pipeline = TFMDatasetPipeline(output_dir=tmp_path, telemetrics=hw)
        pipeline.generate_dataset(num_samples=1, day_index=1, split_name="train")
        pipeline.generate_dataset(num_samples=1, day_index=1, split_name="val")

        d_train = torch.load(
            tmp_path / "train" / "train_day1_sample0.pt", weights_only=False
        )
        d_val = torch.load(tmp_path / "val" / "val_day1_sample0.pt", weights_only=False)

        assert not torch.equal(d_train.x, d_val.x), (
            "train y val generan el circuito IDÉNTICO para el mismo día — "
            "leakage de seeds entre splits (fix de _split_seed_offset no aplicado)."
        )
