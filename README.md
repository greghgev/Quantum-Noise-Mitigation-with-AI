# TFM-Quantum: Mitigación de Ruido Cuántico mediante Deep Learning

Pipeline de Deep Learning para mitigación integral de errores en hardware cuántico NISQ (IBM Eagle, 127 qubits). Separa el ruido de puertas lógicas (dinámico, pre-ejecución) del ruido de lectura (estocástico, post-ejecución) mediante dos módulos especializados entrenados de forma desacoplada.

---

## Pipeline Arquitectónico

```
Circuito → [GEM] → Δ predicho → IBM QPU (intacto) → [REM] → ⟨O⟩_mit = ⟨O⟩_noisy − Δ
```

### Módulo 1 — GEM (Gate Error Mitigation)

Predice Δ **antes** de ejecutar el circuito en el QPU. El circuito se representa como un DAG donde cada nodo es una puerta lógica con un vector de 19 features (tipo de puerta, qubits involucrados, error de puerta, longitud de puerta, día del experimento, posición en profundidad...).

Arquitectura: **Graph Transformer** con un nodo virtual QCR (Quantum Circuit Representative) conectado a todos los nodos del grafo. El QCR actúa como resumen global del circuito — su embedding se usa directamente para predecir Δ. Sin message-passing (estilo GTraQEM, no MPNN).

Ventaja diferencial: al predecir Δ antes de la ejecución, el GEM puede actuar como **filtro de calidad del circuito** (rechazar circuitos con Δ demasiado alto) y mantiene una separación limpia entre tipos de error.

### Módulo 2 — REM (Readout Error Mitigation)

Mitiga el ruido de lectura **después** de recibir los counts del QPU. Predice matrices locales de confusión 2×2 para pares de qubits vecinos (no la matriz global 2ⁿ×2ⁿ). Resuelve el sistema lineal con **GMRES Matrix-Free** limitado al subespacio de bitstrings observados → O(N) en memoria en lugar de O(2^{3n}).

### Post-procesado

```
⟨O⟩_mit = ⟨O⟩_noisy − Δ
```

El valor esperado mitigado se calcula restando al resultado del circuito ruidoso la desviación predicha por el GEM, tras pasar la distribución por el REM.

---

## Estructura del Proyecto

```
TFM-Quantum/
├── configs/
│   ├── gem_config.yaml         # Hiperparámetros GEM (Graph Transformer)
│   └── rem_config.yaml         # Hiperparámetros REM (GNN + GMRES)
├── data/
│   ├── raw/                    # Calibración IBM (ibm_kyiv_calib.json, historial T1/T2)
│   └── processed/
│       ├── train/              # Circuitos random, HEA, Trotterized TFIM (días 1–7)
│       ├── val/                # In-distribution validation (día 6 OOD temporal)
│       ├── test/               # In-distribution test (días 8–10)
│       ├── zeroshot_qaoa/      # QAOA — nunca visto en entrenamiento
│       └── zeroshot_qft/       # QFT  — nunca visto en entrenamiento (contribución original)
├── doc/
│   ├── SOTA/
│   │   ├── comparative_analysis.md   # Análisis comparativo de 13 papers vs decisiones TFM
│   │   ├── TFM-SOTA-Parte1.pdf
│   │   └── TFM-SOTA-Parte2.pdf
│   ├── dataset_info.md               # Decisiones de dataset y métricas
│   ├── src_info.md                   # Documentación de src/
│   ├── notebooks_info.md             # Guía de notebooks
│   └── fallas_y_soluciones.md        # Bugs conocidos y soluciones
├── figures/                    # Gráficas generadas (radar, fidelidad, drift)
├── logs/                       # Artefactos MLflow
├── models/                     # Checkpoints entrenados (.pt)
├── notebooks/
│   ├── 01_eda_ibm_telemetry.ipynb              # EDA + drift step-like
│   ├── 02_quantum_circuit_prototyping.ipynb    # Validación de los 5 tipos de circuito
│   ├── 03_gem_transformer_evaluation.ipynb     # Métricas GEM por split
│   ├── 04_rem_gnn_matrix_free.ipynb            # Métricas REM por split
│   └── 05_final_pipeline_tradeoffs.ipynb       # Pipeline completo + ablación 70/30/80/20/90/10
├── scripts/                    # Entry-points ejecutables (importan de src/)
│   ├── generate.py             # Genera el dataset (--mini o --full)
│   ├── train_gem.py            # Entrena el GEM
│   ├── train_rem.py            # Entrena el REM
│   └── evaluate.py             # Evalúa el pipeline completo por split
├── src/                        # Código fuente modular (lógica de negocio)
│   ├── config.py               # Rutas, constantes físicas, semillas
│   ├── quantum_gen.py          # CircuitFactory, HardwareTelemetrics, TFMDatasetPipeline
│   ├── dataset.py              # PyG Dataset + DataLoaders (in-dist + zero-shot)
│   ├── gem_model.py            # Graph Transformer con nodo QCR
│   ├── rem_model.py            # GNN local 2×2 + GMRES Matrix-Free
│   ├── train.py                # Entrenamiento desacoplado GEM/REM + MLflow + normalización Δ
│   ├── inference.py            # Pipeline producción: GEM → QPU → REM
│   └── utils.py                # Métricas, GMRES wrapper, encodings
├── .env                        # IBM_TOKEN + IBM_INSTANCE_CRN (no versionado)
├── .gitignore
├── .pre-commit-config.yaml     # black + flake8 + checks automáticos
├── .dvcignore
├── CLAUDE.md                   # Instrucciones de contexto para Claude Code
├── IDEA_GENERAL_TFM.md         # Descripción completa del TFM en lenguaje claro
├── IDEAS_FUTURAS.md            # 7 ideas de mejora post-TFM
├── ROADMAP.md                  # Estado de implementación por tareas
├── environment.yml             # Dependencias conda (Python 3.10, Qiskit 1.x, PyG)
└── README.md
```

---

## Dataset y Estrategia de Entrenamiento

### Tipos de circuito

| Tipo | Split | Descripción |
|------|-------|-------------|
| Random | Train (40%) | Circuitos aleatorios sobre el coupling map de ibm_kyiv |
| HEA | Train (40%) | Hardware-Efficient Ansatz — bloques Ry+CNOT por capa |
| Trotterized TFIM | Train (20%) | Transverse-Field Ising Model — benchmark estándar QEM-Bench |
| QAOA | **Zero-shot A** | Nunca en entrenamiento — reservado para test de generalización |
| QFT | **Zero-shot B** | Nunca en entrenamiento — **contribución original** del TFM |

La proporción 80/20 (random+HEA / TFIM) se ablacionará: 70/30, 80/20, 90/10.

### Split temporal (OOD)

10 días simulados de calibración IBM con **drift step-like** (Hirasaki et al., APL 2023): 2–3 saltos aleatorios de ±15–30% en T1/T2/readout_error a lo largo del timeline. El modelo debe generalizar a días de calibración no vistos.

| Split | Días | Propósito |
|-------|------|-----------|
| Train | 1–5 | Aprendizaje |
| Validation | 6 | Early stopping + tuning |
| Test OOD | 8–10 | Evaluación in-distribution |
| Zero-shot QAOA | — | Generalización a circuito nuevo |
| Zero-shot QFT | — | Generalización a circuito nuevo |

### Drift step-like

```
Error (T1/T2)
│
│  ─────┐
│       │
│       └──────┐
│              │
│              └─────────
└──────────────────────── tiempo (días)
```

Modelo basado en evidencia empírica de hardware IBM real (vs. decay lineal previo, descartado).

---

## Métricas de Evaluación

### GEM

| Métrica | Descripción |
|---------|-------------|
| MAE | Error absoluto medio en predicción de Δ |
| RMSE | Error cuadrático medio en predicción de Δ |
| R² | Coeficiente de determinación |
| Mejora relativa | `(MAE_noisy − MAE_mit) / MAE_noisy` |

### REM

| Métrica | Descripción |
|---------|-------------|
| Hellinger Fidelity (HF) | Similaridad entre distribuciones de bitstrings (primaria) |
| Total Variation Distance (TVD) | Distancia L1 entre distribuciones |
| Ratio HF | `HF_mit / HF_noisy` — mejora relativa de la distribución |

Todas las métricas se reportan separadas por split: in-distribution / zero-shot QAOA / zero-shot QFT / por día.

---

## MLOps

- **DVC**: versiona datasets y modelos grandes. Remote local en `~/TFM/dvc-remote/`
- **MLflow**: tracking de experimentos, métricas y checkpoints
- **pre-commit**: `black` (formateo) + `flake8` (linting) se ejecutan automáticamente en cada commit
- **configs/**: YAMLs de hiperparámetros para reproducibilidad total

---

## Instalación

```bash
# 1. Crear el entorno conda
conda env create -f environment.yml
conda activate tfm

# 2. Instalar hooks de pre-commit
pre-commit install

# 3. Configurar credenciales IBM (no commitear)
cp .env.example .env   # añadir IBM_TOKEN e IBM_INSTANCE_CRN
```

## Uso

```bash
# Generar dataset pequeño (150 muestras, validación rápida)
conda run -n tfm python scripts/generate.py --mini

# Generar dataset completo (5000 muestras, 10 días)
conda run -n tfm python scripts/generate.py --full

# Versionar con DVC
dvc add data/processed/
dvc push

# Entrenar
conda run -n tfm python scripts/train_gem.py
conda run -n tfm python scripts/train_rem.py

# Evaluar (todos los splits)
conda run -n tfm python scripts/evaluate.py \
    --gem models/gem_best.pt \
    --rem models/rem_best.pt \
    --split all
```

---

## Hardware objetivo

- **Dispositivo**: `ibm_kyiv` — familia Eagle, 127 qubits, topología Heavy-Hex
- **Canal IBM**: `ibm_quantum_platform` (nuevo, no el deprecated `ibm_quantum`)
- **Validación real**: 20–30 circuitos en hardware real para medir la brecha simulation-to-real (TAREA 6)

---

## Referencias clave

- **GTraQEM** — Bao et al., ICLR 2025: Graph Transformer con nodo QCR para mitigación pre-ejecución
- **QEMFormer** — Nguyen et al., NeurIPS 2024: Transformer post-ejecución sobre distribución de bitstrings
- **Concept Drift step-like** — Hirasaki et al., APL 2023: Evidencia empírica de saltos abruptos en calibración IBM
- **Q-Cluster** — Benchmark de clustering de circuitos cuánticos
- Lista completa de 13 papers en `doc/SOTA/comparative_analysis.md`
